#!/usr/bin/env python3
#
# G34 First Flight Node
#
# Slow, controlled first flight / tuning script for PX4 Offboard:
#  - pre-offboard streaming
#  - slow takeoff to configurable altitude
#  - hover
#  - optional altitude / attitude tuning modes
#  - final descend / auto-land / placeholder precision-land
#  - CSV logging for analysis (SITL + hardware)
#

import os
import csv
import math
from enum import Enum
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleLandDetected,
)


class FlightPhase(Enum):
    PREFLIGHT = 0         # pre-offboard streaming on ground
    TAKEOFF_ASCEND = 1    # slow ramp to target altitude
    HOVER = 2             # hold altitude
    ALT_TUNING = 3        # altitude step for tuning
    ATT_TUNING = 4        # simple attitude/horizontal excitation (placeholder)
    FINAL_DESCEND = 5     # controlled descend to ground
    FINAL_AUTO_LAND = 6   # handoff to PX4 AUTO.LAND
    FINAL_PREC_LAND = 7   # placeholder hook for precision landing / tracktor-beam
    DONE = 8              # mission complete


class G34FirstFlightNode(Node):
    def __init__(self):
        super().__init__("g34_first_flight_node")

        # === Parameters ===
        # NOTE: do NOT declare use_sim_time here; rclpy/Node does that already.
        self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position_v1")
        self.declare_parameter("takeoff_altitude_m", 0.5)
        self.declare_parameter("tuning_mode", "none")       # "none" | "altitude" | "attitude"
        self.declare_parameter("final_mode", "descend")     # "descend" | "auto_land" | "precision_land"
        self.declare_parameter("enable_csv_logging", True)

        # use_sim_time is special: just read it if present
        if self.has_parameter("use_sim_time"):
            self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        else:
            self.use_sim_time = False

        self.local_position_topic = self.get_parameter(
            "local_position_topic"
        ).get_parameter_value().string_value

        self.takeoff_altitude_m = float(
            self.get_parameter("takeoff_altitude_m").get_parameter_value().double_value
        )

        self.tuning_mode = self.get_parameter(
            "tuning_mode"
        ).get_parameter_value().string_value.lower()

        self.final_mode = self.get_parameter(
            "final_mode"
        ).get_parameter_value().string_value.lower()

        self.enable_csv_logging = self.get_parameter(
            "enable_csv_logging"
        ).get_parameter_value().bool_value

        # Clamp / sanitize parameters
        if self.takeoff_altitude_m < 0.3:
            self.get_logger().warn(
                f"takeoff_altitude_m={self.takeoff_altitude_m:.2f} too low, clamping to 0.3 m"
            )
            self.takeoff_altitude_m = 0.3

        if self.tuning_mode not in ("none", "altitude", "attitude"):
            self.get_logger().warn(
                f"tuning_mode='{self.tuning_mode}' invalid, using 'none'"
            )
            self.tuning_mode = "none"

        if self.final_mode not in ("descend", "auto_land", "precision_land"):
            self.get_logger().warn(
                f"final_mode='{self.final_mode}' invalid, using 'descend'"
            )
            self.final_mode = "descend"

        # === Publishers ===
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

        # === Subscribers (use sensor QoS to match microRTPS topics) ===
        self.local_position = None
        self.alt_up_m = None       # up-positive altitude (m), from local_position.z
        self.vz_ned = None         # vertical velocity in NED (m/s)
        self.yaw_rad = 0.0

        self.landed = True

        self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self.local_position_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self.land_detected_callback,
            qos_profile_sensor_data,
        )

        # === Timing and state machine ===
        self.timer_period = 0.1  # seconds
        self.node_start_time = self.get_clock().now().nanoseconds / 1e9
        self.last_time = self.node_start_time

        self.phase = FlightPhase.PREFLIGHT
        self.phase_start_time = self.node_start_time

        # Offboard / arm bookkeeping
        self.offboard_command_sent = False
        self.arm_command_sent = False

        # Vertical setpoint in NED frame (z positive down)
        self.z_sp_ned = 0.0

        # Simple motion tuning parameters (can be tweaked later)
        self.pre_offboard_stream_time = 1.0      # s of setpoints before OFFBOARD request
        self.takeoff_ascent_rate = 0.30          # m/s effective setpoint ramp
        self.takeoff_timeout = 20.0              # s, max time allowed in TAKEOFF_ASCEND

        self.hover_time = 10.0                   # s in HOVER before tuning / final
        self.alt_tune_step_m = 0.3               # m altitude step for tuning
        self.alt_tune_duration = 10.0            # s at step altitude

        self.descend_rate = 0.25                 # m/s descent setpoint ramp
        self.descend_timeout = 20.0              # s max descend phase
        self.land_alt_threshold = 0.05           # m alt_up considered "landed" for our logic

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        self.setup_csv_logging()

        self.get_logger().info(
            f"tuning_mode='{self.tuning_mode}', final_mode='{self.final_mode}', "
            f"takeoff_altitude_m={self.takeoff_altitude_m:.2f} m"
        )

        # Timer for main control loop
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("G34 First Flight node initialized.")

    # === Helpers ===

    def setup_csv_logging(self):
        if not self.enable_csv_logging:
            return

        try:
            log_dir = os.path.expanduser("~/g34_logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(log_dir, f"g34_first_flight_{timestamp}.csv")
            self.csv_file = open(filename, mode="w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "t_ros",
                "phase",
                "takeoff_altitude_m",
                "tuning_mode",
                "final_mode",
                "alt_up_m",
                "vz_ned",
                "z_sp_ned",
                "landed",
            ])
            self.get_logger().info(f"CSV logging enabled: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to set up CSV logging: {e}")
            self.csv_file = None
            self.csv_writer = None

    def get_px4_timestamp(self) -> int:
        # PX4 timestamp in microseconds
        return int(self.get_clock().now().nanoseconds / 1000)

    def local_position_callback(self, msg: VehicleLocalPosition):
        # PX4 NED: x north, y east, z down. Altitude up ~= -z (ignoring offsets).
        self.local_position = msg
        self.alt_up_m = -msg.z
        self.vz_ned = msg.v_z

        # simple yaw estimate from heading field, if valid
        # some versions use heading (rad, NED), some yaw; adapt as needed.
        if hasattr(msg, "heading"):
            self.yaw_rad = msg.heading

    def land_detected_callback(self, msg: VehicleLandDetected):
        self.landed = bool(msg.landed)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_px4_timestamp()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_px4_timestamp()

        # We keep the vehicle over (0,0) and only move in z for now.
        msg.position = [0.0, 0.0, self.z_sp_ned]
        msg.yaw = self.yaw_rad  # hold current yaw; this helps avoid weird yaw motion.

        self.trajectory_setpoint_pub.publish(msg)

    def send_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ):
        msg = VehicleCommand()
        msg.timestamp = self.get_px4_timestamp()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.confirmation = 0
        self.vehicle_command_pub.publish(msg)

    # === State machine ===

    def timer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time
        self.last_time = now

        # Always stream Offboard setpoints
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        # CSV log
        if self.csv_writer is not None:
            try:
                self.csv_writer.writerow([
                    f"{now:.3f}",
                    self.phase.name,
                    f"{self.takeoff_altitude_m:.3f}",
                    self.tuning_mode,
                    self.final_mode,
                    "" if self.alt_up_m is None else f"{self.alt_up_m:.3f}",
                    "" if self.vz_ned is None else f"{self.vz_ned:.3f}",
                    f"{self.z_sp_ned:.3f}",
                    int(self.landed),
                ])
            except Exception as e:
                self.get_logger().error(f"CSV write failed: {e}")
                self.csv_writer = None

        # Human-readable snapshot every ~1 s
        if int((now - self.node_start_time) * 2) % 10 == 0:  # ~1s, coarse
            alt_str = "N/A" if self.alt_up_m is None else f"{self.alt_up_m:.3f}"
            vz_str = "N/A" if self.vz_ned is None else f"{self.vz_ned:.3f}"
            self.get_logger().info(
                f"Phase={self.phase.name}, z_sp_ned={self.z_sp_ned:.3f} m, "
                f"alt_up={alt_str} m, vz_ned={vz_str} m/s, landed={self.landed}"
            )

        # State handling
        if self.phase == FlightPhase.PREFLIGHT:
            self.handle_preflight(now)

        elif self.phase == FlightPhase.TAKEOFF_ASCEND:
            self.handle_takeoff_ascend(now, dt)

        elif self.phase == FlightPhase.HOVER:
            self.handle_hover(now)

        elif self.phase == FlightPhase.ALT_TUNING:
            self.handle_alt_tuning(now)

        elif self.phase == FlightPhase.ATT_TUNING:
            self.handle_att_tuning(now)

        elif self.phase == FlightPhase.FINAL_DESCEND:
            self.handle_final_descend(now, dt)

        elif self.phase == FlightPhase.FINAL_AUTO_LAND:
            self.handle_final_auto_land(now)

        elif self.phase == FlightPhase.FINAL_PREC_LAND:
            self.handle_final_precision_land(now)

        elif self.phase == FlightPhase.DONE:
            # Just keep streaming neutral setpoints for a few seconds; nothing else.
            return

    # --- Phase handlers ---

    def handle_preflight(self, now: float):
        # Keep z_sp_ned at ground level
        self.z_sp_ned = 0.0

        # Require some local position before we proceed (helps EKF & avoids spamming commands)
        if self.local_position is None:
            self.get_logger().debug("PREFLIGHT: waiting for local_position...")
            return

        elapsed = now - self.phase_start_time

        if elapsed < self.pre_offboard_stream_time:
            # Just streaming to satisfy Offboard requirement
            return

        # Once pre-stream is done, request OFFBOARD mode and ARM (once)
        if not self.offboard_command_sent:
            # PX4 DO_SET_MODE: param1 = base mode (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1),
            # param2 = custom mode (PX4 flight mode enum: 6 = OFFBOARD)
            self.send_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )
            self.offboard_command_sent = True
            self.get_logger().info("Sent OFFBOARD mode command.")

        if not self.arm_command_sent:
            self.send_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,  # arm
            )
            self.arm_command_sent = True
            self.get_logger().info("Sent ARM command.")

        # Move into TAKEOFF_ASCEND state once we've requested offboard+arm
        self.phase = FlightPhase.TAKEOFF_ASCEND
        self.phase_start_time = now
        self.get_logger().info("PREFLIGHT -> TAKEOFF_ASCEND")

    def handle_takeoff_ascend(self, now: float, dt: float):
        # Slowly ramp z_sp_ned from 0 to -takeoff_altitude_m
        target_z = -self.takeoff_altitude_m
        if self.z_sp_ned > target_z:
            self.z_sp_ned -= self.takeoff_ascent_rate * dt
            if self.z_sp_ned < target_z:
                self.z_sp_ned = target_z

        elapsed = now - self.phase_start_time

        # Condition to decide we've "reached" altitude
        reached_alt = (
            self.alt_up_m is not None
            and self.alt_up_m >= 0.9 * self.takeoff_altitude_m
        )

        if reached_alt:
            self.phase = FlightPhase.HOVER
            self.phase_start_time = now
            self.get_logger().info(
                f"TAKEOFF_ASCEND -> HOVER (alt_up={self.alt_up_m:.3f} m)"
            )
            return

        if elapsed > self.takeoff_timeout:
            self.get_logger().warn(
                f"TAKEOFF_ASCEND timeout ({elapsed:.1f}s); continuing anyway into HOVER."
            )
            self.phase = FlightPhase.HOVER
            self.phase_start_time = now

    def handle_hover(self, now: float):
        # Hold altitude at takeoff-altitude
        self.z_sp_ned = -self.takeoff_altitude_m
        elapsed = now - self.phase_start_time

        if elapsed < self.hover_time:
            return  # keep hovering

        # Decide next step based on tuning mode
        if self.tuning_mode == "altitude":
            self.phase = FlightPhase.ALT_TUNING
            self.phase_start_time = now
            # immediate step in altitude setpoint
            self.z_sp_ned = -(self.takeoff_altitude_m + self.alt_tune_step_m)
            self.get_logger().info(
                f"HOVER -> ALT_TUNING (step +{self.alt_tune_step_m:.2f} m altitude)"
            )
        elif self.tuning_mode == "attitude":
            self.phase = FlightPhase.ATT_TUNING
            self.phase_start_time = now
            self.get_logger().info("HOVER -> ATT_TUNING (placeholder lateral test).")
        else:
            # No tuning; move directly to final mode selection
            self.transition_to_final_mode(now)

    def handle_alt_tuning(self, now: float):
        # Hold higher altitude for alt_tune_duration, then go to final mode
        self.z_sp_ned = -(self.takeoff_altitude_m + self.alt_tune_step_m)
        elapsed = now - self.phase_start_time

        if elapsed < self.alt_tune_duration:
            return

        self.get_logger().info("ALT_TUNING complete; returning to HOVER altitude and final mode.")
        # Return to nominal hover altitude before final mode
        self.z_sp_ned = -self.takeoff_altitude_m
        self.transition_to_final_mode(now)

    def handle_att_tuning(self, now: float):
        # For now, keep this simple: we just maintain altitude, no lateral motion.
        # This is a placeholder you can expand into an actual lateral/attitude test.
        self.z_sp_ned = -self.takeoff_altitude_m

        # Simple fixed duration, then final mode
        elapsed = now - self.phase_start_time
        att_tune_duration = 10.0  # s

        if elapsed < att_tune_duration:
            # Here you could implement: periodic small yaw, pitch, or x/y position perturbations.
            return

        self.get_logger().info("ATT_TUNING complete; transitioning to final mode.")
        self.transition_to_final_mode(now)

    def transition_to_final_mode(self, now: float):
        if self.final_mode == "descend":
            self.phase = FlightPhase.FINAL_DESCEND
            self.phase_start_time = now
            self.get_logger().info("Final mode: controlled DESCEND to ground.")

        elif self.final_mode == "auto_land":
            # Hand off to PX4's AUTO.LAND
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.phase = FlightPhase.FINAL_AUTO_LAND
            self.phase_start_time = now
            self.get_logger().info("Final mode: PX4 AUTO.LAND requested.")

        elif self.final_mode == "precision_land":
            # Placeholder hook: either PX4 NAV_PRECLAND or external mode (e.g., tracktor-beam).
            # For now we send NAV_PRECLAND and keep streaming neutral z_sp.
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND)
            self.phase = FlightPhase.FINAL_PREC_LAND
            self.phase_start_time = now
            self.get_logger().info(
                "Final mode: precision land requested (NAV_PRECLAND). "
                "Hook in tracktor-beam / custom mode here."
            )

    def handle_final_descend(self, now: float, dt: float):
        # Slowly bring z_sp_ned back toward 0 (ground)
        if self.z_sp_ned < 0.0:
            self.z_sp_ned += self.descend_rate * dt
            if self.z_sp_ned > 0.0:
                self.z_sp_ned = 0.0

        elapsed = now - self.phase_start_time

        # Consider ourselves "landed" when low altitude and near-zero vertical speed,
        # or when PX4 thinks we're landed.
        alt_ok = self.alt_up_m is not None and self.alt_up_m <= self.land_alt_threshold
        vz_ok = self.vz_ned is not None and abs(self.vz_ned) < 0.1

        if (alt_ok and vz_ok) or self.landed or elapsed > self.descend_timeout:
            # Send DISARM once, then mark DONE
            self.send_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=0.0,  # disarm
            )
            self.phase = FlightPhase.DONE
            self.phase_start_time = now
            self.get_logger().info(
                f"FINAL_DESCEND -> DONE (landed={self.landed}, alt_up={self.alt_up_m}, vz_ned={self.vz_ned})"
            )

    def handle_final_auto_land(self, now: float):
        # PX4 AUTO.LAND handles everything. We just monitor and optionally disarm when landed.
        if self.landed:
            self.send_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=0.0,
            )
            self.phase = FlightPhase.DONE
            self.phase_start_time = now
            self.get_logger().info("AUTO.LAND complete and DISARM sent; DONE.")

    def handle_final_precision_land(self, now: float):
        # Here is where you'd *really* integrate tracktor-beam:
        #   - Either let PX4 NAV_PRECLAND handle it
        #   - Or start the tracktor-beam ROS2 mode node (via separate launch)
        #
        # For now, we just watch for landed flag and disarm.
        if self.landed:
            self.send_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=0.0,
            )
            self.phase = FlightPhase.DONE
            self.phase_start_time = now
            self.get_logger().info("Precision landing complete and DISARM sent; DONE.")


def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        if node.csv_file is not None:
            try:
                node.csv_file.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
