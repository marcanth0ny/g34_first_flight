#!/usr/bin/env python3
import math
import os
import csv
from enum import Enum, auto
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleLandDetected,
    VehicleControlMode,
)


class Phase(Enum):
    PREFLIGHT = auto()
    TAKEOFF_ASCEND = auto()
    HOVER = auto()
    ALT_TUNING = auto()
    ATT_TUNING = auto()
    FINAL_DESCEND = auto()
    FINAL_AUTO_LAND = auto()
    FINAL_PL_PRECISION = auto()
    DONE = auto()


class G34FirstFlightNode(Node):
    def __init__(self):
        super().__init__("g34_first_flight_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position")
        self.declare_parameter("takeoff_altitude_m", 0.5)  # positive up
        self.declare_parameter("tuning_mode", "none")      # none | altitude | attitude | both
        self.declare_parameter("final_mode", "descend")    # descend | auto_land | pl_precision

        self.local_position_topic = (
            self.get_parameter("local_position_topic").get_parameter_value().string_value
        )
        self.takeoff_altitude_m = float(
            self.get_parameter("takeoff_altitude_m").get_parameter_value().double_value
        )
        self.tuning_mode = (
            self.get_parameter("tuning_mode").get_parameter_value().string_value.lower()
        )
        self.final_mode = (
            self.get_parameter("final_mode").get_parameter_value().string_value.lower()
        )

        # Safety clamp on altitude
        if self.takeoff_altitude_m <= 0.1:
            self.get_logger().warn(
                f"takeoff_altitude_m={self.takeoff_altitude_m:.2f} too low, clamping to 0.5 m"
            )
            self.takeoff_altitude_m = 0.5

        # Timing / profile parameters
        self.dt = 0.02  # main loop period [s] (~50 Hz)

        self.preoffboard_duration_s = 1.0     # streaming before requesting OFFBOARD+ARM
        self.ascend_timeout_s = 20.0          # safety
        self.hover_duration_s = 5.0           # base hover time

        # Altitude tuning config
        self.alt_tuning_step_dz_m = 0.2       # +step above base altitude
        self.alt_tuning_duration_s = 6.0      # total duration (up step then back)
        # Attitude tuning via position step
        self.att_tuning_step_x_m = 0.5        # NED X step (~forward)
        self.att_tuning_duration_s = 6.0

        # Landing handling
        self.descend_timeout_s = 15.0
        self.land_alt_threshold_m = 0.05      # ~5 cm
        self.land_vz_threshold_mps = 0.05     # small vertical speed

        # ---------------- Internal state ----------------
        self.phase = Phase.PREFLIGHT
        self.phase_start_time = None
        self.offboard_started = False
        self.offboard_start_time = None

        self.local_position_msg = None
        self.vehicle_status_msg = None
        self.vehicle_control_mode_msg = None
        self.land_detected_msg = None

        self.base_yaw_rad = None
        self.base_x_ned = 0.0
        self.base_y_ned = 0.0

        self.last_status_log_time = 0.0
        self.status_log_period_s = 0.5

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logger()

        self.get_logger().info(
            f"tuning_mode='{self.tuning_mode}', "
            f"final_mode='{self.final_mode}', "
            f"takeoff_altitude_m={self.takeoff_altitude_m:.2f} m"
        )

        # ---------------- QoS Profiles ----------------
        qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- Publishers ----------------
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )

        # ---------------- Subscribers ----------------
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self.local_position_callback,
            qos_best_effort,
        )
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_best_effort,
        )
        self.vehicle_control_mode_sub = self.create_subscription(
            VehicleControlMode,
            "/fmu/out/vehicle_control_mode",
            self.vehicle_control_mode_callback,
            qos_best_effort,
        )
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self.land_detected_callback,
            qos_best_effort,
        )

        # ---------------- Timer ----------------
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info("G34 First Flight node initialized.")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def local_position_callback(self, msg: VehicleLocalPosition):
        self.local_position_msg = msg

        # Capture base yaw & base xy on first useful message
        if self.base_yaw_rad is None:
            heading = float(getattr(msg, "heading", float("nan")))
            if math.isfinite(heading):
                self.base_yaw_rad = heading
            else:
                self.base_yaw_rad = 0.0

            self.base_x_ned = float(msg.x)
            self.base_y_ned = float(msg.y)

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.vehicle_status_msg = msg

    def vehicle_control_mode_callback(self, msg: VehicleControlMode):
        self.vehicle_control_mode_msg = msg

    def land_detected_callback(self, msg: VehicleLandDetected):
        self.land_detected_msg = msg

    # -------------------------------------------------------------------------
    # Helpers: Vehicle state
    # -------------------------------------------------------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def is_armed(self) -> bool:
        if self.vehicle_status_msg is None:
            return False
        # VehicleStatus.arming_state == 2 => ARMED in PX4 1.14
        return int(self.vehicle_status_msg.arming_state) == 2

    def is_landed(self) -> bool:
        if self.land_detected_msg is None:
            return False
        return bool(self.land_detected_msg.landed)

    def get_altitude_up_and_vz(self):
        """
        Returns (alt_up_m, vz_ned_mps).
        alt_up_m: positive up, relative to local origin
        vz_ned_mps: NED vertical velocity (down-positive)
        """
        if self.local_position_msg is None:
            return None, None

        z_ned = float(self.local_position_msg.z)
        vz_ned = float(self.local_position_msg.v_z)
        alt_up = -z_ned
        return alt_up, vz_ned

    # -------------------------------------------------------------------------
    # Helpers: Commands
    # -------------------------------------------------------------------------
    def publish_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = float("nan"),
        param6: float = float("nan"),
        param7: float = float("nan"),
    ):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

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

        self.vehicle_command_pub.publish(msg)

    def set_offboard_mode(self):
        # MAV_CMD_DO_SET_MODE (176)
        # param1: base mode (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1)
        # param2: custom main mode (PX4: 6 = OFFBOARD)
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )
        self.get_logger().info("Sending OFFBOARD mode command.")

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=float(VehicleCommand.ARMING_ACTION_ARM),
        )
        self.get_logger().info("Sending ARM command.")

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=float(VehicleCommand.ARMING_ACTION_DISARM),
        )
        self.get_logger().info("Sending DISARM command.")

    def send_auto_land_command(self):
        # PX4 NAV_LAND command at current position
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Sent VEHICLE_CMD_NAV_LAND (auto land).")

    def send_precision_land_command(self):
        """
        Hook for PX4 precision landing / tracktor-beam.

        Here we send PX4's VEHICLE_CMD_NAV_PRECLAND to request precision landing.
        If tracktor-beam exposes a ROS2 service/action, you can call it here too.
        """
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND)
        self.get_logger().info(
            "Sent VEHICLE_CMD_NAV_PRECLAND (precision landing request). "
            "Integrate with tracktor-beam node here if needed."
        )

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # We are using position control in XYZ, yaw
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(
        self,
        x_ned: float,
        y_ned: float,
        z_ned: float,
        yaw_rad: float,
    ):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Position (NED)
        msg.position[0] = float(x_ned)
        msg.position[1] = float(y_ned)
        msg.position[2] = float(z_ned)

        # Zero velocities (let PX4 handle)
        msg.velocity[0] = 0.0
        msg.velocity[1] = 0.0
        msg.velocity[2] = 0.0

        # Leave accel/jerk/thrust as NaN to indicate "don't care"
        msg.acceleration[0] = float("nan")
        msg.acceleration[1] = float("nan")
        msg.acceleration[2] = float("nan")
        msg.jerk[0] = float("nan")
        msg.jerk[1] = float("nan")
        msg.jerk[2] = float("nan")
        msg.thrust[0] = float("nan")
        msg.thrust[1] = float("nan")
        msg.thrust[2] = float("nan")

        msg.yaw = float(yaw_rad)
        msg.yawspeed = 0.0

        self.trajectory_setpoint_pub.publish(msg)

    # -------------------------------------------------------------------------
    # CSV logging
    # -------------------------------------------------------------------------
    def _init_csv_logger(self):
        try:
            home = os.path.expanduser("~")
            log_dir = os.path.join(home, "g34_logs")
            os.makedirs(log_dir, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"g34_first_flight_{ts}.csv")

            self.csv_file = open(path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)

            self.csv_writer.writerow(
                [
                    "t_s",
                    "phase",
                    "tuning_mode",
                    "final_mode",
                    "alt_up_m",
                    "vz_ned_mps",
                    "cmd_x_ned_m",
                    "cmd_y_ned_m",
                    "cmd_z_ned_m",
                    "takeoff_altitude_m",
                    "armed",
                    "landed",
                ]
            )
            self.csv_file.flush()
            self.get_logger().info(f"CSV logging enabled: {path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to initialize CSV logging: {e}")
            self.csv_file = None
            self.csv_writer = None

    def _log_csv(
        self,
        t: float,
        phase: Phase,
        cmd_x_ned: float,
        cmd_y_ned: float,
        cmd_z_ned: float,
    ):
        if self.csv_writer is None:
            return

        alt_up, vz_ned = self.get_altitude_up_and_vz()
        if alt_up is None:
            alt_up = float("nan")
        if vz_ned is None:
            vz_ned = float("nan")

        try:
            self.csv_writer.writerow(
                [
                    f"{t:.3f}",
                    phase.name,
                    self.tuning_mode,
                    self.final_mode,
                    f"{alt_up:.4f}",
                    f"{vz_ned:.4f}",
                    f"{cmd_x_ned:.4f}",
                    f"{cmd_y_ned:.4f}",
                    f"{cmd_z_ned:.4f}",
                    f"{self.takeoff_altitude_m:.3f}",
                    int(self.is_armed()),
                    int(self.is_landed()),
                ]
            )
            # flush occasionally
            if int(t * 10) % 10 == 0:
                self.csv_file.flush()
        except Exception as e:
            self.get_logger().warn(f"CSV log write failed: {e}")

    def close_csv(self):
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None

    # -------------------------------------------------------------------------
    # Main timer / state machine
    # -------------------------------------------------------------------------
    def timer_callback(self):
        t = self.now_s()

        # Default yaw (hold initial heading)
        yaw_cmd = self.base_yaw_rad if self.base_yaw_rad is not None else 0.0

        # Default command: stay at base position, on ground
        cmd_x_ned = self.base_x_ned
        cmd_y_ned = self.base_y_ned
        cmd_z_ned = 0.0  # z_ned=0 at takeoff point

        # Compute altitude for logic & logging
        alt_up, vz_ned = self.get_altitude_up_and_vz()
        armed = self.is_armed()
        landed = self.is_landed()

        if self.phase_start_time is None:
            self.phase_start_time = t

        phase_time = t - self.phase_start_time

        # ----------------- State machine -----------------
        if self.phase == Phase.PREFLIGHT:
            # Stream neutral setpoint near ground
            cmd_z_ned = 0.0

            # Start OFFBOARD + ARM after preoffboard_duration_s
            if not self.offboard_started and phase_time >= self.preoffboard_duration_s:
                self.set_offboard_mode()
                self.arm()
                self.offboard_started = True
                self.offboard_start_time = t

            # Transition to TAKEOFF when armed
            if self.offboard_started and armed:
                self.phase = Phase.TAKEOFF_ASCEND
                self.phase_start_time = t
                self.get_logger().info("PREFLIGHT -> TAKEOFF_ASCEND")

        elif self.phase == Phase.TAKEOFF_ASCEND:
            # Command vertical position to target altitude (z_ned negative)
            cmd_z_ned = -self.takeoff_altitude_m

            # Transition to HOVER when altitude reached or timeout
            reached = False
            if alt_up is not None:
                if alt_up >= self.takeoff_altitude_m - 0.05:
                    reached = True

            if reached or phase_time >= self.ascend_timeout_s:
                self.phase = Phase.HOVER
                self.phase_start_time = t
                self.get_logger().info(
                    f"TAKEOFF_ASCEND -> HOVER (alt_up={alt_up:.3f} m, reached={reached})"
                    if alt_up is not None
                    else "TAKEOFF_ASCEND -> HOVER (alt_up=N/A)"
                )

        elif self.phase == Phase.HOVER:
            # Maintain takeoff altitude
            cmd_z_ned = -self.takeoff_altitude_m

            if phase_time >= self.hover_duration_s:
                # Decide next step based on tuning_mode
                if self.tuning_mode in ("altitude", "both"):
                    self.phase = Phase.ALT_TUNING
                    self.phase_start_time = t
                    self.get_logger().info("HOVER -> ALT_TUNING")
                elif self.tuning_mode == "attitude":
                    self.phase = Phase.ATT_TUNING
                    self.phase_start_time = t
                    self.get_logger().info("HOVER -> ATT_TUNING")
                else:
                    # no tuning: go straight to final mode
                    self._enter_final_mode(t)

        elif self.phase == Phase.ALT_TUNING:
            # Simple two-step: up by +dz, then back to base
            half = self.alt_tuning_duration_s / 2.0
            if phase_time < half:
                cmd_z_ned = -(self.takeoff_altitude_m + self.alt_tuning_step_dz_m)
            elif phase_time < self.alt_tuning_duration_s:
                cmd_z_ned = -self.takeoff_altitude_m
            else:
                self.get_logger().info("ALT_TUNING complete.")
                if self.tuning_mode == "both":
                    self.phase = Phase.ATT_TUNING
                    self.phase_start_time = t
                    self.get_logger().info("ALT_TUNING -> ATT_TUNING")
                else:
                    self._enter_final_mode(t)

        elif self.phase == Phase.ATT_TUNING:
            # Keep altitude, but step in X to excite roll/pitch
            cmd_z_ned = -self.takeoff_altitude_m

            half = self.att_tuning_duration_s / 2.0
            if phase_time < half:
                # Step forward in NED X
                cmd_x_ned = self.base_x_ned + self.att_tuning_step_x_m
            elif phase_time < self.att_tuning_duration_s:
                cmd_x_ned = self.base_x_ned
            else:
                self.get_logger().info("ATT_TUNING complete.")
                self._enter_final_mode(t)

        elif self.phase == Phase.FINAL_DESCEND:
            # Controlled descend back to z=0 (takeoff altitude reference)
            cmd_z_ned = 0.0

            # Stop condition: near ground + small vertical velocity + landed OR timeout
            near_ground = alt_up is not None and abs(alt_up) <= self.land_alt_threshold_m
            vertical_slow = vz_ned is not None and abs(vz_ned) <= self.land_vz_threshold_mps

            if (near_ground and vertical_slow and landed) or phase_time >= self.descend_timeout_s:
                self.disarm()
                self.phase = Phase.DONE
                self.phase_start_time = t
                self.get_logger().info(
                    f"FINAL_DESCEND -> DONE (alt_up={alt_up:.3f if alt_up is not None else 'N/A'}, "
                    f"vz_ned={vz_ned:.3f if vz_ned is not None else 'N/A'}, landed={landed})"
                )

        elif self.phase == Phase.FINAL_AUTO_LAND:
            # We already sent NAV_LAND; keep neutral setpoint and let PX4 handle.
            cmd_z_ned = 0.0
            # You could optionally monitor nav_state/land_detected here and
            # exit to DONE once fully landed. For now, we just hold until manual stop.

        elif self.phase == Phase.FINAL_PL_PRECISION:
            # Precision landing handled by PX4 / tracktor-beam.
            # Keep a neutral setpoint; this phase is mostly informational.
            cmd_z_ned = -self.takeoff_altitude_m

        elif self.phase == Phase.DONE:
            # Keep sending a safe neutral setpoint; motors should be disarmed.
            cmd_z_ned = 0.0

        # ----------------- Publish setpoints -----------------
        # Always publish offboard mode + trajectory setpoint
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(
            cmd_x_ned,
            cmd_y_ned,
            cmd_z_ned,
            yaw_cmd,
        )

        # ----------------- Status logging (throttled) -----------------
        if t - self.last_status_log_time >= self.status_log_period_s:
            alt_str = f"{alt_up:.3f} m" if alt_up is not None else "N/A"
            vz_str = f"{vz_ned:.3f} m/s" if vz_ned is not None else "N/A"
            self.get_logger().info(
                f"Phase={self.phase.name}, "
                f"cmd_z_ned={cmd_z_ned:+.3f} m, alt_up={alt_str}, "
                f"vz_ned={vz_str}, armed={armed}, landed={landed}"
            )
            self.last_status_log_time = t

        # ----------------- CSV logging -----------------
        self._log_csv(t, self.phase, cmd_x_ned, cmd_y_ned, cmd_z_ned)

    # -------------------------------------------------------------------------
    # Final mode selection helper
    # -------------------------------------------------------------------------
    def _enter_final_mode(self, t_now: float):
        """
        Decide and enter the final mode after tuning/hover.
        """
        self.phase_start_time = t_now

        if self.final_mode == "auto_land":
            self.send_auto_land_command()
            self.phase = Phase.FINAL_AUTO_LAND
            self.get_logger().info("Entering FINAL_AUTO_LAND (PX4 NAV_LAND).")
        elif self.final_mode == "pl_precision":
            self.send_precision_land_command()
            self.phase = Phase.FINAL_PL_PRECISION
            self.get_logger().info(
                "Entering FINAL_PL_PRECISION (precision landing via PX4 / tracktor-beam)."
            )
        else:
            # Default to our own controlled descend/disarm
            self.phase = Phase.FINAL_DESCEND
            self.get_logger().info("Entering FINAL_DESCEND (controlled descend & disarm).")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def destroy_node(self):
        self.close_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        # NOTE: no rclpy.shutdown() here; ros2 launch handles it.
