#!/usr/bin/env python3
import os
import csv
import math
from enum import Enum, auto
from datetime import datetime

import rclpy
from rclpy.node import Node

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleControlMode,
    VehicleStatus,
    VehicleLandDetected,
)

# -------------------------------------------------------------------------
#  Simple phase enum for the mission state machine
# -------------------------------------------------------------------------
class Phase(Enum):
    PREFLIGHT = auto()
    TAKEOFF_ASCEND = auto()
    HOVER = auto()
    ALT_TUNING = auto()
    ATT_TUNING = auto()
    FINAL_DESCEND = auto()
    FINAL_AUTO_LAND = auto()
    FINAL_PREC_LAND = auto()
    DONE = auto()


# -------------------------------------------------------------------------
#  Main node
# -------------------------------------------------------------------------
class G34FirstFlightNode(Node):
    def __init__(self):
        super().__init__("g34_first_flight_node")

        # ------------- Parameters ------------------------------------------------
        # Topic for local position (SITL vs hardware)
        self.local_position_topic = (
            self.declare_parameter(
                "local_position_topic", "/fmu/out/vehicle_local_position"
            )
            .get_parameter_value()
            .string_value
        )

        # Takeoff altitude (m, positive up)
        raw_takeoff_alt = (
            self.declare_parameter("takeoff_altitude_m", 0.5)
            .get_parameter_value()
            .double_value
        )
        # Clamp to a safe range (0.3–2.0 m)
        self.takeoff_altitude_m = max(0.3, min(raw_takeoff_alt, 2.0))

        # Hover duration at takeoff altitude (s)
        self.hover_duration_s = (
            self.declare_parameter("hover_duration_s", 5.0)
            .get_parameter_value()
            .double_value
        )

        # Pre-offboard streaming duration (s)
        self.preoffboard_duration_s = (
            self.declare_parameter("preoffboard_duration_s", 1.0)
            .get_parameter_value()
            .double_value
        )

        # Tuning mode: none / altitude / attitude / both
        raw_tuning_mode = (
            self.declare_parameter("tuning_mode", "none")
            .get_parameter_value()
            .string_value
        )
        self.tuning_mode = raw_tuning_mode.strip().lower()
        valid_tuning_modes = ("none", "altitude", "attitude", "both")
        if self.tuning_mode not in valid_tuning_modes:
            self.get_logger().warn(
                f"Invalid tuning_mode '{raw_tuning_mode}', defaulting to 'none'. "
                f"Valid options: {valid_tuning_modes}"
            )
            self.tuning_mode = "none"

        # Final mode after hover/tuning: descend / auto_land / pl_precision
        raw_final_mode = (
            self.declare_parameter("final_mode", "descend")
            .get_parameter_value()
            .string_value
        )
        self.final_mode = raw_final_mode.strip().lower()
        valid_final_modes = ("descend", "auto_land", "pl_precision")
        if self.final_mode not in valid_final_modes:
            self.get_logger().warn(
                f"Invalid final_mode '{raw_final_mode}', defaulting to 'descend'. "
                f"Valid options: {valid_final_modes}"
            )
            self.final_mode = "descend"

        # Altitude tuning parameters
        self.alt_step_amplitude_m = (
            self.declare_parameter("alt_step_amplitude_m", 0.15)
            .get_parameter_value()
            .double_value
        )
        self.alt_step_period_s = (
            self.declare_parameter("alt_step_period_s", 5.0)
            .get_parameter_value()
            .double_value
        )
        self.tuning_duration_s = (
            self.declare_parameter("tuning_duration_s", 30.0)
            .get_parameter_value()
            .double_value
        )

        # Attitude tuning parameters: simple yaw steps
        self.att_step_deg = (
            self.declare_parameter("att_step_deg", 5.0)
            .get_parameter_value()
            .double_value
        )
        self.att_step_period_s = (
            self.declare_parameter("att_step_period_s", 5.0)
            .get_parameter_value()
            .double_value
        )

        # Takeoff ascend timeout (s) – safety
        self.takeoff_timeout_s = (
            self.declare_parameter("takeoff_timeout_s", 20.0)
            .get_parameter_value()
            .double_value
        )

        # Log directory for CSV
        raw_log_dir = (
            self.declare_parameter("log_directory", "~/g34_logs")
            .get_parameter_value()
            .string_value
        )
        self.log_directory = os.path.expanduser(raw_log_dir)

        self.get_logger().info(
            f"tuning_mode='{self.tuning_mode}', final_mode='{self.final_mode}', "
            f"takeoff_altitude_m={self.takeoff_altitude_m:.2f} m"
        )

        # ------------- Publishers / Subscribers ----------------------------------
        # Command interface to PX4
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )

        # Feedback from PX4
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self.local_position_callback,
            10,
        )
        self.vehicle_control_mode_sub = self.create_subscription(
            VehicleControlMode,
            "/fmu/out/vehicle_control_mode",
            self.vehicle_control_mode_callback,
            10,
        )
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            10,
        )
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self.land_detected_callback,
            10,
        )

        # ------------- Internal state -------------------------------------------
        self.phase = Phase.PREFLIGHT
        self.phase_start_time = None

        self.offboard_started = False
        self.offboard_start_time = None

        self.base_yaw_rad = 0.0  # yaw at takeoff, used as reference for attitude tuning

        self.local_position_msg = None
        self.vehicle_control_mode_msg = None
        self.vehicle_status_msg = None
        self.land_detected_msg = None

        self.alt_tuning_center_m = self.takeoff_altitude_m
        self.did_alt_tuning_in_both = False

        self.final_mode_started = False
        self.sent_disarm_cmd = False

        # For simple log throttling
        self.last_hover_log_time = 0.0
        self.last_done_log_time = 0.0

        # ------------- CSV logging setup ----------------------------------------
        os.makedirs(self.log_directory, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            self.log_directory, f"g34_first_flight_{time_str}.csv"
        )
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "t_ros_s",
                "phase",
                "alt_up_m",
                "vz_ned_mps",
                "cmd_alt_up_m",
                "cmd_z_ned_m",
                "yaw_cmd_rad",
                "tuning_mode",
                "final_mode",
                "landed_flag",
                "arming_state",
            ]
        )
        self.get_logger().info(f"CSV logging enabled: {self.csv_path}")

        # ------------- Timer -----------------------------------------------------
        # 20 Hz timer: drives state machine and offboard setpoints
        self.timer_dt = 0.05
        self.timer = self.create_timer(self.timer_dt, self.timer_callback)

        self.get_logger().info("G34 First Flight node initialized.")

    # -------------------------------------------------------------------------
    #  Helpers for time / msg utilities
    # -------------------------------------------------------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def current_phase_name(self) -> str:
        return self.phase.name

    # -------------------------------------------------------------------------
    #  Subscribers
    # -------------------------------------------------------------------------
    def local_position_callback(self, msg: VehicleLocalPosition):
        self.local_position_msg = msg

    def vehicle_control_mode_callback(self, msg: VehicleControlMode):
        self.vehicle_control_mode_msg = msg

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.vehicle_status_msg = msg

    def land_detected_callback(self, msg: VehicleLandDetected):
        self.land_detected_msg = msg

    # -------------------------------------------------------------------------
    #  PX4 command helpers
    # -------------------------------------------------------------------------
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
        msg.timestamp = self.now_us()
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

    def arm(self):
        self.get_logger().info("Sending ARM command.")
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=float(VehicleCommand.ARMING_ACTION_ARM),
        )

    def disarm(self):
        self.get_logger().info("Sending DISARM command.")
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=float(VehicleCommand.ARMING_ACTION_DISARM),
        )

    def set_offboard_mode(self):
        # Uses VEHICLE_CMD_DO_SET_MODE to request PX4 OFFBOARD
        # MAV_MODE: base_mode, custom_mode not strictly needed here; but we use
        # PX4's custom command for nav state in other paths.
        self.get_logger().info("Sending OFFBOARD mode command.")
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,  # base_mode: MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            param2=6.0,  # custom_main_mode: PX4_MAIN_MODE_OFFBOARD
            param3=0.0,
        )

    def set_auto_land_mode(self):
        # Simplest: use NAV_LAND and let PX4 handle final landing.
        self.get_logger().info("Sending NAV_LAND command (AUTO LAND).")
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # -------------------------------------------------------------------------
    #  Offboard publishers
    # -------------------------------------------------------------------------
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.now_us()
        # We are using position control in XYZ, yaw controlled via TrajectorySetpoint.yaw
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x_ned: float, y_ned: float, z_ned: float, yaw_rad: float):
        msg = TrajectorySetpoint()
        msg.timestamp = self.now_us()
        # Position NED
        msg.position[0] = float(x_ned)
        msg.position[1] = float(y_ned)
        msg.position[2] = float(z_ned)
        # Zero velocity/accel; PX4 position controller fills in
        msg.velocity[0] = 0.0
        msg.velocity[1] = 0.0
        msg.velocity[2] = 0.0
        msg.yaw = float(yaw_rad)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_trajectory_setpoint_z(self, z_ned: float, yaw_rad: float = None):
        # Keep XY at 0, just move in Z
        if yaw_rad is None:
            yaw_rad = self.base_yaw_rad
        self.publish_trajectory_setpoint(0.0, 0.0, z_ned, yaw_rad)

    # -------------------------------------------------------------------------
    #  Altitude / velocity extraction from VehicleLocalPosition
    # -------------------------------------------------------------------------
    def get_altitude_and_vz(self):
        """
        Returns (alt_up_m, vz_ned_mps, valid)
        alt_up_m: altitude above home (positive up)
        vz_ned_mps: vertical velocity in NED (down positive)
        """
        if self.local_position_msg is None:
            return (None, None, False)

        # PX4 local position is NED: z is down, so altitude up is -z.
        alt_up_m = -self.local_position_msg.z
        vz_ned = self.local_position_msg.vz
        return (alt_up_m, vz_ned, True)

    def is_landed(self) -> bool:
        if self.land_detected_msg is None:
            return False
        return bool(self.land_detected_msg.landed)

    def is_armed(self) -> bool:
        if self.vehicle_status_msg is None:
            return False
        return self.vehicle_status_msg.arming_state == VehicleStatus.ARMING_STATE_ARMED

    # -------------------------------------------------------------------------
    #  CSV logging
    # -------------------------------------------------------------------------
    def log_row(self, t_ros_s, cmd_z_ned_m, cmd_alt_up_m):
        phase_name = self.current_phase_name()
        alt_up_m, vz_ned, valid = self.get_altitude_and_vz()
        landed_flag = self.is_landed()
        arming_state = (
            self.vehicle_status_msg.arming_state if self.vehicle_status_msg else -1
        )

        self.csv_writer.writerow(
            [
                f"{t_ros_s:.3f}",
                phase_name,
                f"{alt_up_m:.4f}" if valid else "nan",
                f"{vz_ned:.4f}" if valid else "nan",
                f"{cmd_alt_up_m:.4f}" if cmd_alt_up_m is not None else "nan",
                f"{cmd_z_ned_m:.4f}" if cmd_z_ned_m is not None else "nan",
                f"{self.base_yaw_rad:.4f}",
                self.tuning_mode,
                self.final_mode,
                int(landed_flag),
                int(arming_state),
            ]
        )
        # keep file flushed; this is small enough for our use
        self.csv_file.flush()

    # -------------------------------------------------------------------------
    #  Start final mode helper
    # -------------------------------------------------------------------------
    def start_final_mode(self, now_s: float):
        if self.final_mode_started:
            return

        self.final_mode_started = True

        if self.final_mode == "descend":
            self.get_logger().info("Starting FINAL_DESCEND mode.")
            self.phase = Phase.FINAL_DESCEND
            self.phase_start_time = now_s

        elif self.final_mode == "auto_land":
            self.get_logger().info("Starting FINAL_AUTO_LAND mode.")
            self.set_auto_land_mode()
            self.phase = Phase.FINAL_AUTO_LAND
            self.phase_start_time = now_s

        elif self.final_mode == "pl_precision":
            # TODO: once Tracktor Beam / mode executor integration is wired,
            # this is where you'd send the appropriate mode request or start
            # their precision-landing mode. For now, fall back to auto_land.
            self.get_logger().info(
                "final_mode='pl_precision' requested, "
                "but precision-landing mode integration is not yet configured. "
                "Falling back to AUTO LAND."
            )
            self.set_auto_land_mode()
            self.phase = Phase.FINAL_AUTO_LAND
            self.phase_start_time = now_s

        else:
            # Should not reach here, but in case of typo:
            self.get_logger().warn(
                f"Unknown final_mode='{self.final_mode}', defaulting to FINAL_DESCEND."
            )
            self.phase = Phase.FINAL_DESCEND
            self.phase_start_time = now_s

    # -------------------------------------------------------------------------
    #  Main timer callback (state machine)
    # -------------------------------------------------------------------------
    def timer_callback(self):
        t = self.now_s()

        # Always publish OffboardControlMode – PX4 expects high-rate stream
        self.publish_offboard_control_mode()

        # Default commanded altitude/setpoint for logging
        cmd_alt_up_m = None
        cmd_z_ned = None

        # PREFLIGHT: stream neutral setpoints while we wait to enter OFFBOARD
        if self.phase == Phase.PREFLIGHT:
            if self.phase_start_time is None:
                self.phase_start_time = t
                self.offboard_start_time = t
                self.get_logger().info("Phase=PREFLIGHT (pre-offboard streaming).")

            # Neutral setpoint near ground
            self.publish_trajectory_setpoint_z(0.0)
            cmd_z_ned = 0.0
            cmd_alt_up_m = 0.0

            elapsed = t - self.phase_start_time

            # After a brief pre-offboard period, request OFFBOARD + ARM
            if not self.offboard_started and elapsed >= self.preoffboard_duration_s:
                self.set_offboard_mode()
                self.arm()
                self.offboard_started = True
                self.offboard_start_time = t
                self.base_yaw_rad = 0.0  # could be replaced with actual yaw

            # Once offboard and armed, transition to TAKEOFF_ASCEND
            if (
                self.offboard_started
                and self.is_armed()
                and self.vehicle_control_mode_msg
                and self.vehicle_control_mode_msg.flag_control_offboard_enabled
            ):
                self.phase = Phase.TAKEOFF_ASCEND
                self.phase_start_time = t
                self.get_logger().info("PREFLIGHT -> TAKEOFF_ASCEND")

        # TAKEOFF_ASCEND: climb to takeoff_altitude_m (NED z negative)
        elif self.phase == Phase.TAKEOFF_ASCEND:
            if self.phase_start_time is None:
                self.phase_start_time = t

            cmd_alt_up_m = self.takeoff_altitude_m
            cmd_z_ned = -self.takeoff_altitude_m
            self.publish_trajectory_setpoint_z(cmd_z_ned)

            alt_up, vz_ned, valid = self.get_altitude_and_vz()
            landed_flag = self.is_landed()
            elapsed = t - self.phase_start_time

            if valid:
                self.get_logger().info(
                    f"Phase=TAKEOFF_ASCEND, cmd_z_ned={cmd_z_ned:.3f} m, "
                    f"alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, landed={landed_flag}"
                )

            # Transition to HOVER when close enough to target altitude OR timeout
            altitude_reached = valid and (
                abs(alt_up - self.takeoff_altitude_m) <= 0.05
            )
            if altitude_reached or elapsed >= self.takeoff_timeout_s:
                if not altitude_reached:
                    self.get_logger().warn(
                        f"TAKEOFF_ASCEND timeout reached (elapsed={elapsed:.1f}s). "
                        f"alt_up={alt_up:.3f} m (target={self.takeoff_altitude_m:.3f} m)"
                        if valid
                        else "TAKEOFF_ASCEND timeout reached, but no valid altitude."
                    )
                self.phase = Phase.HOVER
                self.phase_start_time = t
                self.get_logger().info("TAKEOFF_ASCEND -> HOVER")

        # HOVER: hold takeoff altitude for hover_duration_s
        elif self.phase == Phase.HOVER:
            if self.phase_start_time is None:
                self.phase_start_time = t

            cmd_alt_up_m = self.takeoff_altitude_m
            cmd_z_ned = -self.takeoff_altitude_m
            self.publish_trajectory_setpoint_z(cmd_z_ned)

            alt_up, vz_ned, valid = self.get_altitude_and_vz()
            hover_elapsed = t - self.phase_start_time

            # Lightly throttled logging (once per ~1 s)
            if hover_elapsed - self.last_hover_log_time > 1.0:
                self.last_hover_log_time = hover_elapsed
                if valid:
                    self.get_logger().info(
                        f"Phase=HOVER, hover_elapsed={hover_elapsed:.1f}s, "
                        f"alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, "
                        f"tuning_mode={self.tuning_mode}"
                    )
                else:
                    self.get_logger().info(
                        f"Phase=HOVER, hover_elapsed={hover_elapsed:.1f}s, "
                        f"tuning_mode={self.tuning_mode}, alt_up=N/A"
                    )

            # Decide next phase once hover time is done
            if hover_elapsed >= self.hover_duration_s:
                if self.tuning_mode == "altitude":
                    self.get_logger().info(
                        f"HOVER -> ALT_TUNING (hover_elapsed={hover_elapsed:.1f}s)"
                    )
                    self.phase = Phase.ALT_TUNING
                    self.phase_start_time = t
                    self.alt_tuning_center_m = self.takeoff_altitude_m

                elif self.tuning_mode == "attitude":
                    self.get_logger().info(
                        f"HOVER -> ATT_TUNING (hover_elapsed={hover_elapsed:.1f}s)"
                    )
                    self.phase = Phase.ATT_TUNING
                    self.phase_start_time = t

                elif self.tuning_mode == "both":
                    self.get_logger().info(
                        "HOVER -> ALT_TUNING (both; starting with altitude)"
                    )
                    self.phase = Phase.ALT_TUNING
                    self.phase_start_time = t
                    self.alt_tuning_center_m = self.takeoff_altitude_m
                    self.did_alt_tuning_in_both = False

                else:
                    # No tuning requested – go straight to final mode
                    self.get_logger().info(
                        f"HOVER -> final_mode='{self.final_mode}' (no tuning requested)"
                    )
                    self.start_final_mode(t)

        # ALT_TUNING: altitude step tests around center
        elif self.phase == Phase.ALT_TUNING:
            if self.phase_start_time is None:
                self.phase_start_time = t

            tuning_elapsed = t - self.phase_start_time

            # Done with altitude tuning?
            if tuning_elapsed >= self.tuning_duration_s:
                if self.tuning_mode == "both" and not self.did_alt_tuning_in_both:
                    self.get_logger().info("ALT_TUNING complete -> ATT_TUNING (both).")
                    self.did_alt_tuning_in_both = True
                    self.phase = Phase.ATT_TUNING
                    self.phase_start_time = t
                else:
                    self.get_logger().info(
                        f"ALT_TUNING complete -> final_mode='{self.final_mode}'"
                    )
                    self.start_final_mode(t)
                cmd_z_ned = None
                cmd_alt_up_m = None
            else:
                # Step index based on period
                step_index = int(tuning_elapsed / self.alt_step_period_s)
                center = self.alt_tuning_center_m

                if step_index % 2 == 0:
                    alt_cmd_m = center + self.alt_step_amplitude_m
                else:
                    alt_cmd_m = center - self.alt_step_amplitude_m

                cmd_alt_up_m = alt_cmd_m
                cmd_z_ned = -alt_cmd_m
                self.publish_trajectory_setpoint_z(cmd_z_ned)

                self.get_logger().info(
                    f"Phase=ALT_TUNING, t={tuning_elapsed:.1f}s, step_index={step_index}, "
                    f"alt_cmd={alt_cmd_m:.3f} m"
                )

        # ATT_TUNING: simple yaw step tests around base_yaw_rad
        elif self.phase == Phase.ATT_TUNING:
            if self.phase_start_time is None:
                self.phase_start_time = t

            tuning_elapsed = t - self.phase_start_time

            # Done with attitude tuning?
            if tuning_elapsed >= self.tuning_duration_s:
                if self.tuning_mode == "both" and not self.did_alt_tuning_in_both:
                    # Should not normally happen, but keep logic symmetric:
                    self.get_logger().info("ATT_TUNING complete -> ALT_TUNING (both).")
                    self.did_alt_tuning_in_both = True
                    self.phase = Phase.ALT_TUNING
                    self.phase_start_time = t
                else:
                    self.get_logger().info(
                        f"ATT_TUNING complete -> final_mode='{self.final_mode}'"
                    )
                    self.start_final_mode(t)
                cmd_z_ned = None
                cmd_alt_up_m = None
            else:
                # Keep altitude near takeoff_altitude
                alt_cmd_m = self.takeoff_altitude_m
                cmd_alt_up_m = alt_cmd_m
                cmd_z_ned = -alt_cmd_m

                # Yaw step pattern
                step_index = int(tuning_elapsed / self.att_step_period_s)
                yaw_step_rad = math.radians(self.att_step_deg)
                if step_index % 2 == 0:
                    yaw_cmd = self.base_yaw_rad + yaw_step_rad
                else:
                    yaw_cmd = self.base_yaw_rad - yaw_step_rad

                self.publish_trajectory_setpoint(0.0, 0.0, cmd_z_ned, yaw_cmd)

                self.get_logger().info(
                    f"Phase=ATT_TUNING, t={tuning_elapsed:.1f}s, step_index={step_index}, "
                    f"yaw_cmd={yaw_cmd:.3f} rad, alt_cmd={alt_cmd_m:.3f} m"
                )

        # FINAL_DESCEND: offboard-controlled descent to ground (z_ned -> 0)
        elif self.phase == Phase.FINAL_DESCEND:
            if self.phase_start_time is None:
                self.phase_start_time = t

            # Command altitude up = 0 (ground)
            cmd_alt_up_m = 0.0
            cmd_z_ned = 0.0
            self.publish_trajectory_setpoint_z(cmd_z_ned)

            alt_up, vz_ned, valid = self.get_altitude_and_vz()
            landed_flag = self.is_landed()

            if valid:
                self.get_logger().info(
                    f"Phase=FINAL_DESCEND, alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, "
                    f"landed={landed_flag}"
                )
            else:
                self.get_logger().info(
                    f"Phase=FINAL_DESCEND, alt_up=N/A, landed={landed_flag}"
                )

            # If PX4's land detector says we're landed, disarm & go DONE
            if landed_flag and not self.sent_disarm_cmd:
                self.disarm()
                self.sent_disarm_cmd = True
                self.phase_start_time = t  # reuse as disarm_wait_start

            # After disarm command, wait a bit then mark DONE
            if self.sent_disarm_cmd and (t - self.phase_start_time) > 2.0:
                self.phase = Phase.DONE
                self.get_logger().info("FINAL_DESCEND -> DONE (landed & disarm requested).")

        # FINAL_AUTO_LAND: PX4 handles landing after NAV_LAND; we just monitor
        elif self.phase == Phase.FINAL_AUTO_LAND:
            if self.phase_start_time is None:
                self.phase_start_time = t

            # Keep streaming neutral offboard setpoint close to current position
            alt_up, vz_ned, valid = self.get_altitude_and_vz()
            if valid:
                cmd_alt_up_m = alt_up
                cmd_z_ned = -alt_up
                self.publish_trajectory_setpoint_z(cmd_z_ned)
            else:
                cmd_alt_up_m = None
                cmd_z_ned = None

            landed_flag = self.is_landed()

            if valid:
                self.get_logger().info(
                    f"Phase=FINAL_AUTO_LAND, alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, "
                    f"landed={landed_flag}"
                )
            else:
                self.get_logger().info(
                    f"Phase=FINAL_AUTO_LAND, alt_up=N/A, landed={landed_flag}"
                )

            # When PX4 says we're landed, request disarm and go DONE
            if landed_flag and not self.sent_disarm_cmd:
                self.disarm()
                self.sent_disarm_cmd = True
                self.phase_start_time = t

            if self.sent_disarm_cmd and (t - self.phase_start_time) > 2.0:
                self.phase = Phase.DONE
                self.get_logger().info("FINAL_AUTO_LAND -> DONE (landed & disarm requested).")

        # FINAL_PREC_LAND: currently just an alias for FINAL_AUTO_LAND path
        elif self.phase == Phase.FINAL_PREC_LAND:
            # For now we do the same as FINAL_AUTO_LAND; once you know the exact
            # Tracktor Beam mode / service call, you can split this out.
            if self.phase_start_time is None:
                self.phase_start_time = t

            alt_up, vz_ned, valid = self.get_altitude_and_vz()
            if valid:
                cmd_alt_up_m = alt_up
                cmd_z_ned = -alt_up
                self.publish_trajectory_setpoint_z(cmd_z_ned)
            else:
                cmd_alt_up_m = None
                cmd_z_ned = None

            landed_flag = self.is_landed()

            if valid:
                self.get_logger().info(
                    f"Phase=FINAL_PREC_LAND (fallback auto), alt_up={alt_up:.3f} m, "
                    f"vz_ned={vz_ned:.3f} m/s, landed={landed_flag}"
                )
            else:
                self.get_logger().info(
                    f"Phase=FINAL_PREC_LAND (fallback auto), alt_up=N/A, landed={landed_flag}"
                )

            if landed_flag and not self.sent_disarm_cmd:
                self.disarm()
                self.sent_disarm_cmd = True
                self.phase_start_time = t

            if self.sent_disarm_cmd and (t - self.phase_start_time) > 2.0:
                self.phase = Phase.DONE
                self.get_logger().info("FINAL_PREC_LAND -> DONE (landed & disarm requested).")

        # DONE: mission over, keep node alive but neutral
        elif self.phase == Phase.DONE:
            # Neutral setpoint to avoid surprises:
            self.publish_trajectory_setpoint_z(0.0)
            cmd_alt_up_m = 0.0
            cmd_z_ned = 0.0

            # Throttle logging to ~1 Hz
            if t - self.last_done_log_time > 1.0:
                self.last_done_log_time = t
                self.get_logger().info(
                    "Phase=DONE (mission complete). You can Ctrl+C to stop the node."
                )

        # ---------------------------------------------------------------------
        #  Log CSV row
        # ---------------------------------------------------------------------
        self.log_row(t, cmd_z_ned, cmd_alt_up_m)

    # -------------------------------------------------------------------------
    #  Cleanup
    # -------------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down G34 First Flight node.")
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


# -------------------------------------------------------------------------
#  main()
# -------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
