#!/usr/bin/env python3

import os
import csv
import math
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleLandDetected,
    VehicleStatus,
)


class Phase:
    PREFLIGHT       = "PREFLIGHT"
    TAKEOFF_ASCEND  = "TAKEOFF_ASCEND"
    HOVER           = "HOVER"
    ALT_TUNE        = "ALT_TUNE"
    ATT_TUNE        = "ATT_TUNE"
    DESCEND         = "DESCEND"
    AUTO_LAND       = "AUTO_LAND"
    PRECISION_LAND  = "PRECISION_LAND"
    DONE            = "DONE"


class G34FirstFlightNode(Node):
    """
    G34 First Flight ROS2 node

    - Streams position setpoints (0,0,z) in NED for PX4 OFFBOARD position control
    - Mission profile:
        PREFLIGHT (pre-stream)
        → TAKEOFF_ASCEND (to takeoff_altitude_m)
        → HOVER
        → optional ALT_TUNE or ATT_TUNE
        → final_mode: DESCEND | AUTO_LAND | PRECISION_LAND
    - CSV logging for post-flight analysis (altitude, velocities, phase, etc.)
    """

    def __init__(self):
        super().__init__("g34_first_flight_node")

        # === Parameters ===
        self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position_v1")
        self.declare_parameter("takeoff_altitude_m", 0.5)
        self.declare_parameter("tuning_mode", "none")         # "none" | "altitude" | "attitude"
        self.declare_parameter("final_mode", "descend")       # "descend" | "auto_land" | "precision_land"
        self.declare_parameter("enable_csv_logging", True)

        if self.has_parameter("use_sim_time"):
            self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        else:
            self.use_sim_time = False

        self.local_position_topic = self.get_parameter("local_position_topic").get_parameter_value().string_value
        self.takeoff_altitude_m = float(self.get_parameter("takeoff_altitude_m").get_parameter_value().double_value)
        self.tuning_mode = self.get_parameter("tuning_mode").get_parameter_value().string_value.lower()
        self.final_mode = self.get_parameter("final_mode").get_parameter_value().string_value.lower()
        self.enable_csv_logging = self.get_parameter("enable_csv_logging").get_parameter_value().bool_value

        # Clamp / sanity
        if self.takeoff_altitude_m < 0.2:
            self.get_logger().warn("takeoff_altitude_m too small, clamping to 0.2 m")
            self.takeoff_altitude_m = 0.2

        # === PX4 QoS (BEST_EFFORT, depth=1) ===
        self.px4_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # === Publishers ===
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )

        # === Subscribers ===
        # Local position (z up = -z_ned)
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self.local_position_callback,
            self.px4_qos,
        )
        # Landed state
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self.land_detected_callback,
            self.px4_qos,
        )
        # Vehicle status (nav_state, arming_state)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            self.px4_qos,
        )

        # === State variables ===
        self.phase = Phase.PREFLIGHT
        self.phase_start_time = self.get_clock().now().nanoseconds / 1e9
        self.node_start_time = self.phase_start_time

        self.local_position = None
        self.land_detected = None
        self.vehicle_status = None

        # Reference position at liftoff / hover
        self.ref_x = None
        self.ref_y = None

        # Commanded setpoint in NED
        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self.cmd_z_ned = 0.0
        self.cmd_yaw = 0.0

        # Offboard / arming logic
        self.preflight_setpoint_counter = 0
        self.preflight_setpoint_required = 20  # ~1 s at 20 Hz
        self.offboard_request_sent = False
        self.arm_request_sent = False

        # Tuning configs
        # Altitude tuning
        self.alt_tune_step_ampl_m = 0.2   # step amplitude above hover
        self.alt_tune_hold_s = 6.0
        self.alt_tune_done = False
        self.alt_tune_state = None  # "UP" | "HOLD" | "DOWN"
        self.alt_tune_state_start = None

        # Attitude tuning (via lateral position steps)
        self.att_amp_m = 0.3
        self.att_step_duration_s = 4.0
        self.att_center_duration_s = 4.0
        self.att_sequence = [
            ("ROLL_POS",  +self.att_amp_m, 0.0),
            ("ROLL_NEG",  -self.att_amp_m, 0.0),
            ("PITCH_POS", 0.0, +self.att_amp_m),
            ("PITCH_NEG", 0.0, -self.att_amp_m),
        ]
        self.att_step_index = 0
        self.att_step_phase = None  # "STEP" | "CENTER"
        self.att_step_phase_start = None

        # Hover duration before tuning / final
        self.hover_min_time_s = 5.0

        # Descent / landing thresholds
        self.land_alt_threshold_m = 0.05  # below this alt_up consider near ground
        self.land_vz_threshold_mps = 0.10

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        if self.enable_csv_logging:
            self.setup_csv_logging()

        # Logging throttle
        self.last_info_log_time = self.node_start_time
        self.info_log_period = 0.5  # seconds

        # Timer (20 Hz)
        self.timer_period_sec = 0.05
        self.timer = self.create_timer(self.timer_period_sec, self.timer_callback)

        self.get_logger().info(
            f"tuning_mode='{self.tuning_mode}', final_mode='{self.final_mode}', "
            f"takeoff_altitude_m={self.takeoff_altitude_m:.2f} m"
        )
        self.get_logger().info("G34 First Flight node initialized.")

    # =========================
    #   Callbacks
    # =========================

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.local_position = msg
        if self.ref_x is None:
            # Lock reference position for hover and tuning
            self.ref_x = msg.x
            self.ref_y = msg.y

    def land_detected_callback(self, msg: VehicleLandDetected):
        self.land_detected = msg

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.vehicle_status = msg

    # =========================
    #   CSV logging
    # =========================

    def setup_csv_logging(self):
        try:
            home = os.path.expanduser("~")
            log_dir = os.path.join(home, "g34_logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(log_dir, f"g34_first_flight_{timestamp_str}.csv")

            self.csv_file = open(filename, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "t_sec",
                "phase",
                "alt_up_m",
                "cmd_z_ned_m",
                "pos_x_m",
                "pos_y_m",
                "pos_z_ned_m",
                "vel_x_ned_mps",
                "vel_y_ned_mps",
                "vel_z_ned_mps",
                "nav_state",
                "arming_state",
                "landed",
            ])
            self.get_logger().info(f"CSV logging enabled: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to open CSV log: {e}")
            self.csv_file = None
            self.csv_writer = None

    def log_csv(self, t_sec, alt_up, vz_ned):
        if self.csv_writer is None:
            return

        lp = self.local_position
        vs = self.vehicle_status
        ld = self.land_detected

        pos_x = lp.x if lp is not None else float("nan")
        pos_y = lp.y if lp is not None else float("nan")
        pos_z = lp.z if lp is not None else float("nan")
        vx = lp.vx if lp is not None else float("nan")
        vy = lp.vy if lp is not None else float("nan")
        vz = lp.vz if lp is not None else float("nan")

        nav_state = vs.nav_state if vs is not None else -1
        arming_state = vs.arming_state if vs is not None else -1
        landed = ld.landed if ld is not None else False

        self.csv_writer.writerow([
            f"{t_sec:.3f}",
            self.phase,
            f"{alt_up:.3f}" if alt_up is not None else "nan",
            f"{self.cmd_z_ned:.3f}",
            f"{pos_x:.3f}",
            f"{pos_y:.3f}",
            f"{pos_z:.3f}",
            f"{vx:.3f}",
            f"{vy:.3f}",
            f"{vz:.3f}",
            nav_state,
            arming_state,
            int(landed),
        ])
        # Flush occasionally to avoid losing data
        if int(t_sec * 10) % 10 == 0:
            self.csv_file.flush()

    # =========================
    #   PX4 Commands
    # =========================

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
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # [us]
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def request_offboard_mode(self):
        # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
        # PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
            param3=0.0,
        )
        self.get_logger().info("Sending OFFBOARD mode command.")

    def send_arm_command(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=float(VehicleCommand.ARMING_ACTION_ARM),
        )
        self.get_logger().info("Sending ARM command.")

    def send_disarm_command(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=float(VehicleCommand.ARMING_ACTION_DISARM),
        )
        self.get_logger().info("Sending DISARM command.")

    def send_nav_land(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Sending NAV_LAND command (PX4 auto land).")

    def send_nav_precland(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_PRECLAND)
        self.get_logger().info("Sending NAV_PRECLAND command (for precision landing).")

    # =========================
    #   Helpers
    # =========================

    def change_phase(self, new_phase: str):
        now = self.get_clock().now().nanoseconds / 1e9
        self.phase = new_phase
        self.phase_start_time = now
        self.get_logger().info(f"Phase -> {self.phase}")

    def phase_time(self) -> float:
        now = self.get_clock().now().nanoseconds / 1e9
        return now - self.phase_start_time

    def get_altitude_up_and_vz(self):
        """Return (alt_up, vz_ned) where alt_up is positive up from origin."""
        if self.local_position is None:
            return None, None
        alt_up = -self.local_position.z  # NED z down
        vz_ned = self.local_position.vz
        return alt_up, vz_ned

    # =========================
    #   Timer / State Machine
    # =========================

    def timer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9
        t_since_start = now - self.node_start_time

        # Need local position before we do anything
        alt_up, vz_ned = self.get_altitude_up_and_vz()
        if self.local_position is None:
            if now - self.last_info_log_time > 1.0:
                self.get_logger().info("Waiting for /vehicle_local_position...")
                self.last_info_log_time = now
            return

        # Initialize reference position
        if self.ref_x is None:
            self.ref_x = self.local_position.x
            self.ref_y = self.local_position.y

        # === High-level state machine ===
        if self.phase == Phase.PREFLIGHT:
            # stream neutral setpoints at ground level
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = 0.0

            self.preflight_setpoint_counter += 1

            if not self.offboard_request_sent and \
               self.preflight_setpoint_counter >= self.preflight_setpoint_required:
                self.request_offboard_mode()
                self.offboard_request_sent = True
                self.offboard_request_time = now

            if self.offboard_request_sent and not self.arm_request_sent and \
               (now - self.offboard_request_time) > 0.2:
                self.send_arm_command()
                self.arm_request_sent = True

            # Transition to TAKEOFF once we see OFFBOARD + ARMED
            if self.vehicle_status is not None:
                vs = self.vehicle_status
                if vs.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and \
                   vs.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                    self.change_phase(Phase.TAKEOFF_ASCEND)

        elif self.phase == Phase.TAKEOFF_ASCEND:
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = -self.takeoff_altitude_m

            # Wait until we are close enough to target altitude
            if alt_up is not None:
                alt_err = abs(alt_up - self.takeoff_altitude_m)
                if alt_up > 0.9 * self.takeoff_altitude_m and alt_err < 0.05 and self.phase_time() > 2.0:
                    self.change_phase(Phase.HOVER)

            # Safety timeout
            if self.phase_time() > 20.0:
                self.get_logger().warn("TAKEOFF_ASCEND timeout, moving to HOVER.")
                self.change_phase(Phase.HOVER)

        elif self.phase == Phase.HOVER:
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = -self.takeoff_altitude_m

            if self.phase_time() < self.hover_min_time_s:
                pass  # just hold
            else:
                # Decide next mode based on tuning_mode
                if self.tuning_mode == "altitude":
                    self.start_altitude_tuning(now)
                elif self.tuning_mode == "attitude":
                    self.start_attitude_tuning(now)
                else:
                    # no tuning -> final mode
                    self.start_final_mode()

        elif self.phase == Phase.ALT_TUNE:
            self.update_altitude_tuning(now, alt_up)

        elif self.phase == Phase.ATT_TUNE:
            self.update_attitude_tuning(now)

        elif self.phase == Phase.DESCEND:
            # Controlled descend to z=0 in OFFBOARD
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = 0.0

            landed = self.land_detected.landed if self.land_detected is not None else False
            alt_near_ground = (alt_up is not None and abs(alt_up) < self.land_alt_threshold_m)
            vz_small = (vz_ned is not None and abs(vz_ned) < self.land_vz_threshold_mps)

            if landed or (alt_near_ground and vz_small and self.phase_time() > 2.0):
                # We can disarm via PX4 (optional)
                self.send_disarm_command()
                self.change_phase(Phase.DONE)

        elif self.phase == Phase.AUTO_LAND:
            # We do NOT send offboard setpoints here, PX4 handles landing
            # Just monitor for landed and then finish.
            landed = self.land_detected.landed if self.land_detected is not None else False
            if landed and self.phase_time() > 2.0:
                self.get_logger().info("PX4 reports landed (AUTO_LAND). Mission complete.")
                self.change_phase(Phase.DONE)

        elif self.phase == Phase.PRECISION_LAND:
            # Same as AUTO_LAND, but initiated with NAV_PRECLAND
            landed = self.land_detected.landed if self.land_detected is not None else False
            if landed and self.phase_time() > 2.0:
                self.get_logger().info("PX4 reports landed (PRECISION_LAND). Mission complete.")
                self.change_phase(Phase.DONE)

        elif self.phase == Phase.DONE:
            # Hold neutral setpoint just in case; props should be off if disarmed
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = 0.0
            # Node will keep running until user Ctrl-C

        # === Publish OFFBOARD setpoints where appropriate ===
        if self.phase not in (Phase.AUTO_LAND, Phase.PRECISION_LAND):
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint()

        # === Logging (console + CSV) ===
        if now - self.last_info_log_time > self.info_log_period:
            self.get_logger().info(
                f"Phase={self.phase}, cmd_z_ned={self.cmd_z_ned:+.3f} m, "
                f"alt_up={'N/A' if alt_up is None else f'{alt_up:+.3f} m'}, "
                f"vz_ned={'N/A' if vz_ned is None else f'{vz_ned:+.3f} m/s'}"
            )
            self.last_info_log_time = now

        self.log_csv(t_since_start, alt_up if alt_up is not None else float("nan"), vz_ned)

    # =========================
    #   Altitude tuning
    # =========================

    def start_altitude_tuning(self, now: float):
        # Simple 1-step up and back down for vertical response
        self.alt_tune_done = False
        self.alt_tune_state = "UP"
        self.alt_tune_state_start = now
        self.alt_step_target = self.takeoff_altitude_m + self.alt_tune_step_ampl_m

        self.get_logger().info(
            f"Starting altitude tuning: step from {self.takeoff_altitude_m:.2f} m "
            f"to {self.alt_step_target:.2f} m"
        )
        self.change_phase(Phase.ALT_TUNE)

    def update_altitude_tuning(self, now: float, alt_up: float):
        if self.alt_tune_state is None:
            self.alt_tune_state = "UP"
            self.alt_tune_state_start = now

        if self.alt_tune_state == "UP":
            # Command higher altitude
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = -self.alt_step_target

            if alt_up is not None:
                if alt_up > 0.9 * self.alt_step_target and abs(alt_up - self.alt_step_target) < 0.05:
                    # Reached near step target -> start hold
                    self.alt_tune_state = "HOLD"
                    self.alt_tune_state_start = now
                    self.get_logger().info("Altitude tuning: reached step altitude, starting HOLD.")

        elif self.alt_tune_state == "HOLD":
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = -self.alt_step_target

            if (now - self.alt_tune_state_start) > self.alt_tune_hold_s:
                # Step back down to original hover altitude
                self.alt_tune_state = "DOWN"
                self.alt_tune_state_start = now
                self.get_logger().info("Altitude tuning: stepping back down to hover altitude.")

        elif self.alt_tune_state == "DOWN":
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = -self.takeoff_altitude_m

            if alt_up is not None and abs(alt_up - self.takeoff_altitude_m) < 0.05 and \
               (now - self.alt_tune_state_start) > 3.0:
                self.get_logger().info("Altitude tuning complete, proceeding to final mode.")
                self.alt_tune_done = True
                self.start_final_mode()

    # =========================
    #   Attitude tuning (via lateral position steps)
    # =========================

    def start_attitude_tuning(self, now: float):
        self.att_step_index = 0
        self.att_step_phase = "STEP"
        self.att_step_phase_start = now

        self.get_logger().info(
            "Starting attitude tuning: lateral position steps in X/Y "
            "(excites roll/pitch loops)."
        )
        self.change_phase(Phase.ATT_TUNE)

    def update_attitude_tuning(self, now: float):
        if self.att_step_index >= len(self.att_sequence):
            self.get_logger().info("Attitude tuning sequence complete, proceeding to final mode.")
            self.start_final_mode()
            return

        step_name, dx, dy = self.att_sequence[self.att_step_index]

        if self.att_step_phase == "STEP":
            self.cmd_x = self.ref_x + dx
            self.cmd_y = self.ref_y + dy
            self.cmd_z_ned = -self.takeoff_altitude_m

            if (now - self.att_step_phase_start) >= self.att_step_duration_s:
                self.att_step_phase = "CENTER"
                self.att_step_phase_start = now
                self.get_logger().info(f"Attitude tuning: completed {step_name} step, centering.")

        elif self.att_step_phase == "CENTER":
            self.cmd_x = self.ref_x
            self.cmd_y = self.ref_y
            self.cmd_z_ned = -self.takeoff_altitude_m

            if (now - self.att_step_phase_start) >= self.att_center_duration_s:
                self.att_step_index += 1
                self.att_step_phase = "STEP"
                self.att_step_phase_start = now
                if self.att_step_index < len(self.att_sequence):
                    next_name, _, _ = self.att_sequence[self.att_step_index]
                    self.get_logger().info(f"Attitude tuning: starting {next_name} step.")

    # =========================
    #   Final modes
    # =========================

    def start_final_mode(self):
        """
        Decide how to finish the mission:
        - 'descend'       : remain in OFFBOARD, descend to z=0, then disarm.
        - 'auto_land'     : hand back to PX4 with NAV_LAND.
        - 'precision_land': hand back to PX4 with NAV_PRECLAND (Tracktor-Beam / precland).
        """
        if self.final_mode == "descend":
            self.change_phase(Phase.DESCEND)

        elif self.final_mode == "auto_land":
            # Stop OFFBOARD control and let PX4 auto land
            self.send_nav_land()
            self.change_phase(Phase.AUTO_LAND)

        elif self.final_mode == "precision_land":
            # This is where Tracktor-Beam / precland hooks in.
            # We simply request NAV_PRECLAND and let PX4 + your mode plugin handle it.
            self.send_nav_precland()
            self.change_phase(Phase.PRECISION_LAND)

        else:
            self.get_logger().warn(
                f"Unknown final_mode='{self.final_mode}', defaulting to 'descend'."
            )
            self.change_phase(Phase.DESCEND)

    # =========================
    #   Offboard publishers
    # =========================

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.x = float(self.cmd_x)
        msg.y = float(self.cmd_y)
        msg.z = float(self.cmd_z_ned)
        msg.yaw = float(self.cmd_yaw)
        self.trajectory_setpoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
