#!/usr/bin/env python3
import math
import os
import csv
from enum import Enum, auto
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleLandDetected,
    VehicleAttitude,
)

from std_srvs.srv import Trigger


class MissionPhase(Enum):
    PREFLIGHT = auto()
    PRE_OFFBOARD = auto()
    TAKEOFF_ASCEND = auto()
    HOVER = auto()
    ALT_TUNING = auto()
    ATT_TUNING = auto()
    LAND_AUTO = auto()
    PRECISION_LAND = auto()
    DONE = auto()


class G34FirstFlightNode(Node):
    def __init__(self):
        super().__init__("g34_first_flight_node")
        self.get_logger().info("G34 First Flight node initialized.")

        # --- Parameters ------------------------------------------------------
        self.local_position_topic = (
            self.declare_parameter(
                "local_position_topic",
                "/fmu/out/vehicle_local_position_v1"
            )
            .get_parameter_value()
            .string_value
        )

        self.attitude_topic = (
            self.declare_parameter(
                "attitude_topic",
                "/fmu/out/vehicle_attitude"
            )
            .get_parameter_value()
            .string_value
        )

        self.vehicle_status_topic = (
            self.declare_parameter(
                "vehicle_status_topic",
                "/fmu/out/vehicle_status_v1"
            )
            .get_parameter_value()
            .string_value
        )

        self.land_detected_topic = (
            self.declare_parameter(
                "land_detected_topic",
                "/fmu/out/vehicle_land_detected"
            )
            .get_parameter_value()
            .string_value
        )

        self.takeoff_altitude_m = (
            self.declare_parameter("takeoff_altitude_m", 0.5)
            .get_parameter_value()
            .double_value
        )

        self.hover_duration_s = (
            self.declare_parameter("hover_duration_s", 10.0)
            .get_parameter_value()
            .double_value
        )

        self.preoffboard_stream_s = (
            self.declare_parameter("preoffboard_stream_s", 2.0)
            .get_parameter_value()
            .double_value
        )

        self.takeoff_timeout_s = (
            self.declare_parameter("takeoff_timeout_s", 20.0)
            .get_parameter_value()
            .double_value
        )

        self.post_land_latched_wait_s = (
            self.declare_parameter("post_land_latched_wait_s", 3.0)
            .get_parameter_value()
            .double_value
        )

        # Tuning modes: "none", "altitude_step", "attitude_step"
        self.tuning_mode = (
            self.declare_parameter("tuning_mode", "none")
            .get_parameter_value()
            .string_value
        )

        # Altitude step tuning
        self.alt_step_amplitude_m = (
            self.declare_parameter("alt_step_amplitude_m", 0.1)
            .get_parameter_value()
            .double_value
        )
        self.alt_step_period_s = (
            self.declare_parameter("alt_step_period_s", 4.0)
            .get_parameter_value()
            .double_value
        )

        # Attitude (yaw) step tuning
        self.att_step_deg = (
            self.declare_parameter("att_step_deg", 10.0)
            .get_parameter_value()
            .double_value
        )
        self.att_step_period_s = (
            self.declare_parameter("att_step_period_s", 4.0)
            .get_parameter_value()
            .double_value
        )

        self.tuning_duration_s = (
            self.declare_parameter("tuning_duration_s", 20.0)
            .get_parameter_value()
            .double_value
        )

        # Final mode: "auto_land" or "precision_land"
        self.final_mode = (
            self.declare_parameter("final_mode", "auto_land")
            .get_parameter_value()
            .string_value
        )

        # Precision landing service (Tracktor-Beam)
        self.precision_land_service_name = (
            self.declare_parameter(
                "precision_land_service_name",
                "/tracktor_beam/start"
            )
            .get_parameter_value()
            .string_value
        )

        # CSV logging
        self.log_directory = (
            self.declare_parameter("log_directory", "")
            .get_parameter_value()
            .string_value
        )
        if not self.log_directory:
            self.log_directory = os.path.expanduser("~/g34_logs")

        # --- Internal state --------------------------------------------------
        self.mission_phase = MissionPhase.PREFLIGHT

        self.last_local_position = None
        self.last_attitude = None
        self.last_vehicle_status = None
        self.last_land_detected = None

        self.yaw_ref = None
        self.x_hold_ned = None
        self.y_hold_ned = None

        self.preoffboard_start_time = None
        self.takeoff_start_time = None
        self.hover_start_time = None
        self.tuning_start_time = None

        self.land_command_sent = False
        self.landed_latch_time = None
        self.disarm_sent = False

        self.precision_land_client = None
        self.precision_land_future = None

        # --- Logging setup ---------------------------------------------------
        self.log_enabled = False
        self.log_file = None
        self.csv_writer = None
        self._setup_csv_logging()

        # --- QoS and comms ---------------------------------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

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

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self.local_position_callback,
            qos,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            self.attitude_topic,
            self.attitude_callback,
            qos,
        )

        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            self.vehicle_status_topic,
            self.vehicle_status_callback,
            qos,
        )

        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            self.land_detected_topic,
            self.land_detected_callback,
            qos,
        )

        # Precision landing service client (optional)
        self.precision_land_client = self.create_client(
            Trigger, self.precision_land_service_name
        )

        # Main 10 Hz timer
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _setup_csv_logging(self):
        try:
            os.makedirs(self.log_directory, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(
                self.log_directory, f"g34_first_flight_{ts}.csv"
            )
            self.log_file = open(log_path, "w", newline="")
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(
                [
                    "t",
                    "phase",
                    "tuning_mode",
                    "final_mode",
                    "alt_up_m",
                    "vz_ned_mps",
                    "yaw_rad",
                    "cmd_z_ned_m",
                    "yaw_cmd_rad",
                    "landed_flag",
                    "nav_state",
                    "arming_state",
                ]
            )
            self.log_enabled = True
            self.get_logger().info(f"CSV logging enabled: {log_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to set up CSV logging: {e}")
            self.log_enabled = False

    def _log_sample(self, t, cmd_z_ned, yaw_cmd):
        if not self.log_enabled:
            return

        alt_up, vz_ned, landed = self._get_alt_vz_landed()
        yaw = None
        if self.last_attitude is not None:
            yaw = self._quat_to_yaw(self.last_attitude.q)

        nav_state = None
        arming_state = None
        if self.last_vehicle_status is not None:
            nav_state = self.last_vehicle_status.nav_state
            arming_state = self.last_vehicle_status.arming_state

        row = [
            f"{t:.3f}",
            self.mission_phase.name,
            self.tuning_mode,
            self.final_mode,
            f"{alt_up:.3f}" if alt_up is not None else "",
            f"{vz_ned:.3f}" if vz_ned is not None else "",
            f"{yaw:.3f}" if yaw is not None else "",
            f"{cmd_z_ned:.3f}" if cmd_z_ned is not None else "",
            f"{yaw_cmd:.3f}" if yaw_cmd is not None else "",
            "1" if landed else "0" if landed is not None else "",
            f"{nav_state}" if nav_state is not None else "",
            f"{arming_state}" if arming_state is not None else "",
        ]

        try:
            self.csv_writer.writerow(row)
            self.log_file.flush()
        except Exception as e:
            self.get_logger().warn(f"CSV write failed: {e}")
            self.log_enabled = False

    def _get_alt_vz_landed(self):
        alt_up = None
        vz_ned = None
        landed = None

        if self.last_local_position is not None:
            # PX4 NED frame: z is down, so altitude up = -z
            alt_up = -self.last_local_position.z
            vz_ned = self.last_local_position.vz

        if self.last_land_detected is not None:
            landed = bool(self.last_land_detected.landed)

        return alt_up, vz_ned, landed

    @staticmethod
    def _quat_to_yaw(q):
        # q = [w, x, y, z]
        if len(q) != 4:
            return 0.0
        w, x, y, z = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # -------------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------------
    def local_position_callback(self, msg: VehicleLocalPosition):
        self.last_local_position = msg

    def attitude_callback(self, msg: VehicleAttitude):
        self.last_attitude = msg
        if self.yaw_ref is None:
            yaw = self._quat_to_yaw(msg.q)
            self.yaw_ref = yaw
            self.get_logger().info(f"Yaw reference locked at {self.yaw_ref:.3f} rad")

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.last_vehicle_status = msg

    def land_detected_callback(self, msg: VehicleLandDetected):
        self.last_land_detected = msg

    # -------------------------------------------------------------------------
    # Publishing helpers
    # -------------------------------------------------------------------------
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self._now() * 1e6)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x_ned, y_ned, z_ned, yaw_cmd):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self._now() * 1e6)
        msg.position = [float(x_ned), float(y_ned), float(z_ned)]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.acceleration = [0.0, 0.0, 0.0]
        msg.yaw = float(yaw_cmd)
        self.trajectory_setpoint_pub.publish(msg)

    def send_vehicle_command(
        self, command, param1=0.0, param2=0.0, param3=0.0,
        param4=0.0, param5=0.0, param6=0.0, param7=0.0
    ):
        msg = VehicleCommand()
        msg.timestamp = int(self._now() * 1e6)
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

    def send_offboard_mode_command(self):
        # MAV_CMD_DO_SET_MODE: base_mode=1 (custom mode), main=6 (Offboard)
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,   # base mode: MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            param2=6.0,   # main mode: Offboard
            param3=0.0,   # submode not used here
        )
        self.get_logger().info("Sending OFFBOARD mode command.")

    def send_arm_command(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.get_logger().info("Sending ARM command.")

    def send_disarm_command(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
        )
        self.get_logger().info("Sending DISARM command.")

    def send_land_command(self):
        # Land at current location, PX4 handles descent + touchdown detection
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
            # params left at 0 => land at current position
        )
        self.get_logger().info("Sending NAV_LAND command (PX4 auto-land).")

    # -------------------------------------------------------------------------
    # Precision landing handoff (Tracktor-Beam)
    # -------------------------------------------------------------------------
    def start_precision_landing(self, now):
        if self.precision_land_client is None:
            self.get_logger().warn(
                "Precision landing client not created; falling back to auto_land."
            )
            self.send_land_command()
            self.land_command_sent = True
            self.mission_phase = MissionPhase.LAND_AUTO
            return

        # Try once to see if service is there; non-blocking wait
        if not self.precision_land_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                "Precision landing service not available; falling back to auto_land."
            )
            self.send_land_command()
            self.land_command_sent = True
            self.mission_phase = MissionPhase.LAND_AUTO
            return

        req = Trigger.Request()
        self.precision_land_future = self.precision_land_client.call_async(req)
        self.get_logger().info(
            f"Precision landing service '{self.precision_land_service_name}' called."
        )
        # From this point, Tracktor-Beam should own Offboard setpoints.
        self.mission_phase = MissionPhase.PRECISION_LAND

    # -------------------------------------------------------------------------
    # Land monitoring (used both for LAND_AUTO and PRECISION_LAND)
    # -------------------------------------------------------------------------
    def update_land_monitoring(self, now: float):
        _, _, landed = self._get_alt_vz_landed()

        if landed:
            if self.landed_latch_time is None:
                self.landed_latch_time = now
                self.get_logger().info(
                    "Landing detected (VehicleLandDetected.landed=True), "
                    "starting post-land timer."
                )
        # We *do not* reset latch if landed becomes False again; we want a
        # monotonic progression to disarm once we see landed at least once.

        if (
            self.landed_latch_time is not None
            and not self.disarm_sent
            and (now - self.landed_latch_time) >= self.post_land_latched_wait_s
        ):
            self.send_disarm_command()
            self.disarm_sent = True

    # -------------------------------------------------------------------------
    # Mission control
    # -------------------------------------------------------------------------
    def start_final_mode(self, now: float):
        """
        Called once when HOVER / tuning is complete to start the landing phase.
        """
        if self.final_mode == "precision_land":
            self.get_logger().info("Starting precision landing handoff.")
            self.start_precision_landing(now)
        else:
            # Default: PX4 auto-land
            self.get_logger().info("Starting PX4 auto-land.")
            self.send_land_command()
            self.land_command_sent = True
            self.mission_phase = MissionPhase.LAND_AUTO

    # -------------------------------------------------------------------------
    # Main timer (10 Hz)
    # -------------------------------------------------------------------------
    def timer_callback(self):
        now = self._now()
        alt_up, vz_ned, landed = self._get_alt_vz_landed()
        yaw_cmd = self.yaw_ref if self.yaw_ref is not None else 0.0
        cmd_z_ned = None  # what we send this cycle (if any)

        # --- PREFLIGHT: wait for position + attitude ------------------------
        if self.mission_phase == MissionPhase.PREFLIGHT:
            if self.last_local_position is None or self.last_attitude is None:
                self.get_logger().throttle(
                    2000,
                    "Waiting for local position and attitude...",
                )
                self._log_sample(now, cmd_z_ned, yaw_cmd)
                return

            # Lock x,y at current local position; start pre-offboard streaming
            self.x_hold_ned = self.last_local_position.x
            self.y_hold_ned = self.last_local_position.y
            if self.yaw_ref is None:
                self.yaw_ref = self._quat_to_yaw(self.last_attitude.q)

            self.preoffboard_start_time = now
            self.mission_phase = MissionPhase.PRE_OFFBOARD
            self.get_logger().info(
                f"PREFLIGHT -> PRE_OFFBOARD (x_hold={self.x_hold_ned:.2f}, "
                f"y_hold={self.y_hold_ned:.2f}, yaw_ref={self.yaw_ref:.2f})"
            )

        # --- PRE_OFFBOARD: stream setpoints before mode switch --------------
        if self.mission_phase == MissionPhase.PRE_OFFBOARD:
            # Hold just above ground (small negative z)
            z_ned_pre = -0.05
            cmd_z_ned = z_ned_pre

            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(
                self.x_hold_ned, self.y_hold_ned, cmd_z_ned, yaw_cmd
            )

            dt = now - self.preoffboard_start_time
            self.get_logger().throttle(
                2000,
                f"Phase=PRE_OFFBOARD, streaming setpoints for {dt:.1f} s",
            )

            if dt >= self.preoffboard_stream_s:
                self.send_offboard_mode_command()
                self.send_arm_command()
                self.takeoff_start_time = now
                self.mission_phase = MissionPhase.TAKEOFF_ASCEND
                self.get_logger().info("PRE_OFFBOARD -> TAKEOFF_ASCEND")

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- TAKEOFF_ASCEND: rise to takeoff_altitude_m ---------------------
        if self.mission_phase == MissionPhase.TAKEOFF_ASCEND:
            if self.x_hold_ned is None or self.yaw_ref is None:
                # Something went wrong; revert to PREFLIGHT
                self.get_logger().warn(
                    "Missing x_hold or yaw_ref in TAKEOFF_ASCEND, reverting to PREFLIGHT."
                )
                self.mission_phase = MissionPhase.PREFLIGHT
                self._log_sample(now, cmd_z_ned, yaw_cmd)
                return

            target_alt = self.takeoff_altitude_m
            cmd_z_ned = -target_alt

            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(
                self.x_hold_ned, self.y_hold_ned, cmd_z_ned, yaw_cmd
            )

            self.get_logger().info(
                f"Phase=TAKEOFF_ASCEND, cmd_z_ned={cmd_z_ned:+.3f} m, "
                f"alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, "
                f"landed={landed}"
            )

            # Check if we've reached altitude
            if alt_up is not None and alt_up >= 0.9 * target_alt:
                self.hover_start_time = now
                self.mission_phase = MissionPhase.HOVER
                self.get_logger().info(
                    f"TAKEOFF_ASCEND -> HOVER "
                    f"(alt_up={alt_up:.2f} m, target={target_alt:.2f} m)"
                )

            # Timeout safety
            if (
                self.takeoff_start_time is not None
                and (now - self.takeoff_start_time) > self.takeoff_timeout_s
            ):
                self.get_logger().warn(
                    "TAKEOFF_ASCEND timeout reached; continuing but "
                    "please verify altitude / thrust tuning."
                )

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- HOVER: hold altitude and yaw for a fixed duration --------------
        if self.mission_phase == MissionPhase.HOVER:
            target_alt = self.takeoff_altitude_m
            cmd_z_ned = -target_alt

            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(
                self.x_hold_ned, self.y_hold_ned, cmd_z_ned, yaw_cmd
            )

            dt_hover = now - self.hover_start_time
            self.get_logger().info(
                f"Phase=HOVER, t={dt_hover:.1f} s, cmd_z_ned={cmd_z_ned:+.3f} m, "
                f"alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s"
            )

            if dt_hover >= self.hover_duration_s:
                if self.tuning_mode == "altitude_step":
                    self.tuning_start_time = now
                    self.mission_phase = MissionPhase.ALT_TUNING
                    self.get_logger().info("HOVER -> ALT_TUNING")
                elif self.tuning_mode == "attitude_step":
                    self.tuning_start_time = now
                    self.mission_phase = MissionPhase.ATT_TUNING
                    self.get_logger().info("HOVER -> ATT_TUNING")
                else:
                    self.get_logger().info("HOVER complete, starting final mode.")
                    self.start_final_mode(now)

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- ALTITUDE TUNING: step alt around hover -------------------------
        if self.mission_phase == MissionPhase.ALT_TUNING:
            target_alt = self.takeoff_altitude_m
            amp = max(min(self.alt_step_amplitude_m, 0.5 * target_alt), 0.02)
            period = max(self.alt_step_period_s, 1.0)

            phase_t = (now - self.tuning_start_time) % period
            if phase_t < period / 2.0:
                alt_cmd = target_alt + amp / 2.0
            else:
                alt_cmd = target_alt - amp / 2.0
                alt_cmd = max(alt_cmd, 0.2)  # keep some margin above ground

            cmd_z_ned = -alt_cmd
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(
                self.x_hold_ned, self.y_hold_ned, cmd_z_ned, yaw_cmd
            )

            dt_tuning = now - self.tuning_start_time
            self.get_logger().info(
                f"Phase=ALT_TUNING, t={dt_tuning:.1f} s, alt_cmd={alt_cmd:.3f} m, "
                f"alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s"
            )

            if dt_tuning >= self.tuning_duration_s:
                self.get_logger().info("ALT_TUNING complete, starting final mode.")
                self.start_final_mode(now)

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- ATTITUDE (YAW) TUNING -----------------------------------------
        if self.mission_phase == MissionPhase.ATT_TUNING:
            target_alt = self.takeoff_altitude_m
            cmd_z_ned = -target_alt

            # yaw step around yaw_ref
            yaw_step_rad = math.radians(self.att_step_deg)
            period = max(self.att_step_period_s, 1.0)
            phase_t = (now - self.tuning_start_time) % period

            if phase_t < period / 2.0:
                yaw_cmd = self.yaw_ref + yaw_step_rad
            else:
                yaw_cmd = self.yaw_ref - yaw_step_rad

            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(
                self.x_hold_ned, self.y_hold_ned, cmd_z_ned, yaw_cmd
            )

            dt_tuning = now - self.tuning_start_time
            self.get_logger().info(
                f"Phase=ATT_TUNING, t={dt_tuning:.1f} s, yaw_cmd={yaw_cmd:.3f} rad, "
                f"alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s"
            )

            if dt_tuning >= self.tuning_duration_s:
                self.get_logger().info("ATT_TUNING complete, starting final mode.")
                self.start_final_mode(now)

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- LAND_AUTO: PX4 auto-land & disarm ------------------------------
        if self.mission_phase == MissionPhase.LAND_AUTO:
            # Do NOT publish Offboard setpoints here; PX4 is handling descent.
            self.update_land_monitoring(now)

            self.get_logger().throttle(
                2000,
                f"Phase=LAND_AUTO, alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, "
                f"landed={landed}, disarm_sent={self.disarm_sent}",
            )

            # Transition to DONE once disarm has been sent
            if self.disarm_sent:
                self.mission_phase = MissionPhase.DONE
                self.get_logger().info("LAND_AUTO complete -> DONE")

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- PRECISION_LAND: Tracktor-Beam owns Offboard --------------------
        if self.mission_phase == MissionPhase.PRECISION_LAND:
            # We do not publish Offboard setpoints here; Tracktor-Beam does.
            # Just monitor landing and disarm.
            if (
                self.precision_land_future is not None
                and self.precision_land_future.done()
            ):
                try:
                    resp = self.precision_land_future.result()
                    if resp.success:
                        self.get_logger().info(
                            f"Precision landing service reports success: {resp.message}"
                        )
                    else:
                        self.get_logger().warn(
                            f"Precision landing service reports failure: {resp.message}"
                        )
                except Exception as e:
                    self.get_logger().warn(
                        f"Precision landing service call failed: {e}"
                    )
                finally:
                    self.precision_land_future = None

            self.update_land_monitoring(now)

            self.get_logger().throttle(
                2000,
                f"Phase=PRECISION_LAND, alt_up={alt_up:.3f} m, vz_ned={vz_ned:.3f} m/s, "
                f"landed={landed}, disarm_sent={self.disarm_sent}",
            )

            if self.disarm_sent:
                self.mission_phase = MissionPhase.DONE
                self.get_logger().info("PRECISION_LAND complete -> DONE")

            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

        # --- DONE: everything finished --------------------------------------
        if self.mission_phase == MissionPhase.DONE:
            # No control, just logging every so often
            self.get_logger().throttle(
                5000,
                "Mission DONE. Node is idle; you can stop the launch when ready.",
            )
            self._log_sample(now, cmd_z_ned, yaw_cmd)
            return

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down G34 First Flight node.")
        try:
            if self.log_file is not None:
                self.log_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
