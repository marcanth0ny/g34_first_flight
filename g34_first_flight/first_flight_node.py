#!/usr/bin/env python3

import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleLandDetected,
    VehicleAttitude,
)


class FlightPhase(Enum):
    PREFLIGHT = auto()
    TAKEOFF_ASCEND = auto()
    HOVER = auto()
    DESCEND = auto()
    FINAL_LAND = auto()
    DONE = auto()


class G34FirstFlightNode(Node):
    def __init__(self):
        super().__init__('g34_first_flight_node')

        # ---------------- Parameters ----------------
        # Basic altitude mission
        self.declare_parameter('takeoff_height_m', 0.5)
        self.declare_parameter('altitude_tolerance_m', 0.05)
        self.declare_parameter('takeoff_min_time_s', 1.5)
        self.declare_parameter('takeoff_timeout_s', 10.0)
        self.declare_parameter('position_hold_time_s', 5.0)
        self.declare_parameter('offboard_warmup_setpoints', 20)

        # Landing detection / timing
        self.declare_parameter('landing_trigger_alt_m', 0.15)    # when to send NAV_LAND
        self.declare_parameter('landing_vz_thresh_mps', 0.10)    # small vertical speed
        self.declare_parameter('landing_done_delay_s', 0.5)      # delay after landed=True before disarm
        self.declare_parameter('landing_phase_timeout_s', 20.0)  # safety timeout in FINAL_LAND

        # Optional tuning
        self.declare_parameter('tuning_mode', 'none')            # 'none', 'altitude', 'attitude'
        self.declare_parameter('tuning_step_amplitude_m', 0.2)
        self.declare_parameter('tuning_step_duration_s', 3.0)

        # Topic for local position (v1 vs non-v1)
        self.declare_parameter('local_position_topic', '/fmu/out/vehicle_local_position_v1')

        # Read parameters
        self.takeoff_height_m = float(self.get_parameter('takeoff_height_m').value)
        self.altitude_tolerance_m = float(self.get_parameter('altitude_tolerance_m').value)
        self.takeoff_min_time_s = float(self.get_parameter('takeoff_min_time_s').value)
        self.takeoff_timeout_s = float(self.get_parameter('takeoff_timeout_s').value)
        self.position_hold_time_s = float(self.get_parameter('position_hold_time_s').value)
        self.offboard_warmup_setpoints = int(self.get_parameter('offboard_warmup_setpoints').value)

        self.landing_trigger_alt_m = float(self.get_parameter('landing_trigger_alt_m').value)
        self.landing_vz_thresh_mps = float(self.get_parameter('landing_vz_thresh_mps').value)
        self.landing_done_delay_s = float(self.get_parameter('landing_done_delay_s').value)
        self.landing_phase_timeout_s = float(self.get_parameter('landing_phase_timeout_s').value)

        self.tuning_mode = str(self.get_parameter('tuning_mode').value).lower()
        self.tuning_step_amplitude_m = float(self.get_parameter('tuning_step_amplitude_m').value)
        self.tuning_step_duration_s = float(self.get_parameter('tuning_step_duration_s').value)

        local_position_topic = self.get_parameter('local_position_topic').value

        # ---------------- PX4 QoS ----------------
        qos_px4 = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- Publishers to PX4 ----------------
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4
        )

        # ---------------- Subscribers from PX4 ----------------
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, local_position_topic, self.local_position_callback, qos_px4
        )
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected, '/fmu/out/vehicle_land_detected', self.land_detected_callback, qos_px4
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_px4
        )

        # ---------------- State variables ----------------
        self.flight_phase = FlightPhase.PREFLIGHT
        self.offboard_setpoint_counter = 0

        self.current_altitude_up = None     # m (up-positive)
        self.current_vz_ned = None          # m/s (down-positive)
        self.landed_flag = False            # PX4 land detector

        self.current_yaw = None             # rad
        self.initial_yaw = None             # rad (captured pre-takeoff)

        self.last_status_print_time = 0.0

        self.mission_start_time = None
        self.hover_start_time = None
        self.descend_start_time = None
        self.final_land_start_time = None
        self.landed_time = None             # when PX4 first says landed=True

        # Main loop timer
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info('G34 First Flight node initialized.')

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def local_position_callback(self, msg: VehicleLocalPosition):
        # NED frame: z positive down; altitude up = -z
        self.current_altitude_up = -msg.z
        self.current_vz_ned = msg.vz

    def land_detected_callback(self, msg: VehicleLandDetected):
        prev = self.landed_flag
        self.landed_flag = bool(msg.landed)
        if self.landed_flag and not prev:
            self.landed_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info('PX4 land detector: landed=True')

    def attitude_callback(self, msg: VehicleAttitude):
        # PX4 quaternion order: [w, x, y, z]
        w, x, y, z = msg.q
        # yaw from quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_yaw = yaw
        if self.initial_yaw is None:
            self.initial_yaw = yaw

    # -------------------------------------------------------------------------
    # VehicleCommand helpers
    # -------------------------------------------------------------------------
    def publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.get_logger().info('Sending ARM command.')
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)

    def disarm(self):
        self.get_logger().info('Sending DISARM command.')
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 0.0)

    def set_offboard_mode(self):
        self.get_logger().info('Sending OFFBOARD mode command.')
        # Param1 = 1 (custom), Param2 = 6 (Offboard)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def command_nav_land(self):
        self.get_logger().info('Sending NAV_LAND command to PX4.')
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # Offboard control publishers
    # -------------------------------------------------------------------------
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x_ned: float, y_ned: float, z_ned: float):
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = [float(x_ned), float(y_ned), float(z_ned)]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        # Hold initial yaw to avoid yaw slewing on takeoff
        if self.initial_yaw is not None:
            msg.yaw = float(self.initial_yaw)
        else:
            msg.yaw = math.nan
        self.trajectory_setpoint_pub.publish(msg)

    # -------------------------------------------------------------------------
    # Tuning profiles
    # -------------------------------------------------------------------------
    def altitude_tuning_profile(self, t_rel: float) -> float:
        """
        Altitude step sequence around nominal hover height.
        """
        h0 = self.takeoff_height_m
        amp = self.tuning_step_amplitude_m
        seg = self.tuning_step_duration_s

        if t_rel < seg:
            return h0
        elif t_rel < 2 * seg:
            return h0 + amp
        elif t_rel < 3 * seg:
            return h0 - amp
        else:
            return h0

    def attitude_tuning_profile(self, t_rel: float):
        """
        Lateral position step sequence for indirect attitude tuning.
        Returns (x_cmd, y_cmd) in NED (m).
        """
        amp = self.tuning_step_amplitude_m
        seg = self.tuning_step_duration_s

        if t_rel < seg:
            return 0.0, 0.0
        elif t_rel < 2 * seg:
            return amp, 0.0      # forward
        elif t_rel < 3 * seg:
            return -amp, 0.0     # backward
        elif t_rel < 4 * seg:
            return 0.0, amp      # right
        elif t_rel < 5 * seg:
            return 0.0, -amp     # left
        else:
            return 0.0, 0.0

    # -------------------------------------------------------------------------
    # Main timer / state machine
    # -------------------------------------------------------------------------
    def timer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9  # seconds

        # Always stream Offboard heartbeat
        self.publish_offboard_control_mode()

        # Default NED position
        x_cmd = 0.0
        y_cmd = 0.0
        z_cmd = 0.0

        # Warmup setpoints before Offboard
        if self.offboard_setpoint_counter < self.offboard_warmup_setpoints:
            self.publish_trajectory_setpoint(x_cmd, y_cmd, z_cmd)
            self.offboard_setpoint_counter += 1
            return

        # Enter Offboard + ARM once
        if self.flight_phase == FlightPhase.PREFLIGHT:
            self.set_offboard_mode()
            self.arm()
            self.flight_phase = FlightPhase.TAKEOFF_ASCEND
            self.mission_start_time = now
            self.get_logger().info('PREFLIGHT -> TAKEOFF_ASCEND')

        # ---------------- Phase logic ----------------
        if self.flight_phase == FlightPhase.TAKEOFF_ASCEND:
            # Command target hover altitude
            z_cmd = -self.takeoff_height_m
            time_in_phase = now - (self.mission_start_time or now)

            alt_ok = False
            if self.current_altitude_up is not None:
                alt_err = abs(self.current_altitude_up - self.takeoff_height_m)
                alt_ok = alt_err < self.altitude_tolerance_m

            if alt_ok and time_in_phase > self.takeoff_min_time_s:
                self.flight_phase = FlightPhase.HOVER
                self.hover_start_time = now
                self.get_logger().info(
                    f'TAKEOFF_ASCEND -> HOVER (alt_up={self.current_altitude_up:.2f} m)'
                )
            elif time_in_phase > self.takeoff_timeout_s:
                self.flight_phase = FlightPhase.HOVER
                self.hover_start_time = now
                self.get_logger().warn(
                    'TAKEOFF_ASCEND timeout -> HOVER '
                    f'(alt_up={self.current_altitude_up if self.current_altitude_up is not None else float("nan"):.2f} m)'
                )

        elif self.flight_phase == FlightPhase.HOVER:
            t_hover = now - (self.hover_start_time or now)
            alt_cmd = self.takeoff_height_m

            if self.tuning_mode == 'altitude':
                alt_cmd = self.altitude_tuning_profile(t_hover)

            if self.tuning_mode == 'attitude':
                x_cmd, y_cmd = self.attitude_tuning_profile(t_hover)

            z_cmd = -alt_cmd

            if t_hover > self.position_hold_time_s:
                self.flight_phase = FlightPhase.DESCEND
                self.descend_start_time = now
                self.get_logger().info('HOVER -> DESCEND')

        elif self.flight_phase == FlightPhase.DESCEND:
            # Simple approach: command back to "ground" z=0
            z_cmd = 0.0
            time_in_phase = now - (self.descend_start_time or now)

            # Once we're close to ground, we hand over to native PX4 land logic
            alt_ok_for_land = (
                self.current_altitude_up is not None
                and self.current_altitude_up < self.landing_trigger_alt_m
                and self.current_altitude_up > -0.05  # avoid negative nonsense
            )
            vz_small = (
                self.current_vz_ned is not None
                and abs(self.current_vz_ned) < 1.0  # just to avoid crazy transitions
            )

            if alt_ok_for_land and vz_small:
                self.command_nav_land()
                self.flight_phase = FlightPhase.FINAL_LAND
                self.final_land_start_time = now
                self.get_logger().info(
                    f'DESCEND -> FINAL_LAND (alt_up={self.current_altitude_up:.3f} m)'
                )

        elif self.flight_phase == FlightPhase.FINAL_LAND:
            # In FINAL_LAND we **do not fight PX4**. We just keep a benign setpoint near ground.
            z_cmd = 0.0
            time_in_phase = now - (self.final_land_start_time or now)

            # If PX4 reports landed, disarm after a short delay
            if self.landed_flag and self.landed_time is not None:
                if now - self.landed_time >= self.landing_done_delay_s:
                    self.disarm()
                    self.flight_phase = FlightPhase.DONE
                    self.get_logger().info(
                        'FINAL_LAND -> DONE (PX4 landed=True and disarm sent)'
                    )

            # Safety timeout: if something is weird, but PX4 *still* says not landed,
            # we DO NOT force disarm (to stay safe). The pilot can take over.
            if time_in_phase > self.landing_phase_timeout_s and not self.landed_flag:
                self.get_logger().warn(
                    'FINAL_LAND timeout: PX4 still not landed. '
                    'Holding DONE without disarm; operator intervention required.'
                )
                self.flight_phase = FlightPhase.DONE

        elif self.flight_phase == FlightPhase.DONE:
            z_cmd = 0.0  # keep a harmless setpoint

        # Publish the currently commanded setpoint
        self.publish_trajectory_setpoint(x_cmd, y_cmd, z_cmd)

        # ---------------- Status printout (2 Hz) ----------------
        if now - self.last_status_print_time > 0.5:
            alt_str = (
                f'{self.current_altitude_up:.3f} m'
                if self.current_altitude_up is not None
                else 'N/A'
            )
            vz_str = (
                f'{self.current_vz_ned:.3f} m/s'
                if self.current_vz_ned is not None
                else 'N/A'
            )
            self.get_logger().info(
                f'Phase={self.flight_phase.name}, '
                f'cmd_z_ned={z_cmd:.3f} m, alt_up={alt_str}, vz_ned={vz_str}, '
                f'landed={self.landed_flag}'
            )
            self.last_status_print_time = now


def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
