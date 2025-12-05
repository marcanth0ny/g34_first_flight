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
)


class FlightPhase(Enum):
    PREFLIGHT = auto()
    TAKEOFF_ASCEND = auto()
    HOVER = auto()
    DESCEND = auto()
    DONE = auto()


class G34FirstFlightNode(Node):
    def __init__(self):
        super().__init__('g34_first_flight_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('takeoff_height_m', 0.5)
        self.declare_parameter('altitude_tolerance_m', 0.05)
        self.declare_parameter('takeoff_min_time_s', 1.5)       # must spend at least this long climbing
        self.declare_parameter('takeoff_timeout_s', 8.0)        # max time before forcing HOVER
        self.declare_parameter('position_hold_time_s', 5.0)     # time spent at hover height
        self.declare_parameter('offboard_warmup_setpoints', 20)
        self.declare_parameter('tuning_mode', 'none')           # 'none', 'altitude', 'attitude'
        self.declare_parameter('tuning_step_amplitude_m', 0.2)
        self.declare_parameter('tuning_step_duration_s', 3.0)
        self.declare_parameter('local_position_topic', '/fmu/out/vehicle_local_position_v1')

        self.takeoff_height_m = float(self.get_parameter('takeoff_height_m').value)
        self.altitude_tolerance_m = float(self.get_parameter('altitude_tolerance_m').value)
        self.takeoff_min_time_s = float(self.get_parameter('takeoff_min_time_s').value)
        self.takeoff_timeout_s = float(self.get_parameter('takeoff_timeout_s').value)
        self.position_hold_time_s = float(self.get_parameter('position_hold_time_s').value)
        self.offboard_warmup_setpoints = int(self.get_parameter('offboard_warmup_setpoints').value)
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

        # ---------------- Publishers (to PX4) ----------------
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4
        )

        # ---------------- Subscribers (from PX4) ----------------
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            local_position_topic,
            self.local_position_callback,
            qos_px4,
        )

        # ---------------- State variables ----------------
        self.flight_phase = FlightPhase.PREFLIGHT
        self.offboard_setpoint_counter = 0

        self.current_altitude_up = None  # meters, up-positive
        self.last_altitude_print_time = 0.0

        self.mission_start_time = None
        self.hover_start_time = None

        # Main loop timer (50 ms)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info('G34 First Flight node initialized.')

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def local_position_callback(self, msg: VehicleLocalPosition):
        """
        PX4 local position: NED (z positive DOWN).
        We always use -z as altitude up. No z_valid gating.
        """
        self.current_altitude_up = -msg.z

    # -------------------------------------------------------------------------
    # PX4 VehicleCommand helpers
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
        # param1 = 1 (custom), param2 = 6 (Offboard)
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    # -------------------------------------------------------------------------
    # Offboard message publishers
    # -------------------------------------------------------------------------
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        # Position-only Offboard; PX4 handles altitude & attitude
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x_ned: float, y_ned: float, z_ned: float):
        """
        Position-only setpoint; yaw is set to NaN so PX4 *does not change yaw*.
        This minimizes initial yaw motion and keeps mission stable.
        """
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = [float(x_ned), float(y_ned), float(z_ned)]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = math.nan  # let PX4 keep current yaw
        self.trajectory_setpoint_pub.publish(msg)

    # -------------------------------------------------------------------------
    # Tuning profiles (optional)
    # -------------------------------------------------------------------------
    def altitude_tuning_profile(self, t_rel: float) -> float:
        """
        Altitude step sequence around nominal hover height.
        t_rel: time [s] since start of HOVER phase.
        Returns commanded altitude [m, up] relative to home.
        """
        h0 = self.takeoff_height_m
        amp = self.tuning_step_amplitude_m
        seg = self.tuning_step_duration_s

        # h0  -> h0+amp -> h0-amp -> h0
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
        Lateral position (x,y) step sequence for indirect attitude tuning.
        t_rel: time [s] since start of HOVER phase.
        Returns (x_cmd, y_cmd) in NED [m].
        """
        amp = self.tuning_step_amplitude_m
        seg = self.tuning_step_duration_s

        # Forward, back, right, left
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
    # Main loop / state machine
    # -------------------------------------------------------------------------
    def timer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9  # seconds

        # Always stream offboard heartbeat
        self.publish_offboard_control_mode()

        # Default commanded position in NED
        x_cmd = 0.0
        y_cmd = 0.0

        # 1) Warmup: must send setpoints before switching to Offboard
        if self.offboard_setpoint_counter < self.offboard_warmup_setpoints:
            z_cmd = 0.0  # stay at "ground" during warmup
            self.publish_trajectory_setpoint(x_cmd, y_cmd, z_cmd)
            self.offboard_setpoint_counter += 1
            return

        # 2) After warmup: enter OFFBOARD and arm (once)
        if self.flight_phase == FlightPhase.PREFLIGHT:
            self.set_offboard_mode()
            self.arm()
            self.flight_phase = FlightPhase.TAKEOFF_ASCEND
            self.mission_start_time = now
            self.get_logger().info('PREFLIGHT -> TAKEOFF_ASCEND')

        # 3) Phase logic
        if self.flight_phase == FlightPhase.TAKEOFF_ASCEND:
            # Command target altitude; PX4 climbs to it
            z_cmd = -self.takeoff_height_m  # NED: up is negative

            # Altitude-based transition with min-time + timeout safety
            time_in_phase = now - (self.mission_start_time or now)
            alt_ok = False
            if self.current_altitude_up is not None:
                alt_err = abs(self.current_altitude_up - self.takeoff_height_m)
                alt_ok = alt_err < self.altitude_tolerance_m

            if (
                alt_ok and time_in_phase > self.takeoff_min_time_s
            ) or time_in_phase > self.takeoff_timeout_s:
                self.flight_phase = FlightPhase.HOVER
                self.hover_start_time = now
                self.get_logger().info(
                    f'TAKEOFF_ASCEND -> HOVER (time_in_phase={time_in_phase:.1f}s, alt_up='
                    f'{self.current_altitude_up if self.current_altitude_up is not None else float("nan"):.2f} m)'
                )

        elif self.flight_phase == FlightPhase.HOVER:
            t_hover = now - (self.hover_start_time or now)

            # Start at nominal altitude
            alt_cmd = self.takeoff_height_m

            # Optional altitude tuning
            if self.tuning_mode == 'altitude':
                alt_cmd = self.altitude_tuning_profile(t_hover)

            # Optional lateral (indirect attitude) tuning
            if self.tuning_mode == 'attitude':
                x_cmd, y_cmd = self.attitude_tuning_profile(t_hover)

            z_cmd = -alt_cmd

            # After hold time, begin descent
            if t_hover > self.position_hold_time_s:
                self.flight_phase = FlightPhase.DESCEND
                self.get_logger().info('HOVER -> DESCEND')

        elif self.flight_phase == FlightPhase.DESCEND:
            # Command back to “ground level” z=0 in NED
            z_cmd = 0.0

            # Use altitude (if available) to decide when to disarm
            if self.current_altitude_up is not None and self.current_altitude_up < 0.05:
                self.disarm()
                self.flight_phase = FlightPhase.DONE
                self.get_logger().info('DESCEND -> DONE (disarmed)')

        elif self.flight_phase == FlightPhase.DONE:
            # Keep commanding ground position so PX4 stays happy in Offboard
            z_cmd = 0.0

        else:
            # Fallback
            z_cmd = 0.0

        # Publish trajectory setpoint for the current phase
        self.publish_trajectory_setpoint(x_cmd, y_cmd, z_cmd)

        # 4) Live altitude printout (2 Hz)
        if now - self.last_altitude_print_time > 0.5:
            alt_str = (
                f'{self.current_altitude_up:.3f} m'
                if self.current_altitude_up is not None
                else 'N/A'
            )
            self.get_logger().info(
                f'Phase={self.flight_phase.name}, '
                f'cmd_z_ned={z_cmd:.3f} m, alt_up={alt_str}'
            )
            self.last_altitude_print_time = now


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
