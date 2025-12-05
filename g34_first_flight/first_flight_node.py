#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    DistanceSensor,
    VehicleAttitude,
    VehicleAttitudeSetpoint,
    VehicleLocalPosition,
)


class G34FirstFlightNode(Node):
    """
    First autonomous hop flight for PLEUAV with two tuning sequences:
      - Altitude step tuning (vertical dynamics)
      - Attitude step tuning (roll & pitch)

    State machine:

      PRE_OFFBOARD -> TAKEOFF -> ALT_HOLD
        -> (optional) ALT_TUNING
        -> (optional) ATT_TUNING
        -> DESCEND -> DONE
    """

    def __init__(self):
        super().__init__('g34_first_flight_node')

        # === Parameters ===
        self.declare_parameter('takeoff_height_m', 0.50)
        self.declare_parameter('descent_height_m', 0.10)
        self.declare_parameter('hover_time_s', 5.0)
        self.declare_parameter('vertical_speed_limit_mps', 0.4)
        self.declare_parameter('altitude_kp', 1.0)
        self.declare_parameter('range_min_valid_m', 0.15)
        self.declare_parameter('range_max_valid_m', 2.00)

        self.declare_parameter('run_altitude_tuning', True)
        self.declare_parameter('run_attitude_tuning', True)
        self.declare_parameter('attitude_step_deg', 5.0)
        self.declare_parameter('hover_thrust_norm', 0.40)

        self.takeoff_height = float(self.get_parameter('takeoff_height_m').value)
        self.descent_height = float(self.get_parameter('descent_height_m').value)
        self.hover_time_s = float(self.get_parameter('hover_time_s').value)
        self.vert_speed_limit = float(self.get_parameter('vertical_speed_limit_mps').value)
        self.altitude_kp = float(self.get_parameter('altitude_kp').value)
        self.range_min_valid = float(self.get_parameter('range_min_valid_m').value)
        self.range_max_valid = float(self.get_parameter('range_max_valid_m').value)

        self.run_altitude_tuning = bool(self.get_parameter('run_altitude_tuning').value)
        self.run_attitude_tuning = bool(self.get_parameter('run_attitude_tuning').value)
        self.attitude_step_deg = float(self.get_parameter('attitude_step_deg').value)
        self.hover_thrust_norm = float(self.get_parameter('hover_thrust_norm').value)

        # === Internal state ===
        self.state = 'PRE_OFFBOARD'
        self.state_start_time = self.get_clock().now()
        self.offboard_setpoint_counter = 0

        self.altitude_m = None
        self.vehicle_status = VehicleStatus()
        self.vehicle_attitude = VehicleAttitude()
        self.altitude_lpos_m = None         # from VehicleLocalPosition (NED -> up)

        self.alt_tuning_done = False
        self.att_tuning_done = False

        # === QoS ===
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # === Publishers ===
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10
        )
        self.attitude_setpoint_pub = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', 10
        )

        # === Subscribers ===
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_sensor,
        )

        self.rangefinder_sub = self.create_subscription(
            DistanceSensor,
            '/fmu/out/distance_sensor',
            self.distance_sensor_callback,
            qos_sensor,
        )

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_sensor,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_sensor,
        )

        # === Timer @ 50 Hz ===
        self.timer_period_s = 0.02
        self.timer = self.create_timer(self.timer_period_s, self.timer_callback)

        self.get_logger().info('G34 First Flight node initialized.')

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def distance_sensor_callback(self, msg: DistanceSensor):
        if (
            msg.current_distance > self.range_min_valid
            and msg.current_distance < self.range_max_valid
        ):
            self.altitude_m = msg.current_distance

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        self.vehicle_attitude = msg
    
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """
        Use PX4 local position as a fallback altitude source.

        VehicleLocalPosition is in NED frame:
          x: north (m)
          y: east (m)
          z: down (m, positive down)

        We define altitude_m as "upwards from ground" -> -z.
        """
        # Save NED->up altitude
        self.altitude_lpos_m = -float(msg.z)

        # If we don't have rangefinder, use local position altitude
        if self.altitude_m is None:
            self.altitude_m = self.altitude_lpos_m

    def get_altitude_m(self) -> float | None:
        """
        Choose the altitude source:
          - Prefer rangefinder if available (low alt hardware flights)
          - Otherwise use local position (SITL, higher alt)
        """
        if self.altitude_m is not None:
            return self.altitude_m
        if self.altitude_lpos_m is not None:
            return self.altitude_lpos_m
        return None

    # -------------------------------------------------------------------------
    # Helpers: time, mode, and commands
    # -------------------------------------------------------------------------

    def current_state_time(self) -> float:
        dt_ns = (self.get_clock().now() - self.state_start_time).nanoseconds
        return dt_ns * 1e-9

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000

        # Choose control mode based on state
        if self.state in ['PRE_OFFBOARD', 'TAKEOFF', 'ALT_HOLD',
                          'ALT_TUNING', 'DESCEND']:
            msg.position = False
            msg.velocity = True    # vertical velocity
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = False

        elif self.state in ['ATT_TUNING']:
            msg.position = False
            msg.velocity = False
            msg.acceleration = False
            msg.attitude = True    # attitude setpoints
            msg.body_rate = False

        elif self.state in ['DONE']:
            # neutral, but still publish something
            msg.position = False
            msg.velocity = True
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = False

        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, vz_ned: float):
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000

        # Position: don't control position -> NaN
        msg.position = [math.nan, math.nan, math.nan]

        # Velocity: (vx, vy, vz) in NED
        # zero horizontal, only vertical
        msg.velocity = [0.0, 0.0, float(vz_ned)]

        # We’re not using acceleration / jerk here (set to NaN)
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]

        # Yaw: keep 0 (aligned with default forward)
        msg.yaw = 0.0
        msg.yawspeed = 0.0

        self.trajectory_setpoint_pub.publish(msg)


    def publish_attitude_setpoint(self, roll: float, pitch: float,
                                  yaw: float, thrust_norm: float):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000

        # Convert RPY -> quaternion (w, x, y, z), NED/FRD convention (PX4 Hamilton) :contentReference[oaicite:2]{index=2}
        qw, qx, qy, qz = self.rpy_to_quaternion(roll, pitch, yaw)
        msg.q_d[0] = qw
        msg.q_d[1] = qx
        msg.q_d[2] = qy
        msg.q_d[3] = qz

        msg.yaw_sp_move_rate = 0.0

        # thrust_body: [0, 0, negative_thrust] for multicopters (upwards) :contentReference[oaicite:3]{index=3}
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -float(thrust_norm)

        self.attitude_setpoint_pub.publish(msg)

    @staticmethod
    def rpy_to_quaternion(roll: float, pitch: float, yaw: float):
        """Convert roll, pitch, yaw to quaternion (w, x, y, z)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qw, qx, qy, qz

    def arm(self):
        self.get_logger().info('Sending ARM command.')
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.param1 = 1.0  # 1 = arm, 0 = disarm
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.vehicle_command_pub.publish(cmd)

    def disarm(self):
        self.get_logger().info('Sending DISARM command.')
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.param1 = 0.0  # 1 = arm, 0 = disarm
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.vehicle_command_pub.publish(cmd)        
        
    def set_offboard_mode(self):
        self.get_logger().info('Sending OFFBOARD mode command.')
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0  # main mode

        cmd.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.vehicle_command_pub.publish(cmd)
        

    # -------------------------------------------------------------------------
    # Vertical control
    # -------------------------------------------------------------------------

    def compute_vz_from_altitude_error(self, target_height_m: float) -> float:
        """
        P controller: altitude error (meters, up) -> vertical velocity (m/s, NED).

        Uses get_altitude_m():
          - Rangefinder when available
          - Otherwise VehicleLocalPosition z
        """
        alt = self.get_altitude_m()
        if alt is None:
            # No valid altitude yet: don't move
            return 0.0

        error_up = target_height_m - alt  # + if too low
        v_up = self.altitude_kp * error_up
        v_up = max(min(v_up, self.vert_speed_limit), -self.vert_speed_limit)

        # Convert up velocity to NED z (down-positive)
        vz_ned = -v_up
        return vz_ned


    # -------------------------------------------------------------------------
    # Tuning sequences
    # -------------------------------------------------------------------------

    def run_altitude_tuning(self) -> float:
        """
        Simple altitude step sequence around takeoff height.

        0-2s: base height
        2-4s: +0.10 m
        4-6s: -0.10 m
        6-8s: base height
        """
        t = self.current_state_time()
        base_h = self.takeoff_height

        if t < 2.0:
            target_h = base_h
        elif t < 4.0:
            target_h = base_h + 0.10
        elif t < 6.0:
            target_h = base_h - 0.10
        elif t < 8.0:
            target_h = base_h
        else:
            target_h = base_h
            self.alt_tuning_done = True

        return self.compute_vz_from_altitude_error(target_h)

    def run_attitude_tuning(self):
        """
        Attitude step sequence around hover attitude with constant thrust.

        Time schedule:
          0-2s:   level
          2-4s:   +roll step
          4-6s:   -roll step
          6-8s:   level
          8-10s:  +pitch step
          10-12s: -pitch step
          12-14s: level, then done
        """
        t = self.current_state_time()
        step_rad = math.radians(self.attitude_step_deg)

        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        if t < 2.0:
            roll, pitch = 0.0, 0.0
        elif t < 4.0:
            roll, pitch = step_rad, 0.0     # roll +
        elif t < 6.0:
            roll, pitch = -step_rad, 0.0    # roll -
        elif t < 8.0:
            roll, pitch = 0.0, 0.0
        elif t < 10.0:
            roll, pitch = 0.0, step_rad     # pitch +
        elif t < 12.0:
            roll, pitch = 0.0, -step_rad    # pitch -
        elif t < 14.0:
            roll, pitch = 0.0, 0.0
        else:
            roll, pitch = 0.0, 0.0
            self.att_tuning_done = True

        self.publish_attitude_setpoint(roll, pitch, yaw, self.hover_thrust_norm)

    # -------------------------------------------------------------------------
    # Main timer callback
    # -------------------------------------------------------------------------

    def timer_callback(self):
        # Always send OffboardControlMode at high rate
        self.publish_offboard_control_mode()

        # Default: hold vertical velocity at zero, unless state overrides
        vz_cmd_ned = 0.0

        # PRE_OFFBOARD: stream neutral for ~1s, then switch to offboard + arm
        if self.state == 'PRE_OFFBOARD':
            self.offboard_setpoint_counter += 1
            vz_cmd_ned = 0.0

            if self.offboard_setpoint_counter == 50:
                self.set_offboard_mode()
                self.arm()
                self.state = 'TAKEOFF'
                self.state_start_time = self.get_clock().now()
                self.get_logger().info('PRE_OFFBOARD -> TAKEOFF')

        elif self.state == 'TAKEOFF':
            vz_cmd_ned = self.compute_vz_from_altitude_error(self.takeoff_height)

            if (
                self.altitude_m is not None
                and abs(self.altitude_m - self.takeoff_height) < 0.05
                and self.current_state_time() > 1.0
            ):
                self.state = 'ALT_HOLD'
                self.state_start_time = self.get_clock().now()
                self.get_logger().info('TAKEOFF -> ALT_HOLD')

        elif self.state == 'ALT_HOLD':
            # P altitude controller around takeoff height
            vz_cmd_ned = self.compute_vz_from_altitude_error(self.takeoff_height)

            if self.current_state_time() > self.hover_time_s:
                if self.run_altitude_tuning:
                    self.state = 'ALT_TUNING'
                    self.state_start_time = self.get_clock().now()
                    self.alt_tuning_done = False
                    self.get_logger().info('ALT_HOLD -> ALT_TUNING')
                elif self.run_attitude_tuning:
                    self.state = 'ATT_TUNING'
                    self.state_start_time = self.get_clock().now()
                    self.att_tuning_done = False
                    self.get_logger().info('ALT_HOLD -> ATT_TUNING')
                else:
                    self.state = 'DESCEND'
                    self.state_start_time = self.get_clock().now()
                    self.get_logger().info('ALT_HOLD -> DESCEND')

        elif self.state == 'ALT_TUNING':
            vz_cmd_ned = self.run_altitude_tuning()
            if self.alt_tuning_done:
                if self.run_attitude_tuning:
                    self.state = 'ATT_TUNING'
                    self.state_start_time = self.get_clock().now()
                    self.att_tuning_done = False
                    self.get_logger().info('ALT_TUNING -> ATT_TUNING')
                else:
                    self.state = 'DESCEND'
                    self.state_start_time = self.get_clock().now()
                    self.get_logger().info('ALT_TUNING -> DESCEND')

        elif self.state == 'ATT_TUNING':
            # Here we control attitude directly (no vertical velocity command).
            self.run_attitude_tuning()
            if self.att_tuning_done:
                self.state = 'DESCEND'
                self.state_start_time = self.get_clock().now()
                self.get_logger().info('ATT_TUNING -> DESCEND')

        elif self.state == 'DESCEND':
            vz_cmd_ned = self.compute_vz_from_altitude_error(self.descent_height)

            if self.altitude_m is not None and self.altitude_m <= (self.descent_height + 0.02):
                vz_cmd_ned = 0.0
                self.disarm()
                self.state = 'DONE'
                self.state_start_time = self.get_clock().now()
                self.get_logger().info('DESCEND -> DONE')

        elif self.state == 'DONE':
            vz_cmd_ned = 0.0

        # Only publish TrajectorySetpoint in velocity-based states
        if self.state in ['PRE_OFFBOARD', 'TAKEOFF', 'ALT_HOLD',
                          'ALT_TUNING', 'DESCEND', 'DONE']:
            self.publish_trajectory_setpoint(vz_cmd_ned)


def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down G34 First Flight node.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
