#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    DistanceSensor,
    VehicleLocalPosition,
    VehicleAttitudeSetpoint,
)


class FlightPhase(Enum):
    IDLE = auto()
    PRE_OFFBOARD = auto()
    TAKEOFF = auto()
    ALT_HOLD = auto()
    ALT_TUNING = auto()
    ATT_TUNING_ROLL = auto()
    ATT_TUNING_PITCH = auto()
    LAND = auto()
    DONE = auto()


class G34FirstFlightNode(Node):
    """
    G34 First Flight / Altitude + Attitude Tuning Node

    Sequence:
      1. PRE_OFFBOARD: stream OffboardControlMode + zero setpoints.
      2. Send OFFBOARD + ARM.
      3. TAKEOFF: climb to target height using vertical velocity (NED).
      4. ALT_HOLD: hold altitude for hover_time_s.
      5. ALT_TUNING (optional): small ±Δh altitude steps for vertical tuning.
      6. ATT_TUNING_ROLL (optional): roll step sequence around hover.
      7. ATT_TUNING_PITCH (optional): pitch step sequence around hover.
      8. LAND: gentle descent and disarm.
      9. DONE: keep streaming zero setpoints for safety.

    Altitude source:
      - Prefer rangefinder (DistanceSensor) when valid.
      - Otherwise use VehicleLocalPosition (NED z → altitude up).

    Attitude tuning:
      - Uses VehicleAttitudeSetpoint (roll/pitch commands, constant hover thrust).
    """

    def __init__(self):
        super().__init__('g34_first_flight_node')

        # === High-level parameters ===
        self.declare_parameter('takeoff_height_m', 0.5)
        self.declare_parameter('hover_time_s', 3.0)

        # Offboard / topic config
        self.declare_parameter('use_rangefinder', True)
        # Adjust to '/fmu/out/vehicle_local_position_v1' if that is your topic
        self.declare_parameter('local_position_topic', '/fmu/out/vehicle_local_position')

        # Altitude controller (velocity mode)
        self.declare_parameter('altitude_kp', 1.0)
        self.declare_parameter('vz_max_up', 0.4)       # max climb speed (m/s, up)
        self.declare_parameter('vz_max_down', 0.4)     # max descent speed (m/s, down)
        self.declare_parameter('alt_tolerance_m', 0.05)

        # Altitude tuning phase
        self.declare_parameter('enable_alt_tuning', True)
        self.declare_parameter('alt_tuning_step_m', 0.1)
        self.declare_parameter('alt_tuning_step_time_s', 3.0)

        # Attitude tuning phase
        self.declare_parameter('enable_att_tuning', True)
        self.declare_parameter('att_step_deg', 5.0)
        self.declare_parameter('att_step_time_s', 2.0)
        self.declare_parameter('att_num_cycles', 2)
        # Hover thrust estimate for attitude mode (normalized 0–1)
        self.declare_parameter('hover_thrust', 0.4)
        # Small P-like correction from altitude error to thrust
        self.declare_parameter('altitude_kp_thrust', 0.5)

        # === Fetch parameters ===
        self.takeoff_height_m = float(self.get_parameter('takeoff_height_m').value)
        self.hover_time_s = float(self.get_parameter('hover_time_s').value)

        self.use_rangefinder = bool(self.get_parameter('use_rangefinder').value)
        self.local_position_topic = self.get_parameter('local_position_topic').value

        self.altitude_kp = float(self.get_parameter('altitude_kp').value)
        self.vz_max_up = float(self.get_parameter('vz_max_up').value)
        self.vz_max_down = float(self.get_parameter('vz_max_down').value)
        self.alt_tolerance_m = float(self.get_parameter('alt_tolerance_m').value)

        self.enable_alt_tuning = bool(self.get_parameter('enable_alt_tuning').value)
        self.alt_tuning_step_m = float(self.get_parameter('alt_tuning_step_m').value)
        self.alt_tuning_step_time_s = float(self.get_parameter('alt_tuning_step_time_s').value)

        self.enable_att_tuning = bool(self.get_parameter('enable_att_tuning').value)
        self.att_step_deg = float(self.get_parameter('att_step_deg').value)
        self.att_step_time_s = float(self.get_parameter('att_step_time_s').value)
        self.att_num_cycles = int(self.get_parameter('att_num_cycles').value)
        self.hover_thrust = float(self.get_parameter('hover_thrust').value)
        self.altitude_kp_thrust = float(self.get_parameter('altitude_kp_thrust').value)

        # Derived attitude tuning sequences (roll & pitch)
        self.att_step_rad = math.radians(self.att_step_deg)
        base_seq = [0.0, self.att_step_rad, 0.0, -self.att_step_rad, 0.0]
        self.roll_step_sequence = base_seq * self.att_num_cycles
        self.pitch_step_sequence = base_seq * self.att_num_cycles

        # Altitude tuning sequence around takeoff height
        self.alt_tuning_targets = [
            self.takeoff_height_m + self.alt_tuning_step_m,
            self.takeoff_height_m - self.alt_tuning_step_m,
        ]
        self.alt_tuning_index = 0

        # === Internal state ===
        self.phase = FlightPhase.PRE_OFFBOARD
        self.state_start_time = self.get_clock().now()
        self.offboard_setpoint_counter = 0

        # Altitude tracking
        self.alt_range_m = None   # from DistanceSensor (up)
        self.alt_lpos_m = None    # from VehicleLocalPosition (up)
        self.alt_valid = False

        # PX4 status
        self.vehicle_status = VehicleStatus()

        # Rangefinder bounds
        self.range_min_valid = 0.02
        self.range_max_valid = 5.0

        # Attitude tuning indices
        self.att_step_index = 0

        # Debug print throttling
        self.last_debug_print_time = 0.0

        # === QoS ===
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # === Publishers ===
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos
        )
        self.attitude_setpoint_pub = self.create_publisher(
            VehicleAttitudeSetpoint,
            '/fmu/in/vehicle_attitude_setpoint',
            qos
        )

        # === Subscribers ===
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos
        )

        self.rangefinder_sub = self.create_subscription(
            DistanceSensor,
            '/fmu/out/distance_sensor',
            self.distance_sensor_callback,
            qos
        )

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self.local_position_callback,
            qos
        )

        # === Timer ===
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz

        self.get_logger().info('G34 First Flight node initialized.')

    # -------------------------------------------------------------------------
    # Time helpers
    # -------------------------------------------------------------------------

    def now(self):
        return self.get_clock().now()

    def seconds_since(self, t):
        return (self.now() - t).nanoseconds * 1e-9

    def seconds_since_start(self):
        return self.now().nanoseconds * 1e-9

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def distance_sensor_callback(self, msg: DistanceSensor):
        if not self.use_rangefinder:
            return

        d = float(msg.current_distance)
        if self.range_min_valid < d < self.range_max_valid:
            self.alt_range_m = d

    def local_position_callback(self, msg: VehicleLocalPosition):
        # NED frame: z is down; altitude up is -z
        self.alt_lpos_m = -float(msg.z)

    # -------------------------------------------------------------------------
    # Altitude utilities
    # -------------------------------------------------------------------------

    def get_altitude_up_m(self):
        """
        Returns altitude in meters (upward), preferring rangefinder if valid,
        otherwise using local position in NED.
        """
        if self.use_rangefinder and self.alt_range_m is not None:
            self.alt_valid = True
            return self.alt_range_m

        if self.alt_lpos_m is not None:
            self.alt_valid = True
            return self.alt_lpos_m

        self.alt_valid = False
        return None

    # -------------------------------------------------------------------------
    # Vehicle command helpers
    # -------------------------------------------------------------------------

    def send_vehicle_command(self,
                             command: int,
                             param1: float = 0.0,
                             param2: float = 0.0,
                             param3: float = 0.0,
                             param4: float = 0.0,
                             param5: float = 0.0,
                             param6: float = 0.0,
                             param7: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.now().nanoseconds / 1000)

        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_pub.publish(msg)

    def send_offboard_mode(self):
        self.get_logger().info('Sending OFFBOARD mode command.')
        # VEHICLE_CMD_DO_SET_MODE: param1=1 (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
        # param2=6 (PX4_CUSTOM_MAIN_MODE_OFFBOARD)
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0
        )

    def send_arm(self):
        self.get_logger().info('Sending ARM command.')
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0
        )

    def send_disarm(self):
        self.get_logger().info('Sending DISARM command.')
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0
        )

    # -------------------------------------------------------------------------
    # Offboard control mode / setpoint publishers
    # -------------------------------------------------------------------------

    def publish_offboard_control_mode_velocity(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.now().nanoseconds / 1000)

        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False

        self.offboard_control_mode_pub.publish(msg)

    def publish_offboard_control_mode_attitude(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.now().nanoseconds / 1000)

        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True   # Attitude mode
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False

        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, vz_ned: float):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.now().nanoseconds / 1000)

        msg.position = [math.nan, math.nan, math.nan]
        msg.velocity = [0.0, 0.0, float(vz_ned)]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [0.0, 0.0, 0.0]
        msg.yaw = 0.0
        msg.yawspeed = 0.0

        self.trajectory_setpoint_pub.publish(msg)

    # -------------------------------------------------------------------------
    # Attitude setpoint for tuning
    # -------------------------------------------------------------------------

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float):
        """
        Converts roll, pitch, yaw (rad) to quaternion [w, x, y, z].
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y, z]

    def publish_attitude_setpoint(self,
                                  roll_rad: float,
                                  pitch_rad: float,
                                  yaw_rad: float,
                                  thrust_norm: float):
        """
        Publish a VehicleAttitudeSetpoint for attitude tuning.
        thrust_norm in [0,1]; thrust_body[2] is negative throttle demand for MC.
        """
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = int(self.now().nanoseconds / 1000)

        msg.roll_body = roll_rad
        msg.pitch_body = pitch_rad
        msg.yaw_body = yaw_rad
        msg.yaw_sp_move_rate = 0.0

        msg.q_d = self.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

        thrust_norm = max(0.0, min(1.0, thrust_norm))
        msg.thrust_body = [0.0, 0.0, -thrust_norm]

        msg.reset_integral = False

        self.attitude_setpoint_pub.publish(msg)

    # -------------------------------------------------------------------------
    # Vertical control laws
    # -------------------------------------------------------------------------

    def compute_vz_for_altitude_target(self, target_alt_up_m: float):
        """
        P controller: altitude error (upwards) -> vertical velocity (NED).
        """
        alt_up = self.get_altitude_up_m()
        if alt_up is None:
            v_up = self.vz_max_up
            err = float('nan')
        else:
            err = target_alt_up_m - alt_up
            v_up_raw = self.altitude_kp * err
            v_up = max(-self.vz_max_down, min(self.vz_max_up, v_up_raw))

        vz_ned = -v_up  # up to NED (down-positive)

        now_s = self.seconds_since_start()
        if alt_up is not None and (now_s - self.last_debug_print_time) > 0.2:
            phase_name = self.phase.name
            self.get_logger().info(
                f"[{phase_name}] alt_up={alt_up:.3f} m, "
                f"target={target_alt_up_m:.3f} m, "
                f"vz_ned={vz_ned:.3f} m/s"
            )
            self.last_debug_print_time = now_s

        return vz_ned, alt_up, err

    def compute_thrust_for_hover(self, target_alt_up_m: float):
        """
        Simple hover thrust with small altitude correction for attitude tuning.
        """
        alt_up = self.get_altitude_up_m()
        if alt_up is None:
            thrust = self.hover_thrust
            err = float('nan')
        else:
            err = target_alt_up_m - alt_up
            thrust_raw = self.hover_thrust + self.altitude_kp_thrust * err
            thrust = max(0.1, min(0.9, thrust_raw))

        now_s = self.seconds_since_start()
        if alt_up is not None and (now_s - self.last_debug_print_time) > 0.2:
            phase_name = self.phase.name
            self.get_logger().info(
                f"[{phase_name}] alt_up={alt_up:.3f} m, "
                f"target={target_alt_up_m:.3f} m, "
                f"thrust={thrust:.3f}"
            )
            self.last_debug_print_time = now_s

        return thrust, alt_up, err

    # -------------------------------------------------------------------------
    # Main timer / state machine
    # -------------------------------------------------------------------------

    def timer_callback(self):
        self.offboard_setpoint_counter += 1

        # Defaults
        vz_cmd_ned = 0.0
        use_velocity_mode = True
        roll_cmd = 0.0
        pitch_cmd = 0.0
        yaw_cmd = 0.0
        thrust_cmd = self.hover_thrust

        if self.phase == FlightPhase.PRE_OFFBOARD:
            # Stream zeros; after ~1 second, send OFFBOARD + ARM and go to TAKEOFF
            vz_cmd_ned = 0.0

            if self.offboard_setpoint_counter == 50:
                self.send_offboard_mode()
                self.send_arm()
                self.phase = FlightPhase.TAKEOFF
                self.state_start_time = self.now()
                self.get_logger().info('PRE_OFFBOARD -> TAKEOFF')

        elif self.phase == FlightPhase.TAKEOFF:
            vz_cmd_ned, alt_up, err = self.compute_vz_for_altitude_target(self.takeoff_height_m)

            if (
                alt_up is not None
                and abs(self.takeoff_height_m - alt_up) < self.alt_tolerance_m
                and self.seconds_since(self.state_start_time) > 1.0
            ):
                self.phase = FlightPhase.ALT_HOLD
                self.state_start_time = self.now()
                self.get_logger().info('TAKEOFF -> ALT_HOLD')

        elif self.phase == FlightPhase.ALT_HOLD:
            vz_cmd_ned, alt_up, err = self.compute_vz_for_altitude_target(self.takeoff_height_m)

            if self.seconds_since(self.state_start_time) > self.hover_time_s:
                if self.enable_alt_tuning:
                    self.phase = FlightPhase.ALT_TUNING
                    self.state_start_time = self.now()
                    self.alt_tuning_index = 0
                    self.get_logger().info('ALT_HOLD -> ALT_TUNING')
                elif self.enable_att_tuning:
                    self.phase = FlightPhase.ATT_TUNING_ROLL
                    self.state_start_time = self.now()
                    self.att_step_index = 0
                    self.get_logger().info('ALT_HOLD -> ATT_TUNING_ROLL')
                else:
                    self.phase = FlightPhase.LAND
                    self.state_start_time = self.now()
                    self.get_logger().info('ALT_HOLD -> LAND')

        elif self.phase == FlightPhase.ALT_TUNING:
            target = self.alt_tuning_targets[self.alt_tuning_index]
            vz_cmd_ned, alt_up, err = self.compute_vz_for_altitude_target(target)

            if self.seconds_since(self.state_start_time) > self.alt_tuning_step_time_s:
                self.alt_tuning_index += 1
                if self.alt_tuning_index >= len(self.alt_tuning_targets):
                    if self.enable_att_tuning:
                        self.phase = FlightPhase.ATT_TUNING_ROLL
                        self.state_start_time = self.now()
                        self.att_step_index = 0
                        self.get_logger().info('ALT_TUNING -> ATT_TUNING_ROLL')
                    else:
                        self.phase = FlightPhase.LAND
                        self.state_start_time = self.now()
                        self.get_logger().info('ALT_TUNING -> LAND')
                else:
                    self.state_start_time = self.now()
                    self.get_logger().info(
                        f'ALT_TUNING: next target={self.alt_tuning_targets[self.alt_tuning_index]:.3f} m'
                    )

        elif self.phase == FlightPhase.ATT_TUNING_ROLL:
            use_velocity_mode = False  # attitude + thrust mode
            thrust_cmd, alt_up, err = self.compute_thrust_for_hover(self.takeoff_height_m)

            if self.att_step_index >= len(self.roll_step_sequence):
                # Done with roll steps
                self.phase = FlightPhase.ATT_TUNING_PITCH
                self.state_start_time = self.now()
                self.att_step_index = 0
                self.get_logger().info('ATT_TUNING_ROLL -> ATT_TUNING_PITCH')
            else:
                roll_cmd = self.roll_step_sequence[self.att_step_index]
                pitch_cmd = 0.0
                yaw_cmd = 0.0

                # Advance step after att_step_time_s
                if self.seconds_since(self.state_start_time) > self.att_step_time_s:
                    self.att_step_index += 1
                    self.state_start_time = self.now()
                    self.get_logger().info(
                        f'ATT_TUNING_ROLL: step index -> {self.att_step_index}'
                    )

        elif self.phase == FlightPhase.ATT_TUNING_PITCH:
            use_velocity_mode = False
            thrust_cmd, alt_up, err = self.compute_thrust_for_hover(self.takeoff_height_m)

            if self.att_step_index >= len(self.pitch_step_sequence):
                # Done with pitch steps
                self.phase = FlightPhase.LAND
                self.state_start_time = self.now()
                self.get_logger().info('ATT_TUNING_PITCH -> LAND')
            else:
                roll_cmd = 0.0
                pitch_cmd = self.pitch_step_sequence[self.att_step_index]
                yaw_cmd = 0.0

                if self.seconds_since(self.state_start_time) > self.att_step_time_s:
                    self.att_step_index += 1
                    self.state_start_time = self.now()
                    self.get_logger().info(
                        f'ATT_TUNING_PITCH: step index -> {self.att_step_index}'
                    )

        elif self.phase == FlightPhase.LAND:
            # Simple constant descent
            vz_cmd_ned = self.vz_max_down
            alt_up = self.get_altitude_up_m()
            if alt_up is not None:
                now_s = self.seconds_since_start()
                if (now_s - self.last_debug_print_time) > 0.2:
                    self.get_logger().info(
                        f"[LAND] alt_up={alt_up:.3f} m, vz_ned={vz_cmd_ned:.3f} m/s"
                    )
                    self.last_debug_print_time = now_s

                if alt_up < 0.05:
                    self.send_disarm()
                    self.phase = FlightPhase.DONE
                    self.state_start_time = self.now()
                    self.get_logger().info('LAND -> DONE (disarmed)')

        elif self.phase == FlightPhase.DONE:
            vz_cmd_ned = 0.0

        # ---------------------------------------------------------------------
        # Publish OffboardControlMode + setpoints
        # ---------------------------------------------------------------------

        if use_velocity_mode:
            self.publish_offboard_control_mode_velocity()
            self.publish_trajectory_setpoint(vz_cmd_ned)
        else:
            self.publish_offboard_control_mode_attitude()
            self.publish_attitude_setpoint(
                roll_cmd,
                pitch_cmd,
                yaw_cmd,
                thrust_cmd
            )


def main(args=None):
    rclpy.init(args=args)
    node = G34FirstFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down G34 First Flight node.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
