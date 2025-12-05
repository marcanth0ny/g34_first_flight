from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='g34_first_flight',
            executable='g34_first_flight_node',
            name='g34_first_flight_node',
            output='screen',
            parameters=[
                {'takeoff_height_m': 0.5},
                {'descent_height_m': 0.1},
                {'hover_time_s': 5.0},
                {'vertical_speed_limit_mps': 0.4},
                {'altitude_kp': 1.0},
                {'range_min_valid_m': 0.15},
                {'range_max_valid_m': 2.0},
                {'run_altitude_tuning': True},
                {'run_attitude_tuning': True},
                {'attitude_step_deg': 5.0},
                {'hover_thrust_norm': 0.30},
            ],
        )
    ])
