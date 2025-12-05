from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # === Launch Arguments ===
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo / PX4 SITL) clock if true'
    )

    local_position_topic_arg = DeclareLaunchArgument(
        'local_position_topic',
        default_value='/fmu/out/vehicle_local_position_v1',
        description=(
            'Topic for vehicle local position. '
            'For PX4 uXRCE-DDS this is typically '
            '"/fmu/out/vehicle_local_position_v1".'
        )
    )

    takeoff_altitude_arg = DeclareLaunchArgument(
        'takeoff_altitude_m',
        default_value='0.5',
        description='Target takeoff altitude [m] above home (e.g. 0.5–0.7)'
    )

    tuning_mode_arg = DeclareLaunchArgument(
        'tuning_mode',
        default_value='none',
        description=(
            'Tuning mode selector:\n'
            '  "none"         : simple takeoff → hover → final mode\n'
            '  "altitude"     : includes vertical/altitude step tests\n'
            '  "attitude"     : includes small lateral/attitude step tests\n'
            '  "comprehensive": (optional) combined altitude + attitude tests\n'
        )
    )

    final_mode_arg = DeclareLaunchArgument(
        'final_mode',
        default_value='descend',
        description=(
            'Final mode after hover / tuning:\n'
            '  "descend"        : offboard-controlled descent + disarm\n'
            '  "auto_land"      : switch PX4 to AUTO.LAND\n'
            '  "precision_land" : hand over to precision landing (Tracktor-beam)\n'
        )
    )

    # Optional: allow enabling or disabling CSV logging from launch
    csv_logging_arg = DeclareLaunchArgument(
        'enable_csv_logging',
        default_value='true',
        description='If true, node writes CSV log for tuning analysis'
    )

    # Optional: log directory (the node can ignore if not used)
    log_dir_arg = DeclareLaunchArgument(
        'log_directory',
        default_value='/home/${env:USER}/g34_logs',
        description='Directory where CSV logs will be stored'
    )

    # === Launch Configurations ===
    use_sim_time = LaunchConfiguration('use_sim_time')
    local_position_topic = LaunchConfiguration('local_position_topic')
    takeoff_altitude_m = LaunchConfiguration('takeoff_altitude_m')
    tuning_mode = LaunchConfiguration('tuning_mode')
    final_mode = LaunchConfiguration('final_mode')
    enable_csv_logging = LaunchConfiguration('enable_csv_logging')
    log_directory = LaunchConfiguration('log_directory')

    # === G34 First Flight Node ===
    first_flight_node = Node(
        package='g34_first_flight',
        executable='g34_first_flight_node',
        name='g34_first_flight_node',
        output='screen',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'local_position_topic': local_position_topic,
                'takeoff_altitude_m': takeoff_altitude_m,
                'tuning_mode': tuning_mode,
                'final_mode': final_mode,
                'enable_csv_logging': enable_csv_logging,
                'log_directory': log_directory,
            }
        ]
    )

    # Build LaunchDescription
    ld = LaunchDescription()

    # Declare all arguments
    ld.add_action(use_sim_time_arg)
    ld.add_action(local_position_topic_arg)
    ld.add_action(takeoff_altitude_arg)
    ld.add_action(tuning_mode_arg)
    ld.add_action(final_mode_arg)
    ld.add_action(csv_logging_arg)
    ld.add_action(log_dir_arg)

    # Add the main node
    ld.add_action(first_flight_node)

    return ld
