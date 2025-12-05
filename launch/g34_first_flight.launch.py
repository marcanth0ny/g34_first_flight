from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    local_position_topic = LaunchConfiguration("local_position_topic")
    takeoff_altitude_m = LaunchConfiguration("takeoff_altitude_m")
    tuning_mode = LaunchConfiguration("tuning_mode")
    final_mode = LaunchConfiguration("final_mode")
    enable_csv_logging = LaunchConfiguration("enable_csv_logging")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use /clock (true in SITL).",
        ),
        DeclareLaunchArgument(
            "local_position_topic",
            default_value="/fmu/out/vehicle_local_position_v1",
            description="PX4 local position topic to use for altitude / velocity.",
        ),
        DeclareLaunchArgument(
            "takeoff_altitude_m",
            default_value="0.5",
            description="Takeoff / hover altitude in meters.",
        ),
        DeclareLaunchArgument(
            "tuning_mode",
            default_value="none",
            description="Tuning mode: 'none', 'altitude', or 'attitude'.",
        ),
        DeclareLaunchArgument(
            "final_mode",
            default_value="descend",
            description="Final phase: 'descend', 'auto_land', or 'precision_land'.",
        ),
        DeclareLaunchArgument(
            "enable_csv_logging",
            default_value="true",
            description="Enable CSV logging to ~/g34_logs.",
        ),

        Node(
            package="g34_first_flight",
            executable="g34_first_flight_node",
            name="g34_first_flight_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "local_position_topic": local_position_topic,
                "takeoff_altitude_m": takeoff_altitude_m,
                "tuning_mode": tuning_mode,
                "final_mode": final_mode,
                "enable_csv_logging": enable_csv_logging,
            }],
        ),
    ])
