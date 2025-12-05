from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    local_position_topic = LaunchConfiguration("local_position_topic")
    takeoff_altitude_m = LaunchConfiguration("takeoff_altitude_m")
    tuning_mode = LaunchConfiguration("tuning_mode")
    final_mode = LaunchConfiguration("final_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "local_position_topic",
                default_value="/fmu/out/vehicle_local_position",
                description="PX4 local position topic (e.g. /fmu/out/vehicle_local_position_v1)",
            ),
            DeclareLaunchArgument(
                "takeoff_altitude_m",
                default_value="0.5",
                description="Takeoff / hover altitude (m, positive up).",
            ),
            DeclareLaunchArgument(
                "tuning_mode",
                default_value="none",
                description="none | altitude | attitude | both",
            ),
            DeclareLaunchArgument(
                "final_mode",
                default_value="descend",
                description="descend | auto_land | pl_precision",
            ),
            Node(
                package="g34_first_flight",
                executable="g34_first_flight_node",
                name="g34_first_flight_node",
                output="screen",
                parameters=[
                    {
                        "local_position_topic": local_position_topic,
                        "takeoff_altitude_m": takeoff_altitude_m,
                        "tuning_mode": tuning_mode,
                        "final_mode": final_mode,
                    }
                ],
            ),
        ]
    )
