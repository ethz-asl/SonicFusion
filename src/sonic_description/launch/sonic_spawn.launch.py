import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

def generate_launch_description():
    xacro_file = os.path.join(get_package_share_directory(
        'sonic_description'), 'urdf/sonic.urdf.xacro')

    world_file = os.path.join(get_package_share_directory(
        'sonic_description'), 'urdf/world_circ.sdf')

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc, mappings={"spawn":"0.0 0.0 0.3"}) #7.0 -1.25 0.3
    robot_desc = {'robot_description': doc.toxml()}
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_desc]
    )

    gazebo_launch = ExecuteProcess(
                        cmd=['ros2', 'launch', 'gazebo_ros', 
                        'gazebo.launch.py', 'world:='+str(world_file)]
    )
    
    spawn_sonic = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'sonic'],
                        output='screen')

    return LaunchDescription([
        gazebo_launch,
        node_robot_state_publisher,
        TimerAction(
            period=3.0,
            actions=[spawn_sonic],
        ),
    ])
