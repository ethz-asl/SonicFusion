import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    config_file = os.path.join(
        get_package_share_directory('sonic_description'),
        'cfg', 'sensor_cfgs.yaml')

    with open(config_file) as f:
        sensor_cfgs_array = yaml.safe_load(f)
    sensor_cfgs = {e['id']: e for e in sensor_cfgs_array}
        
    node=Node(
        package = 'sonic_fusion',
        name = 'sonic_fusion_node',
        executable = 'sonic_fusion_node.py',
        parameters= [
            {"sensor_cfgs": repr(sensor_cfgs)},
        ]
    )
    ld.add_action(node)
    return ld