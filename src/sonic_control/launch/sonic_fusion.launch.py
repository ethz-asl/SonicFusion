import os
import yaml
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def get_world_object_data(world_file):
    world_doc = ET.parse(world_file)
    doc_root = world_doc.getroot()

    object_pose_scale_t = {}
    object_shape_t = {}
    object_data = {}

    for model in doc_root.iter('model'):
        if not model.attrib['name'] == 'ground_plane':
            geometry = model.findall('./link/collision/geometry')
            if not geometry:
                pose = model.find('pose')
                scale = model.find('scale')
                object_pose_scale_t[model.attrib['name']] = {'pose': [float(x) for x in pose.text.split(" ")],'scale': [float(x) for x in scale.text.split(" ")]}
            else:
                shape = geometry[0][0]
                if shape.tag == 'cylinder':
                    object_shape_t[model.attrib['name']] = {'shape': 'cylinder', 'radius': float(shape.find('radius').text), 'length': float(shape.find('length').text)}
                elif shape.tag == 'box':
                    object_shape_t[model.attrib['name']] = {'shape': 'box', 'size': [float(x) for x in shape.find('size').text.split(" ")]}
    
    for key, value in object_pose_scale_t.items():
        object_data[key] = {**value, **object_shape_t[key]}

    return object_data

def generate_launch_description():
    ld = LaunchDescription()

    config_file = os.path.join(
        get_package_share_directory('sonic_description'),
        'cfg', 'sensor_cfgs.yaml')

    with open(config_file) as f:
        sensor_cfgs_array = yaml.safe_load(f)
    sensor_cfgs = {e['id']: e for e in sensor_cfgs_array}

    world_file = os.path.join(get_package_share_directory(
        'sonic_description'), 'urdf/world_circ.sdf')

    gt_data = get_world_object_data(world_file)
        
    node=Node(
        package = 'sonic_fusion',
        name = 'sonic_fusion_node',
        executable = 'sonic_fusion_node.py',
        parameters= [
            {"sensor_cfgs": repr(sensor_cfgs)},
            {"gt_data": repr(gt_data)},
        ]
    )
    ld.add_action(node)
    return ld