#! /usr/bin/env python3

import threading
import shapely.ops as shops

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles as QoSPP

from sensor_msgs.msg import Range

from sonic_fusion.uss_model import USSensorModel


class SonicFusion(Node):

    _sensor_subs = dict()
    _range_data = dict()
    _sensors = dict()

    def __init__(self) -> None:
        super().__init__('sonic_fusion')

        self._update_func_map = {'update_main': self.update_main}

        # ROS parameter declarations
        self.declare_parameter('sensor_cfgs', '')

        # Get sensor configs from dict string representation param
        sensor_cfgs = eval(self.get_parameter('sensor_cfgs').value)

        for sid,scfg in sensor_cfgs.items():
            # Initialize Sensor ROS Subscribers
            topic = 'sonic/uss_'+sid
            self._sensor_subs[sid] = self.create_subscription(
                Range, topic, self._construct_sensor_callback(sid), QoSPP.SENSOR_DATA.value)

            # Initialize Sensor configurations
            self._sensors[sid] = USSensorModel(scfg,empty_thr=0.1)

            # Initialize range data
            self._range_data[sid] = [scfg['max_rng']]

    def _construct_sensor_callback(self, sensor_id):
        def sensor_callback(msg):
            # TODO: min/max range , buffering data
            self._range_data[sensor_id] = [msg.range]
        return sensor_callback

    def get_range_data(self):
        # TODO: update if buffering data
        current_data = {sid: buffer[0] for sid,buffer in self._range_data.items()}
        return current_data

    def update_main(self):
        range_data = self.get_range_data()
        
        # TODO: improve empty and arc by combining (less computations)
        empty_segs = [scfg.get_empty_seg(range_data[sid]) 
                        for sid,scfg in self._sensors.items()]
        empty_regions = shops.unary_union(empty_segs)

        sensor_arcs = [scfg.get_arc(range_data[sid]) 
                        for sid,scfg in self._sensors.items() 
                        if range_data[sid]<scfg.rng[1]-scfg.empty_thr]

        # Resolve model conflicts
        new_arcs = [e.difference(empty_regions) for e in sensor_arcs]
        
        # construct gaussians and bayesian full func (THINK ABOUT UPDATE TRANSFORM)
        
        # TODO: REMOVE AFTER DEBUGGING
        self.get_logger().info(str(type(new_arcs[0])))

    def loop(self, rate: float, update_type: str):
        ros_rate = self.create_rate(rate)
        while rclpy.ok():
            self._update_func_map[update_type]()
            ros_rate.sleep()


def main(args=None):
    rclpy.init(args=args)

    sonic_fusion = SonicFusion()

    # Loop updates in a fixed rate thread 
    # (https://answers.ros.org/question/358343/rate-and-sleep-function-in-rclpy-library-for-ros2/)
    thread_update = threading.Thread(target=sonic_fusion.loop, args=(1, 'update_main'), daemon=True)
    thread_update.start()

    rclpy.spin(sonic_fusion)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rclpy.shutdown()
    sonic_fusion.destroy_node()
    thread_update.join()
    return

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass