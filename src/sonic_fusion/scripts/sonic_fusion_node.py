#! /usr/bin/env python3

import numpy as np
import threading
import shapely.ops as shops
import shapely.constructive as shcon
import shapely.geometry as geom
import shapely.predicates as shpred

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles as QoSPP

from geometry_msgs.msg import Point
from sensor_msgs.msg import Range
from nav_msgs.msg import GridCells

from sonic_fusion.uss_model import USSensorModel
from sonic_fusion.utils import Utils


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

            # Initialize Sensor configurations TODO: maybe better to init on first message?
            self._sensors[sid] = USSensorModel(scfg)

            # Initialize range data
            self._range_data[sid] = [scfg['max_rng']]

        # TODO: Compute max empty_region

        self._gridvis_pub = self.create_publisher(GridCells, 'sonic/vis/regions', 10)

    def _construct_sensor_callback(self, sensor_id):
        def sensor_callback(msg):
            # TODO: min/max range , buffering data
            self._range_data[sensor_id] = [msg.range]
        return sensor_callback

    def get_range_data(self):
        # TODO: update if buffering data
        current_data = {sid: buffer[0] for sid,buffer in self._range_data.items()}
        return current_data

    def get_fused_proba(self, proba_funcs):
        def fused_proba(X,Y):
            eval = proba_funcs[0](X,Y)
            for pf in proba_funcs[1:]:
                eval = eval + pf(X,Y) - eval*pf(X,Y)
            return eval
        return fused_proba

    def visualize(self,proba_map):
        #env = shcon.envelope(empty_regions)

        # build grid and evaluate
        x = np.linspace(0,4.5,100)
        y = np.linspace(-4,4,100)
        X, Y = np.meshgrid(x,y)
        Pxy = proba_map(X,Y)
        xyp = np.vstack([X.ravel(), Y.ravel(), Pxy.ravel()])
        xy_array = [(xyp[0,i],xyp[1,i]) for i in range(10000) if xyp[2,i]>0.1]

        msg = GridCells()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.cell_height = 4.5/100
        msg.cell_width = 8/100
        msg.cells = []
        
        for pxy in xy_array:
            point = Point()
            point.x = float(pxy[0])
            point.y = float(pxy[1])
            point.z = 0.0
            msg.cells.append(point)

        self._gridvis_pub.publish(msg)

    def update_main(self):
        range_data = self.get_range_data()
        
        # TODO: improve empty and arc by combining (less computations)
        empty_segs = [scfg.get_empty_seg_body(range_data[sid]) 
                        for sid,scfg in self._sensors.items()]
        empty_regions = shops.unary_union(empty_segs)

        sensor_arcs = {sid: scfg.get_arc_body(range_data[sid]) 
                        for sid,scfg in self._sensors.items() 
                        if range_data[sid]<scfg.rng[1]-scfg.empty_thr}

        # Resolve model conflicts and unpack all arcs if they were divided (drop empty)
        for sid,arc in sensor_arcs.items():
            sensor_arcs[sid] = [a for a in Utils.geometric_difference((arc,empty_regions)) 
                                    if not shpred.is_empty(a)]
        
        # Construct gaussians (body frame)
        gaussians = dict.fromkeys(sensor_arcs.keys(), [])
        for sid, narcs in sensor_arcs.items():
            gaussians[sid] = [self._sensors[sid].get_gauss_body(narc, range_data[sid]) 
                                for narc in narcs]
        gauss_flat = Utils.flatten(list(gaussians.values()))
        proba_map = self.get_fused_proba(gauss_flat) if gauss_flat else lambda X,Y: 0*X+0*Y

        self.visualize(proba_map)

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