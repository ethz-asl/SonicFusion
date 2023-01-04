#! /usr/bin/env python3

import time
import numpy as np
import threading
import shapely
import shapely.ops as shops
import shapely.constructive as shcon
import shapely.geometry as geom
import shapely.predicates as shpred
import shapely.affinity as shaffin

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles as QoSPP

import geometry_msgs.msg as geomsg
from sensor_msgs.msg import Range
from nav_msgs.msg import GridCells, Odometry
from visualization_msgs.msg import MarkerArray, Marker

from sonic_fusion.uss_model import USSensorModel
from sonic_fusion.utils import Utils


class SonicFusion(Node):

    _sensor_subs = dict()
    _range_data = dict()
    _sensors = dict()
    _gt_objects = dict()
    _fusion_data = dict()
    _fusion_vidata = dict()

    def __init__(self):
        super().__init__('sonic_fusion')

        self._update_func_map = {'update_fusion': self.update_fusion,
                                'update_eval': self.update_eval,
                                'update_virtual': self.update_virtual,}

        # ROS parameter declarations
        self.declare_parameter('sensor_cfgs', '')
        self.declare_parameter('gt_data', '')

        # Init ground truth object data (static) TODO: encode dimensions in names and get pose from model state
        gt_data = eval(self.get_parameter('gt_data').value)
        for key,value in gt_data.items():
            poly = None
            if value["shape"] == 'cylinder':
                poly = geom.Point(value["pose"][0], value["pose"][1]).buffer(value["radius"])
            elif value["shape"] == 'box':
                box = geom.box(-value["size"][0]/2,-value["size"][1]/2,value["size"][0]/2,value["size"][1]/2)
                poly = shaffin.affine_transform(box, [1, 0, 0, 1, value["pose"][0], value["pose"][1]])
                poly = shaffin.scale(poly, xfact=value["scale"][0], yfact=value["scale"][1])
                poly = shaffin.rotate(poly, value["pose"][5], use_radians=True)
            self._gt_objects[key] = poly

        # Get sensor configs from dict string representation param
        sensor_cfgs = eval(self.get_parameter('sensor_cfgs').value)

        max_empty_segs = []
        for sid,scfg in sensor_cfgs.items():
            # Initialize Sensor ROS Subscribers
            topic = 'sonic/uss_'+sid
            self._sensor_subs[sid] = self.create_subscription(
                Range, topic, self._construct_sensor_callback(sid), QoSPP.SENSOR_DATA.value)
            # Initialize Sensor configurations TODO: maybe better to init on first message?
            self._sensors[sid] = USSensorModel(scfg)
            # Initialize range data
            self._range_data[sid] = [scfg['max_rng']]
            # Get max empty region
            max_empty_segs.append(self._sensors[sid].get_empty_seg_body(scfg['max_rng']))

        # Compute max empty_region TODO: store and find angle etc...
        self._max_empty_region = shops.unary_union(max_empty_segs)
        self._observable = Utils.get_observable_region(self._max_empty_region) #(max_r, phi, -phi)

        # Init Robot Odometry
        self._robot_odom = [0.0 for _ in range(6)] # init at 'world' frame for objects
        self._odom_sub = self.create_subscription(
            Odometry, 'sonic/odom', self.odom_callback, QoSPP.SENSOR_DATA.value
        )

        # Init Fusion data
        self._fusion_data = {'fused_proba': lambda X,Y: 0*X + 0*Y,
                            'rois': [], 'odom': self._robot_odom}
        self._fusion_vidata = {"integr_proba":lambda X,Y: 0*X + 0*Y,**self._fusion_data}
        self._buffer_proba = []

        # Publishers
        self._gridvis_pub = self.create_publisher(GridCells, 'sonic/vis/regions', 10)
        self._roivis_pubs = [self.create_publisher(geomsg.PolygonStamped, 'sonic/vis/rois0', 10),
                            self.create_publisher(geomsg.PolygonStamped, 'sonic/vis/rois1', 10),
                            self.create_publisher(geomsg.PolygonStamped, 'sonic/vis/rois2', 10),
                            self.create_publisher(geomsg.PolygonStamped, 'sonic/vis/rois3', 10),
                            self.create_publisher(geomsg.PolygonStamped, 'sonic/vis/rois4', 10),]
        self._front_pub = self.create_publisher(MarkerArray, 'sonic/vis/front', 10)
        self._frontreg_pub = self.create_publisher(geomsg.PolygonStamped, 'sonic/vis/front_reg', 10)

    def _construct_sensor_callback(self, sensor_id):
        def sensor_callback(msg):
            # TODO: min/max range , buffering data
            self._range_data[sensor_id] = [msg.range]
        return sensor_callback

    def odom_callback(self, msg):
        curr_odom = Utils.odommsg_to_list(msg)
        for key,value in self._gt_objects.items():
            theta = self._robot_odom[5] - curr_odom[5]
            matrix = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta), 
                    curr_odom[0] - self._robot_odom[0], curr_odom[1] - self._robot_odom[1]]
            self._gt_objects[key] = shaffin.affine_transform(value,matrix)
        self._robot_odom = curr_odom

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

    def get_virtual_proba(self):
        fused_proba = self._fusion_data['fused_proba']
        phi = self._fusion_data['odom'][-1] - self._robot_odom[-1]
        xo = self._robot_odom[0] - self._fusion_data['odom'][0]
        yo = self._robot_odom[1] - self._fusion_data['odom'][1]
        def virtual_proba(X,Y):
            X_last_b = np.cos(phi)*(X - xo) + np.sin(phi)*(Y - yo)
            Y_last_b = -np.sin(phi)*(X - xo) + np.cos(phi)*(Y - yo)
            return fused_proba(X_last_b,Y_last_b)
        return virtual_proba

    def get_virtual_rois(self):
        rois = self._fusion_data['rois']
        phi = self._fusion_data['odom'][-1] - self._robot_odom[-1]
        matrix = [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi),
                self._robot_odom[0] - self._fusion_data['odom'][0], 
                self._robot_odom[1] - self._fusion_data['odom'][1]]
        vrois = [shaffin.affine_transform(r,matrix) for r in rois]
        return vrois

    def visualize(self,proba_map,rois,front,front_region):

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
            point = geomsg.Point()
            point.x = float(pxy[0])
            point.y = float(pxy[1])
            point.z = 0.0
            msg.cells.append(point)

        self._gridvis_pub.publish(msg)

        # roi to poly msg
        for roi,publ in zip(rois,self._roivis_pubs[:min(5,len(rois))]):
            msg_ply = Utils.geoply_to_plymsg(roi)
            msg_ply.header.frame_id = 'base_link'
            msg_ply.header.stamp = self.get_clock().now().to_msg()
            publ.publish(msg_ply)

        # front to markers
        msg_front = MarkerArray()

        marker_arr = []
        for i,p in enumerate(front):
            msg_marker = Marker()
            msg_marker.header.frame_id = 'base_link'
            msg_marker.header.stamp = self.get_clock().now().to_msg()
            msg_marker.type = Marker.SPHERE
            msg_marker.action = Marker.ADD
            msg_marker.ns = "window"
            msg_marker.id = i
            msg_marker.pose.position.x = float(p[0])
            msg_marker.pose.position.y = float(p[1])
            msg_marker.pose.position.z = 0.0
            msg_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            msg_marker.scale.x = 0.03
            msg_marker.scale.y = 0.03
            msg_marker.scale.z = 0.03
            msg_marker.color.a = 1.0
            msg_marker.color.r = 0.8
            msg_marker.color.g = 0.05
            msg_marker.color.b = 0.2
            marker_arr.append(msg_marker)

        msg_front.markers = marker_arr

        self._front_pub.publish(msg_front)

        # front region
        msg_freg = Utils.geoply_to_plymsg(front_region)
        msg_freg.header.frame_id = 'base_link'
        msg_freg.header.stamp = self.get_clock().now().to_msg()
        self._frontreg_pub.publish(msg_freg)

    def update_fusion(self): 
        """ Measured over 20 Hz possible """
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
        
        # Construct gaussians (body frame) and regions of interest (roi: TODO param)
        rois = []
        gaussians = dict.fromkeys(sensor_arcs.keys(), [])
        for sid, narcs in sensor_arcs.items():
            rois += [shcon.buffer(narc, 0.3, cap_style="square", join_style="bevel") for narc in narcs]
            gaussians[sid] = [self._sensors[sid].get_gauss_body(narc, range_data[sid]) 
                                for narc in narcs]
        rois = list(Utils.geometric_union(rois))
        gauss_flat = Utils.flatten(list(gaussians.values()))
        fused_proba = self.get_fused_proba(gauss_flat) if gauss_flat else lambda X,Y: 0*X+0*Y

        self._fusion_data['fused_proba'] = fused_proba
        self._fusion_data['rois'] = rois
        self._fusion_data['odom'] = self._robot_odom

    def update_virtual(self):
        # TODO test vidata fusion and continue impl
        self._fusion_vidata['fused_proba'] = self.get_virtual_proba()

        # update pos of rois
        self._fusion_vidata['rois'] = self.get_virtual_rois()
        
        # TODO integrate proba map (collect array of probas and sum up?)
        self._buffer_proba.append(self._fusion_vidata['fused_proba'])
        self._buffer_proba = self._buffer_proba[-5:]

        self._fusion_vidata['integr_proba'] = self.get_fused_proba(self._buffer_proba)


    def update_eval(self):
        # TODO param search angle and max range
        front = Utils.sample_front(self._fusion_vidata['rois'], self._observable[0],
                                   (self._observable[1],self._observable[2]), 
                                   self._fusion_vidata['fused_proba'])
        front_ply = [(0,0)]+front
        if len(front_ply) >= 4:
            front_region = geom.Polygon(geom.LinearRing(front_ply))

            nearest_errors = Utils.compute_error('nearest_object', self._gt_objects, 
                                points=[f for f in front if f[0]**2 + f[1]**2 < (self._observable[0]-0.2)**2])
            
            exact_front_region = shapely.intersection(front_region,self._max_empty_region)
            area_errors = Utils.compute_error('area_errors',self._gt_objects,ref_area=exact_front_region)

            # TODO trajectory check

            # TODO vis
            self.visualize(self._fusion_vidata['integr_proba'],self._fusion_vidata['rois'],front,front_region)

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
    thread_fusion = threading.Thread(target=sonic_fusion.loop, args=(1, 'update_fusion'), daemon=True)
    thread_fusion.start()

    thread_virtual = threading.Thread(target=sonic_fusion.loop, args=(5, 'update_virtual'), daemon=True)
    thread_virtual.start()

    thread_eval = threading.Thread(target=sonic_fusion.loop, args=(5, 'update_eval'), daemon=True)
    thread_eval.start()

    rclpy.spin(sonic_fusion)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rclpy.shutdown()
    sonic_fusion.destroy_node()
    thread_fusion.join()
    thread_virtual.join()
    thread_eval.join()
    return

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass