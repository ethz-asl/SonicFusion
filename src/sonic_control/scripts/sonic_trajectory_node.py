#! /usr/bin/env python3

import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles as QoSPP


class SonicTrajectory(Node):

    def __init__(self):
        super().__init__('sonic_fusion')

    def loop(self, rate: float):
        ros_rate = self.create_rate(rate)
        while rclpy.ok():
            # TODO Things
            ros_rate.sleep()


def main(args=None):
    rclpy.init(args=args)

    sonic_traj = SonicTrajectory()

    # Loop updates in a fixed rate thread 
    # (https://answers.ros.org/question/358343/rate-and-sleep-function-in-rclpy-library-for-ros2/)
    thread_traj = threading.Thread(target=sonic_traj.loop, args=(30,), daemon=True)
    thread_traj.start()

    rclpy.spin(sonic_traj)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    rclpy.shutdown()
    sonic_traj.destroy_node()
    thread_traj.join()
    return

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass