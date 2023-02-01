# USS_obstacle_detection

Workspace repository for semester thesis at ETH about ultrasonic sensor fusion for obstacle detection on mobile warehouse robot.

## Install

Requirements: 
- Ubuntu 22.04, 
- ROS2 Rolling Ridley, 
- Shapely (https://shapely.readthedocs.io/en/stable/manual.html)

Copy the packages in src/ to your colcon workspace and build.

## Run

Source the workspace and set `export PYTHONOPTIMIZE=1` in your working terminal to suppress assertions caused by inf values in range messages. (ROS2 issue)

To run the simulation:

`ros2 launch sonic_description sonic_spawn.launch.py`

and in another terminal run:

`ros2 run sonic_fusion sonic_fusion.launch.py`

To visualize the results use rviz2 and the config src/sonic_description/launch/sonic.rviz

The robot can be moved by running the teleop command in a separate terminal:

`ros2 run teleop_twist_keyboard teleop_twist_keyboard cmd_vel:=/sonic/cmd_vel`
