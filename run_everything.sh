#!/bin/bash

cd ~/ws_drone_racing/

# Source ROS2
source /opt/ros/galactic/setup.bash
colcon build
source install/setup.bash

# Source gazebo
export GAZEBO_MODEL_PATH=${PWD}/install/tello_gazebo/share/tello_gazebo/models
source /usr/share/gazebo/setup.sh

#Launch 
ros2 launch tello_race launch.py
