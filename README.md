# tello_race
A tello drone simulation group project using ROS2 and Gazebo simulator for Aerial Robotics Course work at University of Turku, Finland.
...
# Pre-Requisites
## Install ROS2 Galactic
```https://docs.ros.org/ with the `ros-galactic-desktop` option.```

## Install gazebo
```
sudo apt install gazebo11 libgazebo11 libgazebo11-dev
```

## Add the following
```
sudo apt install libasio-dev
sudo apt install ros-galactic-cv-bridge ros-galactic-camera-calibration-parsers
sudo apt install libignition-rendering3
pip3 install transformations
```

## Clone the Packages
```
mkdir -p ~/ws_drone_racing/src
cd ~/tello_ros_ws/src
git clone https://github.com/TIERS/tello-ros2-gazebo.git
git clone https://github.com/abdulmalik0103/tello_race.git
```

## Run simulation
```
cd tello_race 
./run_everything.sh
```
