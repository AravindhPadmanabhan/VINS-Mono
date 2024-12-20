cmake_minimum_required(VERSION 3.0.2)
project(cotracker_pkg)

## Find dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
)

## Declare this package as a catkin package
catkin_package()

## Mark scripts for installation
catkin_install_python(PROGRAMS
  scripts/cotracker_node.py
  # scripts/window_tracker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

