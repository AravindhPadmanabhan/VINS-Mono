cmake_minimum_required(VERSION 3.0.2)
project(tapnext_pkg)

## Find dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  message_generation # Include this for custom messages/services
)

## Declare custom service files
add_service_files(
  FILES
  tapnext.srv
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## Declare this package as a catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)

## Mark scripts for installation
catkin_install_python(PROGRAMS
  scripts/tapnext_node.py
  # scripts/tapnext_tracker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
