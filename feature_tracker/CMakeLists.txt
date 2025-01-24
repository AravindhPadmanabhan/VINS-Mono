cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    message_filters
    cotracker_pkg
)

find_package(OpenCV REQUIRED)

catkin_package(
    CATKIN_DEPENDS
        roscpp
        std_msgs
        sensor_msgs
        cv_bridge
        camera_model
        message_filters
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    src/utils.cpp
)

target_link_libraries(feature_tracker
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)
