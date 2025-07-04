#!/bin/bash

# Usage info
usage() {
    echo "Usage: $0 [euroc_bags_directory] [branch (name to append to sequence. If branch="eval", recorded bag will look like MH_01_eval)] [rosbag_playback_rate]"
    exit 1
}

# Read arguments
BAG_DIR=${1:-"/home/data/euroc"}  
BRANCH=${2:-"eval"}
PLAYBACK_RATE=${3:-0.5}  

# Check if the bag directory exists
if [ ! -d "$BAG_DIR" ]; then
    echo "Error: Directory $BAG_DIR does not exist."
    exit 1
fi

EVAL_DIR="${BAG_DIR}/${BRANCH}"  # Directory to save the recorded bags

if [ ! -d "$EVAL_DIR" ]; then
    echo "Directory does not exist. Creating: $EVAL_DIR"
    mkdir -p "$EVAL_DIR"
else
    echo "Directory already exists: $EVAL_DIR"
fi

echo "Playing rosbag files from: $BAG_DIR"
echo "Playback rate: $PLAYBACK_RATE"
echo "Branch: $BRANCH"

source ../../devel/setup.bash  # Source the ROS workspace

# Loop through all .bag files in the directory
for BAG_FILE in "$BAG_DIR"/*.bag; do
    # Extract the base file name (without extension)
    BAG_NAME=$(basename "$BAG_FILE" .bag)

    # Append branch name after an underscore
    RECORDED_BAG="${EVAL_DIR}/${BAG_NAME:0:5}_${BRANCH}.bag"

    echo "Processing bag file: $BAG_NAME"

    roslaunch trackon_pkg trackon.launch &
    TRACKON_PID=$! & # Store the trackon process ID
    sleep 5

    # Start the ROS launch file in the background
    roslaunch vins_estimator euroc_trackon_eval.launch bag_name:=$BAG_NAME recorded_bag_path:=$RECORDED_BAG & 
    LAUNCH_PID=$! &  # Store the launch file's process ID
    sleep 5

    # Play the current bag file at the specified rate
    rosbag play --clock --rate $PLAYBACK_RATE "$BAG_FILE"

    # Once rosbag finishes, kill the launch file
    echo "ROS bag $BAG_NAME finished. Shutting down the launch file..."
    kill -SIGINT $LAUNCH_PID
    sleep 2
    rosnode kill /rosbag_record
    kill $TRACKON_PID

    # Ensure all ROS nodes are shut down properly
    sleep 2
    rosnode kill -a
    pkill -9 -f ros  # Force kill


    echo "Finished processing $BAG_NAME"
    echo "-----------------------------------"
done


echo "All bag files processed!"
echo "Recorded bags are saved in: $EVAL_DIR"