#!/bin/bash

# Check if the mandatory "branch" argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <branch> [bag_directory] [playback_rate]"
    exit 1
fi

# Read arguments
BRANCH=$1  # Mandatory: Branch name to append
BAG_DIR=${2:-"/home/data/euroc"}  # Optional: Bag directory (default: /home/data/euroc)
PLAYBACK_RATE=${3:-1.0}  # Optional: Playback speed (default: 1.0x)

# Check if the bag directory exists
if [ ! -d "$BAG_DIR" ]; then
    echo "Error: Directory $BAG_DIR does not exist."
    exit 1
fi

EVAL_DIR="/home/data/euroc/${BRANCH}"  # Directory to save the recorded bags

if [ ! -d "$EVAL_DIR" ]; then
    echo "Directory does not exist. Creating: $EVAL_DIR"
    mkdir -p "$EVAL_DIR"
else
    echo "Directory already exists: $EVAL_DIR"
fi

echo "Playing rosbag files from: $BAG_DIR"
echo "Playback rate: $PLAYBACK_RATE"
echo "Branch: $BRANCH"

# Loop through all .bag files in the directory
for BAG_FILE in "$BAG_DIR"/*.bag; do
    # Extract the base file name (without extension)
    BAG_NAME=$(basename "$BAG_FILE" .bag)

    # Append branch name after an underscore
    RECORDED_BAG="${EVAL_DIR}/${BAG_NAME:0:5}_${BRANCH}.bag"

    echo "Processing bag file: $BAG_NAME"

    roslaunch tapnext_pkg tapnext.launch &
    TAPNEXT_PID=$! & # Store the tapnext process ID
    sleep 5

    # Start the ROS launch file in the background
    roslaunch vins_estimator euroc_tapnext_eval.launch bag_name:=$BAG_NAME recorded_bag_path:=$RECORDED_BAG & 
    LAUNCH_PID=$! &  # Store the launch file's process ID
    sleep 5

    # Play the current bag file at the specified rate
    rosbag play --clock --rate $PLAYBACK_RATE "$BAG_FILE"

    # Once rosbag finishes, kill the launch file
    echo "ROS bag $BAG_NAME finished. Shutting down the launch file..."
    kill -SIGINT $LAUNCH_PID
    sleep 2
    rosnode kill /rosbag_record
    kill $TAPNEXT_PID

    # Ensure all ROS nodes are shut down properly
    sleep 2
    rosnode kill -a
    pkill -9 -f ros  # Force kill


    echo "Finished processing $BAG_NAME"
    echo "-----------------------------------"
done

echo "All bag files processed!"
