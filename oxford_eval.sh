#!/bin/bash

# Usage info
usage() {
    echo "Usage: $0 [oxford-spires_bags_directory] [branch (name to append to sequence. If branch="eval", recorded bag will look like MH_01_eval)] [rosbag_playback_rate] [metric (RPE or ATE)]"
    exit 1
}

# Read arguments
BAG_DIR=${1:-"/home/data/oxford"}  
BRANCH=${2:-"eval"}
PLAYBACK_RATE=${3:-0.3}  
METRIC=${4:-"RPE"}

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
    RECORDED_BAG="${EVAL_DIR}/${BAG_NAME}_${BRANCH}.bag"

    echo "Processing bag file: $BAG_NAME"

    roslaunch trackon_pkg trackon.launch &
    TRACKON_PID=$! & # Store the trackon process ID
    sleep 5

    # Start the ROS launch file in the background
    roslaunch vins_estimator oxford_eval.launch recorded_bag_path:=$RECORDED_BAG & 
    LAUNCH_PID=$! &  # Store the launch file's process ID
    sleep 5

    # Play the current bag file at the specified rate
    rosbag play --clock -s 20 --rate $PLAYBACK_RATE "$BAG_FILE"

    # Once rosbag finishes, kill the launch file
    echo "ROS bag $BAG_NAME finished. Shutting down the launch file..."
    kill -SIGINT $LAUNCH_PID
    sleep 2
    rosnode kill /rosbag_record
    kill $TRACKON_PID

    # Ensure all ROS nodes are shut down properly
    sleep 2
    # rosnode kill -a
    # pkill -9 -f ros  # Force kill


    echo "Finished processing $BAG_NAME"
    echo "-----------------------------------"
done

echo "All bag files processed!"

sleep 5

RAW_DIR="$EVAL_DIR/raw_results/"
TRANSFORMED_DIR="$EVAL_DIR/transformed_results/"

if [ ! -d "$RAW_DIR" ]; then
    mkdir -p "$RAW_DIR"
fi

if [ ! -d "$TRANSFORMED_DIR" ]; then
    mkdir -p "$TRANSFORMED_DIR"
fi

# Loop through all .bag files in the directory
for BAG_FILE in "$EVAL_DIR/"*.bag; do
    BAG_NAME=$(basename "$BAG_FILE" .bag)
    RAW_NAME="${RAW_DIR}/${BAG_NAME}.tum"
    TRANSFORMED_NAME="${TRANSFORMED_DIR}/${BAG_NAME}.tum"
    evo_traj bag $BAG_FILE /vins_estimator/odometry --save_as_tum --silent
    mv vins_estimator_odometry.tum  $RAW_NAME
    python3 ../../../spires/transform_to_base.py --input_file $RAW_NAME --output_file $TRANSFORMED_NAME

    PREFIX="${BAG_NAME%%_*}"
    GT_NAME="../../../spires/gt/${PREFIX}_gt.txt"
    
    echo "---------------------------------"
    echo "Results of $BAG_NAME"
    echo "---------------------------------"
    if [ "$METRIC" = "RPE" ]; then
        evo_rpe tum $GT_NAME $TRANSFORMED_NAME -a -d 1 -u m
    elif [ "$METRIC" = "APE" ]; then
        evo_ape tum $GT_NAME $TRANSFORMED_NAME -a
    fi
done

echo "Evaluation complete"