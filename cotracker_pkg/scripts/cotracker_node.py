#!/usr/bin/env python

import os
import sys
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import numpy as np

sys.path.append(os.path.dirname(__file__))

from window_tracker import CoTrackerWindow

def create_pointcloud_msg(points, status, image_stamp):  
    # Define PointCloud2 fields (x, y, z, channel)
    fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="status", offset=12, datatype=PointField.UINT8, count=1),
        ]

    # Create the PointCloud2 message
    header = rospy.Header()
    header.frame_id = "map"
    header.stamp = image_stamp

    points_with_status = []
    if points is not None and status is not None:
        points = points.numpy()
        status = np.array(status, dtype=np.uint8)

        # Add z = 0 and channel to the points
        points_with_status = np.hstack([
            points,  # x, y
            np.zeros((points.shape[0], 1)),  # z
            status.reshape(-1, 1)  # channel
        ])


    pointcloud_msg = pc2.create_cloud(header, fields, points_with_status)
    return pointcloud_msg

class ImageProcessorNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('cotracker_node')

        # Create a subscriber to the input topic
        self.image_subscriber = rospy.Subscriber("/cam0/image_raw", Image, self.image_callback)
        self.query_subscriber = rospy.Subscriber("/feature_tracker/queries", PointCloud2, self.queries_callback)

        # Create a publisher for the output topic
        self.points_publisher = rospy.Publisher('forward_points', PointCloud2, queue_size=10)
        self.debug_publisher = rospy.Publisher('debug_image', Image, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize the CoTrackerWindow object
        self.window = CoTrackerWindow(checkpoint='/home/tap/co-tracker/checkpoints/scaled_online.pth', device='cuda')
        self.debug = True

        rospy.loginfo("Image Processor Node is running.")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Should be `uint8`
        cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)
        self.window.add_image(cv_image)
        forw_pts, status = self.window.track()            
        forw_pts_msg = create_pointcloud_msg(forw_pts, status, msg.header.stamp)
        self.points_publisher.publish(forw_pts_msg)

        if self.debug:
            debug_image = self.window.debug_tracks()
            ros_debug_image = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_publisher.publish(ros_debug_image)
            # print("trajectories shape: ", results[0].shape)
            # print("visibility shape: ", results[1].shape)

        rospy.loginfo("Image processed and published.")

    def queries_callback(self, msg):
        # Specify the fields to extract: "x", "y", "z", and an additional channel (e.g., "intensity")
        field_names = ["x", "y", "z", "indices"]

        # Extract points and additional channel from the PointCloud2 message
        points = []
        indices = []
        for p in pc2.read_points(msg, field_names=field_names, skip_nans=True):
            points.append([p[0], p[1]])  # x, y
            indices.append(p[3])          # intensity (or other channel)

        self.window.update_queries(points, indices)


if __name__ == '__main__':
    try:
        node = ImageProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Image Processor Node.")
