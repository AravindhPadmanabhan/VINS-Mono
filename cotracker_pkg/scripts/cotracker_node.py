#!/usr/bin/env python

import os
import sys
import rospy
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import numpy as np

sys.path.append(os.path.dirname(__file__))

from cotracker_pkg.srv import cotracker, cotrackerResponse 
from window_tracker import CoTrackerWindow

def create_pointcloud_msg(points, status, image_stamp):
    # Create the PointCloud message
    pointcloud_msg = PointCloud()
    pointcloud_msg.header.frame_id = "map"
    pointcloud_msg.header.stamp = image_stamp

    if points is not None and status is not None:
        # Convert points and status to numpy arrays
        points = points.cpu().numpy()
        status = np.array(status.cpu(), dtype=np.float32)
        # print("number of successful tracks: ", status.sum())
        
        # Create a channel for status
        status_channel = ChannelFloat32()
        status_channel.name = "status"

        for i in range(points.shape[0]):
            point = Point32()
            point.x = points[i, 0]
            point.y = points[i, 1]
            point.z = 0.0  # Set z = 0
            pointcloud_msg.points.append(point)
            status_channel.values.append(status[i])

        pointcloud_msg.channels.append(status_channel)

    return pointcloud_msg
class CoTrackerNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('cotracker_service')
        self.service = rospy.Service("cotracker", cotracker, self.track_callback)

        # Create a publisher for the output topic
        self.debug_publisher = rospy.Publisher('/cotracker/debug_image', Image, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize the CoTrackerWindow object
        self.window = CoTrackerWindow(checkpoint='/home/co-tracker/checkpoints/scaled_online.pth',
                                      offline_checkpoint='/home/co-tracker/checkpoints/scaled_offline.pth',
                                      device='cuda')
        self.debug = True

        rospy.loginfo("CoTracker Node is running.")

    def track_callback(self, request):
        if (len(self.window.video)==0):
            forward_points_msg = self.image_callback(request.image)
            return cotrackerResponse(forward_points_msg)
            
        self.queries_callback(request.new_queries)
        forward_points_msg = self.image_callback(request.image)

        if self.debug:
            debug_image = self.window.debug_tracks()
            ros_debug_image = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_publisher.publish(ros_debug_image)
        
        return cotrackerResponse(forward_points_msg)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Should be `uint8`
        cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)
        self.window.add_image(cv_image)
        forw_pts, status = self.window.track()            
        forw_pts_msg = create_pointcloud_msg(forw_pts, status, msg.header.stamp)
        return forw_pts_msg

    def queries_callback(self, msg):
        # Extract points and additional channel from the PointCloud2 message
        points = []
        indices = []
        if not msg.channels or len(msg.channels[0].values) != len(msg.points):
            rospy.logwarn("PointCloud message has inconsistent channels or no data.")
            return
        
        for point, channel_value in zip(msg.points, msg.channels[0].values):
            points.append([point.x, point.y])  # Extract x, y
            indices.append(int(channel_value))          # intensity (or other channel)

        self.window.update_queries(points, indices)
        # print(self.window.queries)
        # rospy.loginfo("Queries updated.")


if __name__ == '__main__':
    try:
        node = CoTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down CoTracker Node.")
