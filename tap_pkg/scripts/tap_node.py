#!/usr/bin/env python

import os
import sys
import rospy
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import numpy as np

sys.path.append(os.path.dirname(__file__))

from tap_pkg.srv import tap, tapResponse 
from cotracker_window import CoTrackerWindow
from trackon_tracker import TrackOnTracker
from tapnext_tracker import TAPNextTracker

def create_pointcloud_msg(points, status, image_stamp):
    pointcloud_msg = PointCloud()
    pointcloud_msg.header.frame_id = "map"
    pointcloud_msg.header.stamp = image_stamp

    if points is not None and status is not None:
        points = points.cpu().numpy()
        status = np.array(status.cpu(), dtype=np.float32)
        
        status_channel = ChannelFloat32()
        status_channel.name = "status"

        for i in range(points.shape[0]):
            point = Point32()
            point.x = points[i, 0]
            point.y = points[i, 1]
            point.z = 0.0
            pointcloud_msg.points.append(point)
            status_channel.values.append(status[i])

        pointcloud_msg.channels.append(status_channel)

    return pointcloud_msg

class TAPNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('tap_service')
        self.service = rospy.Service("tap", tap, self.callback)
        self.debug_publisher = rospy.Publisher('/tap/debug_image', Image, queue_size=10)
        self.bridge = CvBridge()

        self.model = rospy.get_param('~model', "cotracker")

        if self.model == 'cotracker':
            checkpoint = rospy.get_param('~cotracker/checkpoint', "/home/TAP-VINS/catkin_ws/tap/co-tracker/checkpoints/scaled_online.pth")
            offline_checkpoint = rospy.get_param('~cotracker/offline_checkpoint', "/home/TAP-VINS/catkin_ws/tap/co-tracker/checkpoints/scaled_offline.pth")
            device = rospy.get_param('~device', 'cuda')
            debug = rospy.get_param('~debug', False)

            local_grid_size = rospy.get_param('~cotracker/local_grid_size', 0)
            local_grid_extent = rospy.get_param('~cotracker/local_grid_extent', 0)

            self.tracker = CoTrackerNode(checkpoint=checkpoint,
                                         offline_checkpoint=offline_checkpoint,
                                         local_grid_size=local_grid_size,
                                         local_grid_extent=local_grid_extent,
                                         device=device,
                                         debug=debug)

        elif self.model == 'trackon':
            checkpoint = rospy.get_param('~trackon/checkpoint', "/home/TAP-VINS/catkin_ws/tap/track_on/checkpoints/track_on_checkpoint.pt")
            device = rospy.get_param('~device', 'cuda')
            debug = rospy.get_param('~debug', False)

            self.tracker = TrackOnNode(checkpoint=checkpoint, device=device, debug=debug)

        elif self.model == 'tapnext':
            checkpoint = rospy.get_param('~tapnext/checkpoint', "/home/TAP-VINS/catkin_ws/tap/tapnet/checkpoints/bootstapnext_ckpt.npz")
            device = rospy.get_param('~device', 'cuda')
            debug = rospy.get_param('~debug', False)

            self.tracker = TAPNextNode(checkpoint=checkpoint, device=device, debug=debug)

        rospy.loginfo("TAP Node is running.")

    def callback(self, request):
        cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)
        forw_pts, status, debug_img = self.tracker.track_callback(cv_image, request.queries, request.removed_indices)
        forw_pts_msg = create_pointcloud_msg(forw_pts, status, request.image.header.stamp)

        if debug_img is not None:
            ros_debug_image = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            self.debug_publisher.publish(ros_debug_image)
            
        return tapResponse(forw_pts_msg)

class CoTrackerNode:
    def __init__(self, checkpoint, offline_checkpoint, local_grid_size, local_grid_extent, device, debug):
        self.window = CoTrackerWindow(checkpoint=checkpoint,
                                      offline_checkpoint=offline_checkpoint,
                                      local_grid_size=local_grid_size,
                                      local_grid_extent=local_grid_extent,
                                      device=device)

        self.debug = debug

    def track_callback(self, image, queries, removed_indices):
        if (len(self.window.video)==0):
            forw_pts, status = self.image_callback(image)
            return forw_pts, status, None
            
        self.queries_callback(queries, removed_indices)
        forw_pts, status = self.image_callback(image)

        debug_image = None
        if self.debug:
            debug_image = self.window.debug_tracks()
        
        return forw_pts, status, debug_image

    def image_callback(self, image):
        self.window.add_image(image)
        forw_pts, status = self.window.track()            
        return forw_pts, status

    def queries_callback(self, new_queries, removed_indices):
        points = []
        indices = []
        
        for point in new_queries:
            points.append([point.x, point.y])  # Extract x, y
        
        for index in removed_indices:
            indices.append(index)

        self.window.update_queries(points, indices)

class TrackOnNode:
    def __init__(self, checkpoint, device, debug):
        self.tracker = TrackOnTracker(checkpoint=checkpoint,
                                      device=device)

        self.debug = debug

        rospy.loginfo("TrackOn Node is running.")

    def track_callback(self, image, queries, removed_indices):
        if (self.tracker.frame_no==-1):
            forw_pts, status = self.image_callback(image, first_image=True)
            return forw_pts, status, None
            
        self.queries_callback(queries, removed_indices)
        forw_pts, status = self.image_callback(image)

        debug_image = None
        if self.debug:
            debug_image = self.tracker.debug_tracks()
        
        return forw_pts, status, debug_image

    def image_callback(self, image, first_image=False):
        if first_image:
            self.tracker.init_image(image)
            forw_pts = None
            status = None
        else:
            forw_pts, status = self.tracker.track(image)            
        return forw_pts, status

    def queries_callback(self, new_queries, removed_indices):
        points = []
        indices = []
        
        for point in new_queries:
            points.append([point.x, point.y])  # Extract x, y
        
        for index in removed_indices:
            indices.append(index)

        self.tracker.update_queries(points, indices)

class TAPNextNode:
    def __init__(self, checkpoint, device, debug):
        self.tracker = TAPNextTracker(checkpoint=checkpoint,
                                        device=device)

        self.debug = debug

    def track_callback(self, image, queries, removed_indices):
        if (self.tracker.frame_no==-1):
            forw_pts, status = self.image_callback(image, first_image=True)
            return forw_pts, status, None
            
        self.queries_callback(queries, removed_indices)
        forw_pts, status = self.image_callback(image)

        debug_image = None
        if self.debug:
            debug_image = self.tracker.debug_tracks()
        
        return forw_pts, status, debug_image

    def image_callback(self, image, first_image=False):
        if first_image:
            self.tracker.init_image(image)
            forw_pts = None
            status = None
        else:
            forw_pts, status = self.tracker.track(image)
        return forw_pts, status

    def queries_callback(self, new_queries, removed_indices):
        points = []
        indices = []
        
        for point in new_queries:
            points.append([point.x, point.y])  # Extract x, y
        
        for index in removed_indices:
            indices.append(index)

        self.tracker.update_queries(points, indices)

if __name__ == '__main__':
    try:
        node = TAPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down TAP Node.")
