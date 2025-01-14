#!/usr/bin/env python

import os
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

sys.path.append(os.path.dirname(__file__))

from window_tracker import CoTrackerWindow

            

class ImageProcessorNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('cotracker_node')

        # Create a subscriber to the input image topic
        self.subscription = rospy.Subscriber(
            "/cam0/image_raw",  # Replace with your input topic
            Image,
            self.image_callback
        )

        # Create a publisher for the output image topic
        self.publisher = rospy.Publisher(
            '/output_image',  # Replace with your output topic
            Image,
            queue_size=10
        )

        self.debug_publisher = rospy.Publisher(
            '/debug_image',  # Replace with your output topic
            Image,
            queue_size=10
        )

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize the CoTrackerWindow object
        self.window = CoTrackerWindow(checkpoint='/home/tap/co-tracker/checkpoints/scaled_online.pth', device='cuda')
        self.debug = True

        rospy.loginfo("Image Processor Node is running.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Should be `uint8`

            cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)

            # (Optional) Process the image here if needed
            # For now, we'll just pass it through unchanged.
            self.window.add_image(cv_image)
            results = self.window.track()

            if self.debug:
                debug_image = self.window.debug_tracks()
                ros_debug_image = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                self.debug_publisher.publish(ros_debug_image)
                print("trajectories shape: ", results[0].shape)
                print("visibility shape: ", results[1].shape)

            # Convert OpenCV image back to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

            # Publish the image
            self.publisher.publish(ros_image)
            rospy.loginfo("Image processed and published.")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


if __name__ == '__main__':
    try:
        node = ImageProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Image Processor Node.")
