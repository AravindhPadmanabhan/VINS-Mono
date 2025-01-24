#include "feature_tracker.h"


pair<vector<cv::Point2f>, vector<uchar>> readResponse(const sensor_msgs::PointCloudConstPtr& msg)
{
    // Vectors to store the extracted x, y, and status values
    vector<cv::Point2f> points2d;    // For storing (x, y) pairs
    vector<uchar> status_values;    // For storing status values (uchar)

    if (msg->points.size() != msg->channels[0].values.size()) {
        ROS_ERROR("Mismatch between points and channel values size in PointCloud!");
        return make_pair(points2d, status_values);
    }

    // Iterate over the points in the PointCloud
    for (size_t i = 0; i < msg->points.size(); ++i) {
        const geometry_msgs::Point32& point = msg->points[i];
        float status = msg->channels[0].values[i];  // Assuming a single channel for status

        // Add the x, y coordinates to the vector
        points2d.emplace_back(point.x, point.y);

        // Store the status (cast to uchar)
        status_values.push_back(static_cast<uchar>(status));
    }

    return make_pair(points2d, status_values);
}

cotracker_pkg::cotracker createRequest(const vector<cv::Point2f>& queries, const vector<int>& removed_indices, const cv::Mat& img, const std_msgs::Header& header)
{
    cotracker_pkg::cotracker srv;

    if (queries.size() != removed_indices.size()) {
        ROS_ERROR("Size of queries and removed_indices must match!");
        return;
    }

    // Create the PointCloud message
    sensor_msgs::PointCloud pointcloud_msg;
    pointcloud_msg.header = header;

    // Resize the points and channels
    pointcloud_msg.points.resize(queries.size());
    pointcloud_msg.channels.resize(1);  // Assuming a single channel for indices
    pointcloud_msg.channels[0].name = "indices";
    pointcloud_msg.channels[0].values.resize(queries.size());

    // Fill the points and channels
    for (size_t i = 0; i < queries.size(); ++i) {
        // Assign x, y, z coordinates
        pointcloud_msg.points[i].x = queries[i].x;
        pointcloud_msg.points[i].y = queries[i].y;
        pointcloud_msg.points[i].z = 0.0;  // Set z = 0

        // Store the channel value
        pointcloud_msg.channels[0].values[i] = static_cast<float>(removed_indices[i]);
    }

    // Create image message:
    sensor_msgs::Image img_msg;
    try {
        // Use cv_bridge to convert the cv::Mat to a ROS Image message
        cv_bridge::CvImage cv_img;
        cv_img.header = header;               // Use the provided header
        cv_img.encoding = sensor_msgs::image_encodings::BGR8; // Adjust encoding based on your image type
        cv_img.image = img;                   // Assign the OpenCV image

        // Convert to sensor_msgs::Image
        img_msg = *cv_img.toImageMsg();
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return srv; // Return empty service object if conversion fails
    }

    srv.request.image = img_msg;          // Assign the image message
    srv.request.new_queries = pointcloud_msg;

    return srv;
}

