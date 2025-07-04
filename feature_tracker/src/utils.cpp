#include "feature_tracker.h"


pair<vector<cv::Point2f>, vector<uchar>> readResponse(const sensor_msgs::PointCloud& msg)
{
    // Vectors to store the extracted x, y, and status values
    vector<cv::Point2f> points2d;    // For storing (x, y) pairs
    vector<uchar> status_values;    // For storing status values (uchar)

    if (msg.points.size() != msg.channels[0].values.size()) {
        ROS_ERROR("Mismatch between points and channel values size in PointCloud!");
        return make_pair(points2d, status_values);
    }

    // Iterate over the points in the PointCloud
    for (size_t i = 0; i < msg.points.size(); i++) {
        const geometry_msgs::Point32& point = msg.points[i];
        float status = msg.channels[0].values[i];  // Assuming a single channel for status

        // Add the x, y coordinates to the vector
        points2d.emplace_back(point.x, point.y);

        // Store the status (cast to uchar)
        status_values.push_back(static_cast<uchar>(status));
    }

    return make_pair(points2d, status_values);
}

trackon_pkg::trackon createRequest(const vector<cv::Point2f>& queries, const vector<int>& removed_indices, const cv::Mat& img, const std_msgs::Header& header)
{
    trackon_pkg::trackon srv;

    if (queries.size() != removed_indices.size()) {
        ROS_DEBUG("Size of queries and removed_indices are different: %lu and %lu", queries.size(), removed_indices.size());
    }

    srv.request.queries.clear();
    srv.request.removed_indices.clear();

    for (int i = 0; i < static_cast<int>(queries.size()); i++) {
        geometry_msgs::Point32 p;
        p.x = queries[i].x;
        p.y = queries[i].y;
        p.z = 1;
        srv.request.queries.push_back(p);
    }

    for (int i = 0; i < static_cast<int>(removed_indices.size()); i++) {
        srv.request.removed_indices.push_back(removed_indices[i]);
    }

    ROS_DEBUG_STREAM("New queries: " << queries.size() << ", Removed indices: " << removed_indices.size());
    

    // Create image message:
    sensor_msgs::Image img_msg;
    // cv::Mat bgr_img;
    // cv::cvtColor(img, bgr_img, cv::COLOR_GRAY2BGR);
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

    return srv;
}