#ifndef LA3DM_CASSIE_UTIL_H
#define LA3DM_CASSIE_UTIL_H

#include <fstream>
#include <math.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <octomap/ColorOcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>


class CassieData {
  public:
    CassieData(ros::NodeHandle& nh,
             double resolution, double block_depth,
             double sf2, double ell,
             int num_class, double free_thresh,
             double occupied_thresh, float var_thresh, 
             double ds_resolution,
             double free_resolution, double max_range,
             std::string map_topic)
      : nh_(nh)
      , resolution_(resolution)
      , ds_resolution_(ds_resolution)
      , free_resolution_(free_resolution)
      , max_range_(max_range)
      , counter_(0) {
        map_ = new la3dm::SemanticBGKOctoMap(resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, 0.001, 0.001);
        //m_pub_ = new la3dm::MarkerArrayPub(nh_, map_topic, resolution);
        color_octomap_publisher_ = nh_.advertise<octomap_msgs::Octomap>("color_octomap_out", 10);
      }

    // Data preprocess
    void PointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg) {
      long long cloud_msg_time = (long long)(round((double)cloud_msg->header.stamp.toNSec() / 1000.0) + 0.1);

      // Save pcd files
      pcl::PointCloud<pcl::PointXYZL> cloud;
      la3dm::point3f origin;
      
      for (int i = 0; i < cloud_msg->points.size(); ++i) {
        pcl::PointXYZL pt;
        pt.x = cloud_msg->points[i].x;
        pt.y = cloud_msg->points[i].y;
        pt.z = cloud_msg->points[i].z;
        pt.label = cloud_msg->channels[0].values[i];
  
        if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
          continue;
        if (pt.label == 0 || pt.label == 13)  // Note: don't project background and sky
          continue;
        cloud.push_back(pt);
      }

      // Fetch the tf transform and write to a file
      tf::StampedTransform transform;
      try {
        listener_.lookupTransform("/odom",
                                  cloud_msg->header.frame_id,
                                  cloud_msg->header.stamp,
                                  transform);
      } catch (tf::TransformException ex) {
        std::cout<<"tf look for failed\n";
        ROS_ERROR("%s",ex.what());
        return;
      }

      Eigen::Affine3d t_eigen;
      tf::transformTFToEigen(transform, t_eigen);

      // Transform point cloud
      pcl::transformPointCloud(cloud, cloud, t_eigen);
      origin.x() = t_eigen.matrix()(0, 3);
      origin.y() = t_eigen.matrix()(1, 3);
      origin.z() = t_eigen.matrix()(2, 3);
      map_->insert_pointcloud(cloud, origin, ds_resolution_, free_resolution_, max_range_);
      if (counter_ == 1) {
        publish_map();
        counter_ = 0;
      } else
        counter_++; 
      std::cout << "Inserted point cloud at " << cloud_msg_time <<std::endl;
    }

    void publish_map() {
      
      octomap::ColorOcTree* octomap = new octomap::ColorOcTree(resolution_+0.01);
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          //m_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), it.get_node().get_semantics());
          octomap::point3d endpoint(p.x(), p.y(), p.z());
          octomap::ColorOcTreeNode* n = octomap->updateNode(endpoint, true);
          std_msgs::ColorRGBA color = la3dm::NCLTSemanticMapColor(it.get_node().get_semantics());
          n->setColor(color.r*255, color.g*255, color.b*255);
          //octomap->setNodeColor(p.x(), p.y(), p.z(), 1, 0, 0);
        }
      }
      octomap_msgs::Octomap cmap_msg;
      cmap_msg.binary = 0 ;
      cmap_msg.resolution = resolution_+0.01;
      octomap_msgs::fullMapToMsg(*octomap, cmap_msg);
      cmap_msg.header.frame_id = "/map";
      color_octomap_publisher_.publish(cmap_msg);

      //m_pub_->publish();
    }

  private:
    ros::NodeHandle nh_;
    double resolution_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;
    la3dm::SemanticBGKOctoMap* map_;
    la3dm::MarkerArrayPub* m_pub_;
    ros::Publisher color_octomap_publisher_;
    tf::TransformListener listener_;
    int counter_;
};

#endif // LA3DM_CASSIE_UTIL_H
