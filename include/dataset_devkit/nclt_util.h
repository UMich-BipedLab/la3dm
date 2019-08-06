#ifndef LA3DM_NCLT_UTIL_H
#define LA3DM_NCLT_UTIL_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/common/transforms.h>

#include "PointSegmentedDistribution.h"
//#include "markerarray_pub.h"

template <unsigned int NUM_CLASS>
class NCLTData {
  public:
    NCLTData(ros::NodeHandle& nh,
             double resolution, double block_depth,
             double sf2, double ell,
             int num_class,
             double free_thresh, double occupied_thresh,
             float var_thresh)
      : nh_(nh) {
        map_ = new la3dm::SemanticBGKOctoMap(resolution, block_depth, sf2, ell, 15, free_thresh, occupied_thresh, var_thresh, 0.01, 0.01);
        pc_publisher_ = nh.advertise<sensor_msgs::PointCloud2>("cloud_out", 10);
      }
  
  inline void PointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg) {

    pcl::PointCloud<pcl::PointXYZL> cloud;
    la3dm::point3f origin;

    cloud.header.frame_id = cloud_msg->header.frame_id;
    for (int i = 0; i < cloud_msg->points.size(); ++i) {
      pcl::PointXYZL pt;
      pt.x = cloud_msg->points[i].x;
      pt.y = cloud_msg->points[i].y;
      pt.z = cloud_msg->points[i].z;
      pt.label = cloud_msg->channels[0].values[i] + 1;
      if (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z > 20*20)
        continue;
      cloud.push_back(pt);
    }
    std::cout<<"At time "<<cloud_msg->header.stamp.toSec()<<", # of lidar pts is "<<cloud.points.size()<<std::endl;

    // fetch the tf transform at that time
    tf::StampedTransform transform;
    try {
      listener_.lookupTransform("/map",
                               cloud_msg->header.frame_id,
                               cloud_msg->header.stamp,
                               transform);
    } catch (tf::TransformException ex) {
      std::cout<<"tf look for failed\n";
      ROS_ERROR("%s",ex.what());
      return;
    }
    Eigen::Affine3d T_map2body_eigen;
    tf::transformTFToEigen (transform,T_map2body_eigen);


    // Transform point cloud
    pcl::transformPointCloud (cloud, cloud, T_map2body_eigen);
    origin.x() = T_map2body_eigen.translation().x();
    origin.y() = T_map2body_eigen.translation().y();
    origin.z() = T_map2body_eigen.translation().z();

    // Publish point cloud msg
    sensor_msgs::PointCloud2 cloud_msg_out;
    pcl::toROSMsg(cloud, cloud_msg_out);
    cloud_msg_out.header = cloud_msg->header;
    cloud_msg_out.header.frame_id = "/map";
    pc_publisher_.publish(cloud_msg_out);

    map_->insert_pointcloud(cloud, origin, 0.1, 20, 20);
  
    la3dm::MarkerArrayPub m_pub(nh_, "/occupied_cells_vis_array", 0.1f);
    for (auto it = map_->begin_leaf(); it != map_->end_leaf(); it++) {
      if (it.leaf_it.tree == nullptr)
        std::cout << "null" << std::endl;
      la3dm::point3f p = it.get_loc();
      la3dm::SemanticOcTreeNode node = map_->search(p);
      if (node.get_state() == la3dm::State::OCCUPIED) {
        //la3dm::point3f p = it.get_loc();
        int semantics = node.get_semantics();
        //std::cout << "semanitcs: " << semantics << std::endl;
        m_pub.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), semantics);
      }
    }
        m_pub.publish();

    /*for (int i = 0; i < cloud_msg->points.size(); ++i) {
      pcl::PointXYZL pt;
      pt.x = cloud_msg->points[i].x;
      pt.y = cloud_msg->points[i].y;
      pt.z = cloud_msg->points[i].z;
      la3dm::SemanticOcTreeNode node = map_->search(pt.x, pt.y, pt.z);
      if (node.get_state() == la3dm::State::OCCUPIED) {
        int semantics = node.get_semantics();
        std::cout << "semanitcs: " << semantics << std::endl;
        m_pub.insert_point3d_semantics(pt.x, pt.y, pt.z, 0.1, semantics);
        m_pub.publish();
      }
    }*/
  }  

  private:
  ros::NodeHandle nh_;
  la3dm::SemanticBGKOctoMap* map_;
  tf::TransformListener listener_;
  ros::Publisher pc_publisher_;



};



#endif // LA3DM_NCLT_UTIL_H
