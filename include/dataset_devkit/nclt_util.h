#ifndef LA3DM_NCLT_UTIL_H
#define LA3DM_NCLT_UTIL_H

#include <fstream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

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

      long long cloud_msg_time = (long long) (round( ((double)cloud_msg->header.stamp.toNSec()) / 1000.0 ) + 0.1);

      std::cout << cloud_msg_time << std::endl;
      if ( cloud_msg_time < 1335704231918265 || cloud_msg_time > 1335704347924201)
        return;

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
      std::string cloud_name = std::to_string(cloud_msg_time) + ".pcd";
      pcl::io::savePCDFileASCII (cloud_name, cloud);
      std::cout<<"At time "<< std::setprecision(16) << cloud_msg->header.stamp.toSec()<<", # of lidar pts is "<<cloud.points.size()<<std::endl;

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
      
      Eigen::Affine3d t_eigen;
      tf::transformTFToEigen (transform, t_eigen);
      Eigen::Matrix4d t_matrix = t_eigen.matrix();

      myfile_.open("lidar_poses.txt", std::ios::app);
      myfile_ << std::to_string(cloud_msg_time);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
          myfile_ << " " << t_matrix(i,j);
      myfile_ << "\n";
      myfile_.close();
    } 

     void process_input(std::string input_prefix) {
       pcl::PointCloud<pcl::PointXYZL> cloud;
       la3dm::point3f origin;
       la3dm::MarkerArrayPub m_pub(nh_, "/occupied_cells_vis_array", 0.1f);
       
       //for (int i = 0; i < input_pose_map_.size(); ++i) {
       for (auto it  = input_pose_map_.begin(); it != input_pose_map_.end(); ++it) {
         std::string input_name = input_prefix + "/" + std::to_string(it->first) + ".pcd";
         Eigen::Matrix4d t_matrix = it->second;
         pcl::PCLPointCloud2 cloud2;
         Eigen::Vector4f _origin;
         Eigen::Quaternionf orientaion;
         pcl::io::loadPCDFile(input_name, cloud2, _origin, orientaion);
         pcl::fromPCLPointCloud2(cloud2, cloud);
         pcl::transformPointCloud (cloud, cloud , t_matrix);
         origin.x() = t_matrix(0, 3);
         origin.y() = t_matrix(1, 3);
         origin.z() = t_matrix(2, 3);
         map_->insert_pointcloud(cloud, origin, 0.1, 20, 20);
         std::cout << "inserted " << input_name << std::endl;
       
         for (auto it = map_->begin_leaf(); it != map_->end_leaf(); it++) {
         la3dm::point3f p = it.get_loc();
         la3dm::SemanticOcTreeNode node = map_->search(p);
         if (node.get_state() == la3dm::State::OCCUPIED) {
           //la3dm::point3f p = it.get_loc();
           int semantics = node.get_semantics();
           m_pub.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), semantics);
         }
       }
       m_pub.publish();

       
       }
     }


     bool read_input_list(const std::string input_list_name) {
      if (std::ifstream(input_list_name)) {
        std::ifstream fInputs;
        fInputs.open(input_list_name.c_str());
        while (!fInputs.eof()) {
          std::string s;
          std::getline(fInputs, s);
          if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            long long time;
            ss >> time;
            Eigen::Matrix4d t_matrix = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 4; ++j)
                ss >> t_matrix(i, j);
            input_pose_map_[time] = t_matrix;
          }
        }
        fInputs.close();

        return true;
      } else {
        ROS_ERROR_STREAM("Cannot open evaluation list file " << input_list_name);
        return false;
      }
    } 

  private:
    ros::NodeHandle nh_;
    la3dm::SemanticBGKOctoMap* map_;
    tf::TransformListener listener_;
    ros::Publisher pc_publisher_;
    std::unordered_map<long long, Eigen::Matrix4d> input_pose_map_;
    int counter = 0;
    std::ofstream myfile_;



};



#endif // LA3DM_NCLT_UTIL_H
