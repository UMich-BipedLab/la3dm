#ifndef LA3DM_NCLT_UTIL_H
#define LA3DM_NCLT_UTIL_H

#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#define START_TIME 1335704231918265
#define END_TIME 1335704347924201

template <unsigned int NUM_CLASS>
class NCLTData {
  public:
    NCLTData(ros::NodeHandle& nh,
             double resolution, double block_depth,
             double sf2, double ell,
             int num_class, double free_thresh,
             double occupied_thresh, float var_thresh, 
             double free_resolution, double max_range,
             std::string map_topic)
      : nh_(nh)
      , resolution_(resolution)
      , free_resolution_(free_resolution)
      , max_range_(max_range) {
        map_ = new la3dm::SemanticBGKOctoMap(resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, 0.001, 0.001);
        m_pub_ = new la3dm::MarkerArrayPub(nh_, map_topic, 0.1f);
      }

    // Data preprocess
    void PointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg) {
      long long cloud_msg_time = (long long)(round((double)cloud_msg->header.stamp.toNSec() / 1000.0) + 0.1);
      if (cloud_msg_time < START_TIME || cloud_msg_time > END_TIME)
        return;

      // Save pcd files
      pcl::PointCloud<pcl::PointXYZL> cloud;
      for (int i = 0; i < cloud_msg->points.size(); ++i) {
        pcl::PointXYZL pt;
        pt.x = cloud_msg->points[i].x;
        pt.y = cloud_msg->points[i].y;
        pt.z = cloud_msg->points[i].z;
        pt.label = cloud_msg->channels[0].values[i];
        cloud.push_back(pt);
      }

      // Fetch the tf transform and write to a file
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

      std::string cloud_name = std::to_string(cloud_msg_time) + ".pcd";
      pcl::io::savePCDFileASCII(cloud_name, cloud);
      
      Eigen::Affine3d t_eigen;
      tf::transformTFToEigen(transform, t_eigen);
      Eigen::Matrix4d t_matrix = t_eigen.matrix();
      pose_file_.open("LidarPoses.txt", std::ios::app);
      pose_file_ << std::to_string(cloud_msg_time);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
          pose_file_ << " " << t_matrix(i, j);
      pose_file_ << "\n";
      pose_file_.close();
    } 

    bool read_lidar_poses(const std::string lidar_pose_name) {
      if (std::ifstream(lidar_pose_name)) {
        std::ifstream fPoses;
        fPoses.open(lidar_pose_name.c_str());
        while (!fPoses.eof()) {
          std::string s;
          std::getline(fPoses, s);
          if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            long long time;
            ss >> time;
            Eigen::Matrix4d t_matrix = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 4; ++j)
                ss >> t_matrix(i, j);
            time_pose_map_[time] = t_matrix;
          }
        }
        fPoses.close();
        return true;
        } else {
         ROS_ERROR_STREAM("Cannot open evaluation list file " << lidar_pose_name);
         return false;
      }
    } 

    bool process_scans(std::string dir) {
      pcl::PointCloud<pcl::PointXYZL> cloud;
      la3dm::point3f origin;
      for (auto it  = time_pose_map_.begin(); it != time_pose_map_.end(); ++it) {
        std::string scan_name = dir + std::to_string(it->first) + ".pcd";
        Eigen::Matrix4d transform = it->second;

        if (pcl::io::loadPCDFile<pcl::PointXYZL>(scan_name, cloud) == -1) {
          ROS_ERROR_STREAM ("Couldn't read file " << scan_name);
          return 0;
        }
        pcl::transformPointCloud (cloud, cloud, transform);
        origin.x() = transform(0, 3);
        origin.y() = transform(1, 3);
        origin.z() = transform(2, 3);
        map_->insert_pointcloud(cloud, origin, resolution_, free_resolution_, max_range_);
        std::cout << "Inserted " << scan_name << std::endl;
        publish_map();
        query_scans(it->first);
      }
      return 1;
    }

    void publish_map() {
      for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          m_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), it.get_node().get_semantics());
        }
      }
      m_pub_->publish();
    }

    bool read_evaluation_list(const std::string evaluation_list_name) {
      if (std::ifstream(evaluation_list_name)) {
        std::ifstream fScans;
        fScans.open(evaluation_list_name.c_str());
        int counter = 0;
        while (!fScans.eof()) {
          std::string s;
          std::getline(fScans, s);
          if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            long long t;
            ss >> t;
            evaluation_list_.push_back(t);
            counter++;
          }
        }
        fScans.close();
        return true;
      } else {
        ROS_ERROR_STREAM("Cannot open evaluation list file " << evaluation_list_name);
        return false;
      }
    }

    void set_up_evaluation(const std::string evaluation_folder) {
      evaluation_folder_ = evaluation_folder;
    }

    void query_scans(long long current_time) {
      if (check_element_in_vector(current_time, evaluation_list_) < 0)
        return;
      for (int i = 0 ; i < evaluation_list_.size(); ++i) {
        long long query_time = evaluation_list_[i];
        if (query_time <= current_time)
          query_scan(query_time);
      }
    }

    void query_scan(long long time) {
      std::string gt_name = evaluation_folder_ + std::to_string(time) + ".pcd";
      std::string result_name = evaluation_folder_ + std::to_string(time) + ".txt";

      pcl::PointCloud<pcl::PointXYZL> cloud;
      pcl::io::loadPCDFile<pcl::PointXYZL> (gt_name, cloud);
      Eigen::Matrix4d transform = time_pose_map_[time];
      pcl::transformPointCloud (cloud, cloud, transform);

      std::ofstream result_file;
      result_file.open(result_name);
      for (int i = 0; i < cloud.points.size(); ++i) {
        la3dm::SemanticOcTreeNode node = map_->search(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
        if (node.get_state() == la3dm::State::OCCUPIED){
          int pred_label = node.get_semantics();
          result_file << cloud.points[i].label << " " << pred_label - 1  << "\n";
        }
      }
      result_file.close();
    }
  
  private:
    ros::NodeHandle nh_;
    double resolution_;
    double free_resolution_;
    double max_range_;
    la3dm::SemanticBGKOctoMap* map_;
    la3dm::MarkerArrayPub* m_pub_;
    tf::TransformListener listener_;
    std::ofstream pose_file_;
    std::string evaluation_folder_;

    std::map<long long, Eigen::Matrix4d> time_pose_map_;
    std::vector<long long> evaluation_list_;

    int check_element_in_vector(const long long element, const std::vector<long long>& vec_check) {
      for (int i = 0; i < vec_check.size(); ++i)
        if (element == vec_check[i])
          return i;
      return -1;
    }
};

#endif // LA3DM_NCLT_UTIL_H
