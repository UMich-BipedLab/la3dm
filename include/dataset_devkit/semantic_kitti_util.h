#ifndef LA3DM_SEMANTIC_KITTI_UTIL_H
#define LA3DM_SEMANTIC_KITTI_UTIL_H

#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

class SemanticKITTIData {
  public:
    SemanticKITTIData(ros::NodeHandle& nh,
             double resolution, double block_depth,
             double sf2, double ell,
             int num_class, double free_thresh,
             double occupied_thresh, float var_thresh, 
	     double ds_resolution,
             double free_resolution, double max_range,
             std::string map_topic)
      : nh_(nh)
      , ds_resolution_(ds_resolution)
      , free_resolution_(free_resolution)
      , max_range_(max_range) {
        map_ = new la3dm::SemanticBGKOctoMap(resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, 0.001, 0.001);
        m_pub_ = new la3dm::MarkerArrayPub(nh_, map_topic, 0.1f);
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
            Eigen::Matrix4d t_matrix = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 4; ++j)
                ss >> t_matrix(i, j);
            lidar_poses_.push_back(t_matrix);
          }
        }
        fPoses.close();
        return true;
        } else {
         ROS_ERROR_STREAM("Cannot open evaluation list file " << lidar_pose_name);
         return false;
      }
    } 

    bool process_scans(std::string input_data_dir, std::string input_label_dir, int scan_num) {
      la3dm::point3f origin;
      for (int scan_id  = 0; scan_id < scan_num; ++scan_id) {
        char scan_id_c[256];
        sprintf(scan_id_c, "%06d", scan_id);
        std::string scan_name = input_data_dir + std::string(scan_id_c) + ".bin";
        std::string label_name = input_label_dir + std::string(scan_id_c) + ".label";
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = kitti2pcl(scan_name, label_name);
        Eigen::Matrix4d transform = lidar_poses_[scan_id];
        Eigen::Matrix4d calibration;
        calibration <<  -0.001857739385241, -0.999965951350955, -0.008039975204516, -0.004784029760483,
                        -0.006481465826011,  0.008051860151134, -0.999946608177406, -0.073374294642306,
                        0.999977309828677, -0.001805528627661, -0.006496203536139, -0.333996806443304,
       	                 0                ,  0                ,  0                ,  1.000000000000000;
        //calibration << 0.000427680238558, -0.999967248494602, -0.008084491683471, -0.011984599277133,
	//	      -0.007210626507497,  0.008081198471645, -0.999941316450383, -0.054039847297480,
	//	       0.999973864590328,  0.000485948581039, -0.007206933692422, -0.292196864868591,
	//	       0                ,  0                ,  0                ,  1.000000000000000;
	Eigen::Matrix4d new_transform = transform * calibration;
        pcl::transformPointCloud (*cloud, *cloud, new_transform);
        origin.x() = transform(0, 3);
        origin.y() = transform(1, 3);
        origin.z() = transform(2, 3);
        map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_);
        std::cout << "Inserted point cloud at " << scan_name << std::endl;
        //publish_map();
        for (int query_id = scan_id - 10; query_id >= 0 && query_id <= scan_id; ++query_id)
          query_scan(input_data_dir, query_id);
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

    void set_up_evaluation(const std::string gt_label_dir, const std::string evaluation_result_dir) {
      gt_label_dir_ = gt_label_dir;
      evaluation_result_dir_ = evaluation_result_dir;
    }

    void query_scan(std::string input_data_dir, int scan_id) {
      char scan_id_c[256];
      sprintf(scan_id_c, "%06d", scan_id);
      std::string scan_name = input_data_dir + std::string(scan_id_c) + ".bin";
      std::string gt_name = gt_label_dir_ + std::string(scan_id_c) + ".label";
      std::string result_name = evaluation_result_dir_ + std::string(scan_id_c) + ".txt";
      pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = kitti2pcl(scan_name, gt_name);
      Eigen::Matrix4d transform = lidar_poses_[scan_id];
      Eigen::Matrix4d calibration;
      calibration <<  -0.001857739385241, -0.999965951350955, -0.008039975204516, -0.004784029760483,
                      -0.006481465826011,  0.008051860151134, -0.999946608177406, -0.073374294642306,
                       0.999977309828677, -0.001805528627661, -0.006496203536139, -0.333996806443304,
                       0                ,  0                ,  0                ,  1.000000000000000;
      //calibration << 0.000427680238558, -0.999967248494602, -0.008084491683471, -0.011984599277133,
	//	      -0.007210626507497,  0.008081198471645, -0.999941316450383, -0.054039847297480,
	//	       0.999973864590328,  0.000485948581039, -0.007206933692422, -0.292196864868591,
	//	       0                ,  0                ,  0                ,  1.000000000000000;
      Eigen::Matrix4d new_transform = transform * calibration;
      pcl::transformPointCloud (*cloud, *cloud, new_transform);

      std::ofstream result_file;
      result_file.open(result_name);
      for (int i = 0; i < cloud->points.size(); ++i) {
        la3dm::SemanticOcTreeNode node = map_->search(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        int pred_label = 0;
	if (node.get_state() == la3dm::State::OCCUPIED)
	  pred_label = node.get_semantics();
        result_file << cloud->points[i].label << " " << pred_label << "\n";
      }
      result_file.close();
    }

  
  private:
    ros::NodeHandle nh_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;
    la3dm::SemanticBGKOctoMap* map_;
    la3dm::MarkerArrayPub* m_pub_;
    tf::TransformListener listener_;
    std::ofstream pose_file_;
    std::vector<Eigen::Matrix4d> lidar_poses_;
    std::string gt_label_dir_;
    std::string evaluation_result_dir_;

    int check_element_in_vector(const long long element, const std::vector<long long>& vec_check) {
      for (int i = 0; i < vec_check.size(); ++i)
        if (element == vec_check[i])
          return i;
      return -1;
    }

    pcl::PointCloud<pcl::PointXYZL>::Ptr kitti2pcl(std::string fn, std::string fn_label) {
      FILE* fp_label = std::fopen(fn_label.c_str(), "r");
      if (!fp_label) {
        std::perror("File opening failed");
      }
      std::fseek(fp_label, 0L, SEEK_END);
      std::rewind(fp_label);
      FILE* fp = std::fopen(fn.c_str(), "r");
      if (!fp) {
        std::perror("File opening failed");
      }
      std::fseek(fp, 0L, SEEK_END);
      size_t sz = std::ftell(fp);
      std::rewind(fp);
      int n_hits = sz / (sizeof(float) * 4);
      pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
      for (int i = 0; i < n_hits; i++) {
        pcl::PointXYZL point;
        float intensity;
        if (fread(&point.x, sizeof(float), 1, fp) == 0) break;
        if (fread(&point.y, sizeof(float), 1, fp) == 0) break;
        if (fread(&point.z, sizeof(float), 1, fp) == 0) break;
        if (fread(&intensity, sizeof(float), 1, fp) == 0) break;
        if (fread(&point.label, sizeof(float), 1, fp_label) == 0) break;
        pc->push_back(point);
      }
      std::fclose(fp);
      std::fclose(fp_label);
      return pc;
    }
};

#endif // LA3DM_SEMANTIC_KITTI_UTIL_H
