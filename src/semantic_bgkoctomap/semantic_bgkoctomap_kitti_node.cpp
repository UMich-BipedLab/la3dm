#include <string>
#include <iostream>
#include <ros/ros.h>
#include "semantic_bgkoctomap.h"
#include "markerarray_pub.h"

#include "kitti_util.h"


int main(int argc, char **argv) {
    ros::init(argc, argv, "semantic_bgkoctomap_kitti_node");
    ros::NodeHandle nh("~");

    std::string map_topic("/occupied_cells_vis_array");
    double resolution = 0.1;
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    double free_resolution = 0.5;
    double ds_resolution = 0.1;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    double min_z = 0;
    double max_z = 0;
    bool original_size = false;
    float var_thresh = 1.0f;
    float prior_A = 1.0f;
    float prior_B = 1.0f;
    
    // KITTI 05
    std::string dir;
    std::string left_img_prefix;
    std::string depth_img_prefix;
    std::string label_bin_prefix;
    int scan_num = 0;
    double max_range = -1;
    int image_width = 1226;
    int image_height = 370;
    float focal_x = 707.0912;
    float focal_y = 707.0912;
    float center_x = 601.8873;
    float center_y = 183.1104;
    float depth_scaling = 2000;
    int num_class = 12;

    nh.param<std::string>("topic", map_topic, map_topic);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);
    nh.param<bool>("original_size", original_size, original_size);
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<float>("prior_A", prior_A, prior_A);
    nh.param<float>("prior_B", prior_B, prior_B);
    
    // KITTI
    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("left_img_prefix", left_img_prefix, left_img_prefix);
    nh.param<std::string>("depth_img_prefix", depth_img_prefix, depth_img_prefix);
    nh.param<std::string>("label_bin_prefix", label_bin_prefix, label_bin_prefix);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<int>("image_width", image_width, image_width);
    nh.param<int>("image_height", image_height, image_height);
    nh.param<float>("focal_x", focal_x, focal_x);
    nh.param<float>("focal_y", focal_y, focal_y);
    nh.param<float>("center_x", center_x, center_x);
    nh.param<float>("center_y", center_y, center_y);
    nh.param<float>("depth_scaling", depth_scaling, depth_scaling);
    nh.param<int>("num_class", num_class, num_class);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
	    "topic: " << map_topic << std::endl <<
      "resolution: " << resolution << std::endl <<
      "block_depth: " << block_depth << std::endl <<
      "sf2: " << sf2 << std::endl <<
      "ell: " << ell << std::endl <<
      "free_resolution: " << free_resolution << std::endl <<
      "ds_resolution: " << ds_resolution << std::endl <<
      "free_thresh: " << free_thresh << std::endl <<
      "occupied_thresh: " << occupied_thresh << std::endl <<
      "min_z: " << min_z << std::endl <<
      "max_z: " << max_z << std::endl <<
      "original_size: " << original_size << std::endl <<
      "var_thresh: " << var_thresh << std::endl <<
      "prior_A: " << prior_A << std::endl <<
      "prior_B: " << prior_B << std::endl <<

      "KITTI:" << std::endl <<
      "dir: " << dir << std::endl <<
      "left_img_prefix: " << left_img_prefix << std::endl <<
      "depth_img_prefix: " << depth_img_prefix << std::endl <<
      "label_bin_prefix: " << label_bin_prefix << std::endl <<
      "scan_sum: " << scan_num << std::endl <<
      "max_range: " << max_range << std::endl <<
      "image_width: " << image_width << std::endl <<
      "image_height: " << image_height << std::endl <<
      "focal_x: " << focal_x << std::endl <<
      "focal_y: " << focal_y << std::endl <<
      "center_x: " << center_x << std::endl <<
      "center_y: " << center_y << std::endl <<
      "depth_scaling: " << depth_scaling << std::endl <<
      "num_class: " << num_class
      );


    KITTIData kitti_data(image_width, image_height, focal_x, focal_y, center_x, center_y, depth_scaling, num_class);
    kitti_data.read_all_poses(dir + "/" + "CameraTrajectory.txt", scan_num);
    la3dm::SemanticBGKOctoMap map(resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, prior_A, prior_B);
    la3dm::MarkerArrayPub m_pub(nh, map_topic, 0.1f);

    ///////// Build Map /////////////////////
    ros::Time start = ros::Time::now();
    for (int scan_id = 0; scan_id <= scan_num; ++scan_id) {
      la3dm::PCLPointCloud cloud;
      la3dm::point3f origin;
        
      char scan_id_c[256];
      sprintf(scan_id_c, "%06d", scan_id);  // format into 6 digit
      std::string scan_id_s(scan_id_c);
      std::string left_img_name(dir + "/" + left_img_prefix + "/" + scan_id_s + ".png");
      std::string depth_img_name(dir + "/" + depth_img_prefix + "/" + scan_id_s + ".png");
      std::string label_bin_name(dir + "/" + label_bin_prefix + "/" + scan_id_s + ".bin");

    	cv::Mat left_img = cv::imread(left_img_name, 1);
    	cv::Mat depth_img = cv::imread(depth_img_name, CV_LOAD_IMAGE_ANYDEPTH);
    	kitti_data.read_label_prob_bin(label_bin_name);
      kitti_data.process_depth_img(scan_id, left_img, depth_img,cloud, origin);
      std::cout << "Process depth img done." << std::endl;
      
      map.insert_pointcloud(cloud, origin, resolution, free_resolution, max_range);
      ROS_INFO_STREAM("Scan " << scan_id << " done");
     
      kitti_data.reproject_images(scan_id, map); 
     // cv::Mat depth_img_0 = cv::imread("/home/ganlu/la3dm_ws/src/semantic_3d_mapping/grid_sensor/data_kitti_15/depth_img/000000.png", CV_LOAD_IMAGE_ANYDEPTH);
      //kitti_data.reproject_to_images(0, depth_img_0, map);
      std::cout << "Reprojection done." << std::endl;
      
      ///////// Publish Map /////////////////////
      /*for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
          la3dm::point3f p = it.get_loc();
          int semantics = it.get_node().get_semantics();
          m_pub.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), semantics);
        }
      }
      m_pub.publish();
      std::cout << "Publish done." << std::endl;*/
    }
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");
        
    ros::spin();

    return 0;
}
