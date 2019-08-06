#include <string>
#include <iostream>
#include <ros/ros.h>
#include "semantic_bgkoctomap.h"
#include "markerarray_pub.h"

#include "nclt_util.h"


int main(int argc, char **argv) {
    ros::init(argc, argv, "semantic_bgkoctomap_nclt_node");
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
    double max_range = -1;
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
    nh.param<double>("max_range", max_range, max_range);
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
      "max_range: " << max_range << std::endl <<
      "num_class: " << num_class
      );

    
    ///////// Build Map /////////////////////
    //la3dm::SemanticBGKOctoMap* map = new la3dm::SemanticBGKOctoMap(resolution, block_depth, sf2, ell, 15, free_thresh, occupied_thresh, var_thresh, prior_A, prior_B);
    
    NCLTData<14> nclt_data(nh, resolution, block_depth, sf2, ell, 15, free_thresh, occupied_thresh, var_thresh);
    ros::Subscriber sub = nh.subscribe("/labeled_pointcloud", 100, &NCLTData<14>::PointCloudCallback, &nclt_data);

    
        
    ros::spin();
    return 0;
}
