#include <string>
#include <iostream>
#include <ros/ros.h>
#include "semantic_bgkoctomap.h"
#include "markerarray_pub.h"

#include "kitti_util.h"

void load_pcd(std::string filename, la3dm::point3f &origin, la3dm::PCLPointCloud &cloud) {
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f _origin;
    Eigen::Quaternionf orientaion;
    pcl::io::loadPCDFile(filename, cloud2, _origin, orientaion);
    pcl::fromPCLPointCloud2(cloud2, cloud);
    for (size_t i = 0; i < cloud.points.size (); ++i) {
      cloud.points[i].x = cloud.points[i].x + _origin[0];
      cloud.points[i].y = cloud.points[i].y + _origin[1];
      cloud.points[i].z = cloud.points[i].z + _origin[2];
    }
    origin.x() = _origin[0];
    origin.y() = _origin[1];
    origin.z() = _origin[2];
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "semantic_bgkoctomap_static_node");
    ros::NodeHandle nh("~");

    std::string dir;
    std::string prefix;
    int scan_num = 0;
    std::string map_topic("/occupied_cells_vis_array");
    double max_range = -1;
    double resolution = 0.1;
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    int nc = 3;
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

    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("prefix", prefix, prefix);
    nh.param<std::string>("topic", map_topic, map_topic);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<int>("nc", nc, nc);
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

    ROS_INFO_STREAM("Parameters:" << std::endl <<
            "dir: " << dir << std::endl <<
            "prefix: " << prefix << std::endl <<
            "topic: " << map_topic << std::endl <<
            "scan_sum: " << scan_num << std::endl <<
            "max_range: " << max_range << std::endl <<
            "resolution: " << resolution << std::endl <<
            "block_depth: " << block_depth << std::endl <<
            "sf2: " << sf2 << std::endl <<
            "ell: " << ell << std::endl <<
            "nc: " << nc << std::endl <<
            "free_resolution: " << free_resolution << std::endl <<
            "ds_resolution: " << ds_resolution << std::endl <<
            "free_thresh: " << free_thresh << std::endl <<
            "occupied_thresh: " << occupied_thresh << std::endl <<
            "min_z: " << min_z << std::endl <<
            "max_z: " << max_z << std::endl <<
            "original_size: " << original_size << std::endl <<
            "var_thresh: " << var_thresh << std::endl <<
            "prior_A: " << prior_A << std::endl <<
            "prior_B: " << prior_B
            );


    // Process KITTI Data
    float depth_scaling = 2000;
    Eigen::Matrix3f calibration_mat;
    calibration_mat << 707.0912, 0, 601.8873, // kitti sequence 5
		                   0, 707.0912, 183.1104,
			                 0,   0,   1;
    KITTIData kitti_data(depth_scaling, calibration_mat);
    cv::Mat rgb_img = cv::imread("/home/ganlu/la3dm_ws/src/semantic_3d_mapping/grid_sensor/data_kitti/rgb_img/000000.png", 1);
    cv::Mat depth_img = cv::imread("/home/ganlu/la3dm_ws/src/semantic_3d_mapping/grid_sensor/data_kitti/depth_img/000000.png", CV_LOAD_IMAGE_ANYDEPTH);
    MatrixXf_row frame_label_prob;
    frame_label_prob.resize(1226*370, 11); 
    kitti_data.read_label_prob_bin("/home/ganlu/la3dm_ws/src/semantic_3d_mapping/grid_sensor/data_kitti/label_binary/000000.bin", frame_label_prob);


    la3dm::SemanticBGKOctoMap map(resolution, block_depth, sf2, ell, nc, free_thresh, occupied_thresh, var_thresh, prior_A, prior_B);

    ros::Time start = ros::Time::now();
    for (int scan_id = 1; scan_id <= 1; ++scan_id) {
        la3dm::PCLPointCloud cloud;
        la3dm::point3f origin;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        
        kitti_data.process_depth_img(rgb_img, depth_img, frame_label_prob, cloud);
        origin.x() = 0;
        origin.y() = 0;
        origin.z() = 0;
        
        //load_pcd(filename, origin, cloud);

        map.insert_pointcloud(cloud, origin, resolution, free_resolution, 100);
        ROS_INFO_STREAM("Scan " << scan_id << " done");
    }
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");

    ///////// Compute Frontiers /////////////////////
    // ROS_INFO_STREAM("Computing frontiers");
    // la3dm::MarkerArrayPub f_pub(nh, "frontier_map", resolution);
    // for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
    //     la3dm::point3f p = it.get_loc();
    //     if (p.z() > 1.0 || p.z() < 0.3)
    //         continue;


    //     if (it.get_node().get_var() > 0.02 &&
    //         it.get_node().get_prob() < 0.3) {
    //         f_pub.insert_point3d(p.x(), p.y(), p.z());
    //     }
    // }
    // f_pub.publish();

    //////// Test Raytracing //////////////////
    /*la3dm::MarkerArrayPub ray_pub(nh, "/ray", resolution);
    la3dm::SemanticBGKOctoMap::RayCaster ray(&map, la3dm::point3f(1, 1, 0.3), la3dm::point3f(6, 7, 8));
    while (!ray.end()) {
        la3dm::point3f p;
        la3dm::SemanticOcTreeNode node;
        la3dm::BlockHashKey block_key;
        la3dm::OcTreeHashKey node_key;
        if (ray.next(p, node, block_key, node_key)) {
            ray_pub.insert_point3d(p.x(), p.y(), p.z());
        }
    }
    ray_pub.publish();*/

    ///////// Publish Map /////////////////////
    la3dm::MarkerArrayPub m_pub(nh, map_topic, 0.1f);
    if (min_z == max_z) {
        la3dm::point3f lim_min, lim_max;
        map.get_bbox(lim_min, lim_max);
        min_z = lim_min.z();
        max_z = lim_max.z();
    }

    // Find max and min variance
    float max_var = std::numeric_limits<float>::min();
    float min_var = std::numeric_limits<float>::max(); 
    for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
    	 if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
	     if (original_size) {
	     	int semantics = it.get_node().get_semantics();
		std::vector<float> vars = it.get_node().get_vars();
		if (vars[semantics] > max_var)
		  max_var = vars[semantics];
		if (vars[semantics] < min_var)
		  min_var = vars[semantics];
	     }
	 }
    }
    std::cout << "max_var: " << max_var << std::endl;
    std::cout << "min_var: " << min_var << std::endl;


    for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
        if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
            if (original_size) {
                la3dm::point3f p = it.get_loc();
                int semantics = it.get_node().get_semantics();
		            std::vector<float> vars = it.get_node().get_vars();
                m_pub.insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size(), semantics);
		            //std::cout << vars[semantics] << std::endl;
		            //m_pub.insert_point3d_var(p.x(), p.y(), p.z(), min_var, std::min(var_thresh, max_var), it.get_size(), vars[semantics]);
              } /*else {
                auto pruned = it.get_pruned_locs();
                for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    m_pub.insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map.get_resolution());
            }*/
        }
    }

    m_pub.publish();
    ros::spin();

    return 0;
}
