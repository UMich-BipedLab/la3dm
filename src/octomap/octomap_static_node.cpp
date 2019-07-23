#include <string>
#include <iostream>
#include <ros/ros.h>
#include <octomap/octomap.h>
#include <octomap/SemanticOcTree.h>
#include <octomap_ros/conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <markerarray_pub.h>
#include <pcl_ros/point_cloud.h>

using std::string;

void load_pcd(std::string filename, octomap::point3d &origin, octomap::Pointcloud &scan) {
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f _origin;
    Eigen::Quaternionf orientaion;
    pcl::io::loadPCDFile(filename, cloud2, _origin, orientaion);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(cloud2, cloud);

    for (auto it = cloud.begin(); it != cloud.end(); ++it) {
        scan.push_back(it->x + _origin[0], 
                       it->y + _origin[1], 
                       it->z + _origin[2]);
    }
    origin.x() = _origin[0];
    origin.y() = _origin[1];
    origin.z() = _origin[2];
}

void insert_semantics(octomap::Pointcloud &scan, octomap::SemanticOcTree &soc) {
  for (auto it = scan.begin(); it != scan.end(); ++it) {
    octomap::SemanticOcTreeNode *node = soc.updateNode(it->x, it->y, it->z, true);
    if (node) {
      //soc.averageNodeSemantics(node, label_dist);
    }
  }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "octomap_static_node");
    ros::NodeHandle nh("~");

    std::string dir;
    std::string prefix;
    int scan_num = 0;
    int class_num = 3;
    std::string map_topic("/occupied_cells_vis_array");
    double max_range = -1;
    double resolution = 0.1;
    double min_z = 0;
    double max_z = 0;

    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("prefix", prefix, prefix);
    nh.param<std::string>("topic", map_topic, map_topic);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<int>("calss_num", class_num, class_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
                    "dir: " << dir << std::endl <<
                    "prefix: " << prefix << std::endl <<
                    "topic: " << map_topic << std::endl <<
                    "scan_num: " << scan_num << std::endl <<
                    "class_num: " << class_num << std::endl <<
                    "max_range: " << max_range << std::endl <<
                    "resolution: " << resolution << std::endl <<
                    "min_z: " << min_z << std::endl <<
                    "max_z: " << max_z << std::endl
    );

    //octomap::OcTree oc(resolution);
    std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
    label2color[1] = std::make_tuple(255, 0, 0);
    label2color[2] = std::make_tuple(0, 255, 0);
    label2color[3] = std::make_tuple(0, 0, 255);
    label2color[4] = std::make_tuple(255, 255, 0);

    octomap::SemanticOcTree soc(resolution, class_num, label2color);

    ros::Time start = ros::Time::now();
    for (int scan_id = 1; scan_id <= scan_num; ++scan_id) {
        octomap::Pointcloud scan;
        octomap::point3d sensor_origin;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        load_pcd(filename, sensor_origin, scan);

        //octomap::SemanticOcTreeNode* node = soc.updateNode()
        //soc.insertPointCloud(scan, sensor_origin, max_range);
        insert_semantics(scan, soc);
        ROS_INFO_STREAM("Scan " << scan_id << " done");
    }
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");


    ///////// Publish Map /////////////////////
    la3dm::MarkerArrayPub m_pub(nh, map_topic, resolution);
    if (min_z == max_z) {
        double min_x, min_y, max_x, max_y;
        soc.getMetricMin(min_x, min_y, min_z);
        soc.getMetricMax(max_x, max_y, max_z);
    }
    for (auto it = soc.begin_leafs(); it != soc.end_leafs(); ++it) {
        if (soc.isNodeOccupied(*it)) {
            m_pub.insert_point3d(it.getX(), it.getY(), it.getZ(), min_z, max_z, resolution);
        }
    }

    m_pub.publish();
    ros::spin();

    return 0;
}
