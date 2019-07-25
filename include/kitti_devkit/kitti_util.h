#ifndef LA3DM_KITTI_UTIL_H
#define LA3DM_KITTI_UTIL_H

#include <fstream>

#include <Eigen/Dense>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/common/transforms.h>

#include "point3f.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

class KITTIData {
  public:
    KITTIData(int im_width, int im_height,
	      float fx, float fy,
              float cx, float cy,
              float depth_scaling, int num_class) 
      : im_width_(im_width)
      , im_height_(im_height)
      , fx_(fx)
      , fy_(fy)
      , cx_(cx)
      , cy_(cy)
      , depth_scaling_(depth_scaling)
      , num_class_(num_class) {
        frame_label_prob_.resize(im_width_*im_height_, num_class_ - 1);  // NOTE: valid label starts from 0
        label_to_color_.resize(num_class_ - 1, 3);
        label_to_color_ << 128,	0,	0,     // 1. building  // our 11 class  //TODO this conversion should be the same with all other places
                           128,	128,	128,// 2. sky
                           128,	64,	128,  // 3. road
                           128,	128,	0,    // 4. vegetation
                           0,	0,	192,  // 5. sidewalk
                           64,	0,	128,  // 6. car
                           64,	64,	0,    // 7. pedestrian
                           0,	128,	192,  // 8 cyclist
                           192,	128,	128,  // 9. signate
                           64,	64,	128,  // 10. fence
                           192,	192,	128;  // 11. pole
        init_trans_to_ground_ << 1, 0, 0, 0,
		                             0, 0, 1, 0,
		                             0,-1, 0, 1,
		                             0, 0, 0, 1;
      }

   ~KITTIData() {}

   bool read_all_poses(const std::string trajectory_file, const int scan_num) {
     int total_img_number = scan_num + 1;  // NOTE: scan id starts from 0
     if (std::ifstream(trajectory_file)) {
       all_poses_.resize(total_img_number, 12);
       std::ifstream fPoses;
       fPoses.open(trajectory_file.c_str());
       int counter=0;
       while (!fPoses.eof()) {
         std::string s;
         std::getline(fPoses,s);
         if (!s.empty()) {
           std::stringstream ss;
           ss << s;
           float t;
           for (int i=0;i<12;i++) {
             ss >> t;
             all_poses_(counter,i)=t;
          }
          counter++;
          if (counter>=total_img_number)
            break;
         }
      }
      fPoses.close();
      return true;
      } else
        return false;
   }

  bool read_label_prob_bin(const std::string label_bin) {// assumed mat size correct 
    if (std::ifstream(label_bin)) {
      std::ifstream fLables(label_bin.c_str(), std::ios::in|std::ios::binary);
      if (fLables.is_open()) {
          int mat_byte_size = sizeof(float)*frame_label_prob_.rows()*frame_label_prob_.cols(); // byte number, make sure size is correct, or can use tellg to get the file size
          float *mat_field = frame_label_prob_.data();
          fLables.read((char*)mat_field, mat_byte_size);
          fLables.close(); 
      } else {
        ROS_ERROR_STREAM("Cannot open bianry label file " << label_bin);
        return false;
      }
      return true;
    } else
      return false;    
  } 
   
  void process_depth_img(const int scan_id, const cv::Mat& rgb_img, const cv::Mat& depth_img,
                         pcl::PointCloud<pcl::PointXYZL>& cloud, la3dm::point3f& origin) {
 
    Eigen::VectorXf curr_posevec = all_poses_.row(scan_id);
    Eigen::MatrixXf curr_pose = Eigen::Map<MatrixXf_row>(curr_posevec.data(), 3, 4);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0,0,3,4) = curr_pose;
    Eigen::Matrix4f new_transform = init_trans_to_ground_ * transform;
    //new_transform(2,3) = 1.0;
   
    //pcl::PointCloud<pcl::PointXYZL> cloud;
    for (int32_t i = 0; i < im_width_ * im_height_; ++i) {
      int ux = i % im_width_;
      int uy = i / im_width_;
      float pix_depth = (float) depth_img.at<uint16_t>(uy, ux);
      pix_depth = pix_depth / depth_scaling_;

      int pix_label;
      frame_label_prob_.row(i).maxCoeff(&pix_label);
      if (pix_label == 1)  // NOTE: don't project sky label
        continue;

      if (pix_depth > 20.0)
        continue;

      if (pix_depth > 0.1) {
        pcl::PointXYZL pt;
        pt.x = (ux - cx_) * (1.0 / fx_) * pix_depth;
        pt.y = (uy - cy_) * (1.0 / fy_) * pix_depth;
        pt.z = pix_depth;
        //pt.r = rgb_img.at<cv::Vec3b>(uy,ux)[2];
        //pt.g = rgb_img.at<cv::Vec3b>(uy,ux)[1];
        //pt.b = rgb_img.at<cv::Vec3b>(uy,ux)[0];
        pt.label = pix_label + 1;  // NOTE: valid label starts from 0
        
        //Eigen::Vector4f global_pt4 = transform * Eigen::Vector4f(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4f global_pt4 = new_transform * Eigen::Vector4f(pt.x, pt.y, pt.z, 1);
        Eigen::Vector3f global_pt3 = global_pt4.head(3) / global_pt4(3);
        pt.x = global_pt3(0);
        pt.y = global_pt3(1);
        pt.z = global_pt3(2);
        
        cloud.points.push_back(pt);
      }
    }
    cloud.width = (uint32_t) cloud.points.size();
    cloud.height = 1;

    // Transform point cloud
    /*Eigen::VectorXf curr_posevec = all_poses_.row(scan_id);
    Eigen::MatrixXf curr_pose = Eigen::Map<MatrixXf_row>(curr_posevec.data(), 3, 4);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0,0,3,4) = curr_pose;
    std::cout << transform << std::endl;
    //pcl::transformPointCloud (cloud, cloud, transform);
    Eigen::Matrix4f new_transform = init_trans_to_ground_ * transform;
    pcl::transformPointCloud (cloud, cloud, new_transform);
    //pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);*/
    
    // Set sensor origin
    origin.x() = new_transform(0, 3);
    origin.y() = new_transform(1, 3);
    origin.z() = new_transform(2, 3);
  }


  void reproject_to_images(const int scan_id, const cv::Mat& depth_img, la3dm::SemanticBGKOctoMap& map) {
 
    Eigen::VectorXf curr_posevec = all_poses_.row(scan_id);
    Eigen::MatrixXf curr_pose = Eigen::Map<MatrixXf_row>(curr_posevec.data(), 3, 4);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0,0,3,4) = curr_pose;
    Eigen::Matrix4f new_transform = init_trans_to_ground_ * transform;
    new_transform(2,3) = 1.0;
   
    cv::Mat reproj_label_maps = cv::Mat(cv::Size(im_width_, im_height_), CV_8UC1,cv::Scalar(255));
    cv::Mat reproj_label_colors = cv::Mat(cv::Size(im_width_, im_height_), CV_8UC3,cv::Scalar(200,200,200));
    for (int32_t i = 0; i < im_width_ * im_height_; ++i) {
      int ux = i % im_width_;
      int uy = i / im_width_;
      if (ux >= im_width_-1 ||  ux <= 0 || uy >= im_height_-1 ||  uy<= 0)
        continue;

      float pix_depth = (float) depth_img.at<uint16_t>(uy, ux);
      pix_depth = pix_depth / depth_scaling_;

      int pix_label;
      frame_label_prob_.row(i).maxCoeff(&pix_label);

      if (pix_depth > 0.1) {
        pcl::PointXYZL pt;
        pt.x = (ux - cx_) * (1.0 / fx_) * pix_depth;
        pt.y = (uy - cy_) * (1.0 / fy_) * pix_depth;
        pt.z = pix_depth;

        // Transform point
        //Eigen::Vector4f global_pt4 = transform * Eigen::Vector4f(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4f global_pt4 = new_transform * Eigen::Vector4f(pt.x, pt.y, pt.z, 1);
        Eigen::Vector3f global_pt3 = global_pt4.head(3) / global_pt4(3);
        la3dm::SemanticOcTreeNode node = map.search(global_pt3(0), global_pt3(1), global_pt3(2));
       
        if (node.get_state() == la3dm::State::OCCUPIED){
          int pix_label = node.get_semantics();
          reproj_label_maps.at<uint8_t>(uy, ux) = (uint8_t) pix_label;
          reproj_label_colors.at<cv::Vec3b>(uy, ux)[0] = (uint8_t) label_to_color_(pix_label - 1, 2);
          reproj_label_colors.at<cv::Vec3b>(uy, ux)[1] = (uint8_t) label_to_color_(pix_label - 1, 1);
          reproj_label_colors.at<cv::Vec3b>(uy, ux)[2] = (uint8_t) label_to_color_(pix_label - 1, 0);
        }
      }
    }
    cv::imwrite( "/home/ganlu/reproj_label_maps.png", reproj_label_maps);
    cv::imwrite( "/home/ganlu/reproj_label_colors.png", reproj_label_colors);
  }

  private:
    int im_width_;
    int im_height_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    float depth_scaling_;
    int num_class_;
    MatrixXf_row frame_label_prob_;
    Eigen::MatrixXi label_to_color_;
    Eigen::Matrix4f init_trans_to_ground_; 
    Eigen::MatrixXf all_poses_;
};

#endif // LA3DM_KITTI_UTIL_H
