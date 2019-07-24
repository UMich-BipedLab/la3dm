#ifndef LA3DM_KITTI_UTIL_H
#define LA3DM_KITTI_UTIL_H

#include <fstream>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

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
   }

   ~KITTIData() {}

 
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
   
  void process_depth_img(const cv::Mat& rgb_img, const cv::Mat& depth_img, pcl::PointCloud<pcl::PointXYZL>& cloud) {
   
    int pix_label;
    float pix_depth;

    //pcl::PointCloud<pcl::PointXYZL> cloud;
    pcl::PointXYZL pt;
    for (int32_t i = 0; i < im_width_ * im_height_; ++i) {
      int ux = i % im_width_;
      int uy = i / im_width_;
      pix_depth = (float) depth_img.at<uint16_t>(uy, ux);
      pix_depth = pix_depth / depth_scaling_;

      frame_label_prob_.row(i).maxCoeff(&pix_label);

      if (pix_depth > 0.1) {
        pt.x = (ux - cx_) * (1.0 / fx_) * pix_depth;
        pt.y = (uy - cy_) * (1.0 / fy_) * pix_depth;
        pt.z = pix_depth;
        //pt.r = rgb_img.at<cv::Vec3b>(uy,ux)[2];
        //pt.g = rgb_img.at<cv::Vec3b>(uy,ux)[1];
        //pt.b = rgb_img.at<cv::Vec3b>(uy,ux)[0];
        pt.label = pix_label + 1;  // NOTE: valid label starts from 0
        cloud.points.push_back(pt);
      }
    }
    cloud.width = (uint32_t) cloud.points.size();
    cloud.height = 1;
    pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
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
};

#endif // LA3DM_KITTI_UTIL_H
