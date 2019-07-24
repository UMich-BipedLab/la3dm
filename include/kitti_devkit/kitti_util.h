#ifndef LA3DM_KITTI_UTIL_H
#define LA3DM_KITTI_UTIL_H

#include <fstream>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

class KITTIData {
  public:
   KITTIData(float depth_scaling, const Eigen::Matrix3f& calibration_mat) : depth_scaling_(depth_scaling) {
     set_up_calibration(calibration_mat);
   }

   ~KITTIData() {}

 
  bool read_label_prob_bin(const std::string label_bin, MatrixXf_row& frame_label_prob) {// assumed mat size correct 
    if (std::ifstream(label_bin)) {
      std::ifstream fLables(label_bin.c_str(), std::ios::in|std::ios::binary);
      if (fLables.is_open()) {
          int mat_byte_size=sizeof(float)*frame_label_prob.rows()*frame_label_prob.cols(); // byte number, make sure size is correct, or can use tellg to get the file size
          float *mat_field=frame_label_prob.data();
          fLables.read((char*)mat_field,mat_byte_size);
          fLables.close(); 
      } else {
        ROS_ERROR_STREAM("Cannot open bianry label file "<<label_bin);
        return false;
      }
      return true;
    } else
      return false;    
  } 
   
  void process_depth_img(const cv::Mat& rgb_img, const cv::Mat& depth_img, const MatrixXf_row& frame_label_prob, pcl::PointCloud<pcl::PointXYZL>& cloud) {
   
    int im_width = depth_img.cols;
    int im_height = depth_img.rows;

    int pix_label;
    float pix_depth;

    //pcl::PointCloud<pcl::PointXYZL> cloud;
    pcl::PointXYZL pt;
    for (int32_t i = 0; i < im_width * im_height; ++i) {
      int ux = i % im_width;
      int uy = i / im_width;
      pix_depth = (float) depth_img.at<uint16_t>(uy, ux);
      pix_depth = pix_depth / depth_scaling_;

      frame_label_prob.row(i).maxCoeff(&pix_label);

      if (pix_depth > 0.1) {
        pt.x = (ux - center_x_) * fx_inv_ * pix_depth;
        pt.y = (uy - center_y_) * fy_inv_ * pix_depth;
        pt.z = pix_depth;
        //pt.r = rgb_img.at<cv::Vec3b>(uy,ux)[2];
        //pt.g = rgb_img.at<cv::Vec3b>(uy,ux)[1];
        //pt.b = rgb_img.at<cv::Vec3b>(uy,ux)[0];
        pt.label = pix_label + 1;  // Note: valid label starts from 0
        cloud.points.push_back(pt);
      }
    }
    cloud.width = (uint32_t) cloud.points.size();
    cloud.height = 1;
    pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
  }

  private:
    float center_x_;
    float center_y_;
    float fx_inv_;
    float fy_inv_;
    float depth_scaling_ = 1000;

    void set_up_calibration(const Eigen::Matrix3f& calibration_mat) {
      center_x_ = calibration_mat(0, 2);  // cx
      center_y_ = calibration_mat(1, 2);  // cy
      fx_inv_ = 1.0 / calibration_mat(0, 0);  // 1/fx
      fy_inv_ = 1.0 / calibration_mat(1, 1);  // 1/fy
    }


};

#endif // LA3DM_KITTI_UTIL_H
