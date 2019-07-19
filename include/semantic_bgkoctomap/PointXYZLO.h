#ifndef LA3DM_POINTXYZLO_H
#define LA3DM_POINTXYZLO_H

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>

namespace pcl {
  struct PointXYZLO
  {
    PCL_ADD_POINT4D;
    int   label;
    int   object;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
  } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

}

#endif // LA3DM_POINTXYZLO_H
