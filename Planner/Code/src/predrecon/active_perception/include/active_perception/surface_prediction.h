#ifndef _SURFACE_PREDICTION_H_
#define _SURFACE_PREDICTION_H_

#include <pcl/io/pcd_io.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <ros/ros.h>

using namespace std;

namespace fast_planner
{
  class EDTEnvironment;
class SurfacePred
{
public:
  SurfacePred();
  ~SurfacePred();

  void init(ros::NodeHandle& nh, const shared_ptr<EDTEnvironment>& edt);
  void warmup();
  void inference(pcl::PointCloud<pcl::PointXYZ>::Ptr& partial, pcl::PointCloud<pcl::PointXYZ>::Ptr& pred, const double& scale);

private:
  shared_ptr<EDTEnvironment> sp_env_;
  string model_path;
  torch::jit::script::Module module;
  torch::Tensor warmup_tensor;
  pcl::PointCloud<pcl::PointXYZ>::Ptr partial_sample;
  pcl::PointCloud<pcl::PointXYZ>::Ptr partial_qualified;
  pcl::PointCloud<pcl::PointXYZ>::Ptr partial_unqualified;
  pcl::PointCloud<pcl::PointXYZ>::Ptr partial_sample_scale;
  pcl::PointCloud<pcl::PointXYZ>::Ptr partial_remainder;
};
}

#endif