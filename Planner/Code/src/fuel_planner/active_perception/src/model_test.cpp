#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Core>
#include <chrono>
#include <vector>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/common.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>

using namespace std;

int main()
{
  string tmp_input = "/home/albert/Network/PCN-PyTorch/results/house3K/partial/input_0_0.pcd";
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_p (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile<pcl::PointXYZ>(tmp_input, *map_p);

  string model_path = "/home/albert/UAV_Planning/fuel_airsim/surface_prediction/prediction_local.pt";
  torch::jit::script::Module module = torch::jit::load(model_path);
  module.to(at::kCUDA);

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_sample (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_demean (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr partial_sample_scale (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::RandomSample<pcl::PointXYZ> rs;
  rs.setInputCloud(map_p);
  rs.setSample(8192);
  rs.filter(*map_sample);

  Eigen::Vector4f map_center;
  pcl::compute3DCentroid(*map_sample, map_center);
  pcl::demeanPointCloud(*map_sample, map_center, *map_demean);

  double scale = 50.0;
  pcl::PointXYZ pt_scale;
  float tmp_cloud[1][8192][3];
  for (int j=0; j<8192; ++j)
    {
        tmp_cloud[0][j][0] = map_demean->points[j].x/scale;
        tmp_cloud[0][j][1] = map_demean->points[j].y/scale;
        tmp_cloud[0][j][2] = map_demean->points[j].z/scale;
        pt_scale.x = map_demean->points[j].x/scale;
        pt_scale.y = map_demean->points[j].y/scale;
        pt_scale.z = map_demean->points[j].z/scale;
        partial_sample_scale->points.push_back(pt_scale);
    }
  
  torch::Tensor array_tensor = torch::from_blob(tmp_cloud, { 1, 8192, 3 }).to(at::kCUDA);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(array_tensor);
  auto output = module.forward(inputs).toTuple();

  torch::Tensor out1 = output->elements()[0].toTensor().squeeze().to(at::kCPU);
  torch::Tensor out2 = output->elements()[1].toTensor().squeeze().to(at::kCPU);
  torch::Tensor out3 = output->elements()[2].toTensor().squeeze().to(at::kCPU);

  std::vector<float> v(out3.data_ptr<float>(), out3.data_ptr<float>() + out3.numel());
  pcl::PointCloud<pcl::PointXYZ>::Ptr pred (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointXYZ tmp_pt;
  for (int i=0; i<out3.size(0); ++i)
  {
    tmp_pt.x = v[3*i];
    tmp_pt.y = v[3*i+1];
    tmp_pt.z = v[3*i+2];
    pred->points.push_back(tmp_pt);
  }

  *pred = *pred + *partial_sample_scale;

  double angle_0 = 5.0*M_PI/180.0;
  double angle_best = 11.0*M_PI/180.0;
  double prx_const = 0.01;
  double ang = -pow(angle_0-angle_best, 2)/(2.0*pow(prx_const,2));
  double f_ang = exp(ang);
  cout << "angle:" << f_ang << endl;
  
  pcl::PCDWriter writer;
  writer.write("/home/albert/model_vis.pcd",*pred);
}