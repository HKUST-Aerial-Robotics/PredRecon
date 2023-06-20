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
#include <active_perception/surface_prediction.h>
#include <pcl/common/common.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>

using namespace std;
using namespace torch;

namespace fast_planner
{
SurfacePred::SurfacePred(){
}

SurfacePred::~SurfacePred(){
}

void SurfacePred::init(ros::NodeHandle& nh, const shared_ptr<EDTEnvironment>& edt)
{
this->sp_env_ = edt;
if (torch::cuda::is_available())
    cout << "CUDA ready!" << endl;

nh.param("surf_pred/model_", model_path, string("null"));
module = torch::jit::load(model_path);
module.to(at::kCUDA);
cout << "Model loaded!" << endl;

}

void SurfacePred::warmup()
{
for (int i=0; i<20; ++i)
{
    warmup_tensor = torch::ones({ 1, 8192, 3 }).to(at::kCUDA);
    std::vector<torch::jit::IValue> inputs_warm;
    inputs_warm.push_back(warmup_tensor);
    auto output_warm = module.forward(inputs_warm).toTuple();
    // Sync device speed up data transfer
    cudaDeviceSynchronize();
}
}

void SurfacePred::inference(pcl::PointCloud<pcl::PointXYZ>::Ptr& partial, pcl::PointCloud<pcl::PointXYZ>::Ptr& pred, const double& scale)
{
partial_sample.reset(new pcl::PointCloud<pcl::PointXYZ>);
partial_sample_scale.reset(new pcl::PointCloud<pcl::PointXYZ>);
int input_size = partial->points.size();
// Input current map
if (input_size < 8192)
{
  partial_remainder.reset(new pcl::PointCloud<pcl::PointXYZ>);
  int num = 8192/input_size;
  int remainder = 8192 - num*input_size;
  pcl::PointXYZ pt;
  for (int i=0; i<num; ++i)
    for (int j=0; j<input_size; ++j)
    {
      pt.x = partial->points[j].x;
      pt.y = partial->points[j].y;
      pt.z = partial->points[j].z;
      partial_sample->points.push_back(pt);
    }
  
  pcl::RandomSample<pcl::PointXYZ> rs_remain;
  rs_remain.setInputCloud(partial);
  rs_remain.setSample(remainder);
  rs_remain.filter(*partial_remainder);

  *partial_sample = *partial_sample + *partial_remainder;
}
else
{
  pcl::RandomSample<pcl::PointXYZ> rs;
  rs.setInputCloud(partial);
  rs.setSample(8192);
  rs.filter(*partial_sample);
}
// ----------

pcl::PointXYZ pt_scale;
float tmp_cloud[1][8192][3];
for (int j=0; j<8192; ++j)
{
    tmp_cloud[0][j][0] = partial_sample->points[j].x/scale;
    tmp_cloud[0][j][1] = partial_sample->points[j].y/scale;
    tmp_cloud[0][j][2] = partial_sample->points[j].z/scale;
    pt_scale.x = partial_sample->points[j].x/scale;
    pt_scale.y = partial_sample->points[j].y/scale;
    pt_scale.z = partial_sample->points[j].z/scale;
    partial_sample_scale->points.push_back(pt_scale);
}

torch::Tensor array_tensor = torch::from_blob(tmp_cloud, { 1, 8192, 3 }).to(at::kCUDA);
auto t1 = std::chrono::high_resolution_clock::now();

std::vector<torch::jit::IValue> inputs;
inputs.push_back(array_tensor);
auto output = module.forward(inputs).toTuple();

auto t2 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> read_ms = t2 - t1;
cout << "infer_ms:" << read_ms.count() << "ms" << std::endl;

auto ts_t1 = std::chrono::high_resolution_clock::now();
torch::Tensor out1 = output->elements()[0].toTensor().squeeze().to(at::kCPU);
torch::Tensor out2 = output->elements()[1].toTensor().squeeze().to(at::kCPU);
torch::Tensor out3 = output->elements()[2].toTensor().contiguous().squeeze().to(at::kCPU);
auto ts_t2 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> ts_ms = ts_t2 - ts_t1;
cout << "ts_ms:" << ts_ms.count() << "ms" << std::endl;

auto vec_t1 = std::chrono::high_resolution_clock::now();
std::vector<float> v(out3.data_ptr<float>(), out3.data_ptr<float>() + out3.numel());
auto vec_t2 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> vec_ms = vec_t2 - vec_t1;
cout << "vec_ms:" << vec_ms.count() << "ms" << std::endl;

pcl::PointXYZ tmp_pt;
for (int i=0; i<out3.size(0); ++i)
{
    tmp_pt.x = v[3*i];
    tmp_pt.y = v[3*i+1];
    tmp_pt.z = v[3*i+2];
    pred->points.push_back(tmp_pt);
}

*pred = *pred + *partial_sample_scale;

}
}