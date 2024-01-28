#ifndef _PRED_RECON_MANAGER_H_
#define _PRED_RECON_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <bspline/non_uniform_bspline.h>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace fast_planner
{

class SDFMap;
class EDTEnvironment;
class PerceptionUtils;
class FastPlannerManager;
class SurfacePred;
class GlobalPlanner;
class LocalPlanner;
class NonUniformBspline;

struct ReconData {
  ros::Time start_time_;
  
  int local_traj_id_;
  Eigen::Vector3d local_next_pos;
  double local_next_yaw;
  double local_next_time_lb;
  vector<Eigen::Vector3d> local_path_next_goal_;
  Eigen::Vector3d local_next_vel;
  Eigen::Vector3d local_next_acc;
  Eigen::Vector3d local_start_yaw;
  double local_end_yaw;

  double relax_time_;
  double duration_;
  /* vector for each segment */
  vector<NonUniformBspline> local_pos_all, local_vel_all, local_acc_all;
  vector<NonUniformBspline> local_yaw_all, local_yawdot_all, local_yawdotdot_all;
  NonUniformBspline pos_traj_, vel_traj_, acc_traj_;
  NonUniformBspline yaw_traj_, yawdot_traj_, yawdotdot_traj_;
  vector<double> duras;
};

enum PLAN_RESULT { GG, PASS, OVER };

class PredReconManager
{
public:
  PredReconManager();
  ~PredReconManager();

  void initialize(ros::NodeHandle& nh);

  int PlanManager(pcl::PointCloud<pcl::PointXYZ>::Ptr& cur_map, const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw);
  
  unique_ptr<GlobalPlanner> global_planner_;
  unique_ptr<LocalPlanner> local_planner_;
  shared_ptr<EDTEnvironment> env_;
  shared_ptr<ReconData> rd_;
  shared_ptr<FastPlannerManager> planner_manager_;
  unique_ptr<PerceptionUtils> percep;
  pcl::PointCloud<pcl::PointXYZ>::Ptr prediction;
  pcl::PointCloud<pcl::PointXYZ>::Ptr last_prediction;
  pcl::PointCloud<pcl::PointXYZ>::Ptr visit;
  Eigen::Vector3d g_dir;
  bool pred_state;
  int pred_count;

private:
  /* Param */
  shared_ptr<SDFMap> map_;
  unique_ptr<SurfacePred> surf_pred_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr part_demean;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_R;
  Eigen::Vector4f map_center;
  Eigen::Vector3d NBV_yaw_local; 
  double map_scale, center_x, radius_max, sphere_radius;
  int end_flag;
  /* Map Utils */
  void getCurMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& map);
  void back_proj();
  void get_visit(const Vector3d& cur_pos, const double& cur_yaw);
  int getFineStates(const Eigen::Vector3i& id, const int& step);
  int getInternal(const Eigen::Vector3i& id, const int& step);
  double interpolation_yaw(const vector<double>& timeset, const vector<double>& yaws, const double& timestamp);
  pcl::PointCloud<pcl::PointXYZ>::Ptr snow_down(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& z_up, float& z_low);
  /* Path Utils */
  void shortenPath(vector<Eigen::Vector3d>& path);
  void ensurePath(vector<Eigen::Vector3d>& path, vector<double>& yaw);
  /* Pred States Utils */
  double map_x, map_y, map_z, resolution;
  int buffer_size;
  vector<char> fine_states;
  pcl::PointCloud<pcl::PointXYZ>::Ptr near_ground;
  /* PCA radius */
  double PCA_diameter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
};
}

#endif