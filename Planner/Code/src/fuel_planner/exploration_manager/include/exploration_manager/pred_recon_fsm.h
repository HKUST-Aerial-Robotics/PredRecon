#ifndef _PRED_RECON_FSM_H_
#define _PRED_RECON_FSM_H_

#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h> 
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <sensor_msgs/PointCloud2.h> 
#include <pcl/point_cloud.h>  
#include <pcl_conversions/pcl_conversions.h>  
#include <pcl/io/pcd_io.h> 
#include <bspline/Bspline.h>
#include <bspline/non_uniform_bspline.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <thread>

using Eigen::Vector3d;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::string;

namespace fast_planner
{
struct FSM_Data
{
  bool trigger_, have_odom_, static_state_;
  vector<string> state_str_;
  
  double scale;
  Vector3d global_NBV_pos;
  double global_NBV_yaw;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;

  NonUniformBspline newest_pos_traj, last_pos_traj, newest_yaw_traj, last_yaw_traj;
  ros::Time newest_start_time, last_start_time;
  double newest_dura, last_dura;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cur_map;
  Eigen::Vector3d cur_pos;
};

struct FSM_Param
{
  double replan_time_;
  double replan_1_;
  double replan_2_;
  double replan_p_;
};

class PredReconManager;
class PlanningVisualization;
class FastExplorationManager;

enum RECON_STATE { INITIAL, WAIT, PLAN, PUB, EXEC, END };

class PredReconFSM
{
public:
  PredReconFSM() {
  }
  ~PredReconFSM() {
  }

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* ImagePose */
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImagePose;
  typedef unique_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  /* planning utils */
  shared_ptr<PredReconManager> pr_manager_;
  shared_ptr<FastExplorationManager> expl_manager_;
  shared_ptr<FSM_Data> fd_;
  shared_ptr<FSM_Param> fp_;
  shared_ptr<PlanningVisualization> vis_utils_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_p;

  RECON_STATE state_;

  /* ROS utils */
  ros::Timer exec_timer_, sync_timer_, safe_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, cloud_sub_;
  ros::Publisher bspline_pub_, replan_pub_, safe_pub_, plan_pub_;

  unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
  unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> pose_sub_;
  SynchronizerImagePose sync_image_pose_;

  /* support functions */
  void transitState(RECON_STATE new_state, string pos_call);

  /* ROS functions */
  void FSMCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void safetyCallback(const ros::TimerEvent& e);
  void triggerCallback(const nav_msgs::PathConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void cloudCallback(const sensor_msgs::PointCloud2& input);
  void imgCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose);
  int callPathPlanner(ros::Time& ts);
  void syncCallback(const ros::TimerEvent& e);
  void visualization();
  int segment_idx(double& t_exec, vector<double>& dura_list);
  double Gaussian_noise(const double& mean, const double& var);

  /* data address */
  string image_folder;
  double fx_, fy_, cx_, cy_;
  double n_mean, n_var;
  bool image_collect_flag;
};
}

#endif