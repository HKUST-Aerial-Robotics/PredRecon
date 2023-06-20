#include <exploration_manager/pred_recon_fsm.h>
#include <active_perception/global_planner.h>
#include <plan_manage/local_planner.h>
#include <traj_utils/planning_visualization.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <exploration_manager/pred_recon_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <exploration_manager/expl_data.h>
#include <plan_manage/planner_manager.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <std_msgs/Int8.h>
#include <random>

using std::vector;
double time_for_img = 0.0;
int img_id = 1;

double rm_x = 0.0, rm_y = 0.0, rm_z = 0.0;
namespace fast_planner
{

void PredReconFSM::init(ros::NodeHandle& nh)
{
  pr_manager_.reset(new PredReconManager);
  pr_manager_->initialize(nh);

  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);

  fd_.reset(new FSM_Data);
  fp_.reset(new FSM_Param);
  vis_utils_.reset(new PlanningVisualization(nh));
  vis_utils_->set_env(pr_manager_->env_);

  state_ = RECON_STATE::INITIAL;

  fd_->have_odom_ = false;
  fd_->state_str_ = { "INITIAL", "WAIT", "PLAN", "PUB", "EXEC", "END" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;

  nh.param("reconfsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("reconfsm/replan_1", fp_->replan_1_, -1.0);
  nh.param("reconfsm/replan_2", fp_->replan_2_, -1.0);
  nh.param("reconfsm/replan_proportion", fp_->replan_p_, -1.0);
  nh.param("reconfsm/img_dir_", image_folder, string("null"));
  nh.param("reconfsm/fx", fx_, -1.0);
  nh.param("reconfsm/fy", fy_, -1.0);
  nh.param("reconfsm/cx", cx_, -1.0);
  nh.param("reconfsm/cy", cy_, -1.0);
  nh.param("reconfsm/img_flag", image_collect_flag, false);
  nh.param("reconfsm/noise_mean", n_mean, -1.0);
  nh.param("reconfsm/noise_variance", n_var, -1.0);
  /* ROS Func */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &PredReconFSM::FSMCallback, this);
  // sync_timer_ = nh.createTimer(ros::Duration(0.1), &PredReconFSM::syncCallback, this);
  safe_timer_ = nh.createTimer(ros::Duration(0.05), &PredReconFSM::safetyCallback, this);
  // frontier_timer_ = nh.createTimer(ros::Duration(0.5), &PredReconFSM::frontierCallback, this);

  trigger_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1, &PredReconFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &PredReconFSM::odometryCallback, this);
  cloud_sub_ = nh.subscribe("/sdf_map/occupancy_all", 1, &PredReconFSM::cloudCallback, this);
  /* -- image collection -- */
  if (image_collect_flag)
  {
    image_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, "/airsim_node/drone_1/front_center/Scene", 50));
    pose_sub_.reset(
        new message_filters::Subscriber<nav_msgs::Odometry>(nh, "/map_ros/pose", 25));
    sync_image_pose_.reset(new message_filters::Synchronizer<PredReconFSM::SyncPolicyImagePose>(
        PredReconFSM::SyncPolicyImagePose(100), *image_sub_, *pose_sub_));
    sync_image_pose_->registerCallback(boost::bind(&PredReconFSM::imgCallback, this, _1, _2));
  }
  
  // Publisher
  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);
  safe_pub_ = nh.advertise<std_msgs::Int8>("/planning/safe_trigger", 10);
  plan_pub_ = nh.advertise<std_msgs::Int8>("/planning/fail_trigger", 10);
}

void PredReconFSM::FSMCallback(const ros::TimerEvent& e)
{
  ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << fd_->state_str_[int(state_)]);
  
  switch (state_)
  {
    case INITIAL: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom.");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT, "FSM");
      break;
    }

    case WAIT: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      break;
    }

    case END: {
      ROS_INFO_THROTTLE(1.0, "finish reconstruction collection.");
      break;
    }
    
    case PLAN:{
      ros::Time time_start;
      if (fd_->static_state_) {
        // Plan from static state (hover)
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();

        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;

        ROS_WARN("STATIC!");
        cout << "start_pt_x:" << fd_->start_pt_(0) << endl;
        cout << "start_pt_y:" << fd_->start_pt_(1) << endl;
        cout << "start_pt_z:" << fd_->start_pt_(2) << endl;
      }
      else
      {
        double t_r = (ros::Time::now() - pr_manager_->rd_->start_time_).toSec() + fp_->replan_time_;
  
        fd_->start_pt_ = pr_manager_->rd_->pos_traj_.evaluateDeBoor(t_r);
        fd_->start_vel_ = pr_manager_->rd_->vel_traj_.evaluateDeBoor(t_r);
        fd_->start_acc_ = pr_manager_->rd_->acc_traj_.evaluateDeBoor(t_r);
        fd_->start_yaw_(0) = pr_manager_->rd_->yaw_traj_.evaluateDeBoor(t_r)[0];
        fd_->start_yaw_(1) = pr_manager_->rd_->yawdot_traj_.evaluateDeBoor(t_r)[0];
        fd_->start_yaw_(2) = pr_manager_->rd_->yawdotdot_traj_.evaluateDeBoor(t_r)[0];
        fd_->last_pos_traj = pr_manager_->rd_->pos_traj_;
        fd_->last_start_time = pr_manager_->rd_->start_time_;
      }
      
      // replan_pub_.publish(std_msgs::Empty());
      int res = callPathPlanner(time_start);
      if (res == PASS)
      {
        transitState(EXEC, "FSM");
        ROS_WARN("EXEC traj!");
      }
      else if (res == GG)
      {
        ROS_WARN("plan fail");
        /* For AirSim Tricky */
        // fd_->static_state_ = true;
      }
      else if (res == OVER)
      {
        transitState(END, "FSM");
        ROS_WARN("finish whole system!");
      }
      break;
    }

    case EXEC:{
      double t_now = (ros::Time::now() - fd_->newest_start_time).toSec();
      double t_old = (ros::Time::now() - fd_->last_start_time).toSec();
      double t_cur = 0.0;
      double time_to_end = 0.0;
      double replan_duration = 0.0;

      if (t_now >= 0)
      {
        t_cur = t_now;
        time_to_end = fd_->newest_dura - t_cur;
        replan_duration = fd_->newest_dura;
      }
      else if (t_now < 0 && t_old > 0 && pr_manager_->rd_->local_traj_id_ > 1)
      {
        t_cur = t_old;
        time_to_end = fd_->last_dura - t_cur;
        replan_duration = fd_->last_dura;
      }
      // Replan if traj is almost fully executed
      if (time_to_end < fp_->replan_1_)
      {
        transitState(PLAN, "FSM");
        ROS_WARN("Replan: traj fully executed=================================");
        return;
      }
      // Replan after some time
      if (t_cur > min(fp_->replan_p_*replan_duration, fp_->replan_2_)) {
        transitState(PLAN, "FSM");
        ROS_WARN("Replan: periodic call=======================================");
      }
      thread vis_thread(&PredReconFSM::visualization, this);
      vis_thread.detach();
      break;
    }
    
  }
}

void PredReconFSM::safetyCallback(const ros::TimerEvent& e)
{
  if (state_ == RECON_STATE::EXEC)
  {
    double dist;
    bool safe = true;

    double t_now = (ros::Time::now() - fd_->newest_start_time).toSec();
    double t_old = (ros::Time::now() - fd_->last_start_time).toSec();
    if (t_now >= 0)
    {
      Eigen::Vector3d cur_pt = fd_->newest_pos_traj.evaluateDeBoorT(t_now);
      double radius = 0.0;
      Eigen::Vector3d fut_pt;
      double fut_t = 0.02;
      while (radius < 10.0 && t_now + fut_t < fd_->newest_dura)
      {
        fut_pt = fd_->newest_pos_traj.evaluateDeBoorT(t_now + fut_t);
        // check safety TODO: inflate occ + internal
        if (pr_manager_->env_->sdf_map_->getInflateOccupancy(fut_pt) == 1 || pr_manager_->env_->sdf_map_->getInternal_check(fut_pt, 5) == 1)
        {
          dist = radius;
          std::cout << "collision at: " << fut_pt.transpose() << std::endl;
          safe = false;
          break;
        }
        radius = (fut_pt - cur_pt).norm();
        fut_t += 0.02;
      }
    }
    
    int safe_trigger = -1;
    std_msgs::Int8 msg;
    if (!safe) 
    {
      ROS_WARN("Replan: collision detected==================================");
      transitState(PLAN, "safetyCallback");
      safe_trigger = 1;
    }
    msg.data = safe_trigger;
    safe_pub_.publish(msg);
  }
}

void PredReconFSM::triggerCallback(const nav_msgs::PathConstPtr& msg)
{
  if (msg->poses[0].pose.position.z < -0.1) return;
  if (state_ != WAIT) return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN, "triggerCallback");
}

void PredReconFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  // AirSim Trick
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0)) + 90.0*M_PI/180.0;

  fd_->have_odom_ = true;
}

void PredReconFSM::cloudCallback(const sensor_msgs::PointCloud2& input)
{
  fd_->cur_map.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointXYZ pt;
  // sensor_msgs::PointCloud2 ----> pcl::PointCloud<T>
  pcl::fromROSMsg(input, cloud);

  for (int i=0; i<cloud.points.size(); ++i)
  {
    if(cloud.points[i].z>0.3)
    {
      pt.x = cloud.points[i].x;
      pt.y = cloud.points[i].y;
      pt.z = cloud.points[i].z;
      fd_->cur_map->points.push_back(pt);
    }
  }
}

void PredReconFSM::imgCallback(const sensor_msgs::ImageConstPtr& img,
                               const nav_msgs::OdometryConstPtr& pose)
{
  // AirSim only
  Eigen::Quaterniond airsim_q_;
  airsim_q_.w() = cos(0.25*M_PI);
  airsim_q_.x() = 0.0;
  airsim_q_.y() = 0.0;
  airsim_q_.z() = sin(0.25*M_PI);
  // Translation
  Eigen::Vector3d body_pos_, camera2body_XYZ, camera_pos_;
  camera2body_XYZ << 0.125, 0, 0;// hyper param
  body_pos_(0) = pose->pose.pose.position.x;
  body_pos_(1) = pose->pose.pose.position.y;
  body_pos_(2) = pose->pose.pose.position.z;
  // Rotation
  Eigen::Quaterniond body_q_, camera_q_;
  Eigen::Matrix3d Rotation_matrix, camera2body_rotation;
  body_q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
                                 pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);
  body_q_ = airsim_q_*body_q_;
  Rotation_matrix = body_q_.toRotationMatrix();
  camera2body_rotation << 0, 0, 1,
                          -1, 0, 0,
                          0, -1, 0;
  Eigen::Quaterniond c_to_b(camera2body_rotation);
  // Extrinsic
  camera_pos_ = Rotation_matrix * camera2body_XYZ + body_pos_;
  camera_q_ = body_q_*c_to_b;

  // Pose
  Eigen::Matrix3d c_to_w;
  c_to_w = camera_q_.toRotationMatrix();
  Eigen::Quaterniond c_q_;
  c_q_ = camera_q_.inverse();
  Eigen::Vector3d c_pos_;
  c_pos_ = -c_to_w.inverse()*camera_pos_;

  double time = ros::Time::now().toSec();
  if ((time-time_for_img) > 0.4)
  {
    time_for_img = time;
    // ros::Time time = img->header.stamp;
    std::string str;
    stringstream ss;
    ss << setfill('0') << setw(4) << img_id;
    string name = ss.str();
    str = image_folder + "images/image_" + name + ".jpg";
    cv_bridge::CvImageConstPtr ptr;
    ptr = cv_bridge::toCvCopy(img, "bgr8");
    cv::imwrite(str, ptr->image);

    ofstream intrinsic_file;
    intrinsic_file.open(image_folder + "created/sparse/cameras.txt", ios_base::app);
    intrinsic_file << to_string(img_id) << " PINHOLE" << " " << to_string(1280) << " " << to_string(960) << " " << to_string(fx_) << " " << to_string(fy_) << " " << to_string(cx_) << " " << to_string(cy_) << "\n";
    intrinsic_file.close();

    ofstream extrinsic_file;
    extrinsic_file.open(image_folder + "created/sparse/images.txt", ios_base::app);
    extrinsic_file << to_string(img_id) << " " << to_string(c_q_.w()) << " " << to_string(c_q_.x()) << " " << to_string(c_q_.y()) << " " << to_string(c_q_.z()) << " " << to_string(c_pos_(0)) << " " << to_string(c_pos_(1)) << " " << to_string(c_pos_(2)) << " 1 image_" << name + ".jpg" << "\n";
    extrinsic_file << "\n";
    extrinsic_file.close();

    img_id++;
  }
}

void PredReconFSM::syncCallback(const ros::TimerEvent& e)
{
  cudaDeviceSynchronize();
}

int PredReconFSM::callPathPlanner(ros::Time& ts)
{
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);
  if (pr_manager_->rd_->local_traj_id_ > 1)
  {
    fd_->last_pos_traj = fd_->newest_pos_traj;
    fd_->last_yaw_traj = fd_->newest_yaw_traj;
    fd_->last_start_time = fd_->newest_start_time;
    fd_->last_dura = fd_->newest_dura;
  } 
  int res = pr_manager_->PlanManager(fd_->cur_map, fd_->start_pt_, fd_->start_vel_, fd_->start_acc_, fd_->start_yaw_);
  
  if (res == GG)
  {
    int fail = 1;
    std_msgs::Int8 msg;
    msg.data = fail;
    plan_pub_.publish(msg);
  }

  if (res == PASS)
  {
    vis_utils_->publishGlobal_Internal();
    pr_manager_->rd_->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;
    
    fd_->newest_pos_traj = pr_manager_->rd_->pos_traj_;
    fd_->newest_yaw_traj = pr_manager_->rd_->yaw_traj_;
    fd_->newest_start_time = pr_manager_->rd_->start_time_;
    fd_->newest_dura = pr_manager_->rd_->duration_;
    
    bspline::Bspline bspline;
    bspline.order = pr_manager_->planner_manager_->pp_.bspline_degree_;
    bspline.start_time = pr_manager_->rd_->start_time_;
    bspline.traj_id = pr_manager_->rd_->local_traj_id_;
    cout << "traj_id:" << bspline.traj_id << "_id!" << endl;
    // Pos traj
    Eigen::MatrixXd pos_pts = pr_manager_->rd_->pos_traj_.getControlPoint();
    for (int j=0; j<pos_pts.rows(); ++j)
    {
      geometry_msgs::Point pt;
      pt.x = pos_pts(j, 0);
      pt.y = pos_pts(j, 1);
      pt.z = pos_pts(j, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = pr_manager_->rd_->pos_traj_.getKnot();
    for (int k=0; k<knots.rows(); ++k)
    {
      bspline.knots.push_back(knots(k));
    }
    // Yaw traj
    Eigen::MatrixXd yaw_pts = pr_manager_->rd_->yaw_traj_.getControlPoint();
    for (int m=0; m<yaw_pts.rows(); ++m)
    {
      double yaw = yaw_pts(m, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = pr_manager_->rd_->yaw_traj_.getKnotSpan();
    fd_->newest_traj_ = bspline;
    bspline_pub_.publish(fd_->newest_traj_);
    fd_->static_state_ = false;

  }

  return res;
}

void PredReconFSM::frontierCallback(const ros::TimerEvent& e)
{
  static int delay = 0;
  if (++delay < 5) return;

  if (state_ == WAIT || state_ == END)
  {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;
    ft->searchFrontiers();
    ft->computeFrontiersToVisit();
    ft->updateFrontierCostMatrix();

    ft->getFrontiers(ed->frontiers_);
    ft->getFrontierBoxes(ed->frontier_boxes_);

    for (int i = 0; i < ed->frontiers_.size(); ++i) 
    {
      vis_utils_->drawCubes(ed->frontiers_[i], 0.1,
                                vis_utils_->getColor(double(i) / ed->frontiers_.size(), 0.4),
                                "frontier", i, 4);
    }
    for (int i = ed->frontiers_.size(); i < 50; ++i) 
    {
      vis_utils_->drawCubes({}, 0.1, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
    }
  }
}

void PredReconFSM::transitState(RECON_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[" + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]
       << endl;
}

void PredReconFSM::visualization()
{
  // visualize pos and yaw
  vis_utils_->drawBspline(pr_manager_->rd_->pos_traj_, 0.1, Eigen::Vector4d(1.0, 0.0, 0.0, 1), false, 0.15,
                            Eigen::Vector4d(1, 1, 0, 1));
  vis_utils_->drawYawTraj(pr_manager_->rd_->pos_traj_, pr_manager_->rd_->yaw_traj_, 0.1);
  // visualize global planning
  vis_utils_->publishGlobalPlanner(pr_manager_->global_planner_->copy_cloud_GHPR, pr_manager_->global_planner_->uni_open_cnv, pr_manager_->global_planner_->global_tour, pr_manager_->global_planner_->NBV.pos_g,
  pr_manager_->global_planner_->global_cur_pos, pr_manager_->global_planner_->global_dir, pr_manager_->global_planner_->NBV.yaw_g);
  vis_utils_->publishGlobalCluster(pr_manager_->global_planner_->uni_cluster, pr_manager_->global_planner_->cls_normal, pr_manager_->global_planner_->uni_center);
  
  pcl::PointCloud<pcl::PointXYZ> vis_pred;
  pcl::PointXYZ pt_;
  for (int i=0; i<pr_manager_->global_planner_->local_regions_[0].size(); ++i)
  {
    pt_.x = pr_manager_->global_planner_->local_regions_[0][i](0);
    pt_.y = pr_manager_->global_planner_->local_regions_[0][i](1);
    pt_.z = pr_manager_->global_planner_->local_regions_[0][i](2);
    vis_pred.push_back(pt_);
  }
  vis_utils_->publishLocalRegion(vis_pred);
  vis_utils_->publishLocal_Tour(pr_manager_->local_planner_->local_points);
  vis_utils_->publishLocal_VP(pr_manager_->local_planner_->local_points, pr_manager_->local_planner_->local_yaws);
}

double PredReconFSM::Gaussian_noise(const double& mean, const double& var)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> dist(mean, var);
  double noise = dist(generator);

  return noise;
}

} 
