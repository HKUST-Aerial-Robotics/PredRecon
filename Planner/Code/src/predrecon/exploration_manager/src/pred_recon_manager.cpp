#include <exploration_manager/pred_recon_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/surface_prediction.h>
#include <active_perception/global_planner.h>
#include <plan_manage/local_planner.h>
#include <plan_manage/planner_manager.h>
#include <path_searching/astar2.h>
#include <chrono>
#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>
#include <bspline/non_uniform_bspline.h>
#include <active_perception/perception_utils.h>
namespace fast_planner
{

PredReconManager::PredReconManager(){

}

PredReconManager::~PredReconManager(){
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

void PredReconManager::initialize(ros::NodeHandle& nh)
{
  map_.reset(new SDFMap);
  map_->initMap(nh);
  env_.reset(new EDTEnvironment);
  env_->setMap(map_);

  percep.reset(new PerceptionUtils(nh));

  rd_.reset(new ReconData);
  rd_->local_traj_id_ = 0;

  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  nh.param("predmanager/map_size", map_scale, -1.0);
  nh.param("exploration/relax_time", rd_->relax_time_, 1.0);
  nh.param("predmanager/far_goal", radius_max, -1.0);
  nh.param("predmanager/sphere_radius", sphere_radius, -1.0);
  nh.param("predmanager/finish_num", end_flag, -1);

  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, env_);
  ViewNode::map_ = map_;

  double resolution_ = map_->getResolution();
  Eigen::Vector3d origin, size;
  map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);
  /* Plan Utils */
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  planner_manager_->path_finder_->max_search_time_ = 1.0;
  global_planner_.reset(new GlobalPlanner(env_, nh));
  local_planner_.reset(new LocalPlanner(env_, nh));
  cout << "Planner Ready!" << endl;
  surf_pred_.reset(new SurfacePred);
  surf_pred_->init(nh, env_);
  /* GPU warm-up */
  cout << "Warm-Up Started!" << endl;
  surf_pred_->warmup();
  cout << "Warm-Up Finished!" << endl;

  nh.param("sdf_map/map_size_x", map_x, -1.0);
  nh.param("sdf_map/map_size_y", map_y, -1.0);
  nh.param("sdf_map/map_size_z", map_z, -1.0);
  nh.param("sdf_map/resolution", resolution, -1.0);

  buffer_size = ceil(map_x / resolution) * ceil(map_y / resolution) * ceil(map_z / resolution);
  fine_states = vector<char>(buffer_size, 0);
  g_dir = Eigen::Vector3d::Zero();
  pred_state = false;
  pred_count = 0;
  last_prediction.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

int PredReconManager::PlanManager(pcl::PointCloud<pcl::PointXYZ>::Ptr& cur_map, const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw)
{
  cout << "-----Planning Start!-----" << endl;
  auto plan_t1 = std::chrono::high_resolution_clock::now();
  
  cout << "-----Surface Prediction Start!-----" << endl;
  auto pred_t1 = std::chrono::high_resolution_clock::now();
  /* Set Fine States */
  Eigen::Vector3d input_pos;
  Eigen::Vector3i input_idx;
  int input_address;
  for (int i=0; i<cur_map->points.size(); ++i)
  {
    input_pos(0) = cur_map->points[i].x;
    input_pos(1) = cur_map->points[i].y;
    input_pos(2) = cur_map->points[i].z;
    env_->sdf_map_->posToIndex(input_pos, input_idx);
    input_address = env_->sdf_map_->toAddress(input_idx);
    fine_states[input_address] = 1;
  }
  
  part_demean.reset(new pcl::PointCloud<pcl::PointXYZ>);
  /* Get Current Map Cloud */
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_sample (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::RandomSample<pcl::PointXYZ> rs;
  rs.setInputCloud(cur_map);
  rs.setSample(8192);
  rs.filter(*map_sample);

  getCurMap(cur_map);
  prediction.reset(new pcl::PointCloud<pcl::PointXYZ>);
  visit.reset(new pcl::PointCloud<pcl::PointXYZ>);
  /* Prediction */
  surf_pred_->inference(part_demean, prediction, map_scale);
  back_proj();
  if (pred_state == true && pred_count >= 2)
  {
    pcl::PointXYZ minPt, maxPt;
	  pcl::getMinMax3D(*prediction, minPt, maxPt);
    Eigen::Vector3d center_p;
    center_p(0) = 0.5*(minPt.x+maxPt.x);
    center_p(1) = 0.5*(minPt.y+maxPt.y);
    center_p(2) = 0.5*(minPt.z+maxPt.z);

    pcl::PointXYZ minPt_l, maxPt_l;
	  pcl::getMinMax3D(*last_prediction, minPt_l, maxPt_l);
    Eigen::Vector3d center_l;
    center_l(0) = 0.5*(minPt_l.x+maxPt_l.x);
    center_l(1) = 0.5*(minPt_l.y+maxPt_l.y);
    center_l(2) = 0.5*(minPt_l.z+maxPt_l.z);
    
    double diff = (center_p-center_l).norm();
    ROS_WARN("Prediction Diff");
    cout << diff << endl;
    // if (diff > 100.0)
    if (diff > 0.3)
    {
      *prediction = *last_prediction;
      ROS_ERROR("Trigger diff mechanism!");
    }
  }
  pred_count++;
  pred_state = true;
  *last_prediction = *prediction;

  get_visit(pos, yaw(0));
  // Finish whole system
  if (visit->points.size() < end_flag || PCA_diameter(visit) < 0.5)
    return OVER;
  auto pred_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> pred_ms = pred_t2 - pred_t1;
  cout << "Surface_Prediction_Time:" << pred_ms.count() << "ms!" << std::endl;
  cout << "-----Surface Prediction Finished!-----" << endl;

  /* Global Planning */
  cout << "-----Global Planning Start!-----" << endl;
  auto global_t1 = std::chrono::high_resolution_clock::now();

  env_->sdf_map_->reset_PredStates();
  bool finish_res = false;
  *prediction = *prediction + *map_sample;
  finish_res = global_planner_->GlobalPathManager(prediction, visit, cur_map, pos, vel, yaw, g_dir);
  if (finish_res == true)
  {
    ROS_ERROR("global stage no more --> over");
    return OVER;
  }
  if (global_planner_->NBV_res == false)
    return GG;
  ROS_WARN("Global to NBV!");
  cout << "dist to NBV:" << (pos-global_planner_->NBV.pos_g).norm() << endl;

  auto global_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> global_ms = global_t2 - global_t1;
  cout << "Global_Planning_Time:" << global_ms.count() << "ms!" << std::endl;
  cout << "-----Global Planning Finished!-----" << endl;
  /* Local Planning */
  auto local_t1 = std::chrono::high_resolution_clock::now();
  
  pcl::PointXYZ pt;
  local_R.reset(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto i:global_planner_->local_regions_[0])
  {
    pt.x = i(0);
    pt.y = i(1);
    pt.z = i(2);
    local_R->points.push_back(pt);
  }

  double diameter = PCA_diameter(local_R);

  if (diameter < 1.0)
  {
    local_planner_->local_points = {pos, global_planner_->NBV.pos_g};
    local_planner_->local_yaws = {yaw(0), global_planner_->NBV.yaw_g};
  }
  else
  {
    NBV_yaw_local(0) = global_planner_->NBV.yaw_g;
    NBV_yaw_local(1) = 0.0;
    NBV_yaw_local(2) = 0.0;
    local_planner_->LocalPathManager(local_R, global_planner_->local_normal_, pos, vel, yaw, 
    global_planner_->NBV.pos_g, NBV_yaw_local);
  }

  auto local_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> local_ms = local_t2 - local_t1;
  cout << "Local_Planning_Time:" << local_ms.count() << "ms!" << std::endl;
  /* Bspline Control Points */
  auto traj_t1 = std::chrono::high_resolution_clock::now();

  rd_->duration_ = 0.0;
  double len = Astar::pathLength(local_planner_->local_points);
  /*-- If path too long! --*/
  if (len > radius_max)
  {
    ROS_WARN("Far Goal!");
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { pos };
    vector<double> truncated_yaw = { local_planner_->local_yaws.front() };
    for (int i=1; i<local_planner_->local_points.size(); ++i)
    {
      auto cur_pt = local_planner_->local_points[i];
      auto cur_yaw = local_planner_->local_yaws[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      if (len2 < radius_max)
      {
        truncated_path.push_back(cur_pt);
        truncated_yaw.push_back(cur_yaw);
      }
      else
      {
        // double offset = 0.8*radius_max+(cur_pt - truncated_path.back()).norm()-len2;
        // if (offset > 1.0)
        // {
        //   Eigen::Vector3d end_pos = truncated_path.back() + (cur_pt - truncated_path.back()).normalized()*offset;
        //   /* yaw */
        //   double change_range = fabs(cur_yaw-truncated_yaw.back());
        //   int sign_r = cur_yaw-truncated_yaw.back() > 0? 1:-1;
        //   double output_yaw = 0.0;
        //   // constrain in semi plane
        //   if (change_range < M_PI)
        //     output_yaw = truncated_yaw.back() + (cur_yaw-truncated_yaw.back())*offset/((cur_pt - truncated_path.back()).norm());
        //   else
        //     output_yaw = truncated_yaw.back() - sign_r * (2*M_PI-change_range)*offset/((cur_pt - truncated_path.back()).norm());
          
        //   while (output_yaw < -M_PI)
        //     output_yaw += 2 * M_PI;
        //   while (output_yaw > M_PI)
        //     output_yaw -= 2 * M_PI;

        //   double end_yaw = output_yaw;
        //   truncated_path.push_back(end_pos);
        //   truncated_yaw.push_back(end_yaw);
        // }
        /* -- add last+1 viewpoint -- */
        truncated_path.push_back(cur_pt);
        truncated_yaw.push_back(cur_yaw);
        break;
      }
    }

    if (truncated_path.size() == 1)
    {
      Eigen::Vector3d local_dir = (local_planner_->local_points[1] - truncated_path.back()).normalized();
      double local_len = (local_planner_->local_points[1] - truncated_path.back()).norm();
      Eigen::Vector3d pseudo_pt = truncated_path.back() + 0.8*radius_max*local_dir;
      /* pseudo yaw */
      double change_range = fabs(local_planner_->local_yaws[1]-truncated_yaw.back());
      int sign_r = local_planner_->local_yaws[1]-truncated_yaw.back() > 0? 1:-1;
      double output_yaw = 0.0;
      // constrain in semi plane
      if (change_range < M_PI)
        output_yaw = truncated_yaw.back() + (local_planner_->local_yaws[1]-truncated_yaw.back())*0.8*radius_max/local_len;
      else
        output_yaw = truncated_yaw.back() - sign_r * (2*M_PI-change_range)*0.8*radius_max/local_len;
      
      while (output_yaw < -M_PI)
        output_yaw += 2 * M_PI;
      while (output_yaw > M_PI)
        output_yaw -= 2 * M_PI;
      double pseudo_yaw = output_yaw;

      truncated_path.push_back(pseudo_pt);
      truncated_yaw.push_back(pseudo_yaw);
    }

    local_planner_->local_points = truncated_path;
    local_planner_->local_yaws = truncated_yaw;
  }
  /*-- If path too short --*/
  if (len < 1.0)
  {
    ROS_ERROR("path too short!");
    return GG;
  }
  // Old waypoints
  vector<Eigen::Vector3d> old_waypts = local_planner_->local_points;
  // Pos
  // A* search each 2 viewpoints
  double len_after = Astar::pathLength(local_planner_->local_points);
  cout << "initial_local_path_length:" << len_after << "m" << endl;
  cout << "initial_stoe:" << (local_planner_->local_points.back() - local_planner_->local_points.front()).norm() << endl;

  vector<Eigen::Vector3d> initial_path;
  vector<Eigen::Vector3d> initial_segment;
  double initial_length = 0.0;
  for (int i=0; i<local_planner_->local_points.size()-1; ++i)
  {
    planner_manager_->path_finder_->reset();
    planner_manager_->path_finder_->setResolution(0.2);
    if (planner_manager_->path_finder_->local_search(local_planner_->local_points[i], local_planner_->local_points[i+1]) != Astar::REACH_END)
    {
      ROS_ERROR("No path to next viewpoint");
      cout << "Current Length: " <<  initial_length << "m." << endl;
      return GG;
    }
    initial_segment = planner_manager_->path_finder_->getPath();
    initial_length += Astar::pathLength(initial_segment);
    if (i == 0)
      initial_path.insert(initial_path.end(), initial_segment.begin(), initial_segment.end());
    else
      initial_path.insert(initial_path.end(), initial_segment.begin()+1, initial_segment.end());
  }
  shortenPath(initial_path);
  local_planner_->local_points = initial_path;

  len = Astar::pathLength(local_planner_->local_points);
  double len_0 = (local_planner_->local_points.back() - local_planner_->local_points.front()).norm();
  ROS_ERROR("Real Length!");
  cout << "real length:" << len << endl;
  cout << "start to end:" << len_0 << endl;

  // Planning time lower bound
  double yaw_time = 0.0;
  for (int i=0; i<local_planner_->local_yaws.size()-1; ++i)
  {
    yaw_time += fabs(local_planner_->local_yaws[i+1] - local_planner_->local_yaws[i])/ViewNode::yd_;
  }
  rd_->local_next_time_lb = max(len/(1.0*ViewNode::vm_), yaw_time);
  // Record global tsp direction  
  g_dir = (local_planner_->local_points.back() - pos).normalized();
  // check local vp
  // ROS_ERROR("Viewpoint Position!");
  // for (int i=0; i<local_planner_->local_points.size(); ++i)
  //   cout << "No." << i << "_vp:" << local_planner_->local_points[i] << "m." << endl;

  planner_manager_->planExploreTraj(local_planner_->local_points, vel, acc, rd_->local_next_time_lb);
  // planner_manager_->planExploreTraj(initial_path, vel, acc, rd_->local_next_time_lb);
  rd_->duration_ = planner_manager_->local_data_.duration_;
  rd_->pos_traj_ = planner_manager_->local_data_.position_traj_;
  rd_->vel_traj_ = planner_manager_->local_data_.velocity_traj_;
  rd_->acc_traj_ = planner_manager_->local_data_.acceleration_traj_;

  // Yaw
  double yaw_change = fabs(local_planner_->local_yaws.back()-local_planner_->local_yaws.front());
  // for (int i=0; i<local_planner_->local_yaws.size(); ++i)
  //   cout << "No." << i << "_init_yaw:" << local_planner_->local_yaws[i]*180.0/M_PI << " deg" << endl;
  // Old Time step
  vector<double> old_timestep = {0.0};
  double old_dist = 0.0;
  double old_len =  Astar::pathLength(old_waypts);
  for (int i=0; i<old_waypts.size()-1; ++i)
  {
    old_dist += (old_waypts[i+1] - old_waypts[i]).norm();
    double old_ts = (old_dist/old_len) * rd_->duration_;
    old_timestep.push_back(old_ts);
  }
  // New Time step
  vector<double> timestep = {0.0};
  double dist = 0.0;
  for (int i=0; i<local_planner_->local_points.size()-1; ++i)
  {
    dist += (local_planner_->local_points[i+1] - local_planner_->local_points[i]).norm();
    double ts = (dist/len) * rd_->duration_;
    timestep.push_back(ts);
  }
  // distribute new yaws
  vector<double> new_yaws = {local_planner_->local_yaws.front()};
  double tmp_yaw = 0.0;
  for (int i=1; i<timestep.size()-1; ++i)
  {
    tmp_yaw = interpolation_yaw(old_timestep, local_planner_->local_yaws, timestep[i]);
    new_yaws.push_back(tmp_yaw);
  }
  new_yaws.push_back(local_planner_->local_yaws.back());

  local_planner_->local_yaws = new_yaws;
  // check yaw angle
  // cout << "start_yaw:" << yaw(0)*180.0/M_PI << " deg" << endl;
  // ROS_ERROR("Yaw Constriants!");
  // for (int i=0; i<local_planner_->local_yaws.size(); ++i)
  //   cout << "No." << i << "_yaw:" << local_planner_->local_yaws[i]*180.0/M_PI << " deg" << endl;

  planner_manager_->planYawExplore(yaw, local_planner_->local_yaws.back(), true, rd_->relax_time_, timestep, local_planner_->local_yaws);
  rd_->yaw_traj_ = planner_manager_->local_data_.yaw_traj_;
  rd_->yawdot_traj_ = planner_manager_->local_data_.yawdot_traj_;
  rd_->yawdotdot_traj_ = planner_manager_->local_data_.yawdotdot_traj_;

  // cout << "end_yaw:" << rd_->yaw_traj_.evaluateDeBoor(rd_->duration_)*180.0/M_PI << " deg" << endl;
  
  rd_->local_traj_id_ += 1;

  auto traj_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> traj_ms = traj_t2 - traj_t1;
  cout << "Traj_Opt_Time:" << traj_ms.count() << "ms!" << std::endl;

  auto plan_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> plan_ms = plan_t2 - plan_t1;
  cout << "Planning_Time:" << plan_ms.count() << "ms!" << std::endl;
  cout << "-----Planning Finished!-----" << endl;

  return PASS;
}

void PredReconManager::getCurMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& map)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr wo_ground (new pcl::PointCloud<pcl::PointXYZ>);
  for (int i=0; i<map->points.size(); ++i)
  {
    if (map->points[i].z > 0.1)
    {
      wo_ground->points.push_back(map->points[i]);
    }
  }
  pcl::compute3DCentroid(*wo_ground, map_center);
  pcl::demeanPointCloud(*wo_ground, map_center, *part_demean);
}

void PredReconManager::back_proj()
{
  for (int i=0; i<prediction->points.size(); ++i)
  {
    double pred_x = prediction->points[i].x*map_scale*1.05 + map_center(0);
    double pred_y = prediction->points[i].y*map_scale*1.15 + map_center(1);
    double pred_z = prediction->points[i].z*map_scale*1.05 + map_center(2);
    Eigen::Vector3d pred_pos(pred_x, pred_y, pred_z);

    if (env_->sdf_map_->isInMap(pred_pos) && pred_z>0.1)
    {
      prediction->points[i].x = pred_x;
      prediction->points[i].y = pred_y;
      prediction->points[i].z = pred_z;
    }
  }
  pcl::PointXYZ minPt, maxPt;
	pcl::getMinMax3D(*prediction, minPt, maxPt);
  double z_lb = minPt.z-0.1;
  // add snow down
  for (int i=0; i<prediction->points.size(); ++i) 
  {
    // prediction->points[i].x -= 1.0;
    // prediction->points[i].y -= 1.25;
    prediction->points[i].z -= z_lb;
  }

  near_ground.reset(new pcl::PointCloud<pcl::PointXYZ>);
  float z_up = 0.8, z_low = 0.4;
  near_ground = snow_down(prediction, z_up, z_low);
  ROS_WARN("near_ground");
  cout << near_ground->points.size() << endl;

  for (int j=0; j<near_ground->points.size(); ++j)
    near_ground->points[j].z -= (z_up-z_low);
  *prediction = *prediction + *near_ground;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PredReconManager::snow_down(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& z_up, float& z_low)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZ> ());
  // z-axis
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new 
    pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GE, z_low)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, z_up)));

  pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
  condrem.setCondition (range_cond);
  condrem.setInputCloud (cloud);
  condrem.setKeepOrganized(false); // remove the points out of range
  // apply filter
  condrem.filter (*cluster); 

  return cluster;
}

void PredReconManager::get_visit(const Vector3d& cur_pos, const double& cur_yaw)
{
  percep->setPose(cur_pos, cur_yaw);
  Eigen::Vector3d tmp_input_pt;
  Eigen::Vector3i tmp_input_idx;
  int tmp_address;
  int step = 5;
  for (int i=0; i<prediction->points.size(); ++i)
  {
    tmp_input_pt(0) = prediction->points[i].x;
    tmp_input_pt(1) = prediction->points[i].y;
    tmp_input_pt(2) = prediction->points[i].z;
    env_->sdf_map_->posToIndex(tmp_input_pt, tmp_input_idx);
    tmp_address = env_->sdf_map_->toAddress(tmp_input_idx);
    if (env_->sdf_map_->getOccupancy(tmp_input_idx) == SDFMap::OCCUPANCY::FREE)
      continue;
    // if (getFineStates(tmp_input_idx, step) == 0 && !percep->insideCtrl(tmp_input_pt))
    if (getFineStates(tmp_input_idx, step) == 0 && tmp_input_pt(2) > 0.2)
    {
      visit->points.push_back(prediction->points[i]);
    }
  }

  ROS_WARN("Visit Cloud!");
  cout << "visit_num:" << visit->points.size() << endl;
  pcl::PointXYZ minPt_, maxPt_;
	pcl::getMinMax3D(*visit, minPt_, maxPt_);
  cout << minPt_.z << endl;
}

int PredReconManager::getFineStates(const Eigen::Vector3i& id, const int& step)
{
  int flag = 0;
  Eigen::Vector3i inflate_idx;
  int inflate_address;
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        inflate_idx(0) = id(0) + x;
        inflate_idx(1) = id(1) + y;
        inflate_idx(2) = id(2) + z;
        if (env_->sdf_map_->isInMap(inflate_idx))
        {
          inflate_address = env_->sdf_map_->toAddress(inflate_idx);
          if (fine_states[inflate_address] == 1)
          {
            flag = 1;
            break;
          }
        }
      }
  
  int inner_num = 0;
  int internal_step = 3;
  Eigen::Vector3i internal_idx;
  for (int x = -internal_step; x <= internal_step; x=x+internal_step)
  {
    if (x == 0)
      continue;
    internal_idx(0) = id(0) + x;
    internal_idx(1) = id(1);
    internal_idx(2) = id(2);
    if (env_->sdf_map_->get_PredStates(internal_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      inner_num++;
  }
  for (int y = -internal_step; y <= internal_step; y=y+internal_step)
  {
    if (y == 0)
      continue;
    internal_idx(0) = id(0);
    internal_idx(1) = id(1) + y;
    internal_idx(2) = id(2);
    if (env_->sdf_map_->get_PredStates(internal_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      inner_num++;
  }
  for (int z = -internal_step; z <= internal_step; z=z+internal_step)
  {
    if (z == 0)
      continue;
    internal_idx(0) = id(0);
    internal_idx(1) = id(1);
    internal_idx(2) = id(2) + z;
    if (env_->sdf_map_->get_PredStates(internal_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      inner_num++;
  }
  
  if (inner_num > 5)
    flag = 1;

  // bool inside = env_->sdf_map_->setcenter_check(id, 2);
  // if (inside == true)
  //   flag = 1;
  
  Eigen::Vector3i inflate_idx_z;
  int inflate_address_z;
  for (int z = 0; z <= 30; ++z) 
  {
    inflate_idx_z(0) = id(0);
    inflate_idx_z(1) = id(1);
    inflate_idx_z(2) = id(2) + z;
    if (env_->sdf_map_->isInMap(inflate_idx_z))
    {
      inflate_address_z = env_->sdf_map_->toAddress(inflate_idx_z);
      if (fine_states[inflate_address_z] == 1)
      {
        flag = 1;
        break;
      }
    }
  }

  // Eigen::Vector3i inflate_idx_y;
  // int inflate_address_y;
  // for (int y = 1; y <= 20; ++y) 
  // {
  //   inflate_idx_y(0) = id(0);
  //   inflate_idx_y(1) = id(1)-y;
  //   inflate_idx_y(2) = id(2);
  //   if (env_->sdf_map_->isInMap(inflate_idx_y))
  //   {
  //     inflate_address_y = env_->sdf_map_->toAddress(inflate_idx_y);
  //     if (fine_states[inflate_address_y] == 1)
  //     {
  //       flag = 1;
  //       break;
  //     }
  //   }
  // }

  // Eigen::Vector3i inflate_idx_x;
  // int inflate_address_x;
  // for (int x = 1; x <= 10; ++x) 
  // {
  //   inflate_idx_x(0) = id(0)-x;
  //   inflate_idx_x(1) = id(1);
  //   inflate_idx_x(2) = id(2);
  //   if (env_->sdf_map_->isInMap(inflate_idx_x))
  //   {
  //     inflate_address_x = env_->sdf_map_->toAddress(inflate_idx_x);
  //     if (fine_states[inflate_address_x] == 1)
  //     {
  //       flag = 1;
  //       break;
  //     }
  //   }
  // }

  return flag;
}

double PredReconManager::interpolation_yaw(const vector<double>& timeset, const vector<double>& yaws, const double& timestamp)
{
  double change_range = 0.0;
  double output_yaw;
  for (int i=0; i<timeset.size(); ++i)
  {
    if (timestamp < timeset[i])
    {
      change_range = fabs(yaws[i]-yaws[i-1]);
      int sign_r = yaws[i]-yaws[i-1] > 0? 1:-1;
      // constrain in semi plane
      if (change_range < M_PI)
        output_yaw = yaws[i-1] + (yaws[i]-yaws[i-1])*(timestamp-timeset[i-1])/(timeset[i]-timeset[i-1]);
      else
        output_yaw = yaws[i-1] - sign_r * (2*M_PI-change_range)*(timestamp-timeset[i-1])/(timeset[i]-timeset[i-1]);
      
      while (output_yaw < -M_PI)
        output_yaw += 2 * M_PI;
      while (output_yaw > M_PI)
        output_yaw -= 2 * M_PI;
      return output_yaw;
    }
  }
}

double PredReconManager::PCA_diameter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
  Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

  Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
	transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
	transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3,3>(0,0)) * (pcaCentroid.head<3>());

  double r_0 = eigenValuesPCA(0);
  double r_1 = eigenValuesPCA(1);
  double r_2 = eigenValuesPCA(2);

  double x1 = max(fabs(r_0), fabs(r_1));  
  double x2 = max(fabs(x1), fabs(r_2)); 

  return x2;
}

void PredReconManager::shortenPath(vector<Eigen::Vector3d>& path)
{
  if (path.empty()) 
  {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // Shorten the tour, only critical intermediate points are reserved.
  const double dist_thresh = 1.0;
  vector<Vector3d> short_tour = { path.front() };
  for (int i=1; i<path.size()-1; ++i)
  {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else
    {
      // Add waypoints to shorten path only to avoid collision
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok())
      {
        if (env_->sdf_map_->getOccupancy(idx) == SDFMap::OCCUPIED || env_->sdf_map_->get_PredStates(idx) == SDFMap::PRED_INTERNAL) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());
  // Ensure at least three points in the path
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  path = short_tour;
}

void PredReconManager::ensurePath(vector<Eigen::Vector3d>& path, vector<double>& yaw)
{
  if (path.size() < 3)
  {
    path.insert(path.begin()+1, 0.5*(path.front()+path.back()));
    yaw.insert(yaw.begin()+1, 0.5*(yaw.front()+yaw.back()));
  }
}
}