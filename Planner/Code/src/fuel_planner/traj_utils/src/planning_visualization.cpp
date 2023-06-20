#include <traj_utils/planning_visualization.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <cstdlib>
#include <ctime>

using std::cout;
using std::endl;
namespace fast_planner {
PlanningVisualization::PlanningVisualization(ros::NodeHandle& nh) {
  node = nh;

  traj_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/trajectory", 100);
  pubs_.push_back(traj_pub_);

  topo_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/topo_path", 100);
  pubs_.push_back(topo_pub_);

  predict_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/prediction", 100);
  pubs_.push_back(predict_pub_);

  visib_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/"
                                                          "visib_constraint",
                                                          100);
  pubs_.push_back(visib_pub_);

  frontier_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/frontier", 10000);
  pubs_.push_back(frontier_pub_);

  yaw_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/yaw", 100);
  pubs_.push_back(yaw_pub_);

  viewpoint_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/viewpoints", 1000);
  pubs_.push_back(viewpoint_pub_);
  // GlobalPlanner
  pred_pub_ = node.advertise<sensor_msgs::PointCloud2>("/global_planning/pred_cloud", 10);
  localReg_pub_ = node.advertise<sensor_msgs::PointCloud2>("/global_planning/local_region", 10);
  global_pub_ = node.advertise<visualization_msgs::MarkerArray>("/global_planning/global_tour", 10);
  vpg_pub_ = node.advertise<visualization_msgs::MarkerArray>("/global_planning/vp_global", 1);
  internal_pub_ =  node.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  global_dir_pub_ = node.advertise<visualization_msgs::Marker>("/global_planning/global_dir", 10);
  global_c_pub_ = node.advertise<sensor_msgs::PointCloud2>("/global_planning/cluster", 1);
  global_n_pub_ = node.advertise<visualization_msgs::MarkerArray>("/global_planning/normals", 1);

  // LocalPlanner
  local_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_tour", 10);
  localob_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/localob_tour", 10);
  localVP_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/vp_local", 10);

  last_topo_path1_num_ = 0;
  last_topo_path2_num_ = 0;
  last_bspline_phase1_num_ = 0;
  last_bspline_phase2_num_ = 0;
  last_frontier_num_ = 0;
}

void PlanningVisualization::set_env(const shared_ptr<EDTEnvironment>& edt)
{
  this->vis_env_ = edt;
}

void PlanningVisualization::publishPredCloud(const pcl::PointCloud<pcl::PointXYZ>& input_cloud)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = input_cloud;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  pred_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishLocalRegion(const pcl::PointCloud<pcl::PointXYZ>& local_region)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = local_region;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  localReg_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishGlobal_VP(const vector<cluster_normal>& cluster)
{
  static int last_size_ = 0;

  visualization_msgs::MarkerArray view_points;
  for (int i = 0; i < last_size_; i++)
  {
    visualization_msgs::Marker vp_del;
    vp_del.header.frame_id = "world";
    vp_del.header.stamp = ros::Time::now();
    vp_del.id = i;
    vp_del.action = visualization_msgs::Marker::DELETE;
    view_points.markers.push_back(vp_del);
  }
  
  vpg_pub_.publish(view_points);

  view_points.markers.clear();
  
  int counter_vpg = 0;
  const double length = 2.0;
  for (int i=0; i<cluster.size(); ++i)
  {
    visualization_msgs::Marker vp;
    vp.header.frame_id = "world";
    vp.header.stamp = ros::Time::now();
    vp.id = counter_vpg;
    vp.type = visualization_msgs::Marker::MESH_RESOURCE;
    vp.mesh_resource = "file:///home/albert/UAV_Planning/fuel_airsim/src/fuel_planner/utils/visual_cone.dae";
    vp.action = visualization_msgs::Marker::ADD;

    vp.pose.orientation.x = 0.0;
    vp.pose.orientation.y = 0.0;
    vp.pose.orientation.z = sin(0.5*cluster[i].vps_global[0].yaw_g);
    vp.pose.orientation.w = cos(0.5*cluster[i].vps_global[0].yaw_g);
    vp.scale.x = 0.01;
    vp.scale.y = 0.007;
    vp.scale.z = 0.007;
    vp.pose.position.x = cluster[i].vps_global[0].pos_g(0);
    vp.pose.position.y = cluster[i].vps_global[0].pos_g(1);
    vp.pose.position.z = cluster[i].vps_global[0].pos_g(2);
    
    // Chartreuse
    vp.color.r = 0.87;
    vp.color.g = 0.99;
    vp.color.b = 0.00;
    vp.color.a = 0.6;

    view_points.markers.push_back(vp);
    counter_vpg++;
  }
  last_size_ = cluster.size();

  vpg_pub_.publish(view_points);
}

void PlanningVisualization::publishGlobal_Tour(const vector<Eigen::Vector3d>& global_init_tour, const Eigen::Vector3d& refined_vp, const double& refined_yaw)
{
  int counter_vpg = 0;
  // initial path
  visualization_msgs::MarkerArray tours;

  visualization_msgs::Marker global_tour_init;
  global_tour_init.header.frame_id = "world";
  global_tour_init.header.stamp = ros::Time::now();
  global_tour_init.id = counter_vpg;
  global_tour_init.type = visualization_msgs::Marker::LINE_STRIP;
  global_tour_init.action = visualization_msgs::Marker::ADD;

  global_tour_init.pose.orientation.w = 1.0;
  global_tour_init.scale.x = 0.2;

  global_tour_init.color.r = 0.2;
  global_tour_init.color.g = 0.8;
  global_tour_init.color.b = 0.4;
  global_tour_init.color.a = 1.0;

  for (int k=0; k<global_init_tour.size(); ++k)
  {
    geometry_msgs::Point p;
    p.x = global_init_tour[k][0];
    p.y = global_init_tour[k][1];
    p.z = global_init_tour[k][2];
    global_tour_init.points.push_back(p);
  }
  counter_vpg++;
  tours.markers.push_back(global_tour_init);
  // refined NBV
  visualization_msgs::Marker global_tour_refine;
  global_tour_refine.header.frame_id = "world";
  global_tour_refine.header.stamp = ros::Time::now();
  global_tour_refine.id = counter_vpg;
  global_tour_refine.type = visualization_msgs::Marker::LINE_STRIP;
  global_tour_refine.action = visualization_msgs::Marker::ADD;

  global_tour_refine.pose.orientation.w = 1.0;
  global_tour_refine.scale.x = 0.3;

  global_tour_refine.color.r = 1.0;
  global_tour_refine.color.g = 1.0;
  global_tour_refine.color.b = 0.4;
  global_tour_refine.color.a = 1.0;

  geometry_msgs::Point p;
  p.x = global_init_tour[0][0];
  p.y = global_init_tour[0][1];
  p.z = global_init_tour[0][2];
  global_tour_refine.points.push_back(p);
  geometry_msgs::Point nbv;
  nbv.x = refined_vp[0];
  nbv.y = refined_vp[1];
  nbv.z = refined_vp[2];
  global_tour_refine.points.push_back(nbv);
  
  counter_vpg++;
  // tours.markers.push_back(global_tour_refine);
  // refined NBV yaw direction
  visualization_msgs::Marker global_nbv_yaw;
  global_nbv_yaw.header.frame_id = "world";
  global_nbv_yaw.header.stamp = ros::Time::now();
  global_nbv_yaw.id = counter_vpg;
  global_nbv_yaw.type = visualization_msgs::Marker::ARROW;
  global_nbv_yaw.action = visualization_msgs::Marker::ADD;

  global_nbv_yaw.pose.orientation.w = 1.0;
  global_nbv_yaw.scale.x = 0.5;
  global_nbv_yaw.scale.y = 0.6;
  global_nbv_yaw.scale.z = 0.7;
  
  geometry_msgs::Point pt_;
  pt_.x = refined_vp[0];
  pt_.y = refined_vp[1];
  pt_.z = refined_vp[2];
  global_nbv_yaw.points.push_back(pt_);

  pt_.x = refined_vp[0] + 1.5*cos(refined_yaw);
  pt_.y = refined_vp[1] + 1.5*sin(refined_yaw);
  pt_.z = refined_vp[2];
  global_nbv_yaw.points.push_back(pt_);
  
  global_nbv_yaw.color.r = 0.0;
  global_nbv_yaw.color.g = 0.0;
  global_nbv_yaw.color.b = 0.0;
  global_nbv_yaw.color.a = 1.0;

  counter_vpg++;
  // tours.markers.push_back(global_nbv_yaw);

  global_pub_.publish(tours);
}

void PlanningVisualization::publishGlobal_Internal()
{
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  
  Eigen::Vector3i min_cut;
  Eigen::Vector3i max_cut;
  vis_env_->sdf_map_->posToIndex(vis_env_->sdf_map_->min_bound, min_cut);
  vis_env_->sdf_map_->posToIndex(vis_env_->sdf_map_->max_bound, max_cut);
  vis_env_->sdf_map_->boundIndex(min_cut);
  vis_env_->sdf_map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (vis_env_->sdf_map_->get_PredStates(Eigen::Vector3i(x, y, z)) == SDFMap::OCCUPANCY::PRED_INTERNAL)
        {
          Eigen::Vector3d pos;
          vis_env_->sdf_map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) < -1.0) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = "world";
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  internal_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishGlobal_Direction(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& direction)
{
  const double length = 3.0;

  visualization_msgs::Marker vp;
  vp.header.frame_id = "world";
  vp.header.stamp = ros::Time::now();
  vp.id = 0;
  vp.type = visualization_msgs::Marker::ARROW;
  vp.action = visualization_msgs::Marker::ADD;

  vp.pose.orientation.w = 1.0;
  vp.scale.x = 0.1;
  vp.scale.y = 0.1;
  vp.scale.z = 0.1;

  geometry_msgs::Point pt_;
  pt_.x = cur_pos(0);
  pt_.y = cur_pos(1);
  pt_.z = cur_pos(2);
  vp.points.push_back(pt_);

  pt_.x = cur_pos(0) + length*direction(0);
  pt_.y = cur_pos(1) + length*direction(1);
  pt_.z = cur_pos(2) + length*direction(2);
  vp.points.push_back(pt_);

  vp.color.r = 0.5;
  vp.color.g = 0.0;
  vp.color.b = 0.5;
  vp.color.a = 1.0;

  global_dir_pub_.publish(vp);
}

void PlanningVisualization::publishGlobalPlanner(const pcl::PointCloud<pcl::PointXYZ>& cloud, const vector<cluster_normal>& cnv, const vector<Eigen::Vector3d>& g_tour, Eigen::Vector3d& NBV,
const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& dir, const double& nbv_yaw)
{
  publishPredCloud(cloud);
  publishGlobal_VP(cnv);
  publishGlobal_Tour(g_tour, NBV, nbv_yaw);
  publishGlobal_Direction(cur_pos, dir);
}

void PlanningVisualization::publishGlobalCluster(const vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters, const vector<Eigen::Vector3d>& normals, const vector<Eigen::Vector3d>& centers)
{
  /* TODO */
  srand((int)time(0));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  colored_pcl_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB p;
  int red, blue, green;
  for (int i=0; i<clusters.size(); ++i)
  {
    red = rand()%255;
    blue = rand()%255;
    green = rand()%255;
    for (int j=0; j<clusters[i]->points.size(); ++j)
    {
      p.x = clusters[i]->points[j].x;
      p.y = clusters[i]->points[j].y;
      p.z = clusters[i]->points[j].z;
      // color
      p.r = red;
      p.g = green;
      p.b = blue;
      colored_pcl_ptr->points.push_back(p);
    }
  }

  colored_pcl_ptr->width = colored_pcl_ptr->points.size();
  colored_pcl_ptr->height = 1;
  colored_pcl_ptr->is_dense = true;
  colored_pcl_ptr->header.frame_id = "world";

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*colored_pcl_ptr, cloud_msg);
  global_c_pub_.publish(cloud_msg);

  visualization_msgs::MarkerArray cluster_normals;
  int counter_vpg = 0;
  for (int i=0; i<normals.size(); ++i)
  {
    visualization_msgs::Marker nm;
    nm.header.frame_id = "world";
    nm.header.stamp = ros::Time::now();
    nm.id = counter_vpg;
    nm.type = visualization_msgs::Marker::ARROW;
    nm.action = visualization_msgs::Marker::ADD;

    nm.pose.orientation.w = 1.0;
    nm.scale.x = 0.2;
    nm.scale.y = 0.1;
    nm.scale.z = 0.3;
    
    geometry_msgs::Point pt_;
    pt_.x = centers[i](0);
    pt_.y = centers[i](1);
    pt_.z = centers[i](2);
    nm.points.push_back(pt_);

    pt_.x = normals[i](0);
    pt_.y = normals[i](1);
    pt_.z = normals[i](2);
    nm.points.push_back(pt_);
    
    nm.color.r = 0.1;
    nm.color.g = 0.2;
    nm.color.b = 0.7;
    nm.color.a = 1.0;

    cluster_normals.markers.push_back(nm);
    counter_vpg++;
  }

  global_n_pub_.publish(cluster_normals);
}

void PlanningVisualization::publishLocal_Tour(const vector<Eigen::Vector3d>& local_vp)
{
  int counter_local = 0;
  // initial path
  visualization_msgs::MarkerArray local_tours;
  visualization_msgs::MarkerArray local_tours_ob;

  visualization_msgs::Marker local_tour_init;
  local_tour_init.header.frame_id = "world";
  local_tour_init.header.stamp = ros::Time::now();
  local_tour_init.id = counter_local;
  local_tour_init.type = visualization_msgs::Marker::LINE_STRIP;
  local_tour_init.action = visualization_msgs::Marker::ADD;

  local_tour_init.pose.orientation.w = 1.0;
  local_tour_init.scale.x = 0.2;

  local_tour_init.color.r = 0.5;
  local_tour_init.color.g = 0.7;
  local_tour_init.color.b = 0.6;
  local_tour_init.color.a = 1.0;

  for (int k=0; k<local_vp.size(); ++k)
  {
    geometry_msgs::Point p;
    p.x = local_vp[k][0];
    p.y = local_vp[k][1];
    p.z = local_vp[k][2];
    local_tour_init.points.push_back(p);
  }
  counter_local++;
  local_tours.markers.push_back(local_tour_init);
  // observation tour
  visualization_msgs::Marker local_tour_observe;
  local_tour_observe.header.frame_id = "world";
  local_tour_observe.header.stamp = ros::Time::now();
  local_tour_observe.id = counter_local;
  local_tour_observe.type = visualization_msgs::Marker::LINE_STRIP;
  local_tour_observe.action = visualization_msgs::Marker::ADD;

  local_tour_observe.pose.orientation.w = 1.0;
  local_tour_observe.scale.x = 0.35;

  local_tour_observe.color.r = 0.6;
  local_tour_observe.color.g = 0.2;
  local_tour_observe.color.b = 0.2;
  local_tour_observe.color.a = 1.0;

  for (int k=1; k<local_vp.size()-1; ++k)
  {
    geometry_msgs::Point p;
    p.x = local_vp[k][0];
    p.y = local_vp[k][1];
    p.z = local_vp[k][2];
    local_tour_observe.points.push_back(p);
  }
  counter_local++;
  local_tours_ob.markers.push_back(local_tour_observe);

  local_pub_.publish(local_tours);
  localob_pub_.publish(local_tours_ob);
}

void PlanningVisualization::publishLocal_VP(const vector<Eigen::Vector3d>& local_vp, const vector<double>& local_yaw)
{
  visualization_msgs::MarkerArray view_points;
  int counter_vpg = 0;
  const double length = 2.0;
  for (int i=0; i<local_vp.size(); ++i)
  {
    visualization_msgs::Marker vp;
    vp.header.frame_id = "world";
    vp.header.stamp = ros::Time::now();
    vp.id = counter_vpg;
    vp.type = visualization_msgs::Marker::ARROW;
    vp.action = visualization_msgs::Marker::ADD;

    vp.pose.orientation.w = 1.0;
    vp.scale.x = 0.2;
    vp.scale.y = 0.3;
    vp.scale.z = 0.4;

    geometry_msgs::Point pt_;
    pt_.x = local_vp[i](0);
    pt_.y = local_vp[i](1);
    pt_.z = local_vp[i](2);
    vp.points.push_back(pt_);

    pt_.x = local_vp[i](0) + length*cos(local_yaw[i]);
    pt_.y = local_vp[i](1) + length*sin(local_yaw[i]);
    pt_.z = local_vp[i](2);
    vp.points.push_back(pt_);

    vp.color.r = 0.5;
    vp.color.g = 0.0;
    vp.color.b = 0.5;
    vp.color.a = 1.0;

    view_points.markers.push_back(vp);
    counter_vpg++;
  }

  localVP_pub_.publish(view_points);
}

void PlanningVisualization::fillBasicInfo(visualization_msgs::Marker& mk, const Eigen::Vector3d& scale,
                                          const Eigen::Vector4d& color, const string& ns, const int& id,
                                          const int& shape) {
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = id;
  mk.ns = ns;
  mk.type = shape;

  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = scale[0];
  mk.scale.y = scale[1];
  mk.scale.z = scale[2];
}

void PlanningVisualization::fillGeometryInfo(visualization_msgs::Marker& mk,
                                             const vector<Eigen::Vector3d>& list) {
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
}

void PlanningVisualization::fillGeometryInfo(visualization_msgs::Marker& mk,
                                             const vector<Eigen::Vector3d>& list1,
                                             const vector<Eigen::Vector3d>& list2) {
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list1.size()); ++i) {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);
    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);
    mk.points.push_back(pt);
  }
}

void PlanningVisualization::drawBox(const Eigen::Vector3d& center, const Eigen::Vector3d& scale,
                                    const Eigen::Vector4d& color, const string& ns, const int& id,
                                    const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, scale, color, ns, id, visualization_msgs::Marker::CUBE);
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  mk.pose.position.x = center[0];
  mk.pose.position.y = center[1];
  mk.pose.position.z = center[2];
  mk.action = visualization_msgs::Marker::ADD;

  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale,
                                        const Eigen::Vector4d& color, const string& ns, const int& id,
                                        const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::SPHERE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // pub new marker
  fillGeometryInfo(mk, list);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawCubes(const vector<Eigen::Vector3d>& list, const double& scale,
                                      const Eigen::Vector4d& color, const string& ns, const int& id,
                                      const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::CUBE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // pub new marker
  fillGeometryInfo(mk, list);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawLines(const vector<Eigen::Vector3d>& list1,
                                      const vector<Eigen::Vector3d>& list2, const double& scale,
                                      const Eigen::Vector4d& color, const string& ns, const int& id,
                                      const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  if (list1.size() == 0) return;

  // pub new marker
  fillGeometryInfo(mk, list1, list2);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawLines(const vector<Eigen::Vector3d>& list, const double& scale,
                                      const Eigen::Vector4d& color, const string& ns, const int& id,
                                      const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  if (list.size() == 0) return;

  // split the single list into two
  vector<Eigen::Vector3d> list1, list2;
  for (int i = 0; i < list.size() - 1; ++i) {
    list1.push_back(list[i]);
    list2.push_back(list[i + 1]);
  }

  // pub new marker
  fillGeometryInfo(mk, list1, list2);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displaySphereList(const vector<Eigen::Vector3d>& list, double resolution,
                                              const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayCubeList(const vector<Eigen::Vector3d>& list, double resolution,
                                            const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayLineList(const vector<Eigen::Vector3d>& list1,
                                            const vector<Eigen::Vector3d>& list2, double line_width,
                                            const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = line_width;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list1.size()); ++i) {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);
    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawBsplinesPhase1(vector<NonUniformBspline>& bsplines, double size) {
  vector<Eigen::Vector3d> empty;

  for (int i = 0; i < last_bspline_phase1_num_; ++i) {
    displaySphereList(empty, size, Eigen::Vector4d(1, 0, 0, 1), BSPLINE + i % 100);
    displaySphereList(empty, size, Eigen::Vector4d(1, 0, 0, 1), BSPLINE_CTRL_PT + i % 100);
  }
  last_bspline_phase1_num_ = bsplines.size();

  for (int i = 0; i < bsplines.size(); ++i) {
    drawBspline(bsplines[i], size, getColor(double(i) / bsplines.size(), 0.2), false, 2 * size,
                getColor(double(i) / bsplines.size()), i);
  }
}

void PlanningVisualization::drawBsplinesPhase2(vector<NonUniformBspline>& bsplines, double size) {
  vector<Eigen::Vector3d> empty;

  for (int i = 0; i < last_bspline_phase2_num_; ++i) {
    drawSpheres(empty, size, Eigen::Vector4d(1, 0, 0, 1), "B-Spline", i, 0);
    drawSpheres(empty, size, Eigen::Vector4d(1, 0, 0, 1), "B-Spline", i + 50, 0);
    // displaySphereList(empty, size, Eigen::Vector4d(1, 0, 0, 1), BSPLINE + (50 + i) % 100);
    // displaySphereList(empty, size, Eigen::Vector4d(1, 0, 0, 1), BSPLINE_CTRL_PT + (50 + i) % 100);
  }
  last_bspline_phase2_num_ = bsplines.size();

  for (int i = 0; i < bsplines.size(); ++i) {
    drawBspline(bsplines[i], size, getColor(double(i) / bsplines.size(), 0.6), false, 1.5 * size,
                getColor(double(i) / bsplines.size()), i);
  }
}

void PlanningVisualization::drawBspline(NonUniformBspline& bspline, double size,
                                        const Eigen::Vector4d& color, bool show_ctrl_pts, double size2,
                                        const Eigen::Vector4d& color2, int id1) {
  if (bspline.getControlPoint().size() == 0) return;

  vector<Eigen::Vector3d> traj_pts;
  double tm, tmp;
  bspline.getTimeSpan(tm, tmp);

  for (double t = tm; t <= tmp; t += 0.01) {
    Eigen::Vector3d pt = bspline.evaluateDeBoor(t);
    traj_pts.push_back(pt);
  }
  // displaySphereList(traj_pts, size, color, BSPLINE + id1 % 100);
  drawSpheres(traj_pts, size, color, "B-Spline", id1, 0);

  // draw the control point
  if (show_ctrl_pts) {
    Eigen::MatrixXd ctrl_pts = bspline.getControlPoint();
    vector<Eigen::Vector3d> ctp;
    for (int i = 0; i < int(ctrl_pts.rows()); ++i) {
      Eigen::Vector3d pt = ctrl_pts.row(i).transpose();
      ctp.push_back(pt);
    }
    // displaySphereList(ctp, size2, color2, BSPLINE_CTRL_PT + id2 % 100);
    drawSpheres(ctp, size2, color2, "B-Spline", id1 + 50, 0);
  }
}

void PlanningVisualization::drawBspline_Local(vector<NonUniformBspline>& bspline, double size,
                                        const Eigen::Vector4d& color, bool show_ctrl_pts, double size2,
                                        const Eigen::Vector4d& color2, int id1)
{
  vector<Eigen::Vector3d> traj_pts;
  for (int i=0; i<bspline.size(); ++i)
  {
    double tm, tmp;
    bspline[i].getTimeSpan(tm, tmp);

    for (double t = tm; t <= tmp; t += 0.01) {
    Eigen::Vector3d pt = bspline[i].evaluateDeBoor(t);
    traj_pts.push_back(pt);
  }
  }
  // displaySphereList(traj_pts, size, color, BSPLINE + id1 % 100);
  drawSpheres(traj_pts, size, color, "B-Spline", id1, 0);
}

void PlanningVisualization::drawTopoGraph(list<GraphNode::Ptr>& graph, double point_size,
                                          double line_width, const Eigen::Vector4d& color1,
                                          const Eigen::Vector4d& color2, const Eigen::Vector4d& color3,
                                          int id) {
  // clear exsiting node and edge (drawn last time)
  vector<Eigen::Vector3d> empty;
  displaySphereList(empty, point_size, color1, GRAPH_NODE, 1);
  displaySphereList(empty, point_size, color1, GRAPH_NODE + 50, 1);
  displayLineList(empty, empty, line_width, color3, GRAPH_EDGE, 1);

  /* draw graph node */
  vector<Eigen::Vector3d> guards, connectors;
  for (list<GraphNode::Ptr>::iterator iter = graph.begin(); iter != graph.end(); ++iter) {
    if ((*iter)->type_ == GraphNode::Guard) {
      guards.push_back((*iter)->pos_);
    } else if ((*iter)->type_ == GraphNode::Connector) {
      connectors.push_back((*iter)->pos_);
    }
  }
  displaySphereList(guards, point_size, color1, GRAPH_NODE, 1);
  displaySphereList(connectors, point_size, color2, GRAPH_NODE + 50, 1);

  /* draw graph edge */
  vector<Eigen::Vector3d> edge_pt1, edge_pt2;
  for (list<GraphNode::Ptr>::iterator iter = graph.begin(); iter != graph.end(); ++iter) {
    for (int k = 0; k < (*iter)->neighbors_.size(); ++k) {
      edge_pt1.push_back((*iter)->pos_);
      edge_pt2.push_back((*iter)->neighbors_[k]->pos_);
    }
  }
  displayLineList(edge_pt1, edge_pt2, line_width, color3, GRAPH_EDGE, 1);
}

void PlanningVisualization::drawTopoPathsPhase2(vector<vector<Eigen::Vector3d>>& paths,
                                                double line_width) {
  // clear drawn paths
  Eigen::Vector4d color1(1, 1, 1, 1);
  for (int i = 0; i < last_topo_path1_num_; ++i) {
    vector<Eigen::Vector3d> empty;
    displayLineList(empty, empty, line_width, color1, SELECT_PATH + i % 100, 1);
    displaySphereList(empty, line_width, color1, PATH + i % 100, 1);
  }

  last_topo_path1_num_ = paths.size();

  // draw new paths
  for (int i = 0; i < paths.size(); ++i) {
    vector<Eigen::Vector3d> edge_pt1, edge_pt2;

    for (int j = 0; j < paths[i].size() - 1; ++j) {
      edge_pt1.push_back(paths[i][j]);
      edge_pt2.push_back(paths[i][j + 1]);
    }

    displayLineList(edge_pt1, edge_pt2, line_width, getColor(double(i) / (last_topo_path1_num_)),
                    SELECT_PATH + i % 100, 1);
  }
}

void PlanningVisualization::drawTopoPathsPhase1(vector<vector<Eigen::Vector3d>>& paths, double size) {
  // clear drawn paths
  Eigen::Vector4d color1(1, 1, 1, 1);
  for (int i = 0; i < last_topo_path2_num_; ++i) {
    vector<Eigen::Vector3d> empty;
    displayLineList(empty, empty, size, color1, FILTERED_PATH + i % 100, 1);
  }

  last_topo_path2_num_ = paths.size();

  // draw new paths
  for (int i = 0; i < paths.size(); ++i) {
    vector<Eigen::Vector3d> edge_pt1, edge_pt2;

    for (int j = 0; j < paths[i].size() - 1; ++j) {
      edge_pt1.push_back(paths[i][j]);
      edge_pt2.push_back(paths[i][j + 1]);
    }

    displayLineList(edge_pt1, edge_pt2, size, getColor(double(i) / (last_topo_path2_num_), 0.2),
                    FILTERED_PATH + i % 100, 1);
  }
}

void PlanningVisualization::drawGoal(Eigen::Vector3d goal, double resolution,
                                     const Eigen::Vector4d& color, int id) {
  vector<Eigen::Vector3d> goal_vec = { goal };
  displaySphereList(goal_vec, resolution, color, GOAL + id % 100);
}

void PlanningVisualization::drawGeometricPath(const vector<Eigen::Vector3d>& path, double resolution,
                                              const Eigen::Vector4d& color, int id) {
  displaySphereList(path, resolution, color, PATH + id % 100);
}

void PlanningVisualization::drawPolynomialTraj(PolynomialTraj poly_traj, double resolution,
                                               const Eigen::Vector4d& color, int id) {
  vector<Eigen::Vector3d> poly_pts;
  poly_traj.getSamplePoints(poly_pts);
  displaySphereList(poly_pts, resolution, color, POLY_TRAJ + id % 100);
}

void PlanningVisualization::drawPrediction(ObjPrediction pred, double resolution,
                                           const Eigen::Vector4d& color, int id) {
  ros::Time time_now = ros::Time::now();
  double start_time = (time_now - ObjHistory::global_start_time_).toSec();
  const double range = 5.6;

  vector<Eigen::Vector3d> traj;
  for (int i = 0; i < pred->size(); i++) {
    PolynomialPrediction poly = pred->at(i);
    if (!poly.valid()) continue;

    for (double t = start_time; t <= start_time + range; t += 0.8) {
      Eigen::Vector3d pt = poly.evaluateConstVel(t);
      traj.push_back(pt);
    }
  }
  displaySphereList(traj, resolution, color, id % 100, 2);
}

void PlanningVisualization::drawVisibConstraint(const Eigen::MatrixXd& ctrl_pts,
                                                const vector<Eigen::Vector3d>& block_pts) {
  int visible_num = ctrl_pts.rows() - block_pts.size();

  /* draw block points, their projection rays and visible pairs */
  vector<Eigen::Vector3d> pts1, pts2, pts3, pts4;
  int n = ctrl_pts.rows() - visible_num;

  for (int i = 0; i < n; ++i) {
    Eigen::Vector3d qb = block_pts[i];

    if (fabs(qb[2] + 10086) > 1e-3) {
      // compute the projection
      Eigen::Vector3d qi = ctrl_pts.row(i);
      Eigen::Vector3d qj = ctrl_pts.row(i + visible_num);
      Eigen::Vector3d dir = (qj - qi).normalized();
      Eigen::Vector3d qp = qi + dir * ((qb - qi).dot(dir));

      pts1.push_back(qb);
      pts2.push_back(qp);
      pts3.push_back(qi);
      pts4.push_back(qj);
    }
  }

  displayCubeList(pts1, 0.1, Eigen::Vector4d(1, 1, 0, 1), 0, 3);
  displaySphereList(pts4, 0.2, Eigen::Vector4d(0, 1, 0, 1), 1, 3);
  displayLineList(pts1, pts2, 0.015, Eigen::Vector4d(0, 1, 1, 1), 2, 3);
  displayLineList(pts3, pts4, 0.015, Eigen::Vector4d(0, 1, 0, 1), 3, 3);
}

void PlanningVisualization::drawVisibConstraint(const Eigen::MatrixXd& pts,
                                                const vector<VisiblePair>& pairs) {
  vector<Eigen::Vector3d> pts1, pts2, pts3, pts4;
  for (auto pr : pairs) {
    Eigen::Vector3d qb = pr.qb_;
    Eigen::Vector3d qi = pts.row(pr.from_);
    Eigen::Vector3d qj = pts.row(pr.to_);
    Eigen::Vector3d dir = (qj - qi).normalized();
    Eigen::Vector3d qp = qi + dir * ((qb - qi).dot(dir));
    pts1.push_back(qb);
    pts2.push_back(qp);
    pts3.push_back(qi);
    pts4.push_back(qj);
  }
  displayCubeList(pts1, 0.1, Eigen::Vector4d(1, 1, 0, 1), 0, 3);
  displaySphereList(pts4, 0.2, Eigen::Vector4d(0, 1, 0, 1), 1, 3);
  displayLineList(pts1, pts2, 0.015, Eigen::Vector4d(0, 1, 1, 1), 2, 3);
  displayLineList(pts3, pts4, 0.015, Eigen::Vector4d(0, 1, 0, 1), 3, 3);
}

void PlanningVisualization::drawViewConstraint(const ViewConstraint& vc) {
  if (vc.idx_ < 0) return;
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = 0;
  mk.type = visualization_msgs::Marker::ARROW;
  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.w = 1.0;
  mk.scale.x = 0.1;
  mk.scale.y = 0.2;
  mk.scale.z = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.5;
  mk.color.b = 0.0;
  mk.color.a = 1.0;

  geometry_msgs::Point pt;
  pt.x = vc.pt_[0];
  pt.y = vc.pt_[1];
  pt.z = vc.pt_[2];
  mk.points.push_back(pt);
  pt.x = vc.pt_[0] + vc.dir_[0];
  pt.y = vc.pt_[1] + vc.dir_[1];
  pt.z = vc.pt_[2] + vc.dir_[2];
  mk.points.push_back(pt);
  pubs_[3].publish(mk);

  vector<Eigen::Vector3d> pts = { vc.pcons_ };
  displaySphereList(pts, 0.2, Eigen::Vector4d(0, 1, 0, 1), 1, 3);
}

void PlanningVisualization::drawFrontier(const vector<vector<Eigen::Vector3d>>& frontiers) {
  for (int i = 0; i < frontiers.size(); ++i) {
    // displayCubeList(frontiers[i], 0.1, getColor(double(i) / frontiers.size(),
    // 0.4), i, 4);
    drawCubes(frontiers[i], 0.1, getColor(double(i) / frontiers.size(), 0.8), "frontier", i, 4);
  }

  vector<Eigen::Vector3d> frontier;
  for (int i = frontiers.size(); i < last_frontier_num_; ++i) {
    // displayCubeList(frontier, 0.1, getColor(1), i, 4);
    drawCubes(frontier, 0.1, getColor(1), "frontier", i, 4);
  }
  last_frontier_num_ = frontiers.size();
}

void PlanningVisualization::drawYawTraj_Local(vector<NonUniformBspline>& pos_list, vector<NonUniformBspline>& yaw_list, const double& dt)
{
  vector<Eigen::Vector3d> pts1, pts2;
  for (int i=0; i<pos_list.size(); ++i)
  {
    double duration = pos_list[i].getTimeSum();

    for (double tc = 0.0; tc <= duration + 1e-3; tc += dt)
    {
      Eigen::Vector3d pc = pos_list[i].evaluateDeBoorT(tc);
      pc[2] += 0.15;
      double yc = yaw_list[i].evaluateDeBoorT(tc)[0];
      Eigen::Vector3d dir(cos(yc), sin(yc), 0);
      Eigen::Vector3d pdir = pc + 1.0 * dir;
      pts1.push_back(pc);
      pts2.push_back(pdir);
    }
  }

  displayLineList(pts1, pts2, 0.04, Eigen::Vector4d(1, 0.5, 0, 1), 0, 5);
}

void PlanningVisualization::drawYawTraj(NonUniformBspline& pos, NonUniformBspline& yaw,
                                        const double& dt) {
  double duration = pos.getTimeSum();
  vector<Eigen::Vector3d> pts1, pts2;

  for (double tc = 0.0; tc <= duration + 1e-3; tc += dt) {
    Eigen::Vector3d pc = pos.evaluateDeBoorT(tc);
    pc[2] += 0.15;
    double yc = yaw.evaluateDeBoorT(tc)[0];
    Eigen::Vector3d dir(cos(yc), sin(yc), 0);
    Eigen::Vector3d pdir = pc + 1.0 * dir;
    pts1.push_back(pc);
    pts2.push_back(pdir);
  }
  displayLineList(pts1, pts2, 0.04, Eigen::Vector4d(1, 0.5, 0, 1), 0, 5);
}

void PlanningVisualization::drawYawPath(NonUniformBspline& pos, const vector<double>& yaw,
                                        const double& dt) {
  vector<Eigen::Vector3d> pts1, pts2;

  for (int i = 0; i < yaw.size(); ++i) {
    Eigen::Vector3d pc = pos.evaluateDeBoorT(i * dt);
    pc[2] += 0.3;
    Eigen::Vector3d dir(cos(yaw[i]), sin(yaw[i]), 0);
    Eigen::Vector3d pdir = pc + 1.0 * dir;
    pts1.push_back(pc);
    pts2.push_back(pdir);
  }
  displayLineList(pts1, pts2, 0.04, Eigen::Vector4d(1, 0, 1, 1), 1, 5);
}

Eigen::Vector4d PlanningVisualization::getColor(const double& h, double alpha) {
  double h1 = h;
  if (h1 < 0.0 || h1 > 1.0) {
    std::cout << "h out of range" << std::endl;
    h1 = 0.0;
  }

  double lambda;
  Eigen::Vector4d color1, color2;
  if (h1 >= -1e-4 && h1 < 1.0 / 6) {
    lambda = (h1 - 0.0) * 6;
    color1 = Eigen::Vector4d(1, 0, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 1, 1);
  } else if (h1 >= 1.0 / 6 && h1 < 2.0 / 6) {
    lambda = (h1 - 1.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 0, 1, 1);
  } else if (h1 >= 2.0 / 6 && h1 < 3.0 / 6) {
    lambda = (h1 - 2.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 1, 1);
  } else if (h1 >= 3.0 / 6 && h1 < 4.0 / 6) {
    lambda = (h1 - 3.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 0, 1);
  } else if (h1 >= 4.0 / 6 && h1 < 5.0 / 6) {
    lambda = (h1 - 4.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 1, 0, 1);
  } else if (h1 >= 5.0 / 6 && h1 <= 1.0 + 1e-4) {
    lambda = (h1 - 5.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 0, 1);
  }

  Eigen::Vector4d fcolor = (1 - lambda) * color1 + lambda * color2;
  fcolor(3) = alpha;

  return fcolor;
}
// PlanningVisualization::
}  // namespace fast_planner