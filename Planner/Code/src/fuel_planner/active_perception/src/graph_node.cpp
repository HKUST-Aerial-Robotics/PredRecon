#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

namespace fast_planner {
// Static data
double ViewNode::vm_;
double ViewNode::am_;
double ViewNode::yd_;
double ViewNode::ydd_;
double ViewNode::w_dir_;
shared_ptr<Astar> ViewNode::astar_;
shared_ptr<RayCaster> ViewNode::caster_;
shared_ptr<SDFMap> ViewNode::map_;

// Graph node for viewpoints planning
ViewNode::ViewNode(const Vector3d& p, const double& y) {
  pos_ = p;
  yaw_ = y;
  parent_ = nullptr;
  vel_.setZero();  // vel is zero by default, should be set explicitly
}

double ViewNode::costTo(const ViewNode::Ptr& node) {
  vector<Vector3d> path;
  double c = ViewNode::computeCost(pos_, node->pos_, yaw_, node->yaw_, vel_, yaw_dot_, path);
  // std::cout << "cost from " << id_ << " to " << node->id_ << " is: " << c << std::endl;
  return c;
}

double ViewNode::localCostTo(const ViewNode::Ptr& node)
{
  /* balance coefficient */
  double bc_ = 0.2;
  /* cost */
  vector<Vector3d> path;
  double c_;
  double kino_cost = ViewNode::compute_globalCost(pos_, node->pos_, yaw_, node->yaw_, vel_, yaw_dot_, path);
  if (vis_ratio_src_ < 2.0 && node->vis_ratio_src_ < 2.0)
  {
    double view_cost = ViewNode::LocalCost(pos_, node->pos_, yaw_, node->yaw_,
    vis_ratio_ref_, node->vis_ratio_src_, node->normal_, node->center_);

    c_ = bc_*view_cost + (1.0-bc_)*kino_cost;
  }  
  else
  {
    c_ = kino_cost;
    // cout << "kino_cost:" << c_ << endl; 
  }
  // cout << "cost:" << c_ << endl;
  return c_;
}

double ViewNode::searchPath(const Vector3d& p1, const Vector3d& p2, vector<Vector3d>& path) {
  // Try connect two points with straight line
  bool safe = true;
  Vector3i idx;
  caster_->input(p1, p2);
  while (caster_->nextId(idx)) {
    if (map_->getInflateOccupancy(idx) == 1 || map_->getOccupancy(idx) == SDFMap::UNKNOWN ||
        !map_->isInBox(idx)) {
      safe = false;
      break;
    }
  }
  if (safe) {
    path = { p1, p2 };
    return (p1 - p2).norm();
  }
  // Search a path using decreasing resolution
  vector<double> res = { 0.4 };
  for (int k = 0; k < res.size(); ++k) {
    astar_->reset();
    astar_->setResolution(res[k]);
    if (astar_->search(p1, p2) == Astar::REACH_END) {
      path = astar_->getPath();
      return astar_->pathLength(path);
    }
  }
  // Use Astar early termination cost as an estimate
  path = { p1, p2 };
  return 1000;
}

double ViewNode::search_globalPath(const Vector3d& p1, const Vector3d& p2, vector<Vector3d>& path)
{
  // Try connect two points with straight line
  bool safe = true;
  Vector3i idx;
  caster_->input(p1, p2);
  while (caster_->nextId(idx)) 
  {
    if (map_->get_PredStates(idx) == SDFMap::PRED_INTERNAL ||
        !map_->isInBox(idx)) {
      safe = false;
      break;
    }
  }

  if (safe) 
  {
    path = { p1, p2 };
    return (p1 - p2).norm();
  }
  // Search a path using decreasing resolution
  vector<double> res = { 2.0 };// speed up for TSP+refine process!!
  for (int k = 0; k < res.size(); ++k) 
  {
    astar_->reset();
    astar_->setResolution(res[k]);
    if (astar_->global_search(p1, p2) == Astar::REACH_END) {
      path = astar_->getPath();
      return astar_->pathLength(path);
    }
  }
  // Use Astar early termination cost as an estimate
  path = { p1, p2 };
  return 1000;
}

double ViewNode::computeCost(const Vector3d& p1, const Vector3d& p2, const double& y1, const double& y2,
                             const Vector3d& v1, const double& yd1, vector<Vector3d>& path) {
  // Cost of position change
  double pos_cost = ViewNode::searchPath(p1, p2, path) / vm_;

  // Consider velocity change
  if (v1.norm() > 1e-3) {
    Vector3d dir = (p2 - p1).normalized();
    Vector3d vdir = v1.normalized();
    double diff = acos(vdir.dot(dir));
    pos_cost += w_dir_ * diff;
  }

  // Cost of yaw change
  double diff = fabs(y2 - y1);
  diff = min(diff, 2 * M_PI - diff);
  double yaw_cost = diff / yd_;
  return max(pos_cost, yaw_cost);
}

// Compute local viewpoint quality cost
double ViewNode::LocalCost(const Vector3d& p1, const Vector3d& p2, const double& y1, const double& y2, 
  const double& vis1, const double& vis2, const Vector3d& normal, const Vector3d& center)
{
  /* visibility score */
  double vis_lb = 0.01;
  double f_vis = 0.5*(vis1 + vis2) + 1e-4; // belong to (0,1)
  f_vis = max(f_vis, vis_lb);
  // cout << "visibility:" << f_vis << endl;
  /* relative distance score */
  double dis_1 = (p1 - center).norm();
  double dis_2 = (p2 - center).norm();
  double min_dis = std::min(dis_1, dis_2);
  double max_dis = std::max(dis_1, dis_2);

  double f_rel = (min_dis/max_dis) + 1e-4;
  // cout << "relative_dist:" << f_rel << endl;
  /* angle score */
  double angle_best = 45.0*M_PI/180.0, prx_const = 0.2, ang_lower = 0.2;
  Vector3d vec1 = p1 - center;
  Vector3d vec2 = p2 - center;
  double angles = vec1.dot(vec2) / (vec1.norm()*vec2.norm());
  double angle_diff = 0.0;
  if (angles > 1.0 - 1e-4)
  {
    angle_diff = 0.0;
    ROS_WARN("Same pos!");
  }
  else
    angle_diff = fabs(acos(angles));
  angle_diff = min(angle_diff, 2 * M_PI - angle_diff);
  // cout << "angle_diff:" << angle_diff*180.0/M_PI << endl;
  double ang_ori = -pow(angle_diff-angle_best, 2)/(2.0*pow(prx_const,2));
  double f_ang = exp(ang_ori) + 1e-4;
  f_ang = max(f_ang, ang_lower);
  /* focus score */
  double foc_const = 0.2;
  Vector3d p1_yaw(cos(y1), sin(y1), 0.0);
  Vector3d p2_yaw(cos(y2), sin(y2), 0.0);
  double beta_1 = fabs(acos(p1_yaw.dot(vec1) / (p1_yaw.norm()*vec1.norm())));
  double beta_2 = fabs(acos(p2_yaw.dot(vec2) / (p2_yaw.norm()*vec2.norm())));
  double foc_1 = exp(-pow(beta_1,2)/(2.0*pow(foc_const,2)));
  double foc_2 = exp(-pow(beta_2,2)/(2.0*pow(foc_const,2)));

  double f_foc = foc_1 * foc_2;
  /* yaw change score */
  double diff = fabs(y2 - y1);
  double yaw_cost = 0.0;
  if (diff >= 5.0*M_PI/12.0)
    yaw_cost = 100.0*diff / yd_;
  else
    yaw_cost = diff / yd_;
  /* total cost */
  double quality_cost = 1.0/(f_vis*f_rel*f_ang) + yaw_cost;

  return quality_cost;
}

double ViewNode::compute_globalCost(const Vector3d& p1, const Vector3d& p2, const double& y1, const double& y2,
                            const Vector3d& v1, const double& yd1, vector<Vector3d>& path)
{
  // Cost of position change
  double pos_cost = ViewNode::search_globalPath(p1, p2, path) / vm_;

  // Consider velocity change
  if (v1.norm() > 1e-3) {
    Vector3d dir = (p2 - p1).normalized();
    Vector3d vdir = v1.normalized();
    double diff = acos(vdir.dot(dir));
    pos_cost += w_dir_ * diff;
  }

  // Cost of yaw change
  double diff = fabs(y2 - y1);
  diff = min(diff, 2 * M_PI - diff);
  double yaw_cost = diff / yd_;
  return max(pos_cost, yaw_cost);
}
}