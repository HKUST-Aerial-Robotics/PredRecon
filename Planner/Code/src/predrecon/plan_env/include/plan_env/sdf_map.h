#ifndef _SDF_MAP_H
#define _SDF_MAP_H

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <queue>
#include <ros/ros.h>
#include <tuple>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;

namespace cv {
class Mat;
}

class RayCaster;

namespace fast_planner {
struct MapParam;
struct MapData;
class MapROS;

class SDFMap {
public:
  SDFMap();
  ~SDFMap();

  enum OCCUPANCY { UNKNOWN, FREE, OCCUPIED, PRED_INTERNAL, PRED_EXTERNAL};

  std::vector<int> recon_poses_idx_;
  std::vector<int> recon_states_;
  Eigen::Vector3d min_bound, max_bound;

  void initMap(ros::NodeHandle& nh);
  void inputPointCloud(const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num,
                       const Eigen::Vector3d& camera_pos);
  void inputPredictionCloud(const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num,
                       const Eigen::Vector3d& viewpoint);

  void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  void boundIndex(Eigen::Vector3i& id);
  int toAddress(const Eigen::Vector3i& id);
  int toAddress(const int& x, const int& y, const int& z);
  bool isInMap(const Eigen::Vector3d& pos);
  bool isInMap(const Eigen::Vector3i& idx);
  bool isInBox(const Eigen::Vector3i& id);
  bool isInBox(const Eigen::Vector3d& pos);
  void boundBox(Eigen::Vector3d& low, Eigen::Vector3d& up);
  int getOccupancy(const Eigen::Vector3d& pos);
  int getOccupancy(const Eigen::Vector3i& id);
  void setOccupied(const Eigen::Vector3d& pos, const int& occ = 1);
  int getInflateOccupancy(const Eigen::Vector3d& pos);
  int getInflateOccupancy(const Eigen::Vector3i& id);
  double getDistance(const Eigen::Vector3d& pos);
  double getDistance(const Eigen::Vector3i& id);
  double getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad);
  void updateESDF3d();
  void resetBuffer();
  void resetBuffer(const Eigen::Vector3d& min, const Eigen::Vector3d& max);

  // prediction related
  int get_PredStates(const Eigen::Vector3d& pos);
  int get_PredStates(const Eigen::Vector3i& id);
  int get_qualified(const Eigen::Vector3i& id, const int& step);
  void reset_PredStates();

  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
  void getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);
  void getUpdatedBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, bool reset = false);
  double getResolution();
  int getVoxelNum();
  void getOccMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
  void inv_address(int& idx, Eigen::Vector3i& pos);
  void inflate_pred(Eigen::Vector3i& idx);
  bool getInternal_check(const Eigen::Vector3d& pos, const int& step);
  bool setcenter_check(const Eigen::Vector3d& pos, const int& step);
  bool setcenter_check(const Eigen::Vector3i& pos, const int& step);

private:
  void clearAndInflateLocalMap();
  void inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts);
  void setCacheOccupancy(const int& adr, const int& occ);
  Eigen::Vector3d closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt);
  template <typename F_get_val, typename F_set_val>
  void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

  unique_ptr<MapParam> mp_;
  unique_ptr<MapData> md_;
  unique_ptr<MapROS> mr_;
  unique_ptr<RayCaster> caster_;

  friend MapROS;

public:
  typedef std::shared_ptr<SDFMap> Ptr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MapParam {
  // map properties
  Eigen::Vector3d map_origin_, map_size_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;
  Eigen::Vector3i map_voxel_num_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_;
  double virtual_ceil_height_, ground_height_;
  Eigen::Vector3i box_min_, box_max_;
  Eigen::Vector3d box_mind_, box_maxd_;
  double default_dist_;
  bool optimistic_, signed_dist_;
  // map fusion
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  // occupancy probability
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;  // logit
  double max_ray_length_;
  double max_ray_length_prediction;
  double local_bound_inflate_;
  int local_map_margin_;
  double unknown_flag_;
  double view_thre_;
};

struct MapData {
  // main map data, occupancy of each voxel and Euclidean distance
  std::vector<double> occupancy_buffer_;
  std::vector<char> occupancy_buffer_inflate_;
  // prediction map states
  std::vector<char> occupancy_buffer_pred_;
  std::vector<double> distance_buffer_neg_;
  std::vector<double> distance_buffer_;
  std::vector<double> tmp_buffer1_;
  std::vector<double> tmp_buffer2_;
  // data for updating
  vector<short> count_hit_, count_miss_, count_hit_and_miss_;
  vector<char> flag_rayend_, flag_visited_;
  char raycast_num_;
  queue<int> cache_voxel_;
  Eigen::Vector3i local_bound_min_, local_bound_max_;
  Eigen::Vector3d update_min_, update_max_;
  bool reset_updated_box_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void SDFMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i)
    id(i) = floor((pos(i) - mp_->map_origin_(i)) * mp_->resolution_inv_);
}

inline void SDFMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i)
    pos(i) = (id(i) + 0.5) * mp_->resolution_ + mp_->map_origin_(i);
}

inline void SDFMap::boundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_->map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_->map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_->map_voxel_num_(2) - 1), 0);
  id = id1;
}

inline int SDFMap::toAddress(const int& x, const int& y, const int& z) {
  return x * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) + y * mp_->map_voxel_num_(2) + z;
}

inline int SDFMap::toAddress(const Eigen::Vector3i& id) {
  return toAddress(id[0], id[1], id[2]);
}

inline void SDFMap::inv_address(int& idx, Eigen::Vector3i& pos)
{
  int x_inv = int(idx/(mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2)));
  int y_inv = int((idx-x_inv*mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2))/mp_->map_voxel_num_(2));
  int z_inv = idx - x_inv * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) - y_inv * mp_->map_voxel_num_(2);

  pos(0) = x_inv;
  pos(1) = y_inv;
  pos(2) = z_inv;
}

inline bool SDFMap::isInMap(const Eigen::Vector3d& pos) {
  if (pos(0) < mp_->map_min_boundary_(0) + 1e-4 || pos(1) < mp_->map_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_->map_min_boundary_(2) + 1e-4)
    return false;
  if (pos(0) > mp_->map_max_boundary_(0) - 1e-4 || pos(1) > mp_->map_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_->map_max_boundary_(2) - 1e-4)
    return false;
  return true;
}

inline bool SDFMap::isInMap(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) return false;
  if (idx(0) > mp_->map_voxel_num_(0) - 1 || idx(1) > mp_->map_voxel_num_(1) - 1 ||
      idx(2) > mp_->map_voxel_num_(2) - 1)
    return false;
  return true;
}

inline bool SDFMap::isInBox(const Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) {
    if (id[i] < mp_->box_min_[i] || id[i] >= mp_->box_max_[i]) {
      return false;
    }
  }
  return true;
}

inline bool SDFMap::isInBox(const Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i) {
    if (pos[i] <= mp_->box_mind_[i] || pos[i] >= mp_->box_maxd_[i]) {
      return false;
    }
  }
  return true;
}

inline void SDFMap::boundBox(Eigen::Vector3d& low, Eigen::Vector3d& up) {
  for (int i = 0; i < 3; ++i) {
    low[i] = max(low[i], mp_->box_mind_[i]);
    up[i] = min(up[i], mp_->box_maxd_[i]);
  }
}

inline int SDFMap::getOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  double occ = md_->occupancy_buffer_[toAddress(id)];
  if (occ < mp_->clamp_min_log_ - 1e-3) return UNKNOWN;
  if (occ > mp_->min_occupancy_log_) return OCCUPIED;
  return FREE;
}

inline int SDFMap::getOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getOccupancy(id);
}

inline int SDFMap::get_PredStates(const Eigen::Vector3i& id)
{
  if (!isInMap(id)) return -1;
  if (md_->occupancy_buffer_pred_[toAddress(id)] == 0) return PRED_EXTERNAL;
  if (md_->occupancy_buffer_pred_[toAddress(id)] == 1) return PRED_INTERNAL;
}

inline int SDFMap::get_PredStates(const Eigen::Vector3d& pos)
{
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return get_PredStates(id);
}

inline void SDFMap::setOccupied(const Eigen::Vector3d& pos, const int& occ) {
  if (!isInMap(pos)) return;
  Eigen::Vector3i id;
  posToIndex(pos, id);
  md_->occupancy_buffer_inflate_[toAddress(id)] = occ;
}

inline int SDFMap::getInflateOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  return int(md_->occupancy_buffer_inflate_[toAddress(id)]);
}

inline int SDFMap::getInflateOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getInflateOccupancy(id);
}

inline double SDFMap::getDistance(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  return md_->distance_buffer_[toAddress(id)];
}

inline double SDFMap::getDistance(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getDistance(id);
}

inline void SDFMap::inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts) {
  int num = 0;

  /* ---------- + shape inflate ---------- */
  // for (int x = -step; x <= step; ++x)
  // {
  //   if (x == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1), pt(2));
  // }
  // for (int y = -step; y <= step; ++y)
  // {
  //   if (y == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1) + y, pt(2));
  // }
  // for (int z = -1; z <= 1; ++z)
  // {
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1), pt(2) + z);
  // }

  /* ---------- all inflate ---------- */
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
      }
}

inline void SDFMap::inflate_pred(Eigen::Vector3i& idx)
{
  md_->occupancy_buffer_pred_[toAddress(idx)] = 1;
  md_->occupancy_buffer_pred_[toAddress(Eigen::Vector3i(idx(0) + 1, idx(1) + 0, idx(2) + 0))] = 1;
  md_->occupancy_buffer_pred_[toAddress(Eigen::Vector3i(idx(0) + 0, idx(1) + 1, idx(2) + 0))] = 1;
  md_->occupancy_buffer_pred_[toAddress(Eigen::Vector3i(idx(0) + 0, idx(1) + 0, idx(2) + 1))] = 1;
  md_->occupancy_buffer_pred_[toAddress(Eigen::Vector3i(idx(0) - 1, idx(1) + 0, idx(2) + 0))] = 1;
  md_->occupancy_buffer_pred_[toAddress(Eigen::Vector3i(idx(0) + 0, idx(1) - 0, idx(2) + 0))] = 1;
  md_->occupancy_buffer_pred_[toAddress(Eigen::Vector3i(idx(0) + 0, idx(1) + 0, idx(2) - 1))] = 1;
}

inline bool SDFMap::getInternal_check(const Eigen::Vector3d& pos, const int& step)
{
  int flag = 0;
  Eigen::Vector3i inflate_idx, id;
  posToIndex(pos, id);
  int inflate_address;
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        inflate_idx(0) = id(0) + x;
        inflate_idx(1) = id(1) + y;
        inflate_idx(2) = id(2) + z;
        if (isInMap(inflate_idx))
        {
          inflate_address = toAddress(inflate_idx);
          if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
          {
            flag = 1;
            break;
          }
        }
      }
  
  return flag;
}

inline bool SDFMap::setcenter_check(const Eigen::Vector3d& pos, const int& step)
{
  int all = 0, hit = 0;
  bool flag = false;
  Eigen::Vector3i id, inflate_idx;
  posToIndex(pos, id);
  // x axis
  for (int x = -step; x<= step; ++x)
  {
    inflate_idx(0) = id(0) + 2*x;
    inflate_idx(1) = id(1);
    inflate_idx(2) = id(2);
    if (isInMap(inflate_idx))
    {
      all++;
      if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      {
        hit++;
      }
    }
  }
  // y axis
  for (int y = -step; y<= step; ++y)
  {
    inflate_idx(0) = id(0);
    inflate_idx(1) = id(1) + 2*y;
    inflate_idx(2) = id(2);
    if (isInMap(inflate_idx))
    {
      all++;
      if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      {
        hit++;
      }
    }
  }
  // z axis
  for (int z = -step; z<= step; ++z)
  {
    inflate_idx(0) = id(0);
    inflate_idx(1) = id(1);
    inflate_idx(2) = id(2) + 2*z;
    if (isInMap(inflate_idx))
    {
      all++;
      if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      {
        hit++;
      }
    }
  }

  double ratio = (double) hit/all;
  if (ratio > 0.8)
    flag = true;
  
  return flag;
}

inline bool SDFMap::setcenter_check(const Eigen::Vector3i& pos, const int& step)
{
  int all = 0, hit = 0;
  bool flag = false;
  Eigen::Vector3i id, inflate_idx;
  id = pos;
  // x axis
  for (int x = -step; x<= step; ++x)
  {
    inflate_idx(0) = id(0) + 2*x;
    inflate_idx(1) = id(1);
    inflate_idx(2) = id(2);
    if (isInMap(inflate_idx))
    {
      all++;
      if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      {
        hit++;
      }
    }
  }
  // y axis
  for (int y = -step; y<= step; ++y)
  {
    inflate_idx(0) = id(0);
    inflate_idx(1) = id(1) + 2*y;
    inflate_idx(2) = id(2);
    if (isInMap(inflate_idx))
    {
      all++;
      if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      {
        hit++;
      }
    }
  }
  // z axis
  for (int z = -step; z<= step; ++z)
  {
    inflate_idx(0) = id(0);
    inflate_idx(1) = id(1);
    inflate_idx(2) = id(2) + 2*z;
    if (isInMap(inflate_idx))
    {
      all++;
      if (get_PredStates(inflate_idx) == SDFMap::OCCUPANCY::PRED_INTERNAL)
      {
        hit++;
      }
    }
  }

  double ratio = (double) hit/all;
  if (ratio > 0.8)
    flag = true;
  
  return flag;
}
}
#endif