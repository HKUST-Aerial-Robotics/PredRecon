#ifndef _GLOBAL_PLANNER_H_
#define _GLOBAL_PLANNER_H_

#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <memory>
#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/common/common.h>
#include <string>
#include <pcl/filters/conditional_removal.h>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>  
#include <pcl/surface/mls.h>         
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h> 
#include <pcl/io/ply_io.h>
#include <boost/thread/thread.hpp>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/convex_hull.h>
#include <math.h>

using std::shared_ptr;
using std::unique_ptr;
using std::normal_distribution;
using std::default_random_engine;
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
using namespace std;

class RayCaster;

namespace fast_planner
{
class EDTEnvironment;
class PerceptionUtils;

struct VP_global {
  // Position and heading
  Eigen::Vector3d pos_g;
  double yaw_g;
  // Fraction of the cluster that can be covered
  int visib_num_g;
  double vis_ratio_ref;
  double vis_ratio_src;
};
struct cluster_normal
{
  vector<Eigen::Vector3d> global_cells_;
  vector<Eigen::Vector3d> local_cells_;
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  vector<VP_global> vps_global;
  int id_g;
  list<vector<Eigen::Vector3d>> paths_;
  list<double> costs_;
};

class GlobalPlanner
{
  public:
    GlobalPlanner(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
    ~GlobalPlanner();

    bool global_path(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    vector<int>& order, const Eigen::Vector3d& cur_dir);
    void global_refine(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw);
    bool GlobalPathManager(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& visit, pcl::PointCloud<pcl::PointXYZ>::Ptr& cur_map, const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw, const Eigen::Vector3d& cur_g_dir);
    
    // NBV INFO
    bool NBV_res, dir_res;
    Eigen::Vector3d global_dir, global_next, global_cur_pos;
    VP_global NBV;
    int one_cluster_time;
    // Vis Data
    pcl::PointCloud<pcl::PointXYZ> copy_cloud_GHPR;
    vector<cluster_normal> uni_open_cnv;
    vector<vector<Eigen::Vector3d>> local_regions_;
    Eigen::Vector3d local_normal_;
    vector<Eigen::Vector3d> global_tour;
    vector<Eigen::Vector3d> refined_n_points; // [0] --> next_pos
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> uni_cluster;
    vector<Eigen::Vector3d> uni_center;
    vector<Eigen::Vector3d> cls_normal;
    // Finish flag
    bool finish_;

    shared_ptr<PerceptionUtils> percep_utils_;
    
  private:
    // Cluster and Normal Func
    int sgn(double& x);
    void PCA_algo(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointType& c, PointType& pcZ, PointType& pcY, PointType& pcX, PointType& pcZ_inv, PointType& pcY_inv);
    pcl::PointCloud<pcl::PointXYZ>::Ptr condition_get(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& x_up, float& x_low,
    float& y_up, float& y_low, float& z_up, float& z_low);
    void get_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_vec, Eigen::Matrix3f& normal_vec, bool& flag);
    double pca_diameter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void uniform_cluster_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_t, 
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results_t);
    // Tools Func
    int normal_judge(Eigen::Vector3d& center, Eigen::Vector3d& pos, const float& coeff);
    int cal_visibility_cells(const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& set);
    // Dual Sampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr surface_recon_visibility(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector4f& viewpoint);
    void conditional_ec(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results);
    void clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results);
    void normal_vis();
    void sample_vp_global_pillar(cluster_normal& set, int& qualified_vp);
    void global_cover_tp(cluster_normal& set, int& qualified_vp);
    // TSP Tool Func
    void CostMat();
    void fullCostMatrix(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    Eigen::MatrixXd& mat, const Eigen::Vector3d& g_dir);
    void get_GPath(const Eigen::Vector3d& pos, const vector<int>& ids, vector<Eigen::Vector3d>& path);
    void get_vpinfo(const Eigen::Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& yaws);

    // Utils
    shared_ptr<EDTEnvironment> edt_env_;
    unique_ptr<RayCaster> raycaster_;
    // Param
    double r_min, r_max, z_size, z_range, phi_sample, a_step, theta_thre; // sample radius and angle range
    int r_num, z_step;
    double min_dist, refine_radius, max_decay, downsample_c, downsample_e, grid_size, gamma, alpha, visible_ratio;
    int refine_num, top_view_num;
    double nbv_lb, nbv_ub;
    string tsp_dir;
    double dist_coefficient, gc_coefficient;
    double pca_diameter_thre;
    // prediction point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prediction;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_GHPR;
    // vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> uni_cluster;
    vector<Eigen::Matrix3f> uni_normal;
    Eigen::Vector4f viewpoint;
    Eigen::Vector4f all_viewpoint;
    // temp point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transform;
    pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inverse;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_cec;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals;
    // uniform cluster container
    vector<cluster_normal> uni_cnv;
    // global planning path
    vector<Eigen::Vector3d> global_points;
    vector<double> global_yaws;
    vector<vector<Eigen::Vector3d>> global_n_points;
    vector<vector<double>> global_n_yaws;
    vector<int> refined_ids;
    vector<Eigen::Vector3d> unrefined_points;
    vector<Eigen::Vector3d> refined_tour;
    vector<double> refined_n_yaws; // [0] --> next_yaw
};
}

#endif