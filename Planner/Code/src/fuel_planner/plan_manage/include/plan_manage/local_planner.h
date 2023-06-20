#ifndef _LOCAL_PLANNER_H_
#define _LOCAL_PLANNER_H_

#include <ros/ros.h>
#include <chrono>
#include <Eigen/Core>
#include <vector>
#include <active_perception/global_planner.h>

using namespace std;
using std::shared_ptr;
using std::unique_ptr;

class RayCaster;

namespace fast_planner
{

struct VP_global;
struct cluster_normal;

class LocalPlanner
{

private:
    /* Cluster and Normal */
    int sgn(double& x);
    void PCA_algo(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointXYZ& c, pcl::PointXYZ& pcZ, pcl::PointXYZ& pcY, pcl::PointXYZ& pcX, pcl::PointXYZ& pcZ_inv, pcl::PointXYZ& pcY_inv);
    pcl::PointCloud<pcl::PointXYZ>::Ptr condition_get(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& x_up, float& x_low,
    float& y_up, float& y_low, float& z_up, float& z_low);
    void get_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_vec, Eigen::Matrix3f& normal_vec, bool& flag);
    double pca_diameter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void uniform_cluster_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_t, 
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results_t);
    int normal_judge(Eigen::Vector3d& center, Eigen::Vector3d& pos, const float& coeff);
    int cal_visibility_cells(const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& set);
    void conditional_ec(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results);
    void clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results);
    void normal_info();
    /* Viewpoints Sampling */
    void sample_vp_pillar(cluster_normal& set, int& qualified_vp, double& head_bias, bool& img_type);
    /* Cost Tools */
    void CostMat();
    void fullCostMatrix(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw,
    Eigen::MatrixXd& mat);
    /* Graph Search */
    void find_order(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw,
    vector<int>& order);
    void view_graph_search(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw);
    /* Container */
    
    vector<cluster_normal> local_cnv;
    vector<cluster_normal> local_visit_cnv;
    /* Utils */
    shared_ptr<EDTEnvironment> env_;
    unique_ptr<RayCaster> raycheck_;
    /* Param */
    double uniform_range, ds_fac, normal_param; // uniform region max size
    double r_min, r_max, z_size, z_range_sp, phi_range, angle_step, theta_upper;
    int r_step, z_step; 
    double visible_lower;
    string tsp_dir_;
    double pseudo_bias;
    double local_interval;
    double local_normal_thre;
    double local_pca_thre;
    /* Data */
    Eigen::Vector4f centroid;// the centroid of local region
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cluster;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_region;
    pcl::PointCloud<pcl::PointNormal>::Ptr ds_normals;
    double region_angle;
public:
    LocalPlanner(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
    ~LocalPlanner();
    /* Data */
    vector<Eigen::Vector3d> local_points;
    vector<double> local_yaws;
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> local_cluster;
    vector<cluster_normal> local_qual_cnv;
    /* Perception */
    shared_ptr<PerceptionUtils> perceptron_;
    /* PathManager */
    void LocalPathManager(pcl::PointCloud<pcl::PointXYZ>::Ptr& local_region, const Eigen::Vector3d& local_normal, const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw);
};

}

#endif