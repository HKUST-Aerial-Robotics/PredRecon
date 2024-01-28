#ifndef _PLANNING_VISUALIZATION_H_
#define _PLANNING_VISUALIZATION_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <bspline/non_uniform_bspline.h>
#include <iostream>
#include <path_searching/topo_prm.h>
#include <plan_env/obj_predictor.h>
#include <poly_traj/polynomial_traj.h>
#include <ros/ros.h>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <active_perception/traj_visibility.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <active_perception/global_planner.h>

using std::vector;
namespace fast_planner {
class EDTEnvironment;
struct cluster_normal;
class PlanningVisualization {
private:
  enum TRAJECTORY_PLANNING_ID {
    GOAL = 1,
    PATH = 200,
    BSPLINE = 300,
    BSPLINE_CTRL_PT = 400,
    POLY_TRAJ = 500
  };

  enum TOPOLOGICAL_PATH_PLANNING_ID {
    GRAPH_NODE = 1,
    GRAPH_EDGE = 100,
    RAW_PATH = 200,
    FILTERED_PATH = 300,
    SELECT_PATH = 400
  };

  /* data */
  /* visib_pub is seperated from previous ones for different info */
  ros::NodeHandle node;
  ros::Publisher traj_pub_;       // 0
  ros::Publisher topo_pub_;       // 1
  ros::Publisher predict_pub_;    // 2
  ros::Publisher visib_pub_;      // 3, visibility constraints
  ros::Publisher frontier_pub_;   // 4, frontier searching
  ros::Publisher yaw_pub_;        // 5, yaw trajectory
  ros::Publisher viewpoint_pub_;  // 6, viewpoint planning
  // GlobalPlanner
  ros::Publisher pred_pub_;       // 7, prediction point cloud
  ros::Publisher localReg_pub_;   // 8. local region point cloud
  ros::Publisher global_pub_;     // 9, global planning tour
  ros::Publisher vpg_pub_;        // 10, global planning viewpoints
  ros::Publisher internal_pub_;   // 11, global prediction internal
  ros::Publisher global_dir_pub_; // 12, global direction
  ros::Publisher global_c_pub_;   // 13, global cluster
  ros::Publisher global_n_pub_;   // 14, global normals
  // LocalPlanner
  ros::Publisher local_pub_;      // 15, local planning tour
  ros::Publisher localob_pub_;    // 15_1, local planning tour
  ros::Publisher localVP_pub_;    // 16, local planning viewpoints

  vector<ros::Publisher> pubs_;   //

  int last_topo_path1_num_;
  int last_topo_path2_num_;
  int last_bspline_phase1_num_;
  int last_bspline_phase2_num_;
  int last_frontier_num_;

public:
  PlanningVisualization(/* args */) {
  }
  ~PlanningVisualization() {
  }
  PlanningVisualization(ros::NodeHandle& nh);
  void set_env(const shared_ptr<EDTEnvironment>& edt);
  shared_ptr<EDTEnvironment> vis_env_;

  // GlobalPlanner
  void publishPredCloud(const pcl::PointCloud<pcl::PointXYZ>& input_cloud);
  void publishLocalRegion(const pcl::PointCloud<pcl::PointXYZ>& input_cloud);
  void publishGlobal_VP(const vector<cluster_normal>& cluster);
  void publishGlobal_Tour(const vector<Eigen::Vector3d>& global_init_tour, const Eigen::Vector3d& refined_vp, const double& refined_yaw);
  void publishGlobal_Internal();
  void publishGlobal_Direction(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& direction);
  void publishGlobalPlanner(const pcl::PointCloud<pcl::PointXYZ>& cloud, const vector<cluster_normal>& cnv, const vector<Eigen::Vector3d>& g_tour, Eigen::Vector3d& NBV,
  const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& dir, const double& nbv_yaw);

  void publishGlobalCluster(const vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters, const vector<Eigen::Vector3d>& normals, const vector<Eigen::Vector3d>& centers);
  
  // LocalPlanner
  void publishLocal_Tour(const vector<Eigen::Vector3d>& local_vp);
  void publishLocal_VP(const vector<Eigen::Vector3d>& local_vp, const vector<double>& local_yaw);

  // new interface
  void fillBasicInfo(visualization_msgs::Marker& mk, const Eigen::Vector3d& scale,
                     const Eigen::Vector4d& color, const string& ns, const int& id, const int& shape);
  void fillGeometryInfo(visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list);
  void fillGeometryInfo(visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list1,
                        const vector<Eigen::Vector3d>& list2);

  void drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale,
                   const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);
  void drawCubes(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
                 const string& ns, const int& id, const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
                 const double& scale, const Eigen::Vector4d& color, const string& ns, const int& id,
                 const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
                 const string& ns, const int& id, const int& pub_id);
  void drawBox(const Eigen::Vector3d& center, const Eigen::Vector3d& scale, const Eigen::Vector4d& color,
               const string& ns, const int& id, const int& pub_id);

  // Deprecated
  // draw basic shapes
  void displaySphereList(const vector<Eigen::Vector3d>& list, double resolution,
                         const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayCubeList(const vector<Eigen::Vector3d>& list, double resolution,
                       const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
                       double line_width, const Eigen::Vector4d& color, int id, int pub_id = 0);
  // draw a piece-wise straight line path
  void drawGeometricPath(const vector<Eigen::Vector3d>& path, double resolution,
                         const Eigen::Vector4d& color, int id = 0);
  // draw a polynomial trajectory
  void drawPolynomialTraj(PolynomialTraj poly_traj, double resolution, const Eigen::Vector4d& color,
                          int id = 0);
  // draw a bspline trajectory
  void drawBspline(NonUniformBspline& bspline, double size, const Eigen::Vector4d& color,
                   bool show_ctrl_pts = false, double size2 = 0.1,
                   const Eigen::Vector4d& color2 = Eigen::Vector4d(1, 1, 0, 1), int id = 0);
  void drawBspline_Local(vector<NonUniformBspline>& bspline, double size, const Eigen::Vector4d& color,
                   bool show_ctrl_pts = false, double size2 = 0.1,
                   const Eigen::Vector4d& color2 = Eigen::Vector4d(1, 1, 0, 1), int id = 0);
  // draw a set of bspline trajectories generated in different phases
  void drawBsplinesPhase1(vector<NonUniformBspline>& bsplines, double size);
  void drawBsplinesPhase2(vector<NonUniformBspline>& bsplines, double size);
  // draw topological graph and paths
  void drawTopoGraph(list<GraphNode::Ptr>& graph, double point_size, double line_width,
                     const Eigen::Vector4d& color1, const Eigen::Vector4d& color2,
                     const Eigen::Vector4d& color3, int id = 0);
  void drawTopoPathsPhase1(vector<vector<Eigen::Vector3d>>& paths, double line_width);
  void drawTopoPathsPhase2(vector<vector<Eigen::Vector3d>>& paths, double line_width);

  void drawGoal(Eigen::Vector3d goal, double resolution, const Eigen::Vector4d& color, int id = 0);
  void drawPrediction(ObjPrediction pred, double resolution, const Eigen::Vector4d& color, int id = 0);

  Eigen::Vector4d getColor(const double& h, double alpha = 1.0);

  typedef std::shared_ptr<PlanningVisualization> Ptr;

  // SECTION developing
  void drawVisibConstraint(const Eigen::MatrixXd& ctrl_pts, const vector<Eigen::Vector3d>& block_pts);
  void drawVisibConstraint(const Eigen::MatrixXd& pts, const vector<VisiblePair>& pairs);
  void drawViewConstraint(const ViewConstraint& vc);
  void drawFrontier(const vector<vector<Eigen::Vector3d>>& frontiers);
  void drawYawTraj_Local(vector<NonUniformBspline>& pos_list, vector<NonUniformBspline>& yaw_list, const double& dt);
  void drawYawTraj(NonUniformBspline& pos, NonUniformBspline& yaw, const double& dt);
  void drawYawPath(NonUniformBspline& pos, const vector<double>& yaw, const double& dt);
};
}  // namespace fast_planner
#endif