#ifndef _VISUALIZE_UTILS_H_
#define _VISUALIZE_UTILS_H_

#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Eigen>

#define FirstTraj 1
#define BestTraj 2
#define FinalTraj 3
#define TrackedTraj 4
#define TreeTraj 5
#define OptimizedTraj 6
#define FORWARD_REACHABLE_POS 1
#define BACKWARD_REACHABLE_POS 2

typedef Eigen::Matrix<double, 9, 1> StatePVA;

class Color : public std_msgs::ColorRGBA {
 public:
  Color() : std_msgs::ColorRGBA() {}
  Color(double red, double green, double blue) : Color(red, green, blue, 1.0) {}
  Color(double red, double green, double blue, double alpha) : Color() {
    r = red;
    g = green;
    b = blue;
    a = alpha;
  }

  static const Color White() { return Color(1.0, 1.0, 1.0); }
  static const Color Black() { return Color(0.0, 0.0, 0.0); }
  static const Color Gray() { return Color(0.5, 0.5, 0.5); }
  static const Color Red() { return Color(1.0, 0.0, 0.0); }
  static const Color Green() { return Color(0.0, 0.7, 0.0); }
  static const Color Blue() { return Color(0.0, 0.0, 1.0); }
  static const Color SteelBlue() { return Color(0.4, 0.7, 1.0); }
  static const Color Yellow() { return Color(1.0, 1.0, 0.0); }
  static const Color Orange() { return Color(1.0, 0.5, 0.0); }
  static const Color Purple() { return Color(0.5, 0.0, 1.0); }
  static const Color Chartreuse() { return Color(0.5, 1.0, 0.0); }
  static const Color Teal() { return Color(0.0, 1.0, 1.0); }
  static const Color Pink() { return Color(1.0, 0.0, 0.5); }
};

class VisualRviz{
public:
    VisualRviz();
    VisualRviz(const ros::NodeHandle &nh);
    
    void visualizeCollision(const Eigen::Vector3d &collision, ros::Time local_time);
    void visualizeKnots(const std::vector<Eigen::Vector3d> &knots, ros::Time local_time);
    void visualizeStates(const std::vector<StatePVA> &x, int trajectory_type, ros::Time local_time);
    void visualizeSampledState(const std::vector<StatePVA> &nodes, ros::Time local_time);
    void visualizeValidSampledState(const std::vector<StatePVA> &nodes, ros::Time local_time);
    void visualizeStartAndGoal(StatePVA start, StatePVA goal, ros::Time local_time);
    void visualizeTopo(const std::vector<Eigen::Vector3d>& p_head, const std::vector<Eigen::Vector3d>& tracks, ros::Time local_time);
    void visualizeOrphans(const std::vector<StatePVA>& ophs, ros::Time local_time);
    void visualizeReachPos(int type, const Eigen::Vector3d& center, const double& diam, ros::Time local_time);
    void visualizeReplanDire(const Eigen::Vector3d &pos, const Eigen::Vector3d &dire, ros::Time local_time);

    typedef std::shared_ptr<VisualRviz> Ptr;

private:
    ros::NodeHandle nh_;
    ros::Publisher rand_sample_pos_point_pub_;
    ros::Publisher rand_sample_vel_vec_pub_;
    ros::Publisher rand_sample_acc_vec_pub_;
    ros::Publisher tree_traj_pos_point_pub_;
    ros::Publisher tree_traj_vel_vec_pub_;
    ros::Publisher tree_traj_acc_vec_pub_;
    ros::Publisher final_traj_pos_point_pub_;
    ros::Publisher final_traj_vel_vec_pub_;
    ros::Publisher final_traj_acc_vec_pub_;
    ros::Publisher first_traj_pos_point_pub_;
    ros::Publisher first_traj_vel_vec_pub_;
    ros::Publisher first_traj_acc_vec_pub_;
    ros::Publisher best_traj_pos_point_pub_;
    ros::Publisher best_traj_vel_vec_pub_;
    ros::Publisher best_traj_acc_vec_pub_;
    ros::Publisher tracked_traj_pos_point_pub_;
    ros::Publisher optimized_traj_pos_point_pub_;
    ros::Publisher optimized_traj_vel_vec_pub_;
    ros::Publisher optimized_traj_acc_vec_pub_;
    ros::Publisher start_and_goal_pub_;
    ros::Publisher topo_pub_;
    ros::Publisher orphans_pos_pub_;
    ros::Publisher orphans_vel_vec_pub_;
    ros::Publisher fwd_reachable_pos_pub_;
    ros::Publisher bwd_reachable_pos_pub_;
    ros::Publisher knots_pub_;
    ros::Publisher collision_pub_;
    ros::Publisher replan_direction_pub_;
};


#endif
