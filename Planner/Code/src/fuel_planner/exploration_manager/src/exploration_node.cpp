#include <ros/ros.h>
#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/pred_recon_fsm.h>
#define BACKWARD_HAS_DW 1
#include <plan_manage/backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

using namespace fast_planner;

int main(int argc, char** argv) {
  ros::init(argc, argv, "exploration_node");
  ros::NodeHandle nh("~");

  PredReconFSM pr_fsm;
  pr_fsm.init(nh);

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
