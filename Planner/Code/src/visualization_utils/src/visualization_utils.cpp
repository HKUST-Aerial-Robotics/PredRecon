#include "visualization_utils/visualization_utils.h"
#include <ros/ros.h>
#include <queue>
#include <string>

VisualRviz::VisualRviz(const ros::NodeHandle &nh) : nh_(nh)
{
    rand_sample_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("rand_sample_pos_points", 1);
    rand_sample_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("rand_sample_vel_vecs", 1);
    rand_sample_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("rand_sample_acc_vecs", 1);
    tree_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_traj_pos", 1);
    tree_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_traj_vel", 1);
    tree_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("tree_traj_acc", 1);
    final_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("a_traj_pos", 1);
    final_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("a_traj_vel", 1);
    final_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("a_traj_acc", 1);
    first_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("first_traj_pos", 1);
    first_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("first_traj_vel", 1);
    first_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("first_traj_acc", 1);
    best_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("best_traj_pos", 1);
    best_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("best_traj_vel", 1);
    best_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("best_traj_acc", 1);
    tracked_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("tracked_traj_acc", 1);
    optimized_traj_pos_point_pub_ = nh_.advertise<visualization_msgs::Marker>("optimized_traj_pos", 1);
    optimized_traj_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("optimized_traj_vel", 1);
    optimized_traj_acc_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("optimized_traj_acc", 1);
    start_and_goal_pub_ = nh_.advertise<visualization_msgs::Marker>("start_and_goal", 1);
    topo_pub_ = nh_.advertise<visualization_msgs::Marker>("topo", 1);
    orphans_pos_pub_ = nh_.advertise<visualization_msgs::Marker>("orphan_pos", 1);
    orphans_vel_vec_pub_ = nh_.advertise<visualization_msgs::Marker>("orphan_vel", 1);
    fwd_reachable_pos_pub_ = nh_.advertise<visualization_msgs::Marker>("fwd_reachable_pos", 1);
    bwd_reachable_pos_pub_ = nh_.advertise<visualization_msgs::Marker>("bwd_reachable_pos", 1);
    knots_pub_ = nh_.advertise<visualization_msgs::Marker>("knots", 1);
    collision_pub_ = nh_.advertise<visualization_msgs::Marker>("collision", 1);
    replan_direction_pub_ = nh_.advertise<visualization_msgs::Marker>("replan_direction", 1);
}

void VisualRviz::visualizeStates(const std::vector<StatePVA> &x, int trajectory_type, ros::Time local_time)
{
    if (x.empty()) return;
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;
    geometry_msgs::Point p, a;
    for (size_t i = 0; i < x.size(); ++i)
    {
        p.x = x[i](0, 0);
        p.y = x[i](1, 0);
        p.z = x[i](2, 0);
        a.x = x[i](0, 0);
        a.y = x[i](1, 0);
        a.z = x[i](2, 0);
        pos_point.points.push_back(p);

        vel_vec.points.push_back(p);
        p.x += x[i](3, 0);
        p.y += x[i](4, 0);
        p.z += x[i](5, 0);
        vel_vec.points.push_back(p);

        acc_vec.points.push_back(a);
        a.x += x[i](6, 0);
        a.y += x[i](7, 0);
        a.z += x[i](8, 0);
        acc_vec.points.push_back(a);
    }
    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 100;
    pos_point.type = visualization_msgs::Marker::LINE_STRIP;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;

    vel_vec.header.frame_id = "world_enu";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "traj";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 200;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;

    acc_vec.header.frame_id = "world_enu";
    acc_vec.header.stamp = local_time;
    acc_vec.ns = "traj";
    acc_vec.action = visualization_msgs::Marker::ADD;
    acc_vec.lifetime = ros::Duration(0);
    acc_vec.pose.orientation.w = 1.0;
    acc_vec.id = 300;
    acc_vec.type = visualization_msgs::Marker::LINE_LIST;
    acc_vec.scale.x = 0.03;
    
    switch (trajectory_type) 
    {
    case FirstTraj:
        pos_point.color = Color::Blue();
        pos_point.color.a = 1.0;
        vel_vec.color = Color::Blue();
        vel_vec.color.a = 1.0;
        acc_vec.color = Color::Blue();
        acc_vec.color.a = 0.2;
        first_traj_pos_point_pub_.publish(pos_point);
        first_traj_vel_vec_pub_.publish(vel_vec);
        first_traj_acc_vec_pub_.publish(acc_vec);
        break;

    case BestTraj:
        pos_point.color = Color::Red();
        pos_point.color.a = 1.0;
        vel_vec.color = Color::Red();
        vel_vec.color.a = 1.0;
        acc_vec.color = Color::Red();
        acc_vec.color.a = 0.2;
        best_traj_pos_point_pub_.publish(pos_point);
        best_traj_vel_vec_pub_.publish(vel_vec);
        best_traj_acc_vec_pub_.publish(acc_vec);
        break;

    case FinalTraj:
        pos_point.color = Color::Purple();
        pos_point.color.a = 1.0;
        vel_vec.color = Color::Purple();
        vel_vec.color.a = 1.0;
        acc_vec.color = Color::Purple();
        acc_vec.color.a = 0.2;
        final_traj_pos_point_pub_.publish(pos_point);
        final_traj_vel_vec_pub_.publish(vel_vec);
        final_traj_acc_vec_pub_.publish(acc_vec);
        break;

    case TrackedTraj:
        pos_point.color = Color::Black();
        pos_point.color.a = 1.0;
        vel_vec.color = Color::Black();
        vel_vec.color.a = 1.0;
        acc_vec.color = Color::Black();
        acc_vec.color.a = 0.2;
        tracked_traj_pos_point_pub_.publish(pos_point);
        break;

    case OptimizedTraj:
        pos_point.color = Color::Orange();
        pos_point.color.a = 1.0;
        vel_vec.color = Color::Orange();
        vel_vec.color.a = 1.0;
        acc_vec.color = Color::Orange();
        acc_vec.color.a = 0.2;
        optimized_traj_pos_point_pub_.publish(pos_point);
        optimized_traj_vel_vec_pub_.publish(vel_vec);
        optimized_traj_acc_vec_pub_.publish(acc_vec);
        break;

    case TreeTraj:
        pos_point.color = Color::SteelBlue();
        pos_point.color.a = 1.0;
        pos_point.scale.x = 0.03;
        pos_point.scale.y = 0.03;
        pos_point.scale.z = 0.03;
        pos_point.type = visualization_msgs::Marker::POINTS;
        vel_vec.color = Color::SteelBlue();
        vel_vec.color.a = 1.0;
        acc_vec.color = Color::SteelBlue();
        acc_vec.color.a = 0.3;
        tree_traj_pos_point_pub_.publish(pos_point);
        tree_traj_vel_vec_pub_.publish(vel_vec);
        tree_traj_acc_vec_pub_.publish(acc_vec);
        break;
    }
}

void VisualRviz::visualizeKnots(const std::vector<Eigen::Vector3d> &knots, ros::Time local_time)
{
    if (knots.empty()) return;
    visualization_msgs::Marker pos_point;
    geometry_msgs::Point p;
    for (size_t i = 0; i < knots.size(); ++i)
    {
        p.x = knots[i](0);
        p.y = knots[i](1);
        p.z = knots[i](2);
        pos_point.points.push_back(p);
    }
    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 400;
    pos_point.type = visualization_msgs::Marker::SPHERE_LIST;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color = Color::Blue();
    pos_point.color.a = 1.0;
    knots_pub_.publish(pos_point);
}

void VisualRviz::visualizeCollision(const Eigen::Vector3d &collision, ros::Time local_time)
{
    visualization_msgs::Marker pos_point;
    geometry_msgs::Point p;
    p.x = collision(0);
    p.y = collision(1);
    p.z = collision(2);
    pos_point.points.push_back(p);
    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "colllision";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 400;
    pos_point.type = visualization_msgs::Marker::SPHERE_LIST;
    pos_point.scale.x = 0.2;
    pos_point.scale.y = 0.2;
    pos_point.scale.z = 0.2;
    pos_point.color = Color::Red();
    pos_point.color.a = 1.0;
    collision_pub_.publish(pos_point);
}

void VisualRviz::visualizeSampledState(const std::vector<StatePVA> &nodes, ros::Time local_time)
{
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;

    geometry_msgs::Point p;
    for (const auto &node : nodes)
    {
        p.x = node[0];
        p.y = node[1];
        p.z = node[2];
        pos_point.points.push_back(p);
        vel_vec.points.push_back(p);
        p.x += node[3] / 2.0;
        p.y += node[4] / 2.0;
        p.z += node[5] / 2.0;
        vel_vec.points.push_back(p);
    }
    
    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "sample";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 10;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.07;
    pos_point.scale.y = 0.07;
    pos_point.scale.z = 0.07;
    pos_point.color = Color::Green();
    pos_point.color.a = 1.0;

    vel_vec.header.frame_id = "world_enu";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "sample";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 20;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.03;
    vel_vec.color = Color::Red();
    vel_vec.color.a = 1.0;

    rand_sample_pos_point_pub_.publish(pos_point);
    rand_sample_vel_vec_pub_.publish(vel_vec);
}

void VisualRviz::visualizeValidSampledState(const std::vector<StatePVA> &nodes, ros::Time local_time)
{
    visualization_msgs::Marker pos_point, vel_vec, acc_vec;

    geometry_msgs::Point p;
    for (const auto &node : nodes)
    {
        p.x = node[0];
        p.y = node[1];
        p.z = node[2];
        pos_point.points.push_back(p);
    }
    
    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "sample";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 10;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.07;
    pos_point.scale.y = 0.07;
    pos_point.scale.z = 0.07;
    pos_point.color = Color::Red();
    pos_point.color.a = 1.0;

    rand_sample_vel_vec_pub_.publish(pos_point);
}

void VisualRviz::visualizeStartAndGoal(StatePVA start, StatePVA goal, ros::Time local_time)
{
    visualization_msgs::Marker pos_point;

    geometry_msgs::Point p;
    p.x = start[0];
    p.y = start[1];
    p.z = start[2];
    pos_point.points.push_back(p);
    p.x = goal[0];
    p.y = goal[1];
    p.z = goal[2];
    pos_point.points.push_back(p);

    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "s_g";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 11;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.4;
    pos_point.scale.y = 0.4;
    pos_point.scale.z = 0.4;
    pos_point.color = Color::Purple();
    pos_point.color.a = 1.0;

    start_and_goal_pub_.publish(pos_point);
}

void VisualRviz::visualizeTopo(const std::vector<Eigen::Vector3d> &p_head,
                               const std::vector<Eigen::Vector3d> &tracks,
                               ros::Time local_time)
{
    if (tracks.empty() || p_head.empty() || tracks.size() != p_head.size())
        return;

    visualization_msgs::Marker topo;
    geometry_msgs::Point p;

    for (int i = 0; i < tracks.size(); i++)
    {
        p.x = p_head[i][0];
        p.y = p_head[i][1];
        p.z = p_head[i][2];
        topo.points.push_back(p);
        p.x += tracks[i][0];
        p.y += tracks[i][1];
        p.z += tracks[i][2];
        topo.points.push_back(p);
    }

    topo.header.frame_id = "world_enu";
    topo.header.stamp = local_time;
    topo.ns = "topo";
    topo.action = visualization_msgs::Marker::ADD;
    topo.lifetime = ros::Duration(0);
    topo.pose.orientation.w = 1.0;
    topo.id = 117;
    topo.type = visualization_msgs::Marker::LINE_LIST;
    topo.scale.x = 0.15;
    topo.color = Color::Green();
    topo.color.a = 1.0;

    topo_pub_.publish(topo);
}

void VisualRviz::visualizeReplanDire(const Eigen::Vector3d &pos, const Eigen::Vector3d &dire, ros::Time local_time)
{
    visualization_msgs::Marker pos_point;

    geometry_msgs::Point p;
    p.x = pos[0];
    p.y = pos[1];
    p.z = pos[2];
    pos_point.points.push_back(p);
    p.x += dire[0];
    p.y += dire[1];
    p.z += dire[2];
    pos_point.points.push_back(p);

    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "traj";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 11;
    pos_point.type = visualization_msgs::Marker::ARROW;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color = Color::Purple();
    pos_point.color.a = 1.0;

    replan_direction_pub_.publish(pos_point);
}

void VisualRviz::visualizeReachPos(int type, const Eigen::Vector3d& center, const double& diam, ros::Time local_time)
{
    visualization_msgs::Marker pos;
    pos.header.frame_id = "world_enu";
    pos.header.stamp = local_time;
    pos.ns = "reachable_pos";
    pos.type = visualization_msgs::Marker::SPHERE;
    pos.action = visualization_msgs::Marker::ADD;
    pos.pose.position.x = center[0];
    pos.pose.position.y = center[1];
    pos.pose.position.z = center[2];
    pos.scale.x = diam;
    pos.scale.y = diam;
    pos.scale.z = diam;
    if (type == FORWARD_REACHABLE_POS)
    {
        pos.id = 1;
        pos.color.a = 0.3; 
        pos.color = Color::Green();
        fwd_reachable_pos_pub_.publish(pos);
    }
    else if(type == BACKWARD_REACHABLE_POS)
    {
        pos.id = 2;
        pos.color.a = 0.3; 
        pos.color = Color::Red();
        bwd_reachable_pos_pub_.publish(pos);
    }
    
}

void VisualRviz::visualizeOrphans(const std::vector<StatePVA> &ophs, ros::Time local_time)
{
    visualization_msgs::Marker pos_point, vel_vec;
    geometry_msgs::Point p;

    for (int i = 0; i < ophs.size(); i++)
    {
        p.x = ophs[i](0, 0);
        p.y = ophs[i](1, 0);
        p.z = ophs[i](2, 0);
        pos_point.points.push_back(p);

        vel_vec.points.push_back(p);
        p.x += ophs[i](3, 0) / 10.0;
        p.y += ophs[i](4, 0) / 10.0;
        p.z += ophs[i](5, 0) / 10.0;
        vel_vec.points.push_back(p);
    }

    pos_point.header.frame_id = "world_enu";
    pos_point.header.stamp = local_time;
    pos_point.ns = "orphan";
    pos_point.action = visualization_msgs::Marker::ADD;
    pos_point.lifetime = ros::Duration(0);
    pos_point.pose.orientation.w = 1.0;
    pos_point.id = 43;
    pos_point.type = visualization_msgs::Marker::POINTS;
    pos_point.scale.x = 0.1;
    pos_point.scale.y = 0.1;
    pos_point.scale.z = 0.1;
    pos_point.color = Color::Green();
    pos_point.color.a = 1.0;

    vel_vec.header.frame_id = "world_enu";
    vel_vec.header.stamp = local_time;
    vel_vec.ns = "orphan";
    vel_vec.action = visualization_msgs::Marker::ADD;
    vel_vec.lifetime = ros::Duration(0);
    vel_vec.pose.orientation.w = 1.0;
    vel_vec.id = 244;
    vel_vec.type = visualization_msgs::Marker::LINE_LIST;
    vel_vec.scale.x = 0.1;
    vel_vec.scale.y = 0.1;
    vel_vec.scale.z = 0.1;
    vel_vec.color = Color::Yellow();
    vel_vec.color.a = 1.0;

    orphans_pos_pub_.publish(pos_point);
    orphans_vel_vec_pub_.publish(vel_vec);
}

