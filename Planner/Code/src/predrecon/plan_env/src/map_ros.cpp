#include <plan_env/sdf_map.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <path_searching/astar2.h>
#include <plan_env/raycast.h>
#include <lkh_tsp_solver/lkh_interface.h>
#include <plan_env/map_ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h> 
#include <visualization_msgs/Marker.h>
#include <cmath>
#include <chrono>

#include <fstream>

namespace fast_planner {
MapROS::MapROS() {
}

MapROS::~MapROS() {
}

void MapROS::setMap(SDFMap* map) {
  this->map_ = map;
}

int MapROS::sgn(double& x)
{
  if (x>=0)
    return 1;
  else
    return -1;
}

void MapROS::init() {
  node_.param("map_ros/fx", fx_, -1.0);
  node_.param("map_ros/fy", fy_, -1.0);
  node_.param("map_ros/cx", cx_, -1.0);
  node_.param("map_ros/cy", cy_, -1.0);
  node_.param("map_ros/depth_filter_maxdist", depth_filter_maxdist_, -1.0);
  node_.param("map_ros/depth_filter_mindist", depth_filter_mindist_, -1.0);
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);

  node_.param("map_ros/esdf_slice_height", esdf_slice_height_, -0.1);
  node_.param("map_ros/visualization_truncate_height", visualization_truncate_height_, -0.1);
  node_.param("map_ros/visualization_truncate_low", visualization_truncate_low_, -0.1);
  node_.param("map_ros/show_occ_time", show_occ_time_, false);
  node_.param("map_ros/show_esdf_time", show_esdf_time_, false);
  node_.param("map_ros/show_all_map", show_all_map_, false);
  node_.param("map_ros/frame_id", frame_id_, string("world"));

  // K matrix
  K_depth_.setZero();
  K_depth_(0, 0) = fx_; //fx
  K_depth_(1, 1) = fy_; //fy
  K_depth_(0, 2) = cx_; //cx
  K_depth_(1, 2) = cy_; //cy
  K_depth_(2, 2) = 1.0;

  proj_points_.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  point_cloud_.points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  // proj_points_.reserve(640 * 480 / map_->mp_->skip_pixel_ / map_->mp_->skip_pixel_);
  proj_points_cnt = 0;

  local_updated_ = false;
  esdf_need_update_ = false;
  fuse_time_ = 0.0;
  esdf_time_ = 0.0;
  max_fuse_time_ = 0.0;
  max_esdf_time_ = 0.0;
  fuse_num_ = 0;
  esdf_num_ = 0;
  depth_image_.reset(new cv::Mat);

  rand_noise_ = normal_distribution<double>(0, 0.1);
  random_device rd;
  eng_ = default_random_engine(rd());

  esdf_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::visCallback, this);

  map_all_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_all", 10);
  map_local_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local", 10);
  map_local_inflate_pub_ =
      node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local_inflate", 10);
  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);
  depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);
  lidar_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/lidar_cloud", 10);
  // global planning vis pub
  arrow_pub_ = node_.advertise<visualization_msgs::MarkerArray>("/sdf_map/normal", 120);
  vp_pub_ = node_.advertise<visualization_msgs::MarkerArray>("/sdf_map/vp_global", 120);
  init_tour_pub_ = node_.advertise<visualization_msgs::MarkerArray>("/sdf_map/init_global", 10);
  pred_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/pred_cloud", 10);

  depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "/map_ros/depth", 50));
  cloud_sub_.reset(
      new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, "/map_ros/cloud", 50));
  pose_sub_.reset(
      new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/map_ros/pose", 25));

  sync_image_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyImagePose>(
      MapROS::SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
  sync_image_pose_->registerCallback(boost::bind(&MapROS::depthPoseCallback, this, _1, _2));
  sync_cloud_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyCloudPose>(
      MapROS::SyncPolicyCloudPose(100), *cloud_sub_, *pose_sub_));
  sync_cloud_pose_->registerCallback(boost::bind(&MapROS::cloudPoseCallback, this, _1, _2));

  map_start_time_ = ros::Time::now();

  // set sample param
  r_min = 1.5;
  r_max = 3.5;
  z_range = 2.5;
  theta_sample = 45.0*M_PI/180.0;
  node_.param("global/sample_phi_range_", phi_sample, -1.0);
  a_step = 30.0*M_PI/180.0;
  theta_thre = 25.0*M_PI/180.0;
  r_num = 2;
  z_step = 2;

  min_dist = 0.75;
  refine_num = 7;
  refine_radius = 5.0;
  top_view_num = 15;
  max_decay = 0.8;
  downsample_c = 10.0;
  percep_utils_.reset(new PerceptionUtils(node_));

  double resolution_ = map_->getResolution();
  Eigen::Vector3d origin, size;
  map_->getRegion(origin, size);
  raycaster_.reset(new RayCaster);
  raycaster_->setParams(resolution_, origin);
}

void MapROS::visCallback(const ros::TimerEvent& e) {
  publishMapLocal();
  if (show_all_map_) {
  // Limit the frequency of all map
    static double tpass = 0.0;
    tpass += (e.current_real - e.last_real).toSec();
    if (tpass > 0.1) {
      publishMapAll();
      tpass = 0.0;
    }
  }
  // publishPred_INTERNAL();
  // publishPredCloud();
  // publishESDF();

  // publishUpdateRange();
  // publishDepth();
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/) {
  if (!esdf_need_update_) return;
  auto t1 = ros::Time::now();

  map_->updateESDF3d();
  esdf_need_update_ = false;

  auto t2 = ros::Time::now();
  esdf_time_ += (t2 - t1).toSec();
  max_esdf_time_ = max(max_esdf_time_, (t2 - t1).toSec());
  esdf_num_++;
  if (show_esdf_time_)
    ROS_WARN("ESDF t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), esdf_time_ / esdf_num_,
             max_esdf_time_);
}

// prediction process
bool customRegionGrowing (const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float /*squared_distance*/)
{
  Eigen::Vector3d N1(
		seedPoint.normal_x,
		seedPoint.normal_y,
		seedPoint.normal_z
	);
	Eigen::Vector3d N2(
		candidatePoint.normal_x,
		candidatePoint.normal_y,
		candidatePoint.normal_z
	);
 
    Eigen::Vector3d D1(
        seedPoint.x,
        seedPoint.y,
        seedPoint.z
    );
    Eigen::Vector3d D2(
        candidatePoint.x,
        candidatePoint.y,
        candidatePoint.z
    );
    Eigen::Vector3d Dis = D1-D2;

	double angle = acos(N1.dot(N2)/(N1.norm()*N2.norm()));
    double distance = Dis.norm();

	const double threshold_angle = 10.0;	//[deg]
  const double angle_deg = abs(angle/M_PI*180.0);
  // use normal ------
	// if(angle_deg < threshold_angle || angle_deg > (180.0-threshold_angle))	return true;//&& (distance < 10.0)
	// else	return false;

  // wo using normal ------
  return true;
}

void MapROS::PCA_algo(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointType& c, PointType& pcZ, PointType& pcY, PointType& pcX, PointType& pcZ_inv, PointType& pcY_inv)
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

  c.x = pcaCentroid(0);
	c.y = pcaCentroid(1);
	c.z = pcaCentroid(2);
  pcZ.x = 5 * eigenVectorsPCA(0, 0) + c.x;
	pcZ.y = 5 * eigenVectorsPCA(1, 0) + c.y;
	pcZ.z = 5 * eigenVectorsPCA(2, 0) + c.z;
  pcZ_inv.x = -5 * eigenVectorsPCA(0, 0) + c.x;
  pcZ_inv.y = -5 * eigenVectorsPCA(1, 0) + c.y;
  pcZ_inv.z = -5 * eigenVectorsPCA(2, 0) + c.z;

  pcY.x = 5 * eigenVectorsPCA(0, 1) + c.x;
	pcY.y = 5 * eigenVectorsPCA(1, 1) + c.y;
	pcY.z = 5 * eigenVectorsPCA(2, 1) + c.z;
  pcY_inv.x = -5 * eigenVectorsPCA(0, 1) + c.x;
	pcY_inv.y = -5 * eigenVectorsPCA(1, 1) + c.y;
	pcY_inv.z = -5 * eigenVectorsPCA(2, 1) + c.z;
  pcX.x = 5 * eigenVectorsPCA(0, 2) + c.x;
	pcX.y = 5 * eigenVectorsPCA(1, 2) + c.y;
	pcX.z = 5 * eigenVectorsPCA(2, 2) + c.z;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MapROS::condition_get(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& x_up, float& x_low,
  float& y_up, float& y_low, float& z_up, float& z_low)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZ> ());

  // x-axis
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::GE, x_low)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::LT, x_up)));
  // y-axis
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::GE, y_low)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::LT, y_up)));
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

void MapROS::get_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_vec, Eigen::Matrix3f& normal_vec, bool& flag)
// true for Z_axis and false for Y_axis
{
  PointType temp_c, pcz, pcy, pcx, pcz_inv, pcy_inv;
  PCA_algo(cloud_vec, temp_c, pcz, pcy, pcx, pcz_inv, pcy_inv);
  Eigen::Matrix3f normal_info;
  // (center[3], normal_1[3], normal_2[3])

  if (flag)
  {
    normal_info << temp_c.x, temp_c.y, temp_c.z,
                    pcz.x, pcz.y, pcz.z,
                    pcz_inv.x, pcz_inv.y, pcz_inv.z;
    normal_vec = normal_info;
  }
  else
  {
    normal_info << temp_c.x, temp_c.y, temp_c.z,
                    pcy.x, pcy.y, pcy.z,
                    pcy_inv.x, pcy_inv.y, pcy_inv.z;
    normal_vec = normal_info;
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MapROS::surface_recon_visibility(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector4f& viewpoint)
{
  auto startT = std::chrono::high_resolution_clock::now();
    // ------process start---------
    float gamma = 0.0005;
    pcl::PointXYZ vp;
    vp.x = viewpoint[0];
    vp.y = viewpoint[1];
    vp.z = viewpoint[2];

  cloud_out.reset(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::demeanPointCloud(*cloud, viewpoint, *cloud_out);// Centralized by viewpoint

    cloud_transform.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_out, *cloud_transform);
    Eigen::Vector3f points;
    float temp_norm = 0.0;
    // step 1: point cloud transform
    for (int i=0; i<cloud_out->points.size(); ++i)
    {
        points[0] = cloud_out->points[i].x;
        points[1] = cloud_out->points[i].y;
        points[2] = cloud_out->points[i].z;
        temp_norm = points.norm();
        // kernel is d^gamma
        cloud_transform->points[i].x = pow(temp_norm, -gamma)*cloud_out->points[i].x/temp_norm;
        cloud_transform->points[i].y = pow(temp_norm, -gamma)*cloud_out->points[i].y/temp_norm;
        cloud_transform->points[i].z = pow(temp_norm, -gamma)*cloud_out->points[i].z/temp_norm;
    }
    // step 2: create convex hull
    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(cloud_transform);                   
    hull.setDimension(3);  
    hull.setComputeAreaVolume(true);

    std::vector<pcl::Vertices> polygons; 
    surface_hull.reset(new pcl::PointCloud<pcl::PointXYZ>);
    hull.reconstruct(*surface_hull, polygons);
    // step 3: inverse transform
    cloud_inverse.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*surface_hull, *cloud_inverse);
    Eigen::Vector3f points_i;
    float temp_norm_i = 0.0;
    float temp_z = 0.0;
    for (int i=0; i<surface_hull->points.size(); ++i)
    {
        points_i[0] = surface_hull->points[i].x;
        points_i[1] = surface_hull->points[i].y;
        points_i[2] = surface_hull->points[i].z;
        temp_norm_i = points_i.norm();

        cloud_inverse->points[i].x = (pow(temp_norm_i, -1/gamma)*surface_hull->points[i].x/temp_norm_i) + viewpoint[0];
        cloud_inverse->points[i].y = (pow(temp_norm_i, -1/gamma)*surface_hull->points[i].y/temp_norm_i) + viewpoint[1];
        cloud_inverse->points[i].z = (pow(temp_norm_i, -1/gamma)*surface_hull->points[i].z/temp_norm_i) + viewpoint[2];
    }
    // ------process end---------
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> vis_ms = endT - startT;
    cout << "vis_time:" << vis_ms.count() << "ms" << std::endl;

    return cloud_inverse;
}

void MapROS::uniform_cluster_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_t, 
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results_t)
{
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results_t);
    // step 1: partition the input point cloud
    pcl::PointXYZ minPt, maxPt;
	  pcl::getMinMax3D(*cloud_t, minPt, maxPt);
    clustered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    float x_low = 0.0, x_up = 0.0, y_low = 0.0, y_up = 0.0, z_low = 0.0, z_up = 0.0;
    float grid_size = 4.0;

    float x_range = maxPt.x - minPt.x;
    float y_range = maxPt.y - minPt.y;
    float z_range = maxPt.z - minPt.z;

    int x_num = std::ceil(x_range/grid_size);
    int y_num = std::ceil(y_range/grid_size);
    int z_num = std::ceil(z_range/grid_size);

    float x_start = minPt.x;
    float y_start = minPt.y;
    float z_start = minPt.z;
    int counter = 0;
    // step 2: generate ground cloud
    // step 3: uniform clustering
    // ----------------
    for (int i=0; i<x_num; ++i)
    {
        for (int j=0; j<y_num; ++j)
        {
            for (int k=0; k<z_num; ++k)
            {
                x_low = x_start + i*grid_size;
                x_up = x_start + (i+1)*grid_size;
                y_low = y_start + j*grid_size;
                y_up = y_start + (j+1)*grid_size;
                z_low = z_start + k*grid_size;
                z_up = z_start + (k+1)*grid_size;
                clustered_cloud = condition_get(cloud_t, x_up, x_low, y_up, y_low, z_up, z_low);
                if (clustered_cloud->points.size()>0)
                {
                  cluster_results_t.push_back(clustered_cloud);
                  counter++;
                }
            }
        }
    }
}

void MapROS::conditional_ec(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results);
  const double leaf_size = map_->getResolution()*downsample_c;

  downsample_cec.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> ds;
  ds.setInputCloud(cloud);
  ds.setLeafSize(leaf_size, leaf_size, leaf_size);
  ds.filter(*downsample_cec);
  // normal estimation
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;	
  ne.setInputCloud(downsample_cec);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZ>);
  ne.setSearchMethod(normal_tree);
  cloud_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*downsample_cec, *cloud_normals);
  ne.setKSearch(10);
  ne.compute(*cloud_normals);
  // Conditional Euclidean Cluster
  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
  pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec(true);
  cec.setInputCloud (cloud_normals);
  cec.setConditionFunction (&customRegionGrowing);
  cec.setClusterTolerance (2.0*leaf_size);
  cec.setMinClusterSize (10);
  cec.setMaxClusterSize (cloud_normals->size () / 5);
  cec.segment (*clusters);
  cec.getRemovedClusters(small_clusters, large_clusters);
  // send to uniform cluster
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> final_cluster_results;

  if (small_clusters->size() > 0)
  {
      pcl::PointCloud<pcl::PointXYZ>::Ptr small_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& small_cluster : (*small_clusters))
        {
        pcl::PointXYZ temp_small;
        for (const auto& j : small_cluster.indices)
        {
            temp_small.x = (*cloud_normals)[j].x;
            temp_small.y = (*cloud_normals)[j].y;
            temp_small.z = (*cloud_normals)[j].z;
            small_cloud->points.push_back(temp_small);
        }
        }
      final_cluster_results.push_back(small_cloud);
  }

  if (clusters->size() > 0)
  {
      for (const auto& cluster : (*clusters))
        {
        pcl::PointCloud<pcl::PointXYZ>::Ptr n_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ temp_n;
        for (const auto& j : cluster.indices)
        {
            temp_n.x = (*cloud_normals)[j].x;
            temp_n.y = (*cloud_normals)[j].y;
            temp_n.z = (*cloud_normals)[j].z;
            n_cloud->points.push_back(temp_n);
        }
        final_cluster_results.push_back(n_cloud);
        }
  }

  if (large_clusters->size() > 0)
  {
      for (const auto& large_cluster : (*large_clusters))
        {
        pcl::PointCloud<pcl::PointXYZ>::Ptr large_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ temp_large;
        for (const auto& j : large_cluster.indices)
        {
            temp_large.x = (*cloud_normals)[j].x;
            temp_large.y = (*cloud_normals)[j].y;
            temp_large.z = (*cloud_normals)[j].z;
            large_cloud->points.push_back(temp_large);
        }
        final_cluster_results.push_back(large_cloud);
        }
  }
  
  for (int i=0; i<final_cluster_results.size(); ++i)
  {
      vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> temp_cluster_results;
      uniform_cluster_with_normal(final_cluster_results[i], temp_cluster_results);
      cluster_results.insert(cluster_results.end(),temp_cluster_results.begin(),temp_cluster_results.end());
  }
}

void MapROS::normal_vis()
{
  vector<cluster_normal>().swap(uni_cnv);
  cluster_normal temp_cnv_p;
  cluster_normal temp_cnv_n;
  int counter_normal = 0;

  visualization_msgs::MarkerArray arrows;
  float alpha = 0.8;
  double sample_coeff = 6.0;
  const double leaf_size = map_->getResolution()*sample_coeff;
  Eigen::Vector3d normal_0;
  Eigen::Vector3d normal_1;
  Eigen::Vector3d center;
  
  for (int i=0; i<uni_cluster.size(); ++i)
  {
    bool have_normal = false;
    bool type_normal = true;
    Eigen::Matrix3f normal_Z;
    Eigen::Matrix3f normal_Y;
    get_normal(uni_cluster[i], normal_Z, type_normal);
    // cout << "No." << i << "_normal_Z" << normal_Z << endl;

    center(0) = normal_Z(0,0);
    center(1) = normal_Z(0,1);
    center(2) = normal_Z(0,2);
    
    visualization_msgs::Marker normal;
    normal.header.frame_id = "world";
    normal.header.stamp = ros::Time::now();
    normal.id = i;
    normal.type = visualization_msgs::Marker::ARROW;
    normal.action = visualization_msgs::Marker::ADD;

    normal.pose.orientation.w = 1.0;
    normal.scale.x = 0.1;
    normal.scale.y = 0.2;
    normal.scale.z = 0.3;
    
    normal_0(0) = normal_Z(1,0);
    normal_0(1) = normal_Z(1,1);
    normal_0(2) = normal_Z(1,2);
    int state_0 = normal_judge(center, normal_0, alpha);
    
    if (state_0 != 1)
    {
      // have normal
      have_normal = true;
      // store in temp cnv

      vector<Eigen::Vector3d> vec_0;
      for (int h=0; h<uni_cluster[i]->points.size(); ++h)
      {
        Eigen::Vector3d temp_cell_0(uni_cluster[i]->points[h].x, uni_cluster[i]->points[h].y, uni_cluster[i]->points[h].z);
        vec_0.push_back(temp_cell_0);
      }
      temp_cnv_p.global_cells_ = vec_0;
      temp_cnv_p.center_ = center;
      temp_cnv_p.normal_ = normal_0-center;
      temp_cnv_p.id_g = counter_normal;
      counter_normal++;
      uni_cnv.push_back(temp_cnv_p);
    }
    normal_1(0) = normal_Z(2,0);
    normal_1(1) = normal_Z(2,1);
    normal_1(2) = normal_Z(2,2);
    int state_1 = normal_judge(center, normal_1, alpha);
    
    if (state_1 != 1)
    {
      // have normal
      have_normal = true;
      // store in temp cnv

      vector<Eigen::Vector3d> vec_1;
      for (int g=0; g<uni_cluster[i]->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1(uni_cluster[i]->points[g].x, uni_cluster[i]->points[g].y, uni_cluster[i]->points[g].z);
        vec_1.push_back(temp_cell_1);
      }
      temp_cnv_n.global_cells_ = vec_1;
      temp_cnv_n.center_ = center;
      temp_cnv_n.normal_ = normal_1-center;
      temp_cnv_n.id_g = counter_normal;
      counter_normal++;
      uni_cnv.push_back(temp_cnv_n);
    }

    if (have_normal == false)
    {
      cout << "No:" << i << "no normal mech!!!!!!!!!!!!" << endl;
      type_normal = false;
      get_normal(uni_cluster[i], normal_Y, type_normal);
      // cout << "No." << i << "_normal_Y" << normal_Y << endl;

      center(0) = normal_Y(0,0);
      center(1) = normal_Y(0,1);
      center(2) = normal_Y(0,2);
      
      normal_0(0) = normal_Y(1,0);
      normal_0(1) = normal_Y(1,1);
      normal_0(2) = normal_Y(1,2);
      int state_0 = normal_judge(center, normal_0, alpha);
      if (state_0 != 1)
      {
        have_normal = true;
        // store in temp cnv

        vector<Eigen::Vector3d> vec_0;
        for (int h=0; h<uni_cluster[i]->points.size(); ++h)
        {
          Eigen::Vector3d temp_cell_0(uni_cluster[i]->points[h].x, uni_cluster[i]->points[h].y, uni_cluster[i]->points[h].z);
          vec_0.push_back(temp_cell_0);
        }
        temp_cnv_p.global_cells_ = vec_0;
        temp_cnv_p.center_ = center;
        temp_cnv_p.normal_ = normal_0-center;
        temp_cnv_p.id_g = counter_normal;
        counter_normal++;
        uni_cnv.push_back(temp_cnv_p);
      }

      normal_1(0) = normal_Y(2,0);
      normal_1(1) = normal_Y(2,1);
      normal_1(2) = normal_Y(2,2);
      int state_1 = normal_judge(center, normal_1, alpha);
      if (state_1 != 1)
      {
        // have normal
        have_normal = true;
        // store in temp cnv

        vector<Eigen::Vector3d> vec_1;
        for (int g=0; g<uni_cluster[i]->points.size(); ++g)
        {
          Eigen::Vector3d temp_cell_1(uni_cluster[i]->points[g].x, uni_cluster[i]->points[g].y, uni_cluster[i]->points[g].z);
          vec_1.push_back(temp_cell_1);
        }
        temp_cnv_n.global_cells_ = vec_1;
        temp_cnv_n.center_ = center;
        temp_cnv_n.normal_ = normal_1-center;
        temp_cnv_n.id_g = counter_normal;
        counter_normal++;
        uni_cnv.push_back(temp_cnv_n);
      }
    }
    if (have_normal == false)
    {
      if (center(2) > viewpoint(2))
      {
        normal_1(0) = center(0) - viewpoint(0);
        normal_1(1) = center(1) - viewpoint(1);
        normal_1(2) = center(2) - viewpoint(2);
      }
      else
      {
        normal_1(0) = center(0) - viewpoint(0);
        normal_1(1) = center(1) - viewpoint(1);
        normal_1(2) = 0.0;
      }

      int state_0 = normal_judge(center, normal_1, alpha);
      if (state_0 != 1)
      {
      vector<Eigen::Vector3d> vec_1;
      for (int g=0; g<uni_cluster[i]->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1(uni_cluster[i]->points[g].x, uni_cluster[i]->points[g].y, uni_cluster[i]->points[g].z);
        vec_1.push_back(temp_cell_1);
      }
      temp_cnv_n.global_cells_ = vec_1;
      temp_cnv_n.center_ = center;
      temp_cnv_n.normal_ = normal_1;
      temp_cnv_n.id_g = counter_normal;
      counter_normal++;
      uni_cnv.push_back(temp_cnv_n);

      have_normal = true;
      }
    }
    if (have_normal == true)
    {
      geometry_msgs::Point pt;
      pt.x = uni_cnv.back().center_(0);
      pt.y = uni_cnv.back().center_(1);
      pt.z = uni_cnv.back().center_(2);
      normal.points.push_back(pt);

      pt.x = 3*uni_cnv.back().normal_.normalized()(0)+uni_cnv.back().center_(0);
      pt.y = 3*uni_cnv.back().normal_.normalized()(1)+uni_cnv.back().center_(1);
      pt.z = 3*uni_cnv.back().normal_.normalized()(2)+uni_cnv.back().center_(2);
      normal.points.push_back(pt);
      
      normal.color.r = 0.0;
      normal.color.g = 0.2;
      normal.color.b = 1.0;
      normal.color.a = 1.0;

      arrows.markers.push_back(normal);
    }
  }
  cout << "arrow_marker:" << arrows.markers.size() << endl;
  arrow_pub_.publish(arrows);
}

int MapROS::normal_judge(Eigen::Vector3d& center, Eigen::Vector3d& pos, float& coeff)
{
  // z_axis constrain
  if (pos[2] < 0.5)
  {
    return 1;
  }
  
  Eigen::Vector3d start_xp;
  Eigen::Vector3d temp_xp;
  float size = 0.1;
  int element = 5;
  start_xp(0) = coeff*(pos(0)-center(0)) + center(0) - element*size;
  start_xp(1) = coeff*(pos(1)-center(1)) + center(1) - element*size;
  start_xp(2) = coeff*(pos(2)-center(2)) + center(2) - element*size;
  for (int i=0; i<2*element; ++i)
  {
    for (int j=0; j<2*element; ++j)
    {
      for (int k=0; k<2*element; ++k)
      {
        temp_xp(0) = start_xp(0) + i*size;
        temp_xp(1) = start_xp(1) + j*size;
        temp_xp(2) = start_xp(2) + k*size;
        int state = map_->get_PredStates(temp_xp);
        if (state == 3)
        {
          return 1;
        }
      }
    }
  }

  return 0;
}
// calculate visible cells of one sampled viewpoint
int MapROS::cal_visibility_cells(const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& set)
{
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  for (auto cell : set)
  {
    bool vis = true;
    Eigen::Vector3i idx;
    if (percep_utils_->insideFOV(cell))
    {
      raycaster_->input(pos, cell);
      while (raycaster_->nextId(idx)) 
      {
        if (map_->get_PredStates(idx) == SDFMap::PRED_INTERNAL ||
            !map_->isInBox(idx)) {
          vis = false;
          break;
        }
      }
    }
    if (vis == true)
      visib_num += 1;
  }

  return visib_num;
}
// sample global viewpoints
void MapROS::sample_vp_global(cluster_normal& set, int& qualified_vp)
{
  Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  Eigen::Vector3d z_axis_nega(0.0, 0.0, -1.0);
  Eigen::Vector3d x_axis(1.0, 0.0, 0.0);
  Eigen::Vector3d x_axis_nega(-1.0, 0.0, 0.0);

  float z_min = 1.2;
  double theta;
  if (set.normal_(2) >= 0)
    theta = acos(set.normal_.dot(z_axis)/(set.normal_.norm()*z_axis.norm()));
  else
    theta = acos(set.normal_.dot(z_axis_nega)/(set.normal_.norm()*z_axis_nega.norm()));

  Eigen::Vector3d normal_proj(set.normal_(0), set.normal_(1), 0.0);
  double phi;
  int sign_x = sgn(normal_proj(0));
  int sign_y = sgn(normal_proj(1));
  int sign_z = sgn(set.normal_(2));
  if (normal_proj(0) >= 0)
    phi = sign_y*acos(normal_proj.dot(x_axis)/(normal_proj.norm()*x_axis.norm()));
  else
    phi = sign_y*acos(normal_proj.dot(x_axis_nega)/(normal_proj.norm()*x_axis_nega.norm()));
  // store max visibility viewpoints
  int vis_cells = 0;
  VP_global tmp_vp;
  // -------------------------------
  for (double rc = r_min, dr = (r_max - r_min) / r_num; rc <= r_max + 1e-3; rc += dr)
  {
    for (double tc = theta-theta_sample; tc < theta+theta_sample+1e-3; tc+=a_step)
    {
      for (double pc = phi-phi_sample; pc < phi+phi_sample+1e-3; pc+=a_step)
      {
        Eigen::Vector3d sample_pos = set.center_ + rc * Eigen::Vector3d(sign_x*sin(tc)*cos(pc), sin(tc)*sin(pc), sign_z*cos(tc));

        float coe_sample = 1.0;
        int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
        if (!map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
          continue;
        // compute average yaw
        auto& pred_cells = set.global_cells_;
        int lower_vis_bound = ceil(0.5*pred_cells.size());
        Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
        double avg_yaw = 0.0;
        for (int r=1; r<pred_cells.size(); ++r)
        {
          Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
          double yaw = acos(dir.dot(start_dir));
          if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
          avg_yaw += yaw;
        }
        avg_yaw = avg_yaw / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
        // constrain yaw
        while (avg_yaw < -M_PI)
          avg_yaw += 2 * M_PI;
        while (avg_yaw > M_PI)
          avg_yaw -= 2 * M_PI;
        
        int visib_num = cal_visibility_cells(sample_pos, avg_yaw, pred_cells);
        if (visib_num > vis_cells)
        {
          vis_cells = visib_num;
          tmp_vp = {sample_pos, avg_yaw, visib_num};
        }
        if (visib_num > lower_vis_bound)
        {
          VP_global vp_g = {sample_pos, avg_yaw, visib_num};
          set.vps_global.push_back(vp_g);
          qualified_vp++;
        }
      }
    }
  }
  if (qualified_vp == 0)
  {
    auto& cells_ = set.global_cells_;
    int vis_all = cells_.size();
    float ratio = (float)vis_cells/vis_all;
    if (ratio > 0)
    {
      cout << "vis_num:" << vis_cells << endl;
      cout << "unqualified_ratio:" << ratio << endl;
      set.vps_global.push_back(tmp_vp);
      qualified_vp++;
    }
    else
    {
      Eigen::Vector3d normal_n = set.normal_.normalized();
      Eigen::Vector3d sample_pos;
      double scale = 0.5*(r_max + r_min);
      sample_pos(0) = set.center_(0) + scale*normal_n(0);
      sample_pos(1) = set.center_(1) + scale*normal_n(1);
      sample_pos(2) = set.center_(2) + scale*normal_n(2);

      auto& pred_cells = set.global_cells_;
      Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
      double avg_yaw_ = 0.0;
        for (int r=1; r<pred_cells.size(); ++r)
        {
          Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
          double yaw = acos(dir.dot(start_dir));
          if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
          avg_yaw_ += yaw;
        }
        avg_yaw_ = avg_yaw_ / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
        // constrain yaw
        while (avg_yaw_ < -M_PI)
          avg_yaw_ += 2 * M_PI;
        while (avg_yaw_ > M_PI)
          avg_yaw_ -= 2 * M_PI;
      
      int vis_num_ = 10;
      VP_global vp_g = {sample_pos, avg_yaw_, vis_num_};
      set.vps_global.push_back(vp_g);
      qualified_vp++;
    }
  }
}

void MapROS::sample_vp_global_pillar(cluster_normal& set, int& qualified_vp)
{
  Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  Eigen::Vector3d z_axis_nega(0.0, 0.0, -1.0);
  Eigen::Vector3d x_axis(1.0, 0.0, 0.0);
  Eigen::Vector3d x_axis_nega(-1.0, 0.0, 0.0);

  float z_min = 1.0;
  double theta;
  if (set.normal_(2) >= 0)
    theta = acos(set.normal_.dot(z_axis)/(set.normal_.norm()*z_axis.norm()));
  else
    theta = acos(set.normal_.dot(z_axis_nega)/(set.normal_.norm()*z_axis_nega.norm()));
  int sign_z = sgn(set.normal_(2));
  // 2 modes for theta sampling
  if (theta > theta_thre)
  {
    Eigen::Vector3d normal_proj(set.normal_(0), set.normal_(1), 0.0);
    double phi;
    int sign_x = sgn(normal_proj(0));
    int sign_y = sgn(normal_proj(1));
    if (normal_proj(0) >= 0)
      phi = sign_y*acos(normal_proj.dot(x_axis)/(normal_proj.norm()*x_axis.norm()));
    else
      phi = sign_y*acos(normal_proj.dot(x_axis_nega)/(normal_proj.norm()*x_axis_nega.norm()));
    // store max visibility viewpoints
    int vis_cells = 0;
    VP_global tmp_vp;
    // -------------------------------
    for (double rc = r_min, dr = (r_max - r_min) / r_num; rc <= r_max + 1e-3; rc += dr)
    {
      for (double zc = 0.5*z_range, dz = z_range / z_step; zc >= -0.5*z_range - 1e-3; zc -= dz)
      {
        for (double pc = phi-phi_sample; pc < phi+phi_sample+1e-3; pc+=a_step)
        {
          Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(rc*sign_x*cos(pc), rc*sin(pc), zc);

          float coe_sample = 1.0;
          int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
          if (!map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;
          // compute average yaw
          auto& pred_cells = set.global_cells_;
          int lower_vis_bound = ceil(0.5*pred_cells.size());
          Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
          double avg_yaw = 0.0;
          for (int r=1; r<pred_cells.size(); ++r)
          {
            Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
            double yaw = acos(dir.dot(start_dir));
            if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
            avg_yaw += yaw;
          }
          avg_yaw = avg_yaw / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
          // constrain yaw
          while (avg_yaw < -M_PI)
            avg_yaw += 2 * M_PI;
          while (avg_yaw > M_PI)
            avg_yaw -= 2 * M_PI;
          
          int visib_num = cal_visibility_cells(sample_pos, avg_yaw, pred_cells);
          if (visib_num > vis_cells)
          {
            vis_cells = visib_num;
            tmp_vp = {sample_pos, avg_yaw, visib_num};
          }
          if (visib_num > lower_vis_bound)
          {
            VP_global vp_g = {sample_pos, avg_yaw, visib_num};
            set.vps_global.push_back(vp_g);
            qualified_vp++;
          }
        }
      }
    }
    if (qualified_vp == 0)
    {
      auto& cells_ = set.global_cells_;
      int vis_all = cells_.size();
      float ratio = (float)vis_cells/vis_all;
      if (ratio > 0.25)
      {
        cout << "vis_num:" << vis_cells << endl;
        cout << "unqualified_ratio:" << ratio << endl;
        set.vps_global.push_back(tmp_vp);
        qualified_vp++;
      }
      else
      {
        cout << "xoy add vis" << endl;
        Eigen::Vector3d normal_n = set.normal_.normalized();
        Eigen::Vector3d sample_pos;
        double scale = 0.5*(r_max + r_min);
        sample_pos(0) = set.center_(0) + scale*normal_n(0);
        sample_pos(1) = set.center_(1) + scale*normal_n(1);
        sample_pos(2) = set.center_(2) + scale*normal_n(2);

        auto& pred_cells = set.global_cells_;
        Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
        double avg_yaw_ = 0.0;
          for (int r=1; r<pred_cells.size(); ++r)
          {
            Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
            double yaw = acos(dir.dot(start_dir));
            if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
            avg_yaw_ += yaw;
          }
          avg_yaw_ = avg_yaw_ / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
          // constrain yaw
          while (avg_yaw_ < -M_PI)
            avg_yaw_ += 2 * M_PI;
          while (avg_yaw_ > M_PI)
            avg_yaw_ -= 2 * M_PI;
        
        int vis_num_ = 10;
        VP_global vp_g = {sample_pos, avg_yaw_, vis_num_};
        set.vps_global.push_back(vp_g);
        qualified_vp++;
      }
    }
  }
  else
  {
    // store max visibility viewpoints
    int vis_cells = 0;
    VP_global tmp_vp;
    // -------------------------------
    for (double rc = 0.2*r_min, dr = (r_max - r_min) / (2*r_num); rc <= r_max + 1e-3; rc += dr)
    {
      for (double zc = sign_z*z_range, dz = sign_z*z_range / z_step; zc >= -1e-3; zc -= dz)
      {
        for (double pc = 0.0; pc < 2*M_PI+1e-3; pc+=2*a_step)
        {
          Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(rc*cos(pc), rc*sin(pc), zc);

          float coe_sample = 1.0;
          int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
          if (!map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;
          // compute average yaw
          auto& pred_cells = set.global_cells_;
          int lower_vis_bound = ceil(0.5*pred_cells.size());
          Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
          double avg_yaw = 0.0;
          for (int r=1; r<pred_cells.size(); ++r)
          {
            Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
            double yaw = acos(dir.dot(start_dir));
            if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
            avg_yaw += yaw;
          }
          avg_yaw = avg_yaw / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
          // constrain yaw
          while (avg_yaw < -M_PI)
            avg_yaw += 2 * M_PI;
          while (avg_yaw > M_PI)
            avg_yaw -= 2 * M_PI;
          
          int visib_num = cal_visibility_cells(sample_pos, avg_yaw, pred_cells);
          if (visib_num > vis_cells)
          {
            vis_cells = visib_num;
            tmp_vp = {sample_pos, avg_yaw, visib_num};
          }
          if (visib_num > lower_vis_bound)
          {
            VP_global vp_g = {sample_pos, avg_yaw, visib_num};
            set.vps_global.push_back(vp_g);
            qualified_vp++;
          }
        }
      }
    }
    if (qualified_vp == 0)
    {
      auto& cells_ = set.global_cells_;
      int vis_all = cells_.size();
      float ratio = (float)vis_cells/vis_all;
      if (ratio > 0.25)
      {
        cout << "vis_num:" << vis_cells << endl;
        cout << "unqualified_ratio:" << ratio << endl;
        set.vps_global.push_back(tmp_vp);
        qualified_vp++;
      }
      else
      {
        cout << "z_axis add vis" << endl;
        Eigen::Vector3d normal_n = set.normal_.normalized();
        Eigen::Vector3d sample_pos;
        double scale = 0.5*(r_max + r_min);
        sample_pos(0) = set.center_(0) + scale*normal_n(0);
        sample_pos(1) = set.center_(1) + scale*normal_n(1);
        sample_pos(2) = set.center_(2) + scale*normal_n(2);

        auto& pred_cells = set.global_cells_;
        Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
        double avg_yaw_ = 0.0;
          for (int r=1; r<pred_cells.size(); ++r)
          {
            Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
            double yaw = acos(dir.dot(start_dir));
            if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
            avg_yaw_ += yaw;
          }
          avg_yaw_ = avg_yaw_ / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
          // constrain yaw
          while (avg_yaw_ < -M_PI)
            avg_yaw_ += 2 * M_PI;
          while (avg_yaw_ > M_PI)
            avg_yaw_ -= 2 * M_PI;
        
        int vis_num_ = 10;
        VP_global vp_g = {sample_pos, avg_yaw_, vis_num_};
        set.vps_global.push_back(vp_g);
        qualified_vp++;
      }
    }
  }
}

void MapROS::draw_vpg()
{
  visualization_msgs::MarkerArray view_points;
  int counter_vpg = 0;
  const double length = 2.0;
  for (int i=0; i<uni_open_cnv.size(); ++i)
  {
    for (int j=0; j<uni_open_cnv[i].vps_global.size(); ++j)
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
    pt_.x = uni_open_cnv[i].vps_global[j].pos_g(0);
    pt_.y = uni_open_cnv[i].vps_global[j].pos_g(1);
    pt_.z = uni_open_cnv[i].vps_global[j].pos_g(2);
    vp.points.push_back(pt_);

    pt_.x = uni_open_cnv[i].vps_global[j].pos_g(0) + length*cos(uni_open_cnv[i].vps_global[j].yaw_g);
    pt_.y = uni_open_cnv[i].vps_global[j].pos_g(1) + length*sin(uni_open_cnv[i].vps_global[j].yaw_g);
    pt_.z = uni_open_cnv[i].vps_global[j].pos_g(2);
    vp.points.push_back(pt_);
    
    vp.color.r = 0.0;
    vp.color.g = 0.5;
    vp.color.b = 0.5;
    vp.color.a = 1.0;

    view_points.markers.push_back(vp);
    counter_vpg++;
    }
  }

  vp_pub_.publish(view_points);
  // path vis
  visualization_msgs::MarkerArray tours;

  visualization_msgs::Marker global_tour_init;
  global_tour_init.header.frame_id = "world";
  global_tour_init.header.stamp = ros::Time::now();
  global_tour_init.id = counter_vpg;
  global_tour_init.type = visualization_msgs::Marker::LINE_STRIP;
  global_tour_init.action = visualization_msgs::Marker::ADD;

  global_tour_init.pose.orientation.w = 1.0;
  global_tour_init.scale.x = 0.2;

  global_tour_init.color.r = 1.0;
  global_tour_init.color.g = 0.0;
  global_tour_init.color.b = 0.0;
  global_tour_init.color.a = 1.0;
  for (int k=0; k<global_tour.size(); ++k)
  {
    geometry_msgs::Point p;
    p.x = global_tour[k][0];
    p.y = global_tour[k][1];
    p.z = global_tour[k][2];
    global_tour_init.points.push_back(p);
  }
  counter_vpg++;
  tours.markers.push_back(global_tour_init);
  // refine_tour
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
  for (int k=0; k<refined_tour.size(); ++k)
  {
    geometry_msgs::Point p;
    p.x = refined_tour[k][0];
    p.y = refined_tour[k][1];
    p.z = refined_tour[k][2];
    global_tour_refine.points.push_back(p);
  }
  tours.markers.push_back(global_tour_refine);

  init_tour_pub_.publish(tours);
}

void MapROS::CostMat()
{
  auto costMat = [](const vector<cluster_normal>::iterator& it1, const vector<cluster_normal>::iterator& it2)
  {
    // Search path from old cluster's top viewpoint to new cluster'
    VP_global& vui = it1->vps_global.front();
    VP_global& vuj = it2->vps_global.front();
    vector<Eigen::Vector3d> path_ij;
    double cost_ij = ViewNode::compute_globalCost(
        vui.pos_g, vuj.pos_g, vui.yaw_g, vuj.yaw_g, Vector3d(0, 0, 0), 0, path_ij);
    
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);                                                                                         
    it2->paths_.push_back(path_ij);
  };

  for (auto it1 = uni_open_cnv.begin(); it1 != uni_open_cnv.end(); ++it1)
  {
    for (auto it2 = it1; it2 != uni_open_cnv.end(); ++it2)
    {
      if (it1 == it2) 
      {
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } 
      else
        costMat(it1, it2);
    }
  }

  // for (auto cnv : uni_open_cnv)
  // {
  //   cout << "No." <<cnv.id_g << ":" << "(" << cnv.costs_.size() << "," << cnv.paths_.size() << ")" << endl;
  // }
}

void MapROS::fullCostMatrix(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    Eigen::MatrixXd& mat)
{
  // Use Asymmetric TSP
  int dims = uni_open_cnv.size();
  mat.resize(dims + 1, dims + 1);
  int i = 1, j = 1;
  for (auto cnv : uni_open_cnv)
  {
    for (auto cost : cnv.costs_)
    {
      mat(i, j++) = cost;
    }
    ++i;
    j=1;
  }

  mat.leftCols<1>().setZero();
  for (auto cnv : uni_open_cnv)
  {
    VP_global vp = cnv.vps_global.front();
    vector<Eigen::Vector3d> path;
    mat(0, j++) =
      ViewNode::compute_globalCost(cur_pos, vp.pos_g, cur_yaw[0], vp.yaw_g, cur_vel, cur_yaw[1], path);
  }
}

void MapROS::get_GPath(const Eigen::Vector3d& pos, const vector<int>& ids, vector<Eigen::Vector3d>& path)
{
  vector<vector<cluster_normal>::iterator> cnv_indexer;
  for (auto it = uni_open_cnv.begin(); it != uni_open_cnv.end(); ++it)
    cnv_indexer.push_back(it);
  // Compute the path from current pos to the first cluster
  vector<Eigen::Vector3d> segment;
  ViewNode::search_globalPath(pos, cnv_indexer[ids[0]]->vps_global.front().pos_g, segment);
  path.insert(path.end(), segment.begin(), segment.end());
  // Get paths of tour passing all clusters
  for (int i=0; i<cnv_indexer.size()-1; ++i)
  {
    // Move to path to next cluster
    auto path_iter = cnv_indexer[ids[i]]->paths_.begin();
    int next_idx = ids[i+1];
    for (int j = 0; j < next_idx; ++j)
      ++path_iter;
    path.insert(path.end(), path_iter->begin(), path_iter->end());
  }
}

void MapROS::global_path(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    vector<int>& order)
{
  vector<Eigen::Vector3d>().swap(global_tour);
  string tsp_dir = "/home/albert/UAV_Planning/fuel_airsim/tsp_dir";
  // Initialize TSP par file
  ofstream par_file(tsp_dir + "/single.par");
  par_file << "PROBLEM_FILE = " << tsp_dir << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << tsp_dir << "/single.txt\n";
  par_file << "RUNS = 1\n";
  par_file.close();

  // cost matrix
  auto cost_t1 = std::chrono::high_resolution_clock::now();
  CostMat();
  auto cost_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cost_ms = cost_t2 - cost_t1;
  cout << "<<cost_calculate_time>>" << cost_ms.count() << "ms" << std::endl;
  // ---------------------------------------------------
  Eigen::MatrixXd cost_mat;
  fullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat);
  const int dimension = cost_mat.rows();
  // write param to tsp file
  ofstream prob_file(tsp_dir + "/single.tsp");
  string prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
  prob_file << prob_spec;

  const int scale = 100;
  // Use Asymmetric TSP
  for (int i = 0; i < dimension; ++i) 
  {
    for (int j = 0; j < dimension; ++j) 
    {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }

  prob_file << "EOF";
  prob_file.close();
  // Call LKH TSP solver
  solveTSPLKH((tsp_dir + "/single.par").c_str());
  // Obtain TSP results
  ifstream res_file(tsp_dir + "/single.txt");
  string res;
  while (getline(res_file, res)) 
  {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  // Read path for ATSP formulation
  while (getline(res_file, res)) 
  {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1) break;
    order.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }
  res_file.close();
  
  get_GPath(cur_pos, order, global_tour);
  // obtain initial viewpoint info
  vector<Eigen::Vector3d>().swap(global_points);
  vector<double>().swap(global_yaws);
  for (auto cnv : uni_open_cnv)
  {
    bool no_view = true;
    for (auto view : cnv.vps_global)
    {
      if ((view.pos_g - cur_pos).norm() < min_dist) continue;
      global_points.push_back(view.pos_g);
      global_yaws.push_back(view.yaw_g);
      no_view = false;
      break;
    }
    if (no_view) 
    {
      auto view = cnv.vps_global.front();
      global_points.push_back(view.pos_g);
      global_yaws.push_back(view.yaw_g);
    }
  }
  // prepare for refinement process
  vector<int>().swap(refined_ids);
  vector<Eigen::Vector3d>().swap(unrefined_points);
  int knum = refine_num;
  for (int i=0; i<knum; ++i)
  {
    auto tmp = global_points[order[i]];
    unrefined_points.push_back(tmp);
    refined_ids.push_back(order[i]);
    if ((tmp - cur_pos).norm() > refine_radius && refined_ids.size() >= 2) break;
  }
}

void MapROS::get_vpinfo(const Eigen::Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& yaws)
{
  points.clear();
  yaws.clear();
  for (auto id : ids)
  {
    for (auto cnv : uni_open_cnv)
    {
      if (cnv.id_g == id)
      {
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        int visib_thresh = cnv.vps_global.front().visib_num_g * max_decay;
        for (auto view : cnv.vps_global)
        {
          if (pts.size() >= view_num || view.visib_num_g <= visib_thresh) break;
          if ((view.pos_g - cur_pos).norm() < min_dist) continue;
          pts.push_back(view.pos_g);
          ys.push_back(view.yaw_g);
        }
        if (pts.empty())
        {
          // All viewpoints are very close, ignore the distance limit
          for (auto view : cnv.vps_global) 
          {
            if (pts.size() >= view_num || view.visib_num_g <= visib_thresh) break;
            pts.push_back(view.pos_g);
            ys.push_back(view.yaw_g);
          }
        }
        points.push_back(pts);
        yaws.push_back(ys);
      }
    }
  }
}

void MapROS::global_refine(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw)
{
  vector<vector<Eigen::Vector3d>>().swap(global_n_points);
  vector<vector<double>>().swap(global_n_yaws);
  vector<Eigen::Vector3d>().swap(refined_n_points);
  vector<double>().swap(refined_n_yaws);
  // get viewpoints info
  get_vpinfo(cur_pos, refined_ids, top_view_num, max_decay, global_n_points, global_n_yaws);
  // Create graph for viewpoints selection
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;
  // Add the current state
  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0]));
  first->vel_ = cur_vel;
  g_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;
  // Add viewpoints
  for (int i=0; i < global_n_points.size(); ++i)
  {
    for (int j=0; j < global_n_points[i].size(); ++j)
    {
      ViewNode::Ptr node(new ViewNode(global_n_points[i][j], global_n_yaws[i][j]));
      g_search.addNode(node);
      for (auto nd : last_group)
        g_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);
      if (i == global_n_points.size() - 1) 
      {
        final_node = node;
        break;
      }
    }
    last_group = cur_group;
    cur_group.clear();
  }
  // Search optimal sequence
  vector<ViewNode::Ptr> path;
  g_search.DijkstraSearch(first->id_, final_node->id_, path);
  for (int i = 1; i < path.size(); ++i) 
  {
    refined_n_points.push_back(path[i]->pos_);
    refined_n_yaws.push_back(path[i]->yaw_);
  }
  vector<Eigen::Vector3d>().swap(refined_tour);
  refined_tour.push_back(cur_pos);
  ViewNode::astar_->lambda_heu_ = 1.0;
  ViewNode::astar_->setResolution(0.2);
  for (auto pt : refined_n_points) 
  {
    vector<Eigen::Vector3d> path;
    if (ViewNode::search_globalPath(refined_tour.back(), pt, path))
      refined_tour.insert(refined_tour.end(), path.begin(), path.end());
    else
      refined_tour.push_back(pt);
  }
  ViewNode::astar_->lambda_heu_ = 10000;
}

void MapROS::depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                               const nav_msgs::OdometryConstPtr& pose) {
  // AirSim only
  Eigen::Quaterniond airsim_q_;
  airsim_q_.w() = cos(0.25*M_PI);
  airsim_q_.x() = 0.0;
  airsim_q_.y() = 0.0;
  airsim_q_.z() = sin(0.25*M_PI);

  Eigen::Vector3d body_pos_, camera2body_XYZ;
  camera2body_XYZ << 0.125, 0, 0;// hyper param
  body_pos_(0) = pose->pose.pose.position.x;
  body_pos_(1) = pose->pose.pose.position.y;
  body_pos_(2) = pose->pose.pose.position.z;

  Eigen::Quaterniond body_q_;
  Eigen::Matrix3d Rotation_matrix, camera2body_rotation;
  body_q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
                                 pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);
  // body_q_ = airsim_q_*body_q_;
  Rotation_matrix = body_q_.toRotationMatrix();
  camera2body_rotation << 0, 0, 1,
                          -1, 0, 0,
                          0, -1, 0;
  Eigen::Quaterniond c_to_b(camera2body_rotation);

  camera_pos_ = Rotation_matrix * camera2body_XYZ + body_pos_;
  camera_q_ = body_q_;

  if (!map_->isInMap(camera_pos_))  // exceed mapped region
    return;
  
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  cv_ptr->image.copyTo(*depth_image_);
  
  auto t1 = ros::Time::now();
  // generate point cloud, update map
  proessDepthImage();
  // ------------------------------------------------

  map_->inputPointCloud(point_cloud_, proj_points_cnt, camera_pos_);
  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }

  auto t2 = ros::Time::now();
  fuse_time_ += (t2 - t1).toSec();
  max_fuse_time_ = max(max_fuse_time_, (t2 - t1).toSec());
  fuse_num_ += 1;
  if (show_occ_time_)
    ROS_WARN("Fusion t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), fuse_time_ / fuse_num_,
             max_fuse_time_);
}

void MapROS::cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                               const nav_msgs::OdometryConstPtr& pose) {
  camera_pos_(0) = pose->pose.pose.position.x;
  camera_pos_(1) = pose->pose.pose.position.y;
  camera_pos_(2) = pose->pose.pose.position.z;
  camera_q_ = Eigen::Quaterniond(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
                                 pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);

  Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg, cloud);

  int num = cloud.points.size();

  map_->inputPointCloud(cloud, num, camera_pos_);

  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }
}

void MapROS::proessDepthImage() {
  proj_points_cnt = 0;

  uint16_t* row_ptr;
  int cols = depth_image_->cols;
  int rows = depth_image_->rows;
  double depth;
  Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
  Eigen::Vector3d pt_cur, pt_world;
  const double inv_factor = 1.0 / k_depth_scaling_factor_;

  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    row_ptr = depth_image_->ptr<uint16_t>(v) + depth_filter_margin_;
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {
      depth = (*row_ptr) * inv_factor;
      row_ptr = row_ptr + skip_pixel_;

      // TODO: simplify the logic here
      if (*row_ptr == 0 || depth > depth_filter_maxdist_)
        depth = depth_filter_maxdist_;
      else if (depth < depth_filter_mindist_)
        continue;
      // fixed depth bug in AirSim
      Eigen::Vector3d normal_uvd = Eigen::Vector3d(u, v, 1.0);
      Eigen::Vector3d normal_xyz = K_depth_.inverse() * normal_uvd;
      double length = normal_xyz.norm();
      Eigen::Vector3d xyz = normal_xyz / length * depth;
      pt_cur << xyz(0), xyz(2), -xyz(1);

      // pt_cur(0) = (u - cx_) * depth / fx_;
      // pt_cur(1) = (v - cy_) * depth / fy_;
      // pt_cur(2) = depth;
      pt_world = camera_r * pt_cur + camera_pos_;
      auto& pt = point_cloud_.points[proj_points_cnt++];
      pt.x = pt_world[0];
      pt.y = pt_world[1];
      pt.z = pt_world[2];
    }
  }

  publishDepth();
}

void MapROS::publishMapAll() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;
  for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
    for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
      for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->min_occupancy_log_) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud1.push_back(pt);
        }
      }
  cloud1.width = cloud1.points.size();
  cloud1.height = 1;
  cloud1.is_dense = true;
  cloud1.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud1, cloud_msg);
  map_all_pub_.publish(cloud_msg);

  // Output time and known volumn
  double time_now = (ros::Time::now() - map_start_time_).toSec();
  double known_volumn = 0;

  for (int x = map_->mp_->box_min_(0) /* + 1 */; x < map_->mp_->box_max_(0); ++x)
    for (int y = map_->mp_->box_min_(1) /* + 1 */; y < map_->mp_->box_max_(1); ++y)
      for (int z = map_->mp_->box_min_(2) /* + 1 */; z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->clamp_min_log_ - 1e-3)
          known_volumn += 0.1 * 0.1 * 0.1;
      }

  ofstream file("/home/boboyu/workspaces/plan_ws/src/fast_planner/exploration_manager/resource/"
                "curve1.txt",
                ios::app);
  file << "time:" << time_now << ",vol:" << known_volumn << std::endl;
}

void MapROS::publishMapLocal() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointCloud<pcl::PointXYZ> cloud2;
  Eigen::Vector3i min_cut = map_->md_->local_bound_min_;
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_;
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  // for (int z = min_cut(2); z <= max_cut(2); ++z)
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = map_->mp_->box_min_(2); z < map_->mp_->box_max_(2); ++z) {
        if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->min_occupancy_log_) {
          // Occupied cells
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
        // else if (map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y, z)] == 1)
        // {
        //   // Inflated occupied cells
        //   Eigen::Vector3d pos;
        //   map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
        //   if (pos(2) > visualization_truncate_height_)
        //     continue;
        //   if (pos(2) < visualization_truncate_low_)
        //     continue;

        //   pt.x = pos(0);
        //   pt.y = pos(1);
        //   pt.z = pos(2);
        //   cloud2.push_back(pt);
        // }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  cloud2.width = cloud2.points.size();
  cloud2.height = 1;
  cloud2.is_dense = true;
  cloud2.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_local_pub_.publish(cloud_msg);
  pcl::toROSMsg(cloud2, cloud_msg);
  map_local_inflate_pub_.publish(cloud_msg);
}

void MapROS::publishPredCloud()
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = copy_cloud_GHPR;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = frame_id_;
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  pred_pub_.publish(cloud_msg);
}

void MapROS::publishPred_INTERNAL() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  // Eigen::Vector3i min_cut = map_->md_->local_bound_min_;
  // Eigen::Vector3i max_cut = map_->md_->local_bound_max_;

  Eigen::Vector3i min_cut;
  Eigen::Vector3i max_cut;
  map_->posToIndex(map_->mp_->map_min_boundary_, min_cut);
  map_->posToIndex(map_->mp_->map_max_boundary_, max_cut);
  map_->boundIndex(max_cut);
  map_->boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        // if ((map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] > map_->mp_->clamp_min_log_ - 1e-3 && map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] < map_->mp_->min_occupancy_log_))
        // if (map_->md_->occupancy_buffer_[map_->toAddress(x, y, z)] < map_->mp_->clamp_min_log_ - 2*1e-3)
        if (map_->md_->occupancy_buffer_pred_[map_->toAddress(x, y, z)] == 1)
        {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          // if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

void MapROS::publishDepth() {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int i = 0; i < proj_points_cnt; ++i) {
    cloud.push_back(point_cloud_.points[i]);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  depth_pub_.publish(cloud_msg);
}

void MapROS::publishUpdateRange() {
  Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
  visualization_msgs::Marker mk;
  map_->indexToPos(map_->md_->local_bound_min_, esdf_min_pos);
  map_->indexToPos(map_->md_->local_bound_max_, esdf_max_pos);

  cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
  cube_scale = esdf_max_pos - esdf_min_pos;
  mk.header.frame_id = frame_id_;
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.id = 0;
  mk.pose.position.x = cube_pos(0);
  mk.pose.position.y = cube_pos(1);
  mk.pose.position.z = cube_pos(2);
  mk.scale.x = cube_scale(0);
  mk.scale.y = cube_scale(1);
  mk.scale.z = cube_scale(2);
  mk.color.a = 0.3;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;

  update_range_pub_.publish(mk);
}

void MapROS::publishESDF() {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;

  Eigen::Vector3i min_cut = map_->md_->local_bound_min_ - Eigen::Vector3i(map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_);
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_ + Eigen::Vector3i(map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_);
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      Eigen::Vector3d pos;
      map_->indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = esdf_slice_height_;
      dist = map_->getDistance(pos);
      dist = min(dist, max_dist);
      dist = max(dist, min_dist);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = -0.2;
      pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);

  // ROS_INFO("pub esdf");
}
}