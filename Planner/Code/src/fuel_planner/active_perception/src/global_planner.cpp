#include <active_perception/global_planner.h>
#include <plan_env/sdf_map.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_search.h>
#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/raycast.h>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <visualization_msgs/MarkerArray.h>

#include <lkh_tsp_solver/lkh_interface.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h> 

using namespace std;

namespace fast_planner
{
GlobalPlanner::GlobalPlanner(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh)
{
  // Utils Init
  this->edt_env_ = edt;
  percep_utils_.reset(new PerceptionUtils(nh));

  double resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_.reset(new RayCaster);
  raycaster_->setParams(resolution_, origin);

  // Params
  nh.param("global/sample_r_min_", r_min, -1.0);
  nh.param("global/sample_r_max_", r_max, -1.0);
  nh.param("global/sample_z_size_", z_size, -1.0);
  nh.param("global/sample_z_range_", z_range, -1.0);
  nh.param("global/sample_phi_range_", phi_sample, -1.0);
  nh.param("global/sample_angle_step_", a_step, -1.0);
  nh.param("global/sample_theta_threshold_", theta_thre, -1.0);
  nh.param("global/sample_r_num_", r_num, -1);
  nh.param("global/sample_z_num_", z_step, -1);
  nh.param("global/sample_min_dist_", min_dist, -1.0);
  nh.param("global/tsp_refine_radius_", refine_radius, -1.0);
  nh.param("global/tsp_max_decay_", max_decay, -1.0);
  nh.param("global/downsample_coeff_", downsample_c, -1.0);
  nh.param("global/downsample_each_", downsample_e, -1.0);
  nh.param("global/uniform_grid_size_", grid_size, -1.0);
  nh.param("global/projection_param_", gamma, -1.0);
  nh.param("global/normal_judge_param_", alpha, -1.0);
  nh.param("global/visible_threshold_", visible_ratio, -1.0);
  nh.param("global/tsp_refine_num_", refine_num, -1);
  nh.param("global/tsp_topvp_num_", top_view_num, -1);
  nh.param("global/tsp_dir_", tsp_dir, string("null"));
  nh.param("global/nbv_distlb_", nbv_lb, -1.0);
  nh.param("global/nbv_distub_", nbv_ub, -1.0);
  nh.param("global/dist_cost_", dist_coefficient, -1.0);
  nh.param("global/consistency_cost_", gc_coefficient, -1.0);
  nh.param("global/cluster_pca_diameter_", pca_diameter_thre, -1.0);
  NBV_res = false;
  dir_res = false;
  finish_ = false;
  one_cluster_time = 0;
}

GlobalPlanner::~GlobalPlanner(){
}
// sign functon
int GlobalPlanner::sgn(double& x)
{
  if (x>=0)
    return 1;
  else
    return -1;
}
// CEC condition constrained by normal
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
// PCA for normal calculation
void GlobalPlanner::PCA_algo(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointType& c, PointType& pcZ, PointType& pcY, PointType& pcX, PointType& pcZ_inv, PointType& pcY_inv)
{
  Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
  Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

  // ROS_WARN("PCA_Algo");
  // cout << "values:" << eigenValuesPCA << "." << endl;

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
// uniform partition
pcl::PointCloud<pcl::PointXYZ>::Ptr GlobalPlanner::condition_get(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& x_up, float& x_low,
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
// get normal info
void GlobalPlanner::get_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_vec, Eigen::Matrix3f& normal_vec, bool& flag)
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
// get PCA max diameter for point cloud
double GlobalPlanner::pca_diameter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
  Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

  double r_0 = eigenValuesPCA(0);
  double r_1 = eigenValuesPCA(1);
  double r_2 = eigenValuesPCA(2);
  
  // ROS_WARN("pca diameter");
  // cout << "diameter:" << r_0 << "," << r_1 << "," << r_2 << "." << endl;

  double x1 = max(fabs(r_0), fabs(r_1));  
  double x2 = max(fabs(x1), fabs(r_2)); 

  return x2;
}
// uniform cluster
void GlobalPlanner::uniform_cluster_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_t, 
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results_t)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results_t);
  // step 1: partition the input point cloud
  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*cloud_t, minPt, maxPt);
  clustered_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  float x_low = 0.0, x_up = 0.0, y_low = 0.0, y_up = 0.0, z_low = 0.0, z_up = 0.0;

  float x_range = (maxPt.x - minPt.x)+0.1;
  float y_range = (maxPt.y - minPt.y)+0.1;
  float z_range = (maxPt.z - minPt.z)+0.1;

  int x_num = std::ceil(x_range/grid_size);
  int y_num = std::ceil(y_range/grid_size);
  int z_num = std::ceil(z_range/grid_size);

  float x_start = maxPt.x+1e-2;
  float y_start = maxPt.y+1e-2;
  float z_start = minPt.z-1e-2;
  int counter = 0;
  // step 2: prepare diameter
  double diameter = 0.0;
  // step 3: uniform clustering
  // ----------------
  for (int i=0; i<x_num; ++i)
  {
      for (int j=0; j<y_num; ++j)
      {
          for (int k=0; k<z_num; ++k)
          {
              x_up = x_start - i*grid_size;
              x_low = x_start - (i+1)*grid_size;
              y_up = y_start - j*grid_size;
              y_low = y_start - (j+1)*grid_size;
              z_low = z_start + k*grid_size;
              z_up = z_start + (k+1)*grid_size;
              clustered_cloud = condition_get(cloud_t, x_up, x_low, y_up, y_low, z_up, z_low);
              if (clustered_cloud->points.size()>3)
              {
                diameter = pca_diameter(clustered_cloud);
                if (diameter>pca_diameter_thre)
                {
                  cluster_results_t.push_back(clustered_cloud);
                  counter++;
                }
              }
          }
      }
  }
}
// Point cloud visibility process
pcl::PointCloud<pcl::PointXYZ>::Ptr GlobalPlanner::surface_recon_visibility(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector4f& viewpoint)
{
  /*
  On the Visibility of Point Clouds: https://openaccess.thecvf.com/content_iccv_2015/papers/Katz_On_the_Visibility_ICCV_2015_paper.pdf
  */
  auto startT = std::chrono::high_resolution_clock::now();
    // ------process start---------
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
    // cout << "vis_time:" << vis_ms.count() << "ms" << std::endl;

    return cloud_inverse;
}
// Conditional Euclidean Cluster
void GlobalPlanner::conditional_ec(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results);
  const double leaf_size = edt_env_->sdf_map_->getResolution()*downsample_c;

  downsample_cec.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> ds;
  ds.setInputCloud(cloud);
  ds.setLeafSize(leaf_size, leaf_size, leaf_size);
  ds.filter(*downsample_cec);
  cout << "downsample_size:" << downsample_cec->points.size() << endl;
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
// clustering
void GlobalPlanner::clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results);
  const double leaf_size = edt_env_->sdf_map_->getResolution()*downsample_c;

  downsample_cec.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> ds;
  ds.setInputCloud(cloud);
  ds.setLeafSize(leaf_size, leaf_size, leaf_size);
  ds.filter(*downsample_cec);

  uniform_cluster_with_normal(downsample_cec, cluster_results);
}
// normal choose for each cluster
void GlobalPlanner::normal_vis()
{
  vector<cluster_normal>().swap(uni_cnv);
  vector<Eigen::Vector3d>().swap(uni_center);
  vector<Eigen::Vector3d>().swap(cls_normal);
  const double voxel_size = edt_env_->sdf_map_->getResolution()*downsample_e;
  cluster_normal temp_cnv_p;
  cluster_normal temp_cnv_n;
  int counter_normal = 0;

  Eigen::Vector3d normal_0;
  Eigen::Vector3d normal_1;
  Eigen::Vector3d center;
  
  for (int i=0; i<uni_cluster.size(); ++i)
  {
    // cout << "uni_cluster:" << uni_cluster[i]->points.size() << endl;
    
    bool have_normal = false;
    bool type_normal = true;
    Eigen::Matrix3f normal_Z;
    Eigen::Matrix3f normal_Y;
    get_normal(uni_cluster[i], normal_Z, type_normal);

    center(0) = normal_Z(0,0);
    center(1) = normal_Z(0,1);
    center(2) = normal_Z(0,2);
    
    normal_0(0) = normal_Z(1,0);
    normal_0(1) = normal_Z(1,1);
    normal_0(2) = normal_Z(1,2);
    int state_0 = normal_judge(center, normal_0, alpha);
    
    if (state_0 != 1)
    {
      // have normal
      have_normal = true;
      // downsample
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds_0 (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> ds_0;
      ds_0.setInputCloud(uni_cluster[i]);
      ds_0.setLeafSize(voxel_size, voxel_size, voxel_size);
      ds_0.filter(*cloud_ds_0);
      
      vector<Eigen::Vector3d> vec_0;
      for (int h=0; h<cloud_ds_0->points.size(); ++h)
      {
        Eigen::Vector3d temp_cell_0(cloud_ds_0->points[h].x, cloud_ds_0->points[h].y, cloud_ds_0->points[h].z);
        vec_0.push_back(temp_cell_0);
      }
      vector<Eigen::Vector3d> vec_0_l;
      for (int h=0; h<uni_cluster[i]->points.size(); ++h)
      {
        Eigen::Vector3d temp_cell_0_l(uni_cluster[i]->points[h].x, uni_cluster[i]->points[h].y, uni_cluster[i]->points[h].z);
        vec_0_l.push_back(temp_cell_0_l);
      }

      temp_cnv_p.global_cells_ = vec_0;
      temp_cnv_p.local_cells_ = vec_0_l;
      temp_cnv_p.center_ = center;
      temp_cnv_p.normal_ = normal_0-center;
      temp_cnv_p.id_g = counter_normal;
      counter_normal++;
      uni_cnv.push_back(temp_cnv_p);
      uni_center.push_back(center);
      cls_normal.push_back(normal_0);
    }
    normal_1(0) = normal_Z(2,0);
    normal_1(1) = normal_Z(2,1);
    normal_1(2) = normal_Z(2,2);
    int state_1 = normal_judge(center, normal_1, alpha);
    
    if (state_1 != 1)
    {
      // have normal
      have_normal = true;
      // downsample
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds_1 (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> ds_1;
      ds_1.setInputCloud(uni_cluster[i]);
      ds_1.setLeafSize(voxel_size, voxel_size, voxel_size);
      ds_1.filter(*cloud_ds_1);

      vector<Eigen::Vector3d> vec_1;
      for (int g=0; g<cloud_ds_1->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1(cloud_ds_1->points[g].x, cloud_ds_1->points[g].y, cloud_ds_1->points[g].z);
        vec_1.push_back(temp_cell_1);
      }
      vector<Eigen::Vector3d> vec_1_l;
      for (int g=0; g<uni_cluster[i]->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1_l(uni_cluster[i]->points[g].x, uni_cluster[i]->points[g].y, uni_cluster[i]->points[g].z);
        vec_1_l.push_back(temp_cell_1_l);
      }

      temp_cnv_n.global_cells_ = vec_1;
      temp_cnv_n.local_cells_ = vec_1_l;
      temp_cnv_n.center_ = center;
      temp_cnv_n.normal_ = normal_1-center;
      temp_cnv_n.id_g = counter_normal;
      counter_normal++;
      uni_cnv.push_back(temp_cnv_n);
      uni_center.push_back(center);
      cls_normal.push_back(normal_1);
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
        // downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds_2 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> ds_2;
        ds_2.setInputCloud(uni_cluster[i]);
        ds_2.setLeafSize(voxel_size, voxel_size, voxel_size);
        ds_2.filter(*cloud_ds_2);

        vector<Eigen::Vector3d> vec_0;
        for (int h=0; h<cloud_ds_2->points.size(); ++h)
        {
          Eigen::Vector3d temp_cell_0(cloud_ds_2->points[h].x, cloud_ds_2->points[h].y, cloud_ds_2->points[h].z);
          vec_0.push_back(temp_cell_0);
        }
        vector<Eigen::Vector3d> vec_0_l;
        for (int h=0; h<uni_cluster[i]->points.size(); ++h)
        {
          Eigen::Vector3d temp_cell_0_l(uni_cluster[i]->points[h].x, uni_cluster[i]->points[h].y, uni_cluster[i]->points[h].z);
          vec_0_l.push_back(temp_cell_0_l);
        }

        temp_cnv_p.global_cells_ = vec_0;
        temp_cnv_p.local_cells_ = vec_0_l;
        temp_cnv_p.center_ = center;
        temp_cnv_p.normal_ = normal_0-center;
        temp_cnv_p.id_g = counter_normal;
        counter_normal++;
        uni_cnv.push_back(temp_cnv_p);
        uni_center.push_back(center);
        cls_normal.push_back(normal_0);
      }

      normal_1(0) = normal_Y(2,0);
      normal_1(1) = normal_Y(2,1);
      normal_1(2) = normal_Y(2,2);
      int state_1 = normal_judge(center, normal_1, alpha);
      if (state_1 != 1)
      {
        // have normal
        have_normal = true;
        // downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds_3 (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> ds_3;
        ds_3.setInputCloud(uni_cluster[i]);
        ds_3.setLeafSize(voxel_size, voxel_size, voxel_size);
        ds_3.filter(*cloud_ds_3);

        vector<Eigen::Vector3d> vec_1;
        for (int g=0; g<cloud_ds_3->points.size(); ++g)
        {
          Eigen::Vector3d temp_cell_1(cloud_ds_3->points[g].x, cloud_ds_3->points[g].y, cloud_ds_3->points[g].z);
          vec_1.push_back(temp_cell_1);
        }
        vector<Eigen::Vector3d> vec_1_l;
        for (int g=0; g<uni_cluster[i]->points.size(); ++g)
        {
          Eigen::Vector3d temp_cell_1_l(uni_cluster[i]->points[g].x, uni_cluster[i]->points[g].y, uni_cluster[i]->points[g].z);
          vec_1_l.push_back(temp_cell_1_l);
        }

        temp_cnv_n.global_cells_ = vec_1;
        temp_cnv_n.local_cells_ = vec_1_l;
        temp_cnv_n.center_ = center;
        temp_cnv_n.normal_ = normal_1-center;
        temp_cnv_n.id_g = counter_normal;
        counter_normal++;
        uni_cnv.push_back(temp_cnv_n);
        uni_center.push_back(center);
        cls_normal.push_back(normal_1);
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

      // int state_0 = normal_judge(center, normal_1, alpha);
      // if (state_0 != 1)
      // {
      // downsample
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds_4 (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> ds_4;
      ds_4.setInputCloud(uni_cluster[i]);
      ds_4.setLeafSize(voxel_size, voxel_size, voxel_size);
      ds_4.filter(*cloud_ds_4);
      
      vector<Eigen::Vector3d> vec_1;
      for (int g=0; g<cloud_ds_4->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1(cloud_ds_4->points[g].x, cloud_ds_4->points[g].y, cloud_ds_4->points[g].z);
        vec_1.push_back(temp_cell_1);
      }
      vector<Eigen::Vector3d> vec_1_l;
      for (int g=0; g<uni_cluster[i]->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1_l(uni_cluster[i]->points[g].x, uni_cluster[i]->points[g].y, uni_cluster[i]->points[g].z);
        vec_1_l.push_back(temp_cell_1_l);
      }

      temp_cnv_n.global_cells_ = vec_1;
      temp_cnv_n.local_cells_ = vec_1_l;
      temp_cnv_n.center_ = center;
      temp_cnv_n.normal_ = normal_1;
      temp_cnv_n.id_g = counter_normal;
      counter_normal++;
      uni_cnv.push_back(temp_cnv_n);
      uni_center.push_back(center);
      cls_normal.push_back(normal_1+center);

      have_normal = true;
      // }
    }
    if (have_normal == false)
    {
      ROS_ERROR("No Normal!");
    }
  }
}
// normal chooser, 0 is qualified and 1 is not.
int GlobalPlanner::normal_judge(Eigen::Vector3d& center, Eigen::Vector3d& pos, const float& coeff)
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
        int state = edt_env_->sdf_map_->get_PredStates(temp_xp);
        if (state == 3)
        {
          return 1;
        }
      }
    }
  }

  return 0;
}
// calculate visible cells for viewpoint sample
int GlobalPlanner::cal_visibility_cells(const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& set)
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
        if (edt_env_->sdf_map_->get_PredStates(idx) == SDFMap::PRED_INTERNAL ||
            !edt_env_->sdf_map_->isInBox(idx) || edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::OCCUPIED) {
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
// dual pillar viewpoints sampling
void GlobalPlanner::sample_vp_global_pillar(cluster_normal& set, int& qualified_vp)
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
      for (double zc = -0.5*z_size, dz = z_size / z_step; zc <= 0.5*z_size - 1e-3; zc += dz)
      {
        for (double pc = phi-phi_sample; pc < phi+phi_sample+1e-3; pc+=a_step)
        {
          Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(rc*sign_x*cos(pc), rc*sin(pc), zc);

          float coe_sample = 1.0;
          int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
          if (!edt_env_->sdf_map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;
          // compute average yaw
          auto& pred_cells = set.global_cells_;
          int lower_vis_bound = ceil(visible_ratio*pred_cells.size());
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
      if (ratio > 0.5*visible_ratio)
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
          // for (int r=1; r<pred_cells.size(); ++r)
          // {
          //   Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
          //   double yaw = acos(dir.dot(start_dir));
          //   if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
          //   avg_yaw_ += yaw;
          // }
          // avg_yaw_ = avg_yaw_ / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
          avg_yaw_ = atan2(-normal_n(1), -normal_n(0));
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
    for (double rc = 0.6*r_min, dr = (r_max - r_min) / (2*r_num); rc <= 0.6*r_max + 1e-3; rc += dr)
    {
      for (double zc = sign_z*z_range, dz = sign_z*z_range / z_step; fabs(zc) >= 0.5*z_range; zc -= dz)
      {
        for (double pc = 0.0; pc < 2*M_PI+1e-3; pc+=2*a_step)
        {
          Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(rc*cos(pc), rc*sin(pc), zc);

          float coe_sample = 1.0;
          int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
          if (!edt_env_->sdf_map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;
          // compute average yaw
          auto& pred_cells = set.global_cells_;
          int lower_vis_bound = ceil(visible_ratio*pred_cells.size());
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
      if (ratio > 0.5*visible_ratio)
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
// global coverage target point
void GlobalPlanner::global_cover_tp(cluster_normal& set, int& qualified_vp)
{
  // validate observation
  // bool inside = edt_env_->sdf_map_->setcenter_check(set.center_, 2);
  if (set.center_(2) < 0.3)
  {
    qualified_vp = -1;
    return;
  }

  float z_min = 1.0;

  Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  Eigen::Vector3d z_axis_nega(0.0, 0.0, -1.0);

  double theta;
  if (set.normal_(2) >= 0)
  {
    theta = acos(set.normal_.dot(z_axis)/(set.normal_.norm()*z_axis.norm()));
    while (theta < -M_PI)
      theta += 2 * M_PI;
    while (theta > M_PI)
      theta -= 2 * M_PI;
    theta = fabs(theta);
  }
  else
  {
    theta = acos(set.normal_.dot(z_axis_nega)/(set.normal_.norm()*z_axis_nega.norm()));
    while (theta < -M_PI)
      theta += 2 * M_PI;
    while (theta > M_PI)
      theta -= 2 * M_PI;
    theta = fabs(theta);
  }
  int sign_z = sgn(set.normal_(2));
  // if theta near plane
  if (theta > theta_thre)
  {
    Eigen::Vector3d sample_pos = set.center_ + r_min * set.normal_.normalized();
    if (sample_pos(2) < 1.0)
    {
      sample_pos(2) = set.center_(2);
    }
    // compute average yaw
    auto& pred_cells = set.global_cells_;
    int lower_vis_bound = ceil(visible_ratio*pred_cells.size());
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
    VP_global vp_g = {sample_pos, avg_yaw, visib_num};
    set.vps_global.push_back(vp_g);
    qualified_vp++;
  }
  // else if (theta <= theta_thre && theta > 20.0*M_PI/180.0)
  // {
  //   // ROS_ERROR("Z-axis Target Point.");
  //   Eigen::Vector3d sample_pos = set.center_ + 0.5*r_min * set.normal_.normalized();
  //   // compute average yaw
  //   auto& pred_cells = set.global_cells_;
  //   int lower_vis_bound = ceil(visible_ratio*pred_cells.size());
  //   Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
  //   double avg_yaw = 0.0;
  //   for (int r=1; r<pred_cells.size(); ++r)
  //   {
  //     Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
  //     double yaw = acos(dir.dot(start_dir));
  //     if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
  //     avg_yaw += yaw;
  //   }
  //   avg_yaw = avg_yaw / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
  //   // constrain yaw
  //   while (avg_yaw < -M_PI)
  //     avg_yaw += 2 * M_PI;
  //   while (avg_yaw > M_PI)
  //     avg_yaw -= 2 * M_PI;
    
  //   int visib_num = cal_visibility_cells(sample_pos, avg_yaw, pred_cells);
  //   VP_global vp_g = {sample_pos, avg_yaw, visib_num};
  //   set.vps_global.push_back(vp_g);
  //   qualified_vp++;
  // }
  // else if (theta <= 20.0*M_PI/180.0)
  else
  {
    for (double pc = 0.0; pc < 2*M_PI+1e-3; pc+=a_step)
    {
    Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(1.4*r_min*cos(pc), 1.4*r_min*sin(pc), 1.5);
    if (sample_pos(2) < 1.0)
    {
      sample_pos(2) = set.center_(2);
    }
    int sample_state = normal_judge(sample_pos, sample_pos, 1.0);
    if (!edt_env_->sdf_map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
      continue;
    // compute average yaw
    auto& pred_cells = set.global_cells_;
    int lower_vis_bound = ceil(visible_ratio*pred_cells.size());
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
    if (visib_num > lower_vis_bound)
    {
      VP_global vp_g = {sample_pos, avg_yaw, visib_num};
      set.vps_global.push_back(vp_g);
      qualified_vp++;
    }
    }
  }
}
// cost mat element
void GlobalPlanner::CostMat()
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
}
// get full cost mat
void GlobalPlanner::fullCostMatrix(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    Eigen::MatrixXd& mat, const Eigen::Vector3d& g_dir)
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

  Eigen::Vector3d candidates_dir;
  double dist_cost = 0.0;
  double consistency_cost = 0.0;
  for (auto cnv : uni_open_cnv)
  {
    VP_global vp = cnv.vps_global.front();
    vector<Eigen::Vector3d> path;
    if (dir_res == false)
    {
      dist_cost = ViewNode::compute_globalCost(cur_pos, vp.pos_g, cur_yaw[0], vp.yaw_g, cur_vel, cur_yaw[1], path);
      mat(0, j++) = dist_cost;
    }
    else if (dir_res == true)
    {
      candidates_dir = (vp.pos_g - cur_pos).normalized();
      double diff_angle = acos(g_dir.dot(candidates_dir));
      while (diff_angle < -M_PI)
        diff_angle += 2 * M_PI;
      while (diff_angle > M_PI)
        diff_angle -= 2 * M_PI;

      consistency_cost = fabs(diff_angle);
      dist_cost = ViewNode::compute_globalCost(cur_pos, vp.pos_g, cur_yaw[0], vp.yaw_g, cur_vel, cur_yaw[1], path);
      // ROS_WARN("Log Global Cost!");
      // cout << "cost_1:" << dist_cost << endl;
      // cout << "cost_2:" << consistency_cost << endl;
      mat(0, j++) = dist_coefficient*dist_cost + gc_coefficient*consistency_cost;
    }
  }
}
// get full path
void GlobalPlanner::get_GPath(const Eigen::Vector3d& pos, const vector<int>& ids, vector<Eigen::Vector3d>& path)
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
  // local_region order
  vector<vector<Eigen::Vector3d>>().swap(local_regions_);
  for (int j=0; j<cnv_indexer.size(); ++j)
  {
    local_regions_.push_back(cnv_indexer[ids[j]]->local_cells_);
  }
  // local_region normal
  local_normal_ = cnv_indexer[ids[0]]->normal_;
}
// gt viewpoints info
void GlobalPlanner::get_vpinfo(const Eigen::Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
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
// initial global path
bool GlobalPlanner::global_path(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    vector<int>& order, const Eigen::Vector3d& cur_dir)
{
  vector<Eigen::Vector3d>().swap(global_tour);
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
  // cout << "<<cost_calculate_time>>" << cost_ms.count() << "ms" << std::endl;
  // ---------------------------------------------------
  Eigen::MatrixXd cost_mat;
  fullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat, cur_dir);
  const int dimension = cost_mat.rows();
  // finish
  if (dimension < 3)
  {
    return true;
  }
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
  /* --- prepare for refinement process --- */
  // vector<int>().swap(refined_ids);
  // vector<Eigen::Vector3d>().swap(unrefined_points);
  // int knum = refine_num;
  // for (int i=0; i<knum; ++i)
  // {
  //   auto tmp = global_points[order[i]];
  //   unrefined_points.push_back(tmp);
  //   refined_ids.push_back(order[i]);
  //   if ((tmp - cur_pos).norm() > refine_radius && refined_ids.size() >= 3) break;
  // }

  return false;
}
// refined global path through Dijkstra
void GlobalPlanner::global_refine(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw)
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
// global planning manager
bool GlobalPlanner::GlobalPathManager(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& visit, pcl::PointCloud<pcl::PointXYZ>::Ptr& cur_map, const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw, const Eigen::Vector3d& cur_g_dir)
{
  global_cur_pos = cur_pos;
  double total_time = 0.0;
  // Prediction Point Cloud Preprocess
  auto pp_t1 = std::chrono::high_resolution_clock::now();
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud_prediction.reset(new pcl::PointCloud<pcl::PointXYZ>);// Instantiation of point cloud
  cloud_GHPR.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*input_cloud, *cloud_prediction);
  //** use visit or whole prediction *//
  pcl::compute3DCentroid(*visit, viewpoint);
  pcl::compute3DCentroid(*cloud_prediction, all_viewpoint);
  //** ----- **//
  auto pp_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> pp_ms = pp_t2 - pp_t1;
  cout << "preprocess_time:" << pp_ms.count() << "ms" << std::endl;
  total_time += pp_ms.count();

  // Conditional Euclidean Clustering
  auto startTime = std::chrono::high_resolution_clock::now();
  clustering(visit, uni_cluster);
  cout << "-----------cluster_size:" << uni_cluster.size() << endl;
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cec_ms = endTime - startTime;
  cout << "CEC_time:" << cec_ms.count() << "ms" << std::endl;
  total_time += cec_ms.count();

  // Visibility Point Cloud
  auto vis_t1 = std::chrono::high_resolution_clock::now();

  // cloud_GHPR = surface_recon_visibility(cloud_prediction, viewpoint);
  cloud_GHPR = cloud_prediction;
  
  pcl::PointXYZ min_;
  pcl::PointXYZ max_;
  // pcl::getMinMax3D(*cloud_GHPR,min_,max_);
  pcl::getMinMax3D(*cloud_prediction,min_,max_);
  double x_mid = 0.5*(min_.x+max_.x);
  double y_mid = 0.5*(min_.y+max_.y);
  double z_mid = 0.5*(min_.z+max_.z);
  double x_back = 0.7*(min_.x+max_.x);
  double y_back = 0.7*(min_.y+max_.y);
  Eigen::Vector3d center(x_mid, y_mid, z_mid);
  // Eigen::Vector3d center(viewpoint[0], viewpoint[1], viewpoint[2]);
  Eigen::Vector3d center_back(x_back, y_back, z_mid);
  // Eigen::Vector3d center_back(all_viewpoint[0], all_viewpoint[1], all_viewpoint[2]);
  Eigen::Vector3d center_ground(x_mid, y_mid, min_.z);
  auto vis_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> vis_ms = vis_t2 - vis_t1;
  cout << "visibility_time:" << vis_ms.count() << "ms" << std::endl;
  total_time += vis_ms.count();

  // Update Prediction States
  auto update_t1 = std::chrono::high_resolution_clock::now();
  cloud = *cloud_GHPR;
  // pcl::copyPointCloud(*cloud_GHPR, copy_cloud_GHPR);
  pcl::copyPointCloud(*visit, copy_cloud_GHPR);
  int num = cloud.points.size();
  cout << "cloud number:" << num << endl;
  edt_env_->sdf_map_->inputPredictionCloud(cloud, num, center);
  edt_env_->sdf_map_->inputPredictionCloud(cloud, num, center_back);
  edt_env_->sdf_map_->inputPredictionCloud(cloud, num, center_ground);
  auto update_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> update_ms = update_t2 - update_t1;
  cout << "update_time:" << update_ms.count() << "ms" << std::endl;
  total_time += update_ms.count();

  // Normal Process
  auto ds_t1 = std::chrono::high_resolution_clock::now();
  normal_vis();
  cout << "partition_size:" << uni_cnv.size() << endl;
  auto ds_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ds_ms = ds_t2 - ds_t1;
  cout << "normal_select_time:" << ds_ms.count() << "ms" << std::endl;
  total_time += ds_ms.count();

  // Dual Sampling: sample viewpoints and send them to open_cnv(have qualified viewpoints)
  auto sp_t1 = std::chrono::high_resolution_clock::now();
  vector<cluster_normal>().swap(uni_open_cnv);
  cluster_normal open_cnv;
  int open_id = 0;
  for (int i=0; i<uni_cnv.size(); ++i)
  {
    int q_vp = 0;
    // sample_vp_global_pillar(uni_cnv[i], q_vp);
    global_cover_tp(uni_cnv[i], q_vp);
    if (q_vp > 0)
    {
      open_cnv.global_cells_ = uni_cnv[i].global_cells_;
      open_cnv.local_cells_ = uni_cnv[i].local_cells_;
      open_cnv.center_ = uni_cnv[i].center_;
      open_cnv.normal_ = uni_cnv[i].normal_;
      open_cnv.vps_global = uni_cnv[i].vps_global;

      std::sort(open_cnv.vps_global.begin(), open_cnv.vps_global.end(),
      [](const VP_global& v1, const VP_global& v2) { return v1.visib_num_g > v2.visib_num_g; });

      open_cnv.id_g = open_id;
      uni_open_cnv.push_back(open_cnv);
      open_id++;
    }
  }
  cout << "qualified_partition_size:" << uni_open_cnv.size() << endl;
  auto sp_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> sp_ms = sp_t2 - sp_t1;
  cout << "sample_time:" << sp_ms.count() << "ms" << std::endl;
  total_time += sp_ms.count();

  // --------------sample end----------------------------------------------
  // Global TSP and Refinement
  auto cost_t1 = std::chrono::high_resolution_clock::now();
  vector<int> tsp_id;

  Eigen::Vector3d next_pos;
  double next_yaw;

  if (uni_open_cnv.size() >= 2)
  {
    finish_ = global_path(cur_pos, cur_vel, cur_yaw, tsp_id, cur_g_dir);
    // global_refine(cur_pos, cur_vel, cur_yaw);

    // Eigen::Vector3d cn_dir;
    // double len_cton;
    // for (int i=0; i<refined_n_points.size(); ++i)
    // {
    //   len_cton = (refined_n_points[i]-cur_pos).norm();
    //   if (len_cton > nbv_lb)
    //   {
    //     ROS_WARN("Qualified NBV!");
    //     cout << "Cur to NBV:" << len_cton << endl;
    //     NBV_res = true;
    //     if (len_cton < nbv_ub)
    //     {
    //       next_pos = refined_n_points[i];
    //       next_yaw = refined_n_yaws[i];
    //     }
    //     else
    //     {
    //       cn_dir = (refined_n_points[i]-cur_pos).normalized();
    //       next_pos = cur_pos + 0.8*cn_dir*nbv_ub;
    //       next_yaw = cur_yaw(0) + 0.8*(refined_n_yaws[i]-cur_yaw(0))*nbv_ub/len_cton;
    //     }
    //     break;
    //   }
    // }
    if ((cur_pos-uni_open_cnv[tsp_id[0]].vps_global.front().pos_g).norm() > 4.0)
    {
      next_pos = uni_open_cnv[tsp_id[0]].vps_global.front().pos_g;
      next_yaw = uni_open_cnv[tsp_id[0]].vps_global.front().yaw_g;
    }
    else if ((cur_pos-uni_open_cnv[tsp_id[1]].vps_global.front().pos_g).norm() > 3.0)
    {
      next_pos = uni_open_cnv[tsp_id[1]].vps_global.front().pos_g;
      next_yaw = uni_open_cnv[tsp_id[1]].vps_global.front().yaw_g;
    }
    else
    {
      next_pos = uni_open_cnv[tsp_id[2]].vps_global.front().pos_g;
      next_yaw = uni_open_cnv[tsp_id[2]].vps_global.front().yaw_g;
    }

    int next_visib = 10;
    NBV_res = true;
    
    NBV = {next_pos, next_yaw, next_visib};
    /* TODO */
    global_dir = (next_pos - cur_pos).normalized();
    dir_res = true;
  }

  else if (uni_open_cnv.size() == 1)
  {
    next_pos = uni_open_cnv[0].vps_global.front().pos_g;
    next_yaw = uni_open_cnv[0].vps_global.front().yaw_g;

    int next_visib = 10;
    NBV_res = true;
    
    NBV = {next_pos, next_yaw, next_visib};
    one_cluster_time++;
  }

  else if (uni_open_cnv.size() == 0)
  {
    return true;
  }

  auto cost_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cost_ms = cost_t2 - cost_t1;
  cout << "TSP_time:" << cost_ms.count() << "ms" << std::endl;
  total_time += cost_ms.count();

  ROS_WARN("global nbv yaw!");
  cout << next_yaw*180.0/M_PI << "deg." << endl;
  cout << "total_global:" << total_time << "ms" << endl;

  if (one_cluster_time > 10)
    return true;

  return false;
}
}