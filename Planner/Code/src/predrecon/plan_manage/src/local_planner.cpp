#include <plan_manage/local_planner.h>
#include <active_perception/global_planner.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_search.h>
#include <active_perception/graph_node.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <plan_env/raycast.h>
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
LocalPlanner::LocalPlanner(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh)
{
  // Utils Init
  this->env_ = edt;
  perceptron_.reset(new PerceptionUtils(nh));

  double resolution_ = env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  raycheck_.reset(new RayCaster);
  raycheck_->setParams(resolution_, origin);
  // Params
  nh.param("local/region_max_size_", uniform_range, -1.0);
  nh.param("local/downsample_factor_", ds_fac, -1.0);
  nh.param("local/normal_length_", normal_param, -1.0);
  nh.param("local/vp_min_radius_", r_min, -1.0);
  nh.param("local/vp_max_radius_", r_max, -1.0);
  nh.param("local/vp_z_size_", z_size, -1.0);
  nh.param("local/vp_z_range_", z_range_sp, -1.0);
  nh.param("local/vp_angle_step_", angle_step, -1.0);
  nh.param("local/vp_phi_range_", phi_range, -1.0);
  nh.param("local/vp_theta_upper_", theta_upper, -1.0);
  nh.param("local/vp_r_num_", r_step, -1);
  nh.param("local/vp_z_num_", z_step, -1);
  nh.param("local/vp_visible_threshold_", visible_lower, -1.0);
  nh.param("local/tsp_dir_", tsp_dir_, string("null"));
  nh.param("local/vp_pseudo_bias_", pseudo_bias, -1.0);
  nh.param("local/interval", local_interval, -1.0);
  nh.param("local/local_normal_threshold", local_normal_thre, -1.0);
  nh.param("local/cluster_pca_diameter_", local_pca_thre, -1.0);
}

LocalPlanner::~LocalPlanner(){
}

/* Condition Func */
bool NormalCondition (const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float /*squared_distance*/)
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
	if(angle_deg < threshold_angle || angle_deg > (180.0-threshold_angle))	return true;//&& (distance < 10.0)
	else	return false;

  // wo using normal ------
  // return true;
}
/* Sign Func */
int LocalPlanner::sgn(double& x)
{
  if (x>=0)
    return 1;
  else
    return -1;
}
/* PCA algorithm */
void LocalPlanner::PCA_algo(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointXYZ& c, pcl::PointXYZ& pcZ, pcl::PointXYZ& pcY, pcl::PointXYZ& pcX, pcl::PointXYZ& pcZ_inv, pcl::PointXYZ& pcY_inv)
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
/* partition region segmentation */
pcl::PointCloud<pcl::PointXYZ>::Ptr LocalPlanner::condition_get(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& x_up, float& x_low,
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
/* Normal generation */
void LocalPlanner::get_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_vec, Eigen::Matrix3f& normal_vec, bool& flag)
{
  pcl::PointXYZ temp_c, pcz, pcy, pcx, pcz_inv, pcy_inv;
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
/* get PCA max diameter for point cloud */
double LocalPlanner::pca_diameter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
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
/* uniform cluster */
void LocalPlanner::uniform_cluster_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_t, 
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results_t)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results_t);
  /* step 1: partition the input point cloud */
  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*cloud_t, minPt, maxPt);
  tmp_cluster.reset(new pcl::PointCloud<pcl::PointXYZ>);
  float x_low = 0.0, x_up = 0.0, y_low = 0.0, y_up = 0.0, z_low = 0.0, z_up = 0.0;

  float x_range = maxPt.x - minPt.x;
  float y_range = maxPt.y - minPt.y;
  float z_range = maxPt.z - minPt.z;

  int x_num = std::ceil(x_range/uniform_range);
  int y_num = std::ceil(y_range/uniform_range);
  int z_num = std::ceil(z_range/uniform_range);

  float x_start = minPt.x;
  float y_start = minPt.y;
  float z_start = minPt.z;
  int counter_l = 0;

  double diameter = 0.0;
  /* step 2: uniform clustering */
  for (int i=0; i<x_num; ++i)
  {
      for (int j=0; j<y_num; ++j)
      {
          for (int k=0; k<z_num; ++k)
          {
              x_low = x_start + i*uniform_range;
              x_up = x_start + (i+1)*uniform_range;
              y_low = y_start + j*uniform_range;
              y_up = y_start + (j+1)*uniform_range;
              z_low = z_start + k*uniform_range;
              z_up = z_start + (k+1)*uniform_range;
              tmp_cluster = condition_get(cloud_t, x_up, x_low, y_up, y_low, z_up, z_low);
              diameter = pca_diameter(tmp_cluster);
              if (tmp_cluster->points.size()>3)
              {
                // if (counter_l > 0)
                // {
                //   if (tmp_cluster->points.size() < 10)
                //     *cluster_results_t.back() = *cluster_results_t.back() + *tmp_cluster;
                //   else if(cluster_results_t.back()->points.size() < 10)
                //     *cluster_results_t.back() = *cluster_results_t.back() + *tmp_cluster;
                //   else
                //   {
                //     cluster_results_t.push_back(tmp_cluster);
                //     counter_l++;
                //   }
                // }
                // else
                // {
                //   cluster_results_t.push_back(tmp_cluster);
                //   counter_l++;
                // }
                diameter = pca_diameter(tmp_cluster);
                if (diameter > local_pca_thre && diameter < uniform_range)
                {
                  cluster_results_t.push_back(tmp_cluster);
                  counter_l++;
                }
              }
          }
      }
  }
}
/* judge normal direction, 1 means hit, 0 means safe. */
int LocalPlanner::normal_judge(Eigen::Vector3d& center, Eigen::Vector3d& pos, const float& coeff)
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
        int state = env_->sdf_map_->get_PredStates(temp_xp);
        if (state == 3)
        {
          return 1;
        }
      }
    }
  }

  return 0;
}
/* compute visible cells */
int LocalPlanner::cal_visibility_cells(const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& set)
{
  perceptron_->setPose(pos, yaw);
  int visib_num = 0;
  for (auto cell : set)
  {
    bool vis = true;
    Eigen::Vector3i idx;
    if (perceptron_->insideFOV(cell))
    {
      raycheck_->input(pos, cell);
      while (raycheck_->nextId(idx)) 
      {
        if (env_->sdf_map_->get_PredStates(idx) == SDFMap::PRED_INTERNAL ||
            !env_->sdf_map_->isInBox(idx) || env_->sdf_map_->getOccupancy(idx) == SDFMap::OCCUPIED) {
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
/* Normal-conditional Euclidean cluster */
void LocalPlanner::conditional_ec(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results);
  const double leaf_size = env_->sdf_map_->getResolution()*ds_fac;

  ds_region.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> ds;
  ds.setInputCloud(cloud);
  ds.setLeafSize(leaf_size, leaf_size, leaf_size);
  ds.filter(*ds_region);
  // normal set
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;	
  ne.setInputCloud(ds_region);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZ>);
  ne.setSearchMethod(normal_tree);
  ds_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*ds_region, *ds_normals);
  ne.setKSearch(10);
  ne.compute(*ds_normals);
  // Conditional Euclidean Cluster
  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
  pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec(true);
  cec.setInputCloud (ds_normals);
  cec.setConditionFunction (&NormalCondition);
  cec.setClusterTolerance (2.0*leaf_size);
  cec.setMinClusterSize (10);
  cec.setMaxClusterSize (ds_normals->size () / 5);
  cec.segment (*clusters);
  cec.getRemovedClusters(small_clusters, large_clusters);
  // send to uniform cluster
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> final_cluster_results;
  // assemble small as one
  if (small_clusters->size() > 0)
  {
      pcl::PointCloud<pcl::PointXYZ>::Ptr small_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& small_cluster : (*small_clusters))
        {
        pcl::PointXYZ temp_small;
        for (const auto& j : small_cluster.indices)
        {
            temp_small.x = (*ds_normals)[j].x;
            temp_small.y = (*ds_normals)[j].y;
            temp_small.z = (*ds_normals)[j].z;
            small_cloud->points.push_back(temp_small);
        }
        }
      final_cluster_results.push_back(small_cloud);
  }
  // clusters
  if (clusters->size() > 0)
  {
      for (const auto& cluster : (*clusters))
        {
        pcl::PointCloud<pcl::PointXYZ>::Ptr n_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ temp_n;
        for (const auto& j : cluster.indices)
        {
            temp_n.x = (*ds_normals)[j].x;
            temp_n.y = (*ds_normals)[j].y;
            temp_n.z = (*ds_normals)[j].z;
            n_cloud->points.push_back(temp_n);
        }
        final_cluster_results.push_back(n_cloud);
        }
  }
  // large clusters
  if (large_clusters->size() > 0)
  {
      for (const auto& large_cluster : (*large_clusters))
        {
        pcl::PointCloud<pcl::PointXYZ>::Ptr large_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ temp_large;
        for (const auto& j : large_cluster.indices)
        {
            temp_large.x = (*ds_normals)[j].x;
            temp_large.y = (*ds_normals)[j].y;
            temp_large.z = (*ds_normals)[j].z;
            large_cloud->points.push_back(temp_large);
        }
        final_cluster_results.push_back(large_cloud);
        }
  }
  // uniform segmentation
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> us_cluster_results;
  for (int i=0; i<final_cluster_results.size(); ++i)
  {
      vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> temp_cluster_results;
      uniform_cluster_with_normal(final_cluster_results[i], temp_cluster_results);
      us_cluster_results.insert(us_cluster_results.end(),temp_cluster_results.begin(),temp_cluster_results.end());
  }
  // Merge
  int cc = 0;
  if (us_cluster_results.size() <= 1)
  {
    for (auto i:us_cluster_results)
    {
      cluster_results.push_back(i);
    }
  }
  else
  {
    for (auto i:us_cluster_results)
    {
      if (i->points.size() > 10)            
        cluster_results.push_back(i);
    }
  }
}
/* clustering */
void LocalPlanner::clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results)
{
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results);
  const double leaf_size = env_->sdf_map_->getResolution()*ds_fac;

  // ds_region.reset(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::VoxelGrid<pcl::PointXYZ> ds;
  // ds.setInputCloud(cloud);
  // ds.setLeafSize(leaf_size, leaf_size, leaf_size);
  // ds.filter(*ds_region);

  uniform_cluster_with_normal(cloud, cluster_results);
}
/* Select correct normal */
void LocalPlanner::normal_info()
{
  vector<cluster_normal>().swap(local_cnv);
  cluster_normal temp_cnv_p;
  cluster_normal temp_cnv_n;
  int counter_normal = 0;

  Eigen::Vector3d normal_0;
  Eigen::Vector3d normal_1;
  Eigen::Vector3d center;

  for (int i=0; i<local_cluster.size(); ++i)
  {
    bool have_normal = false;
    bool type_normal = true;
    Eigen::Matrix3f normal_Z;
    Eigen::Matrix3f normal_Y;
    get_normal(local_cluster[i], normal_Z, type_normal);

    center(0) = normal_Z(0,0);
    center(1) = normal_Z(0,1);
    center(2) = normal_Z(0,2);
    // N_0
    normal_0(0) = normal_Z(1,0);
    normal_0(1) = normal_Z(1,1);
    normal_0(2) = normal_Z(1,2);
    int state_0 = normal_judge(center, normal_0, normal_param);

    if (state_0 != 1)
    {
      // have normal
      have_normal = true;
      vector<Eigen::Vector3d> vec_0;
      for (int h=0; h<local_cluster[i]->points.size(); ++h)
      {
        Eigen::Vector3d temp_cell_0(local_cluster[i]->points[h].x, local_cluster[i]->points[h].y, local_cluster[i]->points[h].z);
        vec_0.push_back(temp_cell_0);
      }
      temp_cnv_p.global_cells_ = vec_0;
      temp_cnv_p.center_ = center;
      temp_cnv_p.normal_ = normal_0-center;
      temp_cnv_p.id_g = counter_normal;
      counter_normal++;
      local_cnv.push_back(temp_cnv_p);
    }
    // N_1
    normal_1(0) = normal_Z(2,0);
    normal_1(1) = normal_Z(2,1);
    normal_1(2) = normal_Z(2,2);
    int state_1 = normal_judge(center, normal_1, normal_param);

    if (state_1 != 1)
    {
      // have normal
      have_normal = true;
      vector<Eigen::Vector3d> vec_1;
      for (int g=0; g<local_cluster[i]->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1(local_cluster[i]->points[g].x, local_cluster[i]->points[g].y, local_cluster[i]->points[g].z);
        vec_1.push_back(temp_cell_1);
      }
      temp_cnv_n.global_cells_ = vec_1;
      temp_cnv_n.center_ = center;
      temp_cnv_n.normal_ = normal_1-center;
      temp_cnv_n.id_g = counter_normal;
      counter_normal++;
      local_cnv.push_back(temp_cnv_n);
    }

    if (have_normal == false)
    {
      cout << "No:" << i << "no normal mech!!!!!!!!!!!!" << endl;
      type_normal = false;
      get_normal(local_cluster[i], normal_Y, type_normal);

      center(0) = normal_Y(0,0);
      center(1) = normal_Y(0,1);
      center(2) = normal_Y(0,2);
      //N_0
      normal_0(0) = normal_Y(1,0);
      normal_0(1) = normal_Y(1,1);
      normal_0(2) = normal_Y(1,2);
      int state_0 = normal_judge(center, normal_0, normal_param);
      if (state_0 != 1)
      {
        have_normal = true;
        vector<Eigen::Vector3d> vec_0;
        for (int h=0; h<local_cluster[i]->points.size(); ++h)
        {
          Eigen::Vector3d temp_cell_0(local_cluster[i]->points[h].x, local_cluster[i]->points[h].y, local_cluster[i]->points[h].z);
          vec_0.push_back(temp_cell_0);
        }
        temp_cnv_p.global_cells_ = vec_0;
        temp_cnv_p.center_ = center;
        temp_cnv_p.normal_ = normal_0-center;
        temp_cnv_p.id_g = counter_normal;
        counter_normal++;
        local_cnv.push_back(temp_cnv_p);
      }
      // N_1
      normal_1(0) = normal_Y(2,0);
      normal_1(1) = normal_Y(2,1);
      normal_1(2) = normal_Y(2,2);
      int state_1 = normal_judge(center, normal_1, normal_param);
      if (state_1 != 1)
      {
        have_normal = true;
        vector<Eigen::Vector3d> vec_1;
        for (int g=0; g<local_cluster[i]->points.size(); ++g)
        {
          Eigen::Vector3d temp_cell_1(local_cluster[i]->points[g].x, local_cluster[i]->points[g].y, local_cluster[i]->points[g].z);
          vec_1.push_back(temp_cell_1);
        }
        temp_cnv_n.global_cells_ = vec_1;
        temp_cnv_n.center_ = center;
        temp_cnv_n.normal_ = normal_1-center;
        temp_cnv_n.id_g = counter_normal;
        counter_normal++;
        local_cnv.push_back(temp_cnv_n);
      }
    }
    if (have_normal == false)
    {
      normal_1(0) = center(0) - centroid(0);
      normal_1(1) = center(1) - centroid(1);
      normal_1(2) = center(2) - centroid(2);

      int state_0 = normal_judge(center, normal_1, normal_param);
      if (state_0 != 1)
      {
      vector<Eigen::Vector3d> vec_1;
      for (int g=0; g<local_cluster[i]->points.size(); ++g)
      {
        Eigen::Vector3d temp_cell_1(local_cluster[i]->points[g].x, local_cluster[i]->points[g].y, local_cluster[i]->points[g].z);
        vec_1.push_back(temp_cell_1);
      }
      temp_cnv_n.global_cells_ = vec_1;
      temp_cnv_n.center_ = center;
      temp_cnv_n.normal_ = normal_1;
      temp_cnv_n.id_g = counter_normal;
      counter_normal++;
      local_cnv.push_back(temp_cnv_n);

      have_normal = true;
      }
    }
  }
}
/* Dual viewpoints sampling, img_type: true for ref and false for src */
void LocalPlanner::sample_vp_pillar(cluster_normal& set, int& qualified_vp, double& head_bias, bool& img_type)
{
  Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  Eigen::Vector3d z_axis_nega(0.0, 0.0, -1.0);
  Eigen::Vector3d x_axis(1.0, 0.0, 0.0);
  Eigen::Vector3d x_axis_nega(-1.0, 0.0, 0.0);

  float z_min = 0.6;
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
  // 2 modes for theta sampling
  if (theta > theta_upper)
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
    double vis_ratio_ref = 0.0;
    double vis_ratio_src = 0.0;
    VP_global tmp_l_vp;
    // phi bias angle
    phi = phi + head_bias;
    // **************
    // -------------------------------
    for (double rc = r_min, dr = (r_max - r_min) / r_step; rc <= r_max + 1e-3; rc += dr){
      for (double zc = 0.5*z_size, dz = z_size / z_step; zc >= -0.5*z_size - 1e-3; zc -= dz){
        for (double pc = phi-phi_range; pc < phi+phi_range+1e-3; pc+=angle_step)
        {
          Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(rc*sign_x*cos(pc), rc*sin(pc), zc);

          if (sample_pos(2) < 1.0)
          {
            sample_pos(2) = set.center_(2);
          }
          
          float coe_sample = 1.0;
          int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
          if (!env_->sdf_map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;
          // compute average yaw
          auto& pred_cells = set.global_cells_;
          int lower_vis_bound = ceil(visible_lower*pred_cells.size());
          double avg_yaw = 0.0;
          Eigen::Vector3d start_dir = (pred_cells.front() - sample_pos).normalized();
          for (int r=1; r<pred_cells.size(); ++r)
          {
            Eigen::Vector3d dir = (pred_cells[r] - sample_pos).normalized();
            double yaw = acos(dir.dot(start_dir));
            if (start_dir.cross(dir)[2] < 0) yaw = -yaw;
            avg_yaw += yaw;
          }
          avg_yaw = avg_yaw / pred_cells.size() + atan2(start_dir[1], start_dir[0]);
          // constrain yaw
          // Eigen::Vector3d to_center = (set.center_ - sample_pos).normalized();
          // avg_yaw = atan2(to_center[1], to_center[0]);
          while (avg_yaw < -M_PI)
            avg_yaw += 2 * M_PI;
          while (avg_yaw > M_PI)
            avg_yaw -= 2 * M_PI;
          // compute visible cells
          int visib_num = cal_visibility_cells(sample_pos, avg_yaw, pred_cells);
          if (visib_num > vis_cells)
          {
            vis_cells = visib_num;
            if (img_type)
              vis_ratio_ref = double(visib_num)/pred_cells.size();
            else
              vis_ratio_src = double(visib_num)/pred_cells.size();
            tmp_l_vp = {sample_pos, avg_yaw, visib_num, vis_ratio_ref, vis_ratio_src};
          }
          if (visib_num > lower_vis_bound)
          {
            if (img_type)
              vis_ratio_ref = double(visib_num)/pred_cells.size();
            else
              vis_ratio_src = double(visib_num)/pred_cells.size();
            VP_global vp_g = {sample_pos, avg_yaw, visib_num, vis_ratio_ref, vis_ratio_src};
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
      double ratio = (double)vis_cells/vis_all;
      if (ratio > 0.5*visible_lower)
      {
        set.vps_global.push_back(tmp_l_vp);
        qualified_vp++;
      }
      else
      {
        cout << "None qualified viewpoints..." << endl;
        Eigen::Vector3d normal_n = set.normal_.normalized();
        Eigen::Vector3d sample_pos;
        double scale = 0.5*(r_min+r_max);
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
        if (img_type)
          vis_ratio_ref = 0.01;
        else
          vis_ratio_src = 0.01;
        VP_global vp_g = {sample_pos, avg_yaw_, vis_num_, vis_ratio_ref, vis_ratio_src};
        set.vps_global.push_back(vp_g);
        qualified_vp++;
      }
    }
  }
  else if (theta <= theta_upper || region_angle < local_normal_thre)
  {
    // store max visibility viewpoints
    int vis_cells = 0;
    double vis_ratio_ref = 0.0;
    double vis_ratio_src = 0.0;
    VP_global tmp_l_vp;
    // -------------------------------
    double scale = 1.4*r_min;
       for (double pc = head_bias; pc < 2*M_PI+head_bias+1e-3; pc+=2*angle_step)
       {
         Eigen::Vector3d sample_pos = set.center_ + Eigen::Vector3d(scale*cos(pc), scale*sin(pc), 1.5);

         if (sample_pos(2) < 1.0)
          {
            sample_pos(2) = set.center_(2);
          }

         float coe_sample = 1.0;
         int sample_state = normal_judge(sample_pos, sample_pos, coe_sample);
         if (!env_->sdf_map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;
         // compute average yaw
         auto& pred_cells = set.global_cells_;
         int lower_vis_bound = ceil(visible_lower*pred_cells.size());
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
            if (img_type)
              vis_ratio_ref = double(visib_num)/pred_cells.size();
            else
              vis_ratio_src = double(visib_num)/pred_cells.size();
            tmp_l_vp = {sample_pos, avg_yaw, visib_num, vis_ratio_ref, vis_ratio_src};
          }
          if (visib_num > lower_vis_bound)
          {
            if (img_type)
              vis_ratio_ref = double(visib_num)/pred_cells.size();
            else
              vis_ratio_src = double(visib_num)/pred_cells.size();
            VP_global vp_g = {sample_pos, avg_yaw, visib_num, vis_ratio_ref, vis_ratio_src};
            set.vps_global.push_back(vp_g);
            qualified_vp++;
          }
       }
    if (qualified_vp == 0)
    {
      auto& cells_ = set.global_cells_;
      int vis_all = cells_.size();
      double ratio = (double)vis_cells/vis_all;
      // if (ratio > 0.5*visible_lower)
      // {
      //   set.vps_global.push_back(tmp_l_vp);
      //   qualified_vp++;
      // }
      // else
      // {
        cout << "None qualified viewpoints..." << endl;
        Eigen::Vector3d normal_n = set.normal_.normalized();
        Eigen::Vector3d sample_pos;
        double scale = 1.2*r_min;

        for (double pc = head_bias; pc < 2*M_PI+head_bias+1e-3; pc+=2*angle_step)
        {
          sample_pos = set.center_ + Eigen::Vector3d(scale*cos(pc), scale*sin(pc), 3.0);

          int sample_state = normal_judge(sample_pos, sample_pos, 1.0);
          if (!env_->sdf_map_->isInBox(sample_pos) || sample_state == 1 || sample_pos[2] < z_min)
            continue;

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
          if (img_type)
            vis_ratio_ref = 0.01;
          else
            vis_ratio_src = 0.01;
          VP_global vp_g = {sample_pos, avg_yaw_, vis_num_, vis_ratio_ref, vis_ratio_src};
          set.vps_global.push_back(vp_g);
          qualified_vp++;
        }
      // }
    }
  }
}
/* Order cost */
void LocalPlanner::CostMat()
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

  for (auto it1 = local_qual_cnv.begin(); it1 != local_qual_cnv.end(); ++it1)
  {
    for (auto it2 = it1; it2 != local_qual_cnv.end(); ++it2)
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
/* Full cost matrix given determined start and end */
void LocalPlanner::fullCostMatrix(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw,
    Eigen::MatrixXd& mat)
{
  // Use Asymmetric TSP
  int dims = local_qual_cnv.size();
  mat.resize(dims + 2, dims + 2);
  int i = 1, j = 1, k = 1;
  for (auto cnv : local_qual_cnv)
  {
    for (auto cost : cnv.costs_)
    {
      mat(i, j++) = cost;
    }
    ++i;
    j=1;
  }
  
  vector<Eigen::Vector3d> s_path_e;
  mat(0, dims+1) = ViewNode::compute_globalCost(cur_pos, nbv_pos, cur_yaw[0], nbv_yaw[0], cur_vel, cur_yaw[1], s_path_e);

  mat(dims+1, dims+1) = 0.0;
  vector<Eigen::Vector3d> path_;
  for (int m=0; m<dims+1; ++m)
  { 
    mat(dims+1, m) = 1000.0;
  }
  mat.leftCols<1>().setZero();

  for (auto cnv : local_qual_cnv)
  {
    VP_global vp = cnv.vps_global.front();
    vector<Eigen::Vector3d> path;
    mat(0, j++) =
      ViewNode::compute_globalCost(cur_pos, vp.pos_g, cur_yaw[0], vp.yaw_g, cur_vel, cur_yaw[1], path);
  }
  for (auto cnv : local_qual_cnv)
  {
    VP_global vp = cnv.vps_global.front();
    vector<Eigen::Vector3d> path;
    mat(k++, dims+1) =
      ViewNode::compute_globalCost(vp.pos_g, nbv_pos, vp.yaw_g, nbv_yaw[0], cur_vel, cur_yaw[1], path);

  }
}
/* Find initial order */
void LocalPlanner::find_order(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw,
    vector<int>& order)
{
  vector<cluster_normal>().swap(local_visit_cnv);
  // Initialize TSP par file
  ofstream par_file(tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";
  par_file.close();
  // CostMat
  CostMat();
  Eigen::MatrixXd cost_mat;
  fullCostMatrix(cur_pos, cur_vel, cur_yaw, nbv_pos, nbv_yaw, cost_mat);
  const int dimension = cost_mat.rows();
  // TSP
  ofstream prob_file(tsp_dir_ + "/single.tsp");
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
  solveTSPLKH((tsp_dir_ + "/single.par").c_str());
  // obtain results
  ifstream res_file(tsp_dir_ + "/single.txt");
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
  order.pop_back();
  // show order and change order
  cluster_normal pseudo_cnv;
  pseudo_cnv.global_cells_ = local_qual_cnv[order[0]].global_cells_;
  pseudo_cnv.center_ = local_qual_cnv[order[0]].center_;
  pseudo_cnv.normal_ = local_qual_cnv[order[0]].normal_;
  pseudo_cnv.id_g = -1;
  int q_p = 0;
  bool img_ = true;
  sample_vp_pillar(pseudo_cnv, q_p, pseudo_bias, img_);
  local_visit_cnv.push_back(pseudo_cnv);

  for (int i=0; i<order.size(); ++i)
  {
    local_visit_cnv.push_back(local_qual_cnv[order[i]]);
  }
  // pseudo NBV cnv
  cluster_normal nbv_cnv;
  VP_global nvb_vp;
  nvb_vp.pos_g = nbv_pos;
  nvb_vp.yaw_g = nbv_yaw[0];
  nbv_cnv.vps_global.push_back(nvb_vp);
  nbv_cnv.id_g = local_visit_cnv.size() - 1;

  local_visit_cnv.push_back(nbv_cnv);
}
/* View-cost graph search */
void LocalPlanner::view_graph_search(const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw)
{
  GraphSearch<ViewNode> v_search;
  vector<ViewNode::Ptr> last_group, cur_group;
  // Add the current state
  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0]));
  first->vel_ = cur_vel;
  first->vis_ratio_src_ = 5.0;
  v_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;
  // Construct Graph
  cout << "---Construct Graph---" << endl;
  int counter_node = 0;
  for (int i=0; i<local_visit_cnv.size(); ++i)
  {
    for(int j=0; j<local_visit_cnv[i].vps_global.size(); ++j)
    {
      // node initialization
      ViewNode::Ptr node(new ViewNode(local_visit_cnv[i].vps_global[j].pos_g, local_visit_cnv[i].vps_global[j].yaw_g));
      node->vis_ratio_src_ = local_visit_cnv[i].vps_global[j].vis_ratio_src;
      node->center_ = local_visit_cnv[i].center_;
      node->normal_ = local_visit_cnv[i].normal_;
      if (i<local_visit_cnv.size()-2)
      {
        int tmp_all = local_visit_cnv[i+1].global_cells_.size();
        int visible_c = cal_visibility_cells(local_visit_cnv[i].vps_global[j].pos_g, local_visit_cnv[i].vps_global[j].yaw_g, local_visit_cnv[i+1].global_cells_);
        node->vis_ratio_ref_ = (double)visible_c/tmp_all;
      }
      // add node
      v_search.addNode(node);
      counter_node++;
      for (auto nd : last_group)
        v_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);
      if (i == local_visit_cnv.size() - 1) 
      {
        node->vis_ratio_src_ = 5.0;
        final_node = node;
        break;
      }
    }
    last_group = cur_group;
    cur_group.clear();
  }
  // Search optimal sequence
  cout << "---Search optimal sequence---" << endl;
  vector<ViewNode::Ptr> path;
  v_search.ViewDijkstraSearch(first->id_, final_node->id_, path);
  cout << "---Search Finish---" << endl;
  if (path.size() == 0)
    ROS_ERROR("Local No Path!");
  // Add viewpoints to local planner output
  vector<Eigen::Vector3d>().swap(local_points);
  vector<double>().swap(local_yaws);
  local_points.push_back(path[0]->pos_);
  local_yaws.push_back(path[0]->yaw_);
  
  if (path.size() > 1)
  {
  for (int i=1; i < path.size()-1; ++i)
  {
    if (path[i]->pos_(2) > 1.0)
    {
    if ((path[i]->pos_ - local_points.back()).norm() > 1.5)
    {
      local_points.push_back(path[i]->pos_);
      local_yaws.push_back(path[i]->yaw_);
    }
    // if ((path[i]->pos_ - local_points.back()).norm() > 4.0)
    // {
    //   Eigen::Vector3d dir_ = (path[i]->pos_ - local_points.back()).normalized();
    //   double yaw_dist = path[i]->yaw_ - local_yaws.back();
    //   vector<Eigen::Vector3d> tmp_pos_;
    //   vector<double> tmp_yaw_;
    //   Eigen::Vector3d pos_inter;
    //   double yaw_inter;

    //   double length = (path[i]->pos_ - local_points.back()).norm();
    //   int step = (length/local_interval)+1;
    //   for (int i=1; i<step; ++i)
    //   {
    //     pos_inter = local_points.back() + step*i*dir_/length;
    //     yaw_inter = local_yaws.back() + step*i*yaw_dist/length;
    //     tmp_pos_.push_back(pos_inter);
    //     tmp_yaw_.push_back(yaw_inter);
    //   }
    //   tmp_pos_.push_back(path[i]->pos_);
    //   tmp_yaw_.push_back(path[i]->yaw_);

    //   local_points.insert(local_points.end(), tmp_pos_.begin(), tmp_pos_.end());
    //   local_yaws.insert(local_yaws.end(), tmp_yaw_.begin(), tmp_yaw_.end());
    // }
    }
  }
  cout << "---Push Finish---" << endl;
  // push back nbv
  local_points.push_back(path.back()->pos_);
  local_yaws.push_back(path.back()->yaw_);
  }
}
/* Local Planning Manager */
void LocalPlanner::LocalPathManager(pcl::PointCloud<pcl::PointXYZ>::Ptr& local_region, const Eigen::Vector3d& local_normal, const Eigen::Vector3d& cur_pos, const Eigen::Vector3d& cur_vel, const Eigen::Vector3d& cur_yaw,
    const Eigen::Vector3d& nbv_pos, const Eigen::Vector3d& nbv_yaw)
{
  cout << "-----Local Planning Start!-----" << endl;
  /* --- compute local region normal angle --- */
  Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  Eigen::Vector3d z_axis_nega(0.0, 0.0, -1.0);
  if (local_normal(0) >= 0)
    region_angle = acos(local_normal.dot(z_axis)/(local_normal.norm()*z_axis.norm()));
  else 
    region_angle = acos(local_normal.dot(z_axis_nega)/(local_normal.norm()*z_axis_nega.norm()));
  /* -- easy mode -- */
  if (region_angle <= 20.0*M_PI/180.0 && (cur_pos-nbv_pos).norm() < 3.0)
  {
    local_points = {cur_pos, nbv_pos};
    local_yaws = {cur_yaw(0), nbv_yaw(0)};
  }
  /* ---------- */
  pcl::compute3DCentroid(*local_region, centroid);
  conditional_ec(local_region, local_cluster);
  normal_info();
  // dual sampling 
  vector<cluster_normal>().swap(local_qual_cnv);
  cluster_normal qual_cnv;
  int qual_id = 0;
  double bias = 0.0;
  bool type = false;
  for (int i=0; i<local_cnv.size(); ++i)
  {
    int q_vp = 0;
    sample_vp_pillar(local_cnv[i], q_vp, bias, type);
    if (q_vp > 0)
    {
      qual_cnv.global_cells_ = local_cnv[i].global_cells_;
      qual_cnv.center_ = local_cnv[i].center_;
      qual_cnv.normal_ = local_cnv[i].normal_;
      qual_cnv.vps_global = local_cnv[i].vps_global;

      std::sort(qual_cnv.vps_global.begin(), qual_cnv.vps_global.end(),
      [](const VP_global& v1, const VP_global& v2) { return v1.visib_num_g > v2.visib_num_g; });
      cout << "i:" << qual_cnv.vps_global.size() << endl;

      qual_cnv.id_g = qual_id;
      local_qual_cnv.push_back(qual_cnv);
      qual_id++;
    }
  }
  // sampling finished!
  // Find order
  if (local_qual_cnv.size() > 0)
  {
    vector<int> local_tsp_id; // viewcost graph search order
    find_order(cur_pos, cur_vel, cur_yaw, nbv_pos, nbv_yaw, local_tsp_id);
    // graph search
    view_graph_search(cur_pos, cur_vel, cur_yaw, nbv_pos, nbv_yaw);
  }
  else
  {
    vector<Eigen::Vector3d>().swap(local_points);
    vector<double>().swap(local_yaws);
    local_points.push_back(cur_pos);
    local_points.push_back(nbv_pos);
    local_yaws.push_back(cur_yaw(0));
    local_yaws.push_back(nbv_yaw(0));
  }

  // ROS_WARN("-----Local Yaws-----");
  // for (int i=0; i<local_yaws.size(); ++i)
  //   cout << "Rank." << i << "_yaw:" << local_yaws[i]*180.0/M_PI << "deg." << endl; 

  // ROS_WARN("-----Local Points-----");
  // for (int i=0; i<local_points.size(); ++i)
  //   cout << "Rank." << i << "_vp:" << local_points[i] << "m." << endl; 

  /* ---------- */
  cout << "-----Local Planning Finished!-----" << endl;
}
}