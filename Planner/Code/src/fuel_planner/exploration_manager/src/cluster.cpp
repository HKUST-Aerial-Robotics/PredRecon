#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/visualization/cloud_viewer.h>
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

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

bool customRegionGrowing (const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float squared_distance)
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
	// if(angle_deg < threshold_angle || angle_deg > (180.0-threshold_angle))	return true;//&& (distance < 10.0)
	// else	return false;
    return true;
}

void PCA_algo(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointType& c, PointType& pcZ, PointType& pcY, PointType& pcX, PointType& pcZ_inv, PointType& pcY_inv)
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

pcl::PointCloud<pcl::PointXYZ>::Ptr condition_get(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float& x_up, float& x_low,
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

void visualization(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_vec, vector<Eigen::Matrix3f>& normal_vec, bool& flag) 
{
  pcl::visualization::PCLVisualizer viewer("cloud");
  viewer.setBackgroundColor(255, 255, 255);
  
  for (int i=0; i<cloud_vec.size(); ++i)
  {
      pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> RandomColor(cloud_vec[i]);
      PointType temp_c, pcz, pcy, pcx, pcz_inv, pcy_inv;
      PCA_algo(cloud_vec[i], temp_c, pcz, pcy, pcx, pcz_inv, pcy_inv);
      Eigen::Matrix3f normal_info;
      // (center[3], normal_1[3], normal_2[3])
      normal_info << temp_c.x, temp_c.y, temp_c.z,
                     pcz.x, pcz.y, pcz.z,
                     pcz_inv.x, pcz_inv.y, pcz_inv.z;
      normal_vec[i] = normal_info;

      viewer.addPointCloud<pcl::PointXYZ>(cloud_vec[i], RandomColor, "sample_"+to_string(i)); 
      if (flag == true)
      {
          viewer.addArrow(pcz, temp_c, 0.0, 0.0, 1.0, false, "arrow_z_"+to_string(i));
        //   viewer.addArrow(pcz_inv, temp_c, 1.0, 0.0, 0.0, false, "arrow_z_inv_"+to_string(i));
      }

      viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "sample_"+to_string(i));
  }
 
  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }
}

void surface_recon_possion(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz)
{
    pcl::PointCloud < pcl::PointXYZRGB > ::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_xyz, *cloud);
    
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(30);
	n.compute(*normals);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree2->setInputCloud(cloud_with_normals);

    pcl::Poisson<pcl::PointXYZRGBNormal> pn;
	pn.setConfidence(true); //是否使用法向量的大小作为置信信息。如果false，所有法向量均归一化。
	pn.setDegree(2); //设置参数degree[1,5],值越大越精细，耗时越久。
	pn.setDepth(8); //树的最大深度，求解2^d x 2^d x 2^d立方体元。由于八叉树自适应采样密度，指定值仅为最大深度。
	pn.setIsoDivide(8); //用于提取ISO等值面的算法的深度
	pn.setManifold(true); //是否添加多边形的重心，当多边形三角化时。 设置流行标志，如果设置为true，则对多边形进行细分三角话时添加重心，设置false则不添加
	pn.setOutputPolygons(false); //是否输出多边形网格（而不是三角化移动立方体的结果）
	pn.setSamplesPerNode(2.0); //设置落入一个八叉树结点中的样本点的最小数量。无噪声，[1.0-5.0],有噪声[15.-20.]平滑
	pn.setScale(1.25); //设置用于重构的立方体直径和样本边界立方体直径的比率。
	pn.setSolverDivide(8); //设置求解线性方程组的Gauss-Seidel迭代方法的深度

    pn.setSearchMethod(tree2);
	pn.setInputCloud(cloud_with_normals);

    pcl::PolygonMesh mesh;

    pn.performReconstruction(mesh);
    pcl::io::savePLYFile("/home/albert/surface_test.ply", mesh);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(mesh, "my");
	viewer->initCameraParameters();
	while (!viewer->wasStopped()){
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void surface_recon_triangle(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    n.setInputCloud (cloud);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setSearchRadius (2.0);
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (1000);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPolygonMesh(triangles, "my");
	viewer->initCameraParameters();
	while (!viewer->wasStopped()){
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

pcl::PointCloud<pcl::PointXYZ>::Ptr surface_recon_visibility(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector4f& viewpoint)
{
    auto startT = std::chrono::high_resolution_clock::now();
    // ------process start---------
    float gamma = 0.0005;
    pcl::compute3DCentroid(*cloud, viewpoint);
    pcl::PointXYZ vp;
    vp.x = viewpoint[0];
    vp.y = viewpoint[1];
    vp.z = viewpoint[2];

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::demeanPointCloud(*cloud, viewpoint, *cloud_out);// Centralized by viewpoint

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transform(new pcl::PointCloud<pcl::PointXYZ>);
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr surface_hull(new pcl::PointCloud<pcl::PointXYZ>);
    hull.reconstruct(*surface_hull, polygons);
    // step 3: inverse transform
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inverse(new pcl::PointCloud<pcl::PointXYZ>);
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

    pcl::visualization::PCLVisualizer viewer("cloud_with_tranform");
    viewer.setBackgroundColor(0, 0, 0);
    // add viewpoint
    viewer.addSphere (vp, 0.5, 0.5, 0.5, 0.0, "sphere");

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> set_color(cloud_out, 125, 0, 125);
    // viewer.addPointCloud<pcl::PointXYZ>(cloud, set_color, "sample_1"); 
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample_1");

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> set_color_a(cloud_out, 125, 0, 50);
    // viewer.addPointCloud<pcl::PointXYZ>(cloud_transform, set_color_a, "sample_2");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample_2");

    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handlerK(surface_hull, 0, 255, 0);
    // viewer.addPointCloud(surface_hull, color_handlerK, "point");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_inverse(cloud_inverse, 0, 255, 0);
    viewer.addPointCloud(cloud_inverse, color_inverse, "inverse");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inverse");

    while (!viewer.wasStopped()) 
    {
        viewer.spinOnce();
    }

    return cloud_inverse;
}

void uniform_cluster_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cluster_results,
vector<Eigen::Matrix3f>& normal_vec)
{
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>().swap(cluster_results);
    vector<Eigen::Matrix3f>().swap(normal_vec);
    
    // step 1: partition the input point cloud
    pcl::PointXYZ minPt, maxPt;
	pcl::getMinMax3D(*cloud, minPt, maxPt);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud(new pcl::PointCloud<pcl::PointXYZ>);
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
    // step 2: uniform clustering
    // count time start
    auto startTime = std::chrono::high_resolution_clock::now();
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
                clustered_cloud = condition_get(cloud, x_up, x_low, y_up, y_low, z_up, z_low);
                if (clustered_cloud->points.size()>0)
                {
                    // if (clustered_cloud->points.size()<60)
                    // {
                    //     if (counter == 0)
                    //     {
                    //         cluster_results.push_back(clustered_cloud);
                    //         counter++;
                    //     }
                    //     else
                    //     {
                    //         last_cloud = cluster_results[counter-1];
                    //         *last_cloud += *clustered_cloud;
                    //         cluster_results[counter-1] = last_cloud;
                    //         // cout << cluster_results[counter-1]->points.size() << endl;
                    //     }
                    // }
                    // else
                    // {
                        cluster_results.push_back(clustered_cloud);
                        counter++;
                    // }
                }
            }
        }
    }
    // step 3: acquire normal
    normal_vec.resize(cluster_results.size());
    bool flag = true;
    // visualization(cluster_results, normal_vec, flag);
     // count time end
    auto endTime = std::chrono::high_resolution_clock::now();
    //
    std::chrono::duration<double, std::milli> fp_ms = endTime - startTime;
    cout << "time:" << fp_ms.count() << "ms" << std::endl;
}

int main()
{
    std::string test_path = "/home/albert/Network/PCN-PyTorch/results/house/pred/output_5_3.pcd";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(test_path, *cloud);

    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_results;
    vector<Eigen::Matrix3f> normal_vec;

    double leaf_size = 1.0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> rs_1;
    rs_1.setInputCloud(cloud);
    rs_1.setLeafSize(leaf_size, leaf_size, leaf_size);
    rs_1.filter(*downsample_1);

    cout << downsample_1->points.size() << endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    // normal estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;	
    ne.setInputCloud(downsample_1);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(normal_tree);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*downsample_1, *cloud_normals);
    ne.setKSearch(10);
    ne.compute(*cloud_normals);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    cout << "normal_time:" << fp_ms.count() << "ms" << std::endl;
    cout << "normal_size:" << cloud_normals->points.size() << endl;
    // Conditional Euclidean Cluster
    auto cec_t1 = std::chrono::high_resolution_clock::now();

    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
    pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec(true);
    cec.setInputCloud (cloud_normals);
    cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (2.0);
    cec.setMinClusterSize (10);
    cec.setMaxClusterSize (cloud_normals->size () / 5);
    cec.segment (*clusters);
    cec.getRemovedClusters(small_clusters, large_clusters);
    cout << "CEC:" << clusters->size() << endl;
    cout << "CEC_s:" << small_clusters->size() << endl;
    cout << "CEC_l:" << large_clusters->size() << endl;

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
        cluster_results.push_back(small_cloud);
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
          cluster_results.push_back(n_cloud);
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
          cluster_results.push_back(large_cloud);
          }
    }
    

    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> final_cluster_results;
    for (auto i : cluster_results)
    {
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> temp_cluster_results;
        vector<Eigen::Matrix3f> temp_normal_vec;
        uniform_cluster_with_normal(i, temp_cluster_results, temp_normal_vec);
        final_cluster_results.insert(final_cluster_results.end(),temp_cluster_results.begin(),temp_cluster_results.end());
    }
    auto cec_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cec_ms = cec_t2 - cec_t1;
    cout << "cec_time:" << cec_ms.count() << "ms" << std::endl;
    cout << "cec_size:" << final_cluster_results.size() << endl;

    bool flag = false;
    normal_vec.resize(final_cluster_results.size());
    visualization(final_cluster_results, normal_vec, flag);
    // vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_results;
    // vector<Eigen::Matrix3f> normal_vec;
    // uniform_cluster_with_normal(downsample_1, cluster_results, normal_vec);
    // cout << "cluster:" << cluster_results.size() << endl;

    // for (auto i : cluster_results)
    // {
    //     cout << "size:" << i->points.size() << endl;
    // }

    // Eigen::Vector4f viewpoint;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_GHPR(new pcl::PointCloud<pcl::PointXYZ>);
    // cloud_GHPR = surface_recon_visibility(downsample_1, viewpoint);
    // pcl::PCDWriter writer;
    // writer.write("/home/albert/surface_vis.pcd",*cloud_GHPR);

    return 0;
}