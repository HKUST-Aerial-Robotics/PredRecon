#include <iostream>  
#include <ros/ros.h>  
#include <pcl/point_cloud.h>  
#include <pcl_conversions/pcl_conversions.h>  
#include <pcl/io/pcd_io.h>  
  
#include <pcl/visualization/cloud_viewer.h>  
  
#include <sensor_msgs/PointCloud2.h>  
using std::cout;  
using std::endl;  
using std::stringstream;  
using std::string;  
  
using namespace pcl;  
  
unsigned int filesNum = 0;  
bool saveCloud(false);  
  
void cloudCB(const sensor_msgs::PointCloud2& input)  
{  
pcl::PointCloud<pcl::PointXYZ> cloud;
pcl::PointCloud<pcl::PointXYZ> after_cloud;
pcl::PointXYZ pt;

pcl::fromROSMsg(input, cloud); // sensor_msgs::PointCloud2 ----> pcl::PointCloud<T>
for (int i=0; i<cloud.points.size(); ++i)
{
  if(cloud.points[i].z>0.4)
  {
    pt.x = cloud.points[i].x;
    pt.y = cloud.points[i].y;
    pt.z = cloud.points[i].z;
    after_cloud.push_back(pt);
  }
}
 
stringstream stream;  
stream << "data/inputCloud"<< filesNum<< ".pcd";  
string filename = stream.str();  
  
io::savePCDFile(filename, after_cloud, true)   
filesNum++;  
cout << filename<<" Saved."<<endl;
  
}  
  
}  
  
int main (int argc, char** argv)  
{  
ros::init(argc, argv, "pcl_write");  
ros::NodeHandle nh;    
  
ros::Subscriber pcl_sub = nh.subscribe("/sdf_map/occupancy_all", 1, cloudCB);  
  
ros::Rate rate(30.0);  
  
while (ros::ok() && ! viewer->wasStopped())  
{  
ros::spinOnce();  
rate.sleep();  
}  
  
return 0;  
}  