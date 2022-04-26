# Point Cloud Compression Message
ROS package for pointcloud2 compression.  
compression and decompression for ROS package.  
 
# Feature
Translate pointcloud2 to hdf5  
Compression and decompression by zstd or nvcomp and Stream  

# Dependency
ROS1  
http://wiki.ros.org/ROS/Installation  
  
Eigen3
> sudo apt-get install libeigen3-dev
  
HDF5  
> sudo apt-get install libhdf5-dev

zstd  
> sudo apt-get install libzstd-dev

# Install and build
clone to your ros work space and catkin_make

# Package
# pointcloud_transport_pcl2hdf
## Topics
pointcloud(in): pointcloud2 message  
pointcloud_hdf5(out): hdf5 and compression message  

## Parameter
compressed: true or false ( default: false )  
compressed_level: int ( default: 3 ) compressed level only for zstd  

# pointcloud_transport_hdf2pcl
## Topics
pointcloud(out): pointcloud2 message  
pointcloud_hdf5(in): hdf5 and compression message  

# Launch file
Please check launch file in launch for start these package.
