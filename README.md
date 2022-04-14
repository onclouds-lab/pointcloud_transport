# pointcloud_transport
ROS package for converting pointcloud2 to binary
This package convert pointcloud2 to binary and binary to pointcloud2.
 
# Feature
reduce amout of size and compressed by zstd

# Dependency
ROS1  
http://wiki.ros.org/ROS/Installation  
  
Eigen3
> sudo apt-get install libeigen3-dev
  
HDF5  
> sudo apt-get install libhdf5-dev

zstd  
> sudo apt-get install libzstd-dev

nvcomp  
https://github.com/NVIDIA/nvcomp

# Install and build
clone to your ros work space and catkin_make

# Topics
pointcloud: pointcloud2 message  
pointcloud_hdf5: hdf5 message converted from pointcloud2  

# Parameter
compressed: true or false ( default: false )  
compressed_level: int ( default: 3 ) compressed level for zstd  
