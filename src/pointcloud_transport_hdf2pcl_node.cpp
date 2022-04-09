#include <pointcloud_transport/pointcloud_transport.h>

void convertToPointCloud( Eigen::MatrixXf mat, pcl::PointCloud<pcl::PointXYZI>::Ptr pc ){
    pc->points.resize(mat.cols());
    Eigen::MatrixXf lmat = pc->getMatrixXfMap();
    lmat.block(0,0,5,mat.cols()) = mat;
    pc->getMatrixXfMap() = lmat;
}

ros::Publisher pc2_pub;
void hdf_cb( const pointcloud_transport::PclHDF5::ConstPtr& HDF_msg){
    std::cout << __func__ << std::endl;

    std::cout << HDF_msg->header.stamp << std::endl;
    std::cout << (int)HDF_msg->type << std::endl;
    std::cout << HDF_msg->compressed << std::endl;
    std::cout << HDF_msg->compressed_level << std::endl;
    std::cout << HDF_msg->hdf5data.size() << std::endl;

    hdf5frame* hdf5data = new hdf5frame(HDF_msg->hdf5data, HDF_msg->compressed, HDF_msg->compressed_level);

    //Eigne mat
    Eigen::MatrixXf m = hdf5data->getMat("/pointcloud");
    std::cout << m.rows() << " x " << m.cols() << std::endl;
    std::cout << m.block(0,0,5,5) << std::endl;

    //convert pcl
    if( HDF_msg->type == pointcloud_transport::PclHDF5::TYPE_XYZI ){
        std::cout << "point clouds type : XYZI" << std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc = static_cast<pcl::PointCloud<pcl::PointXYZI>::Ptr>(new pcl::PointCloud<pcl::PointXYZI>());
        convertToPointCloud( m, pc );
        std::cout << "point size : " << pc->size() << std::endl;

        //publish
        sensor_msgs::PointCloud2 PC2_msg;
        pcl::toROSMsg(*pc.get(), PC2_msg );
        PC2_msg.header = HDF_msg->header;

        pc2_pub.publish(PC2_msg);

    } else {
        std::cout << __func__ << std::endl;
        std::cout << "undetected pointcloud type !" << std::endl;
        exit(1);
    }

    delete hdf5data;
}

int main( int argc, char** argv ){
    ros::init(argc,argv,"pointcloud_transport_hdf2pcl_node");
    ros::NodeHandle nh("~");

    ros::Subscriber hdf_sub = nh.subscribe("pointcloud_hdf", 10, &hdf_cb);
    pc2_pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 10);

    ros::spin();
}