#include <pointcloud_transport/pointcloud_transport.h>

void convertToEigen( pcl::PointCloud<pcl::PointXYZI>::Ptr pc, Eigen::MatrixXf& mat ){
    Eigen::MatrixXf m = pc->getMatrixXfMap();
    mat = m.block(0,0,5,m.cols());
}

ros::Publisher hdf5_pub;
bool compressed;
int compressed_level;
void pointcloud_cb( const sensor_msgs::PointCloud2ConstPtr& pointsMsg){
    std::cout << __func__ << std::endl;

    hdf5frame* hdf5data = new hdf5frame(compressed, compressed_level);
    //type
    uint8_t pcl_type;
    for( int i=0; i<pointsMsg->fields.size(); i++ ){
        if(pointsMsg->fields[i].name.compare("i")==0){
            pcl_type = pointcloud_transport::PclHDF5::TYPE_XYZI;
        } else if( pointsMsg->fields[i].name.compare("intensity")==0 ){
            pcl_type = pointcloud_transport::PclHDF5::TYPE_XYZI;
        }
        std::cout << pointsMsg->fields[i].name << std::endl;
    }

    Eigen::MatrixXf m;
    if( pcl_type == pointcloud_transport::PclHDF5::TYPE_XYZI ){
        std::cout << "point clouds type : XYZI" << std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc = static_cast<pcl::PointCloud<pcl::PointXYZI>::Ptr>(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*pointsMsg, *pc);
        std::cout << "point size : " << pc->size() << std::endl;

        //convert pcl to eigen mat
        convertToEigen( pc, m );
        
    } else {
        std::cout << __func__ << std::endl;
        std::cout << "undetected pointcloud type !" << std::endl;
        exit(1);
    }

    std::cout << m.rows() << " x " << m.cols() << std::endl;
    std::cout << m.block(0,0,5,5) << std::endl;

    //convert eigen mat to hdf5
    hdf5data->input("/pointcloud", m);
    hdf5data->scan();

    //create ros message
    pointcloud_transport::PclHDF5 HDF5_msg;
    HDF5_msg.header = pointsMsg->header;
    HDF5_msg.type =pcl_type;
    HDF5_msg.compressed = hdf5data->getCompressed();
    HDF5_msg.compressed_level = hdf5data->getCompressed_level();
    hdf5data->get_file_image2(HDF5_msg.hdf5data);

    //publish
    hdf5_pub.publish(HDF5_msg);

    delete hdf5data;

}

int main( int argc, char** argv ){
    ros::init(argc,argv,"pointcloud_transport_pcl2hdf_node");
    ros::NodeHandle nh("~");

    nh.param<bool>("compressed", compressed, false);
    nh.param<int>("compressed_level", compressed_level, 3);

    ros::Subscriber pcl_sub = nh.subscribe("pointcloud", 10, &pointcloud_cb);
    hdf5_pub = nh.advertise<pointcloud_transport::PclHDF5>("pointcloud_hdf", 10);

    hdf5frame* hdf5data = new hdf5frame(1, compressed_level);
    std::vector<unsigned char> buf;
    hdf5data->get_file_image2(buf);
    std::cout << "size" << std::endl;
    std::cout << buf.size() << std::endl;
    delete hdf5data;

    hdf5frame* hdf5data2 = new hdf5frame(buf, 1, 1);
    ros::spin();

}