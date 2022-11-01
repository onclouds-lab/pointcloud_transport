#include <pointcloud_transport/pointcloud_transport.h>

void convertToEigen( pcl::PointCloud<pcl::PointXYZI>::Ptr pc, Eigen::MatrixXf& mat ){
    Eigen::MatrixXf m = pc->getMatrixXfMap();
    mat = m.block(0,0,5,m.cols());
}

Eigen::Matrix<short, -1, -1> convertToSHORT( Eigen::MatrixXf mat ){
    Eigen::MatrixXf m = round(mat.array());;
    return(m.cast<short>());
}

ros::Publisher hdf5_pub;
bool compressed;
int compressed_level;
bool float_flag;
float lsb;
float voxelGridSize;
int skipRate;
unsigned int count = 0;
void pointcloud_cb( const sensor_msgs::PointCloud2ConstPtr& pointsMsg){
    std::cout << __func__ << std::endl;

    //skip
    count++;
    if( count%skipRate != 0 ) return;

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

        //Voxel Grid Filter
        if( voxelGridSize > 0.0 ){
            pcl::PointCloud<pcl::PointXYZI>::Ptr pcFiltered = static_cast<pcl::PointCloud<pcl::PointXYZI>::Ptr>(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::VoxelGrid<pcl::PointXYZI> sor;
            sor.setInputCloud( pc );
            sor.setLeafSize (voxelGridSize, voxelGridSize, voxelGridSize);
            //sor.setDownsampleAllData(true);
            sor.filter ( *(pcFiltered) );

            std::cout << "point size : " << pc->size() << std::endl;
            std::cout << "point(filtered) size : " << pcFiltered->size() << std::endl;

            //convert pcl to eigen mat
            convertToEigen( pcFiltered, m );

        } else {
            std::cout << "point size : " << pc->size() << std::endl;
            //convert pcl to eigen mat
            convertToEigen( pc, m );

        }
        
    } else {
        std::cout << __func__ << std::endl;
        std::cout << "undetected pointcloud type !" << std::endl;
        exit(1);
    }

    std::cout << m.rows() << " x " << m.cols() << std::endl;
    std::cout << m.block(0,0,5,5) << std::endl;

    //convert eigen mat to hdf5
    if( float_flag == true ){
        hdf5data->input("/pointcloud", m);
        hdf5data->scan();
    } else {
        Eigen::Matrix<short, -1, -1> xyz_ushort = convertToSHORT( m.block(0,0,3,m.cols())/lsb );
        hdf5data->input<short>("/pointcloud_xyz", xyz_ushort, H5T_NATIVE_SHORT);
        hdf5data->input<float>("/pointcloud_intensity", m.block(4,0,1,m.cols()));
        hdf5data->scan();
    }

    //create ros message
    pointcloud_transport::PclHDF5 HDF5_msg;
    HDF5_msg.header = pointsMsg->header;
    HDF5_msg.type =pcl_type;
    HDF5_msg.compressed = hdf5data->getCompressed();
    HDF5_msg.compressed_level = hdf5data->getCompressed_level();
    HDF5_msg.lsb = lsb;
    hdf5data->get_file_image2(HDF5_msg.hdf5data);

    //publish
    hdf5_pub.publish(HDF5_msg);

    delete hdf5data;

}

int main( int argc, char** argv ){
    ros::init(argc,argv,"pointcloud_transport_pcl2hdf_node");
    ros::NodeHandle nh("~");

    nh.param<bool>("type_float", float_flag, false);
    ROS_INFO("%s: Type Float %d", ros::this_node::getName().c_str(), float_flag);
    nh.param<float>("lsb", lsb, 0.004);
    ROS_INFO("%s: LSB %f", ros::this_node::getName().c_str(), lsb);
    nh.param<float>("voxel_grid_size", voxelGridSize, 0.0);
    ROS_INFO("%s: Voxel Grid Size %f", ros::this_node::getName().c_str(), voxelGridSize);
    nh.param<int>("skip_rate", skipRate, 1);
    ROS_INFO("%s: Skip Rate %d", ros::this_node::getName().c_str(), skipRate);

    nh.param<bool>("compressed", compressed, false);
    ROS_INFO("%s: compressed %d", ros::this_node::getName().c_str(), compressed);
    nh.param<int>("compressed_level", compressed_level, 3);
    ROS_INFO("%s: compressed_level %d", ros::this_node::getName().c_str(), compressed_level);

    ros::Subscriber pcl_sub = nh.subscribe("pointcloud", 10, &pointcloud_cb);
    hdf5_pub = nh.advertise<pointcloud_transport::PclHDF5>("pointcloud_hdf", 10);

    /* Eigen Matrix Test */
    /*
    Eigen::MatrixXf mat(5,2);
    mat(0,0) = 1.0; mat(1,0) = 2.0; mat(2,0) = 3.0; mat(3,0) = 0.0; mat(4,0) = 11;
    mat(0,1) = -4.0; mat(1,1) = -5.0; mat(2,1) = -6.0; mat(3,1) = 0.0; mat(4,1) = 12;

    Eigen::MatrixXf mat2(3, mat.cols());
    mat2.block(0,0,3,mat.cols()) = mat.block(0,0,3,mat.cols())/lsb;
    //mat2.block(4,0,1,mat.cols()) = mat.block(4,0,1,mat.cols());
    mat2 = round(mat2.array()); //ROUND UP
    std::cout << mat2 << std::endl;    
    std::cout << "test" << std::endl;
    std::cout << sizeof(*(mat.data())) << std::endl;

    Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> mat3;
    mat3 = mat2.cast<short>();
    std::cout << sizeof(*(mat3.data())) << std::endl;
    std::cout << mat3 << std::endl;

    hdf5frame* hdf5data = new hdf5frame(compressed, compressed_level);
    hdf5data->input<short>("/pointcloud_xyz", mat3, H5T_NATIVE_SHORT);
    hdf5data->input<float>("pointcloud_intensity", mat.block(4,0,1,mat.cols()));
    hdf5data->scan();

    //Eigen::MatrixXf m = hdf5data->getMat<float>("/pointcloud_xyz");
    Eigen::Matrix<short, -1, -1> mat_ret = hdf5data->getMat<short>("/pointcloud_xyz", H5T_NATIVE_SHORT);
    Eigen::MatrixXf mat_intensity = hdf5data->getMat<float>("/pointcloud_intensity");
    std::cout << "get matrix" << std::endl;
    std::cout << mat_ret << std::endl;
    std::cout << mat_intensity << std::endl;

    Eigen::MatrixXf pc_mat = Eigen::MatrixXf::Zero(5,mat_ret.cols());
    pc_mat.block(0,0,3,pc_mat.cols()) = mat_ret.cast<float>()*lsb;
    pc_mat.block(4,0,1,pc_mat.cols()) = mat_intensity;
    std::cout << pc_mat << std::endl;
    */

    /*
    hdf5frame* hdf5data = new hdf5frame(compressed, compressed_level);
    std::vector<unsigned char> buf;
    hdf5data->get_file_image2(buf);
    std::cout << "size" << std::endl;
    std::cout << buf.size() << std::endl;
    delete hdf5data;
    hdf5frame* hdf5data2 = new hdf5frame(buf, 1, 1);
    */

    ros::spin();

}