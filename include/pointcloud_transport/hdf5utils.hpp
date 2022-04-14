// HDF5
#ifndef HDF5_UTILS_INCLUDE_H_
#define HDF5_UTILS_INCLUDE_H_

#pragma once
#include <iostream>
#include <fstream>

// Eigen
#include <Eigen/Core>

//Opencv
//#include <opencv2/opencv.hpp>

#include "H5Cpp.h"
#include "hdf5_hl.h"

// ZSTD
#include "zstd.h"
#include "zstd_errors.h"

// NvComp
#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    if (err != cudaSuccess) {                                               \
      std::cerr << "Failure" << std::endl;                                \
      exit(1);                                                              \
    }                                                                         \
  } while (false)

class hdf5frame {
public:
  hdf5frame( int comp = false, int comp_lvl = 1 )
  {
    std::cout << __func__ << ":new" << std::endl;
    initialize(comp, comp_lvl);
    createNew("hdf5frame_new");

  }
  hdf5frame( std::vector<unsigned char> buf, int comp = false, int comp_lvl = 1 )
  {
    std::cout << __func__ << ": binary( size " << buf.size() << " )" << std::endl;
    initialize(comp, comp_lvl);
    read_file_image2( buf );
  }
  hdf5frame( std::string filename, int comp = false, int comp_lvl = 1 )
  {
    std::cout << __func__ << ": file ( " << filename << " )" << std::endl;
    initialize(comp, comp_lvl);

    /*Open the stream in binary mode.*/
    std::ifstream bin_file(filename, std::ios::binary);
    if ( !bin_file.good()) {
      std::cout << __func__ << std::endl;
      std::cout << "error: can't open file : " << filename << std::endl;
      exit(1);
    }

    /*Read Binary data using streambuffer iterators.*/
    std::vector<unsigned char> v_buf((std::istreambuf_iterator<char>(bin_file)), (std::istreambuf_iterator<char>()));
    bin_file.close();

    read_file_image2( v_buf );

  }

  ~hdf5frame(){
    std::cout << __func__ << " delete" << std::endl;
    H5Fclose(file_id);

  }

  void set_compressed_level( int l ){ compressed_level = l; }
  int getCompressed( void ){ return( compressed ); }
  int getCompressed_level( void ){ return( compressed_level ); }

  void copy_file_image( std::vector<unsigned char>& dst ){
    //nessesary to flush
    H5Fflush(file_id, H5F_SCOPE_LOCAL);
    ssize_t imgSize=H5Fget_file_image(file_id,NULL,0); // first call to determine size
    dst.resize(imgSize);
    H5Fget_file_image(file_id,dst.data(),imgSize); // second call to actually copy the data into our buffer
  }

  void get_file_image( std::vector<unsigned char>& dst ){
    //nessesary to flush
    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    ssize_t imgSize=H5Fget_file_image(file_id,NULL,0); // first call to determine size
    unsigned char* hdf5_img = new unsigned char[imgSize];
    H5Fget_file_image(file_id,hdf5_img,imgSize); // second call to actually copy the data into our buffer

    if( compressed ){
      size_t const cBuffSize = ZSTD_compressBound(imgSize);
      unsigned char* cBuff = new unsigned char[cBuffSize];
      //void* const cBuff = malloc_orDie(cBuffSize);

      /* Compress.
      * If you are doing many compressions, you may want to reuse the context.
      * See the multiple_simple_compression.c example.
      */
      //size_t const cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize, 1);
      //CHECK_ZSTD(cSize);
      size_t const cSize = ZSTD_compress(cBuff, cBuffSize, hdf5_img, imgSize, compressed_level);
      dst.resize(cSize);
      std::copy(cBuff, cBuff+cSize, dst.begin() );

    } else {
      dst.resize(imgSize);
      std::copy(hdf5_img, hdf5_img+imgSize, dst.begin() );

    }

  }

  void get_file_image2( std::vector<unsigned char>& dst ){
    std::cout << __func__ <<std::endl;
    //nessesary to flush
    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    ssize_t imgSize=H5Fget_file_image(file_id,NULL,0); // first call to determine size
    unsigned char* hdf5_img = new unsigned char[imgSize];
    H5Fget_file_image(file_id,hdf5_img,imgSize); // second call to actually copy the data into our buffer

    if( compressed ){

      //copy host to gpu
      uint8_t* device_input_ptrs;
      CUDA_CHECK(cudaMalloc(&device_input_ptrs, imgSize));
      CUDA_CHECK(cudaMemcpy(device_input_ptrs, hdf5_img, imgSize, cudaMemcpyDefault));      

      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      const int chunk_size = 1 << 16;
      nvcompType_t data_type = NVCOMP_TYPE_CHAR;

      nvcomp::LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
      nvcomp::CompressionConfig comp_config = nvcomp_manager.configure_compression(imgSize);

      uint8_t* comp_buffer;
      CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
      nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
      ssize_t comp_data_len = nvcomp_manager.get_compressed_output_size(comp_buffer);

      uint8_t* comp_host = new uint8_t[comp_data_len];
      CUDA_CHECK(cudaMemcpy(comp_host, comp_buffer, comp_data_len, cudaMemcpyDefault));      

      dst.resize(comp_data_len);
      std::copy(comp_host, comp_host+comp_data_len, dst.begin() );

      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaFree(device_input_ptrs));
      CUDA_CHECK(cudaFree(comp_buffer));

      CUDA_CHECK(cudaStreamDestroy(stream));

    } else {
      dst.resize(imgSize);
      std::copy(hdf5_img, hdf5_img+imgSize, dst.begin() );

    }

  }

  void write( std::string filename )
  {
    //nessesary to flush
    H5Fflush(file_id, H5F_SCOPE_LOCAL);
    
    ssize_t imgSize=H5Fget_file_image(file_id,NULL,0); // first call to determine size
    std::cout << imgSize << std::endl;
    char buf[imgSize];
    H5Fget_file_image(file_id,buf,imgSize); // second call to actually copy the data into our buffer

    if( compressed ){
      size_t const cBuffSize = ZSTD_compressBound(imgSize);
      unsigned char* cBuff = new unsigned char[cBuffSize];
      //void* const cBuff = malloc_orDie(cBuffSize);

      /* Compress.
      * If you are doing many compressions, you may want to reuse the context.
      * See the multiple_simple_compression.c example.
      */
      //size_t const cSize = ZSTD_compress(cBuff, cBuffSize, fBuff, fSize, 1);
      //CHECK_ZSTD(cSize);
      size_t const cSize = ZSTD_compress(cBuff, cBuffSize, buf, imgSize, compressed_level);

      auto hdf5_file = std::fstream(filename, std::ios::out | std::ios::binary);
      hdf5_file.write((char*)cBuff, cSize);
      hdf5_file.close();

    } else {
      auto hdf5_file = std::fstream(filename, std::ios::out | std::ios::binary);
      hdf5_file.write(buf, imgSize);
      hdf5_file.close();

    }

  }

  Eigen::MatrixXf getMat( std::string name, int row_major = false ){

    if( H5Lexists(file_id, name.c_str(), H5P_DEFAULT) <= 0 ){
      std::cout << __func__ << " : not found " << name << std::endl;
      exit(0);
    }

    //hid_t grp_id = H5Gopen(file_id, "/feature", H5P_DEFAULT);
    hid_t ds_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);

    hid_t space = H5Dget_space(ds_id);
    int rank = H5Sget_simple_extent_ndims(space);
    hsize_t* dims = new hsize_t[rank];
    int ndims = H5Sget_simple_extent_dims(space, dims, NULL);

    float* data_buf = new float[dims[0]*dims[1]];
    herr_t ret = H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_buf);
    Eigen::MatrixXf m;
    if( row_major ){
      m = Eigen::Map<Eigen::Matrix<float,-1,-1,Eigen::RowMajor>>(data_buf, dims[0], dims[1]);
    } else {
      m = Eigen::Map<Eigen::MatrixXf>(data_buf, dims[0], dims[1]);
    }

    return(m);

  }

#if 0
  cv::Mat getCvMat( std::string name ){

    if( H5Lexists(file_id, name.c_str(), H5P_DEFAULT) <= 0 ){
      std::cout << __func__ << " : not found " << name << std::endl;
      exit(0);
    }

    //hid_t grp_id = H5Gopen(file_id, "/feature", H5P_DEFAULT);
    hid_t ds_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);

    hid_t space = H5Dget_space(ds_id);
    int rank = H5Sget_simple_extent_ndims(space);
    hsize_t* dims = new hsize_t[rank];
    int ndims = H5Sget_simple_extent_dims(space, dims, NULL);

    int len = dims[0];
    for( int i=1; i<rank; i++ ) len*=dims[i];

    unsigned char* data_buf = new unsigned char[len];
    herr_t ret = H5Dread(ds_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_buf);

    cv::Mat mat(dims[0], dims[1]*dims[2], CV_8UC1, data_buf);
    return(mat.reshape(dims[2]));

  }

  void input( std::string name, cv::Mat mat ){
    std::vector<std::string> name_vec = split(name,'/');
    //Create a group named in the file. */
    std::string group_path;
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT );
    for( int i=0; i<name_vec.size()-1; i++){
      if( name_vec[i].empty() ) continue;

      group_path += "/" + name_vec[i];
      if( H5Lexists(file_id, group_path.c_str(), H5P_DEFAULT) > 0 ){
        group_id = H5Gopen(file_id, group_path.c_str(), H5P_DEFAULT );
      } else {
        group_id = H5Gcreate(file_id, group_path.c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT );
      }
    }
    std::string dname = name_vec.back();

    //std::cout << mat.dims << std::endl;
    //for( int i=0; i<mat.dims; i++ ) std::cout << mat.size[i] << std::endl;
    //std::cout << mat.channels() << std::endl;

    int num_dims = mat.dims+1;
    hsize_t* dims = new hsize_t[num_dims];
    for( int i=0; i<mat.dims; i++ ) dims[i] = mat.size[i];
    //channell
    dims[mat.dims] = mat.channels();
    hid_t space_id = H5Screate_simple( num_dims, dims, NULL );
    hid_t dt_id = H5Tcopy(H5T_NATIVE_UCHAR);

    hid_t ds_id = H5Dcreate(group_id, dname.c_str(), dt_id, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(ds_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data);

  }
#endif

  void input( std::string name, Eigen::VectorXf mat )
  {
    std::vector<std::string> name_vec = split(name,'/');
    //Create a group named in the file. */
    std::string group_path;
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT );
    for( int i=0; i<name_vec.size()-1; i++){
      if( name_vec[i].empty() ) continue;

      group_path += "/" + name_vec[i];
      if( H5Lexists(file_id, group_path.c_str(), H5P_DEFAULT) > 0 ){
        group_id = H5Gopen(file_id, group_path.c_str(), H5P_DEFAULT );
      } else {
        group_id = H5Gcreate(file_id, group_path.c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT );
      }
    }
    std::string dname = name_vec.back();

    hsize_t dims[2] = { mat.rows(), mat.cols()};
    hid_t space_id = H5Screate_simple( 2, dims, NULL );
    hid_t dt_id = H5Tcopy(H5T_NATIVE_FLOAT);
    //status = H5Tset_order(datatype, H5T_ORDER_LE);

    hid_t ds_id = H5Dcreate(group_id, dname.c_str(), dt_id, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());
  }

  void input( std::string name, Eigen::MatrixXf mat )
  {
    std::vector<std::string> name_vec = split(name,'/');
    //Create a group named in the file. */
    std::string group_path;
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT );
    for( int i=0; i<name_vec.size()-1; i++){
      if( name_vec[i].empty() ) continue;

      group_path += "/" + name_vec[i];
      if( H5Lexists(file_id, group_path.c_str(), H5P_DEFAULT) > 0 ){
        group_id = H5Gopen(file_id, group_path.c_str(), H5P_DEFAULT );
      } else {
        group_id = H5Gcreate(file_id, group_path.c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT );
      }
    }
    std::string dname = name_vec.back();

    hsize_t dims[2] = { mat.rows(), mat.cols()};
    hid_t space_id = H5Screate_simple( 2, dims, NULL );
    hid_t dt_id = H5Tcopy(H5T_NATIVE_FLOAT);
    //status = H5Tset_order(datatype, H5T_ORDER_LE);

    hid_t ds_id = H5Dcreate(group_id, dname.c_str(), dt_id, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());
    
  }

  void scan( void ){
    hid_t root_id = H5Gopen(file_id, "/", H5P_DEFAULT );
    char group_name[256];
    int len = H5Iget_name(root_id, group_name, 256);
    printf("Group Name: %s\n",group_name);
    scan_group(root_id);
  }


private:

  int compressed;
  int compressed_level;
  hid_t file_id;

  void initialize( int comp, int comp_lv )
  {
    compressed = comp;
    compressed_level = comp_lv;
  }

  void createNew( std::string hdf5name)
  {
    /* create the HDF5 file image first */
    hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
    herr_t h5err=H5Pset_fapl_core(faplist_id,/* memory increment size: 4M */1<<20,/*backing_store*/false);
    if(h5err<0) throw std::runtime_error("H5P_set_fapl_core failed.");
    hid_t fid = H5Fcreate(hdf5name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, faplist_id);

    /* add data like usual */
    //H5::Group grp=h5file.createGroup("somegroup");
    /* ... */

    /* get the image */
    H5Fflush(fid, H5F_SCOPE_LOCAL);
    ssize_t imgSize=H5Fget_file_image(fid,NULL,0); // first call to determine size
    std::vector<char> buf(imgSize);
    H5Fget_file_image(fid,buf.data(),imgSize); // second call to actually copy the data into our buffer

    file_id = H5LTopen_file_image(buf.data(),imgSize,H5LT_FILE_IMAGE_OPEN_RW);

    H5Fclose(fid);

  }

  void read_file_image( std::vector<unsigned char> v_buf ){

    if( compressed ){
      unsigned long long const buf_size = ZSTD_getFrameContentSize(v_buf.data(), v_buf.size());
      if( buf_size == ZSTD_CONTENTSIZE_ERROR || buf_size == ZSTD_CONTENTSIZE_UNKNOWN ){
        printf("not compressed by zstd!\n");
        exit(1);
      }

      unsigned char* hdf5 = new unsigned char[buf_size];
      unsigned long long const hdf5_size = ZSTD_decompress(hdf5, buf_size, v_buf.data(), v_buf.size());
      std::cout << "size" << std::endl;
      std::cout << hdf5_size << std::endl;
      file_id = H5LTopen_file_image( hdf5, hdf5_size, H5LT_FILE_IMAGE_OPEN_RW);

    } else {
      file_id = H5LTopen_file_image( v_buf.data(), v_buf.size(), H5LT_FILE_IMAGE_OPEN_RW);
    }

  }

  void read_file_image2( std::vector<unsigned char> v_buf ){
    std::cout << __func__ << std::endl;

    if( compressed ){
      //copy host to gpu
      uint8_t* comp_buffer;
      CUDA_CHECK(cudaMalloc(&comp_buffer, v_buf.size()));
      CUDA_CHECK(cudaMemcpy(comp_buffer, v_buf.data(), v_buf.size(), cudaMemcpyDefault));      

      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      const int chunk_size = 1 << 16;
      nvcompType_t data_type = NVCOMP_TYPE_CHAR;

      nvcomp::LZ4Manager nvcomp_manager{chunk_size, data_type, stream};

      nvcomp::DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
      uint8_t* res_decomp_buffer;
      CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

      std::cout << "nvcomp decomp: size " << decomp_config.decomp_data_size << std::endl;

      nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

      uint8_t* hdf5 = new uint8_t[decomp_config.decomp_data_size];
      CUDA_CHECK(cudaMemcpy(hdf5, res_decomp_buffer, decomp_config.decomp_data_size, cudaMemcpyDefault));      

      file_id = H5LTopen_file_image( hdf5, decomp_config.decomp_data_size, H5LT_FILE_IMAGE_OPEN_RW);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaFree(comp_buffer));
      CUDA_CHECK(cudaFree(res_decomp_buffer));

      CUDA_CHECK(cudaStreamDestroy(stream));

    } else {
      file_id = H5LTopen_file_image( v_buf.data(), v_buf.size(), H5LT_FILE_IMAGE_OPEN_RW);
    }

  }

  std::vector<std::string> split(std::string& input, char delimiter)
  {
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter)) {
          result.push_back(field);
    }
    return result;
  }

  void scan_group(hid_t group_id){
    hsize_t num_obj;
    char memb_name[256];
    H5Gget_num_objs(group_id, &num_obj);
	  for (int i = 0; i < num_obj; i++) {
      printf("  Member: %d ",i);
      int len = H5Gget_objname_by_idx(group_id, (hsize_t)i, memb_name, (size_t)256 );
		  printf("   %d ",len);
		  printf("  Member: %s ",memb_name);
      printf("\n");
		  int otype = H5Gget_objtype_by_idx(group_id, (size_t)i );
      switch(otype) {
		    case(H5G_LINK):
        {
			    printf(" SYM_LINK:\n");
			    //do_link(gid,memb_name);
			    break;
        }
		    case(H5G_UDLINK):
        {
			    printf(" UDLINK: Object is a user-defined link. \n");
			    //do_link(gid,memb_name);
			    break;
        }
			  case(H5G_GROUP):
        {
			    printf(" GROUP in %s:\n", memb_name);
			    hid_t grpid = H5Gopen(group_id, memb_name, H5P_DEFAULT);
			    scan_group(grpid);
          printf("end of GROUP %s \n", memb_name);
			    H5Gclose(grpid);
			    break;
         }
			  case(H5G_DATASET):
        {
			    printf(" DATASET:\n");
			    hid_t ds_id = H5Dopen(group_id,memb_name, H5P_DEFAULT);
          scan_dataset(ds_id);
			    //do_dset(dsid);
			    H5Dclose(ds_id);
			    break;
        }
			  case(H5G_TYPE):
        {
			    printf(" DATA TYPE:\n");
			    //typeid = H5Topen(gid,memb_name, H5P_DEFAULT);
			    //do_dtype(typeid);
			    //H5Tclose(typeid);
			    break;
        }
			  default:
				  printf(" unknown? : %d\n", otype);
				  break;
			}
      
    }
    //hid_t dataset_id = H5Dopen(file_id, "/feature", H5P_DEFAULT );    
  }

  void scan_dataset(hid_t ds_id){
    hid_t space = H5Dget_space(ds_id);
    int rank = H5Sget_simple_extent_ndims(space);
    hsize_t* dims = new hsize_t[rank];
    int ndims = H5Sget_simple_extent_dims(space, dims, NULL);
    std::cout << rank << ":";
    for( int i=0; i<rank; i++ ) {
      std::cout << dims[i];
      if( i<rank-1 ){
        std::cout << "x";
      } else {
        std::cout << std::endl;
      }
    }
    H5Sclose(space);
    hid_t tid = H5Dget_type(ds_id);
    H5T_class_t t_class;
    t_class = H5Tget_class(tid);
    switch(t_class){
      case(H5T_INTEGER):
        std::cout << "int data size(bytes)" << H5Tget_size(tid) << std::endl;
        break;
      case(H5T_FLOAT):
        std::cout << "float data size(bytes)" << H5Tget_size(tid) << std::endl;
        break;
    }
    //rdata = (hdset_reg_ref_t *) malloc (dims[0] * sizeof (hdset_reg_ref_t));
    hid_t type_t = H5Tget_native_type( tid, H5T_DIR_ASCEND );
    /*
    if( type_t == H5T_NATIVE_INT ){
      std::cout << "NATIVE_INT" << std::endl;
    } else if( type_t == H5T_NATIVE_FLOAT ){
      std::cout << "NATIVE_FLOAT" << std::endl;
    } else if( type_t == H5T_NATIVE_DOUBLE ){
      std::cout << "NATIVE_DOUBLE" << std::endl;
    } else if( type_t == H5T_NATIVE_B8 ){
      std::cout << "NATIVE_B8" << std::endl;
    } else {
      std::cout << "unkonw" << std::endl;
    }
    */
    H5Tclose(tid);

  }

};

#endif  //HDF5_UTILS_INCLUDE_H_
