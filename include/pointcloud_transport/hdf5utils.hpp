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
    std::cout << file_id << std::endl;
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

  size_t encode_nvheader(size_t chunk_size, size_t batch_size, size_t* host_compressed_bytes, char* header = nullptr) {
    
    //size: chunck + batch + compressed_bytes + compressed
    size_t header_bytes = sizeof(size_t) + sizeof(size_t) + (sizeof(size_t) * batch_size);
    // return size only
    if( header == nullptr ){
      return(header_bytes);
    }

    memcpy(header,&chunk_size,sizeof(size_t));
    memcpy(header+sizeof(size_t),&batch_size,sizeof(size_t));
    memcpy(header+sizeof(size_t)*2,host_compressed_bytes,sizeof(size_t) * batch_size);

    return(header_bytes);

  }

  size_t decode_nvheader(char* bin_data, size_t& chunk_size, size_t& batch_size, size_t* host_compressed_bytes = nullptr) {

    memcpy(&chunk_size, bin_data, sizeof(size_t));
    memcpy(&batch_size, bin_data+sizeof(size_t), sizeof(size_t));

    //size: chunck + batch + compressed_bytes + compressed
    size_t bytes_size = (sizeof(size_t) * batch_size);
    // return size only
    if( host_compressed_bytes == nullptr ){
      return(bytes_size);
    }

    memcpy(host_compressed_bytes, bin_data+sizeof(size_t)*2, bytes_size);

    return(sizeof(size_t)*2+bytes_size);
    
  }

  void execute_nvdecomp(char* bin_data, const size_t bin_bytes, std::vector<unsigned char>& dst ){
    std::cout << __func__ << std::endl;

    size_t chunk_size;
    size_t batch_size;

    size_t bytes_size = decode_nvheader(bin_data, chunk_size, batch_size);
    
    //std::cout << chunk_size << std::endl;
    //std::cout << batch_size << std::endl;

    size_t* host_compressed_bytes;
    cudaMallocHost((void**)&host_compressed_bytes, bytes_size);

    size_t header_size = decode_nvheader(bin_data, chunk_size, batch_size, host_compressed_bytes);

    //std::cout << "host_compressed_bytes" << std::endl;
    //for( int i=0; i<batch_size;  i++ ) std::cout << host_compressed_bytes[i] << std::endl;
 
    size_t in_bytes = bin_bytes - header_size;
    std::cout << "in_bytes" << std::endl;
    std::cout << in_bytes << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    char* device_input_data;
    cudaMalloc((void**)&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, bin_data+header_size, in_bytes, cudaMemcpyHostToDevice, stream);

    // Setup an array of pointers to the start of each chunk
    void ** host_compressed_ptrs;
    cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t)*batch_size);
    host_compressed_ptrs[0] = device_input_data;
    for (size_t ix_chunk = 1; ix_chunk < batch_size; ++ix_chunk) {
      host_compressed_ptrs[ix_chunk] = host_compressed_ptrs[ix_chunk-1] + host_compressed_bytes[ix_chunk-1];
    }

    size_t* device_compressed_bytes;
    void ** device_compressed_ptrs;
    cudaMalloc((void**)&device_compressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc((void**)&device_compressed_ptrs, sizeof(size_t) * batch_size);
  
    cudaMemcpyAsync(device_compressed_bytes, host_compressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

    // Decompression can be similarly performed on a batch of multiple compressed input chunks. 
    // As no metadata is stored with the compressed data, chunks can be re-arranged as well as decompressed 
    // with other chunks that originally were not compressed in the same batch.

    // If we didn't have the uncompressed sizes, we'd need to compute this information here. 
    // We demonstrate how to do this.
    size_t* device_uncompressed_bytes;
    cudaMalloc((void**)&device_uncompressed_bytes, sizeof(size_t) * batch_size);
    
    nvcompBatchedLZ4GetDecompressSizeAsync(
      device_compressed_ptrs,
      device_compressed_bytes,
      device_uncompressed_bytes,
      batch_size,
      stream);

    // allocate space for compressed chunk sizes to be written to

    // Next, allocate output space on the device
    void ** host_uncompressed_ptrs;
    cudaMallocHost((void**)&host_uncompressed_ptrs, sizeof(size_t) * batch_size);
    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      cudaMalloc(&host_uncompressed_ptrs[ix_chunk], chunk_size);
    }

    void** device_uncompressed_ptrs;
    cudaMalloc((void**)&device_uncompressed_ptrs, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(
      device_uncompressed_ptrs, host_uncompressed_ptrs, 
      sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);

    // Next, allocate the temporary buffer 
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);
    void * device_decomp_temp;
    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

    // allocate statuses
    nvcompStatus_t* device_statuses;
    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t)*batch_size);

    // Also allocate an array to store the actual_uncompressed_bytes.
    // Note that we could use nullptr for this. We already have the 
    // actual sizes computed during the call to nvcompBatchedLZ4GetDecompressSizeAsync.
    size_t* device_actual_uncompressed_bytes;
    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t)*batch_size);

    // And finally, call the decompression routine.
    // This decompresses each input, device_compressed_ptrs[i], and places the decompressed
    // result in the corresponding output list, device_uncompressed_ptrs[i]. It also writes
    // the size of the uncompressed data to device_uncompressed_bytes[i].
    nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
      device_compressed_ptrs, 
      device_compressed_bytes, 
      device_uncompressed_bytes, 
      device_actual_uncompressed_bytes, 
      batch_size,
      device_decomp_temp, 
      decomp_temp_bytes, 
      device_uncompressed_ptrs, 
      device_statuses, 
      stream);
  
    if (decomp_res != nvcompSuccess)
    {
      std::cerr << "Failed compression!" << std::endl;
      assert(decomp_res == nvcompSuccess);
    }    

    size_t *host_uncompressed_bytes;
    cudaMallocHost((void**)&host_uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(host_uncompressed_bytes, device_actual_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost, stream);

    //allocate uncompressed
    cudaStreamSynchronize(stream);

    //std::cout << "host_uncompressed_bytes" << std::endl;
    size_t out_bytes = 0;
    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      //std::cout << host_compressed_bytes[ix_chunk] << std::endl;
      //std::cout << (size_t)(host_uncompressed_bytes[ix_chunk]) << std::endl;
      out_bytes += (size_t)(host_uncompressed_bytes[ix_chunk]);
    }
    //std::cout << out_bytes << std::endl;

    char* host_output_data;
    cudaMallocHost((void**)&host_output_data, out_bytes);

    // Setup an array of pointers to the start of each chunk
    void ** host_output_ptrs;
    cudaMallocHost((void**)&host_output_ptrs, sizeof(size_t)*batch_size);
    host_output_ptrs[0] = host_output_data;
    for (size_t ix_chunk = 1; ix_chunk < batch_size; ++ix_chunk) {
      host_output_ptrs[ix_chunk] = host_output_ptrs[ix_chunk-1] + host_uncompressed_bytes[ix_chunk-1];
    }

    //std::cout << "copy device to host" << std::endl;
    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      //std::cout << host_output_ptrs[ix_chunk] << std::endl;
      //std::cout << (size_t)(host_uncompressed_bytes[ix_chunk]) << std::endl;
      //cudaMemcpyAsync(host_output_ptrs[ix_chunk], device_compressed_ptrs, (size_t)(host_compressed_bytes[ix_chunk]), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(host_output_ptrs[ix_chunk], host_uncompressed_ptrs[ix_chunk], (size_t)(host_uncompressed_bytes[ix_chunk]), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    dst.resize(out_bytes);
    std::copy(host_output_data, host_output_data+out_bytes, dst.begin() );

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "decomp finish" << std::endl;

  }


  void execute_nvcomp(char* input_data, const size_t in_bytes, std::vector<unsigned char>& dst){
    std::cout << __func__ << std::endl;
    std::cout << in_bytes << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // First, initialize the data on the host.

    // compute chunk sizes
    size_t* host_uncompressed_bytes;
    const size_t chunk_size = 65536;
    //const size_t chunk_size = 200;
    const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

    char* device_input_data;
    cudaMalloc((void**)&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);

    cudaMallocHost((void**)&host_uncompressed_bytes, sizeof(size_t)*batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      if (i + 1 < batch_size) {
        host_uncompressed_bytes[i] = chunk_size;
      } else {
        // last chunk may be smaller
        host_uncompressed_bytes[i] = in_bytes - (chunk_size*i);
      }
    }

    // Setup an array of pointers to the start of each chunk
    void ** host_uncompressed_ptrs;
    cudaMallocHost((void**)&host_uncompressed_ptrs, sizeof(size_t)*batch_size);
    for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      host_uncompressed_ptrs[ix_chunk] = device_input_data + chunk_size*ix_chunk;
    }

   //for( int i=0; i<batch_size; i++) std::cout << host_uncompressed_bytes[i] << std::endl;

    size_t* device_uncompressed_bytes;
    void ** device_uncompressed_ptrs;
    cudaMalloc((void**)&device_uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc((void**)&device_uncompressed_ptrs, sizeof(size_t) * batch_size);
  
    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

    // Then we need to allocate the temporary workspace and output space needed by the compressor.
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void* device_temp_ptr;
    cudaMalloc(&device_temp_ptr, temp_bytes);

    // get the maxmimum output size for each chunk
    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    // Next, allocate output space on the device
    void ** host_compressed_ptrs;
    cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t) * batch_size);
    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes);
    }

    void** device_compressed_ptrs;
    cudaMalloc((void**)&device_compressed_ptrs, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(
      device_compressed_ptrs, host_compressed_ptrs, 
      sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);

    // allocate space for compressed chunk sizes to be written to
    size_t * device_compressed_bytes;
    cudaMalloc((void**)&device_compressed_bytes, sizeof(size_t) * batch_size);

    // And finally, call the API to compress the data
    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(  
      device_uncompressed_ptrs,    
      device_uncompressed_bytes,  
      chunk_size, // The maximum chunk size  
      batch_size,  
      device_temp_ptr,  
      temp_bytes,  
      device_compressed_ptrs,  
      device_compressed_bytes,  
      nvcompBatchedLZ4DefaultOpts,  
      stream);

    if (comp_res != nvcompSuccess)
    {
      std::cerr << "Failed compression!" << std::endl;
      assert(comp_res == nvcompSuccess);
    }

    // allocate space for compressed chunk sizes to be written to
    size_t *host_compressed_bytes;
    cudaMallocHost((void**)&host_compressed_bytes, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(host_compressed_bytes, device_compressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    size_t out_bytes = 0;
    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        out_bytes += (size_t)(host_compressed_bytes[ix_chunk]);
    }
    std::cout << "out_bytes" << std::endl;
    std::cout << out_bytes << std::endl;

    char* host_output_data;
    cudaMallocHost((void**)&host_output_data, out_bytes);

    // Setup an array of pointers to the start of each chunk
    void ** host_output_ptrs;
    cudaMallocHost((void**)&host_output_ptrs, sizeof(size_t)*batch_size);
    host_output_ptrs[0] = host_output_data;
    std::cout << host_output_ptrs[0] << std::endl;
    for (size_t ix_chunk = 1; ix_chunk < batch_size; ++ix_chunk) {
      host_output_ptrs[ix_chunk] = host_output_ptrs[ix_chunk-1] + host_compressed_bytes[ix_chunk-1];
      std::cout << host_output_ptrs[ix_chunk] << std::endl;
    } 

    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      //std::cout << host_output_ptrs[ix_chunk] << std::endl;
      //std::cout << (size_t)(host_compressed_bytes[ix_chunk]) << std::endl;
      //cudaMemcpyAsync(host_output_ptrs[ix_chunk], device_compressed_ptrs, (size_t)(host_compressed_bytes[ix_chunk]), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(host_output_ptrs[ix_chunk], host_compressed_ptrs[ix_chunk], (size_t)(host_compressed_bytes[ix_chunk]), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    
    //create bin data
    size_t header_bytes = encode_nvheader(chunk_size, batch_size, host_compressed_bytes);
    char* header_data = new char[header_bytes];
    encode_nvheader(chunk_size, batch_size, host_compressed_bytes, header_data);

    size_t bin_bytes = header_bytes + out_bytes;
    //std::cout << bin_bytes << std::endl;    
    char* bin_data = new char[bin_bytes];

    memcpy(bin_data,header_data,header_bytes);
    memcpy(bin_data+header_bytes,host_output_data,out_bytes);

    dst.resize(bin_bytes);
    std::copy(bin_data, bin_data+bin_bytes, dst.begin() );

    //execute_nvdecomp(bin_data, bin_bytes);
    
    //std::cout << device_compressed_bytes[0] << std::endl;
    //std::cout << device_uncompressed_bytes[0] << std::endl;

  }

  void get_file_image2( std::vector<unsigned char>& dst ){
    std::cout << __func__ <<std::endl;
    //nessesary to flush
    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    ssize_t imgSize=H5Fget_file_image(file_id,NULL,0); // first call to determine size
    char* hdf5_img = new char[imgSize];
    H5Fget_file_image(file_id,hdf5_img,imgSize); // second call to actually copy the data into our buffer

    if( compressed ){
      execute_nvcomp(hdf5_img, imgSize, dst);
      std::cout << "finish!!" << std::endl;

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

    H5Dclose(ds_id);

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
    H5Dclose(ds_id);

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

    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    H5Tclose(dt_id); //not nessesary
    H5Sclose(space_id); //not nessesary

    H5Dclose(ds_id);
    H5Gclose(group_id);
    H5Pclose(lcpl); //not nessesary

  }
#endif

  void input( std::string name, Eigen::VectorXf mat )
  {
    std::vector<std::string> name_vec = split(name,'/');
    //Create a group named in the file.
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
  
    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    H5Tclose(dt_id); //not nessesary
    H5Sclose(space_id); //not nessesary

    H5Dclose(ds_id);
    H5Gclose(group_id);
    H5Pclose(lcpl); //not nessesary

  }

  void input( std::string name, Eigen::MatrixXf mat )
  {
    std::vector<std::string> name_vec = split(name,'/');
    //Create a group named in the file.
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

    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    H5Tclose(dt_id); //not nessesary
    H5Sclose(space_id); //not nessesary

    H5Dclose(ds_id);
    H5Gclose(group_id);
    H5Pclose(lcpl); //not nessesary
  }

  void scan( void ){
    hid_t root_id = H5Gopen(file_id, "/", H5P_DEFAULT );
    char group_name[256];
    int len = H5Iget_name(root_id, group_name, 256);
    printf("Group Name: %s\n",group_name);
    scan_group(root_id);

    H5Gclose(root_id);
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

  void createNew( std::string hdf5name )
  {
    /* create the HDF5 file image first */
    hid_t fcplist_id = H5Pcreate(H5P_FILE_CREATE);
    hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
    herr_t h5err=H5Pset_fapl_core(faplist_id,/* memory increment size: 4M */1<<20,/*backing_store*/false);
    if(h5err<0) throw std::runtime_error("H5P_set_fapl_core failed.");
    file_id = H5Fcreate(hdf5name.c_str(), H5F_ACC_TRUNC, fcplist_id, faplist_id);

    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    /*
    ssize_t imgSize = H5Fget_file_image(fid,NULL,0);
    std::vector<unsigned char> buf(imgSize);
    H5Fget_file_image(fid,buf.data(),imgSize);
    H5Fclose(fid);

    open_file_image("hdf5frame_new", buf.data(), imgSize);
    */

  }

  void open_file_image( std::string hdf5name, unsigned char* data, int size ){
    /* create the HDF5 file image first */
    hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
    herr_t h5err=H5Pset_fapl_core(faplist_id,/* memory increment size: 4M */1<<20,/*backing_store*/false);
    if(h5err<0) throw std::runtime_error("H5P_set_fapl_core failed.");
    H5Pset_file_image(faplist_id, data, size);
    file_id = H5Fopen(hdf5name.c_str(), H5F_ACC_RDWR, faplist_id);

    H5Fflush(file_id, H5F_SCOPE_LOCAL);

    H5Pclose(faplist_id);

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
      open_file_image( "hdf5frame_new", hdf5, hdf5_size );

    } else {
      open_file_image( "hdf5frame_new", v_buf.data(), v_buf.size() );
    }

  }

  void read_file_image2( std::vector<unsigned char> v_buf ){
    std::cout << __func__ << std::endl;

    if( compressed ){
      std::vector<unsigned char> hdf5_buf;
      execute_nvdecomp((char*)v_buf.data(), v_buf.size(), hdf5_buf);
      open_file_image( "hdf5frame_new", hdf5_buf.data(), hdf5_buf.size() );

    } else {
      open_file_image( "hdf5frame_new", v_buf.data(), v_buf.size() );
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
