
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define DEBUG
//#define PRINT_MATRIX
#include <vector>
#include <CL/cl2.hpp>
#include "cl_function.hpp"
#include "utility.hpp"
#include <iostream>
#include <fstream>
#include <CL/cl_ext_xilinx.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <stdio.h>
#include <ap_int.h>
#include <cstdlib>
#include <ctime>
#include "spdmm.h"
#include <iterator>
#include <string>
#include <cfloat>

using namespace std;

struct timespec start, stop; 

double run_time;

/* Parallelization factors */
int num_cu = 8; //number of compute units
int number_of_partition = 32; // to keep things simple, let number_of_partition = multiple of num_cu

/* meta-data */
int hidden_dim = 128; //48, 128
int output_dim = 41; //30, 47

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];

    //sparse matrix meta data
    int row_size; //number of sparse matrix rows, excluding the last element
    int nnz_size; //size of value array and col indices array
    //row_size = 2449029;
    //nnz_size = 123718152;

	int hidden_dim = 128;
	int output_dim = 41;

    //struct timespec start, stop; 
	  double kernel_total_time, mm_time, spdmm_time, inter_time  = 0;
    
    std::vector<data_type,aligned_allocator<ap_int<512>>>     value      ;   // (nnz_size);
    std::vector<index_type,aligned_allocator<ap_int<512>>>    col_idx    ;   //(nnz_size);
    std::vector<index_type,aligned_allocator<ap_int<512>>>    row_ptr    ;  // (row_size+1); //+1 is to include the last element
    std::vector<data_type,aligned_allocator<ap_int<512>>>     dense_matrix;

    std::vector<data_type,aligned_allocator<ap_int<512>>>     weight_matrix;
    std::vector<data_type,aligned_allocator<ap_int<512>>>     weight_matrix2;
    std::vector<data_type,aligned_allocator<ap_int<512>>>     padded_weight_matrix;
    std::vector<data_type,aligned_allocator<ap_int<512>>>     padded_weight_matrix2;

    std::vector<data_type,aligned_allocator<ap_int<512>>>     golden_AX;
    std::vector<data_type,aligned_allocator<ap_int<512>>>     golden_H;
    std::vector<data_type,aligned_allocator<ap_int<512>>>     golden_final;

    std::vector<index_type,aligned_allocator<ap_int<512>>> size_buffer;

    read_file <data_type> ("../data/AX.bin", golden_AX);
    read_file <data_type> ("../data/H.bin", golden_H);
    read_file <data_type> ("../data/final.bin", golden_final);

    //read sparse matrix
    //read_file <data_type> ("/home/jason/cluster-gcn/Amazon-2M/data.bin", value);
    //read_file <data_type> ("../reddit/data.bin", value);
    read_file <index_type> ("../reddit/indices.bin", col_idx);
    read_file <index_type> ("../reddit/indptr.bin", row_ptr);
    //read weight
    read_file <data_type> ("/home/jason/cluster-gcn/Amazon-2M/w1.bin", weight_matrix);
    read_file <data_type> ("/home/jason/cluster-gcn/Amazon-2M/w2.bin", weight_matrix2);
    //read dense matrix
    //read_file <data_type> ("/home/jason/cluster-gcn/Amazon-2M/feats.bin", dense_matrix); 

    read_file <index_type> ("../data/shape.bin", size_buffer);
    row_size = size_buffer[0];
    nnz_size = size_buffer[1];

    row_size = 232965;
	nnz_size = 23213838;
	int feature_length = 602;

	for(int i =0; i < row_size ; i++){
		for(int j = 0; j < feature_length; j++){
            dense_matrix.push_back(j*7/13);
		}
	}

    for(int i=0; i < nnz_size; i++)
    {
        value.push_back(i*5/19);
    }

	for(int i = 0; i< hidden_dim; i++){
		for(int j = 0; j < feature_length;j++){
			weight_matrix.push_back(i*feature_length + j);
		}
	} 

	for(int i = 0; i< hidden_dim; i++){
		for(int j = 0; j < output_dim;j++){
            weight_matrix2.push_back(i*output_dim + j);
		}
	} 
    
    int dense_mtx_col; //number of cols
    dense_mtx_col = dense_matrix.size()/row_size;

    int hiddenPn = hidden_dim;
    if(dense_mtx_col%VDATA_SIZE!=0 || hidden_dim%VDATA_SIZE!=0) 
    weight_padding(weight_matrix,padded_weight_matrix, dense_mtx_col,hidden_dim, hiddenPn);

    int outputPn = output_dim;
    if(hiddenPn%VDATA_SIZE!=0 || output_dim%VDATA_SIZE!=0) 
    weight_padding(weight_matrix2,padded_weight_matrix2, hiddenPn, output_dim, outputPn);

    //padding, let (# of dense matrix column) % VDATA_SIZE == 0
    int featurePn; //# of dense matrix col after padding
    bool need_padding = (dense_mtx_col % VDATA_SIZE != 0);
    if(!need_padding) featurePn = dense_mtx_col; 
    else featurePn = (dense_mtx_col/ VDATA_SIZE +1) * VDATA_SIZE;

    //directly copy matrix if padding is not needed, else pad the matrix
    std::vector<data_type,aligned_allocator<ap_int<512>>>    padded_dense_matrix;
    padded_dense_matrix.resize((row_size * featurePn));
    if(!need_padding) for(int i=0; i< dense_matrix.size(); i++) padded_dense_matrix[i] = dense_matrix[i];
    else padding(dense_matrix, dense_mtx_col, featurePn, row_size, padded_dense_matrix);

    //row_ptr partitioning
    int padded_row_size;
    if(row_size % number_of_partition == 0) padded_row_size = row_size;
    else 
    {
        padded_row_size = (row_size/number_of_partition+1)*number_of_partition;
        int iter = padded_row_size - row_size;
        index_type last = row_ptr.back();
        for(int i=0; i<iter; i++) row_ptr.push_back(last); //padding row_ptr
    }

    int chunk_row_size = padded_row_size / number_of_partition; 
    int partition_size = (chunk_row_size/VDATA_SIZE + 1) * VDATA_SIZE; 
    cout << "partition size: " << partition_size << endl;
    cout << "chunck row size:" << chunk_row_size << endl;
    //Let partition row size be multiple of VDATA.
    
    std::vector<index_type,aligned_allocator<ap_int<512>>>  row_ptr_part   [number_of_partition];
    for(int i=0; i < number_of_partition; i++)
    {
        row_ptr_part[i].resize(partition_size);
        for(int j=0; j < partition_size; j++)
        {
            if(j < (chunk_row_size+1)) row_ptr_part[i][j] = row_ptr[i * chunk_row_size + j];
            else row_ptr_part[i][j] = row_ptr[(i+1) * chunk_row_size ];
        }
    }

    std::vector<indexandvalue,aligned_allocator<ap_int<512>>>  IndexAndValue [number_of_partition];
    int nnz [number_of_partition];

    for(int i = 0; i < number_of_partition; i++)
    {
        int start = row_ptr_part[i][0];
        int end   = row_ptr_part[i][chunk_row_size];
        nnz[i] = end - start;
        for(int j = start; j < end; j++)
        {
            indexandvalue element;
            element.data = value[j];
            element.index = col_idx[j];
            IndexAndValue[i].push_back(element);
        }
    }

    cl_int err;
    unsigned fileBufSize;

    std::vector<data_type,aligned_allocator<ap_int<512>>>     AX_results [number_of_partition];
    std::vector<data_type,aligned_allocator<ap_int<512>>>     AXW_results       [number_of_partition];
    std::vector<data_type,aligned_allocator<ap_int<512>>>     AH_results        [number_of_partition];
    std::vector<data_type,aligned_allocator<ap_int<512>>>     AHW_results       [number_of_partition];

    for(int i=0; i < number_of_partition; i++) AX_results[i].resize(partition_size * featurePn);
    for(int i=0; i < number_of_partition; i++) AXW_results[i].resize(partition_size*hiddenPn);
    for(int i=0; i < number_of_partition; i++) AH_results[i].resize(partition_size*hiddenPn);
    for(int i=0; i < number_of_partition; i++) AHW_results[i].resize(partition_size*outputPn);

    //OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE |
                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    char* fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);

    #ifdef DEBUG
        cout << "FPGA binary read..." << endl;
    #endif

    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    #ifdef DEBUG
        cout << "Creating Kernels" << endl;
    #endif
    //Creating Kernel objects
    
    std::vector<cl::Kernel> spdmm_krnls(num_cu);
    std::vector<cl::Kernel> mmult_krnls(num_cu);

    for (int i = 0; i < num_cu; i++) OCL_CHECK(err, spdmm_krnls[i] = cl::Kernel(program, "spdmm", &err)); 
    for (int i = 0; i < num_cu; i++)OCL_CHECK(err, mmult_krnls[i] = cl::Kernel(program, "mmult", &err));
   
    //Creating buffers
    std::vector<cl::Buffer> buffer_in_row_ptr(num_cu);
    std::vector<cl::Buffer> buffer_out_AX (num_cu);
    std::vector<cl::Buffer> buffer_out_H (num_cu);
    std::vector<cl::Buffer> buffer_out_result (num_cu);
    std::vector<cl::Buffer> buffer_in_mm_AX (num_cu);
    std::vector<cl::Buffer> buffer_in_CSRidxAndVal(num_cu);
    std::vector<cl::Buffer> buffer_in_feats(num_cu);
    std::vector<cl::Buffer> buffer_in_weight(num_cu);

    int calc_iter = number_of_partition / num_cu;

    cl::Event event;
    uint64_t nstimestart, nstimeend;
    uint64_t total_time = 0, total_spdmm = 0, total_mmult = 0;

    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    #ifdef DEBUG
        cout << "=======================================" << endl;
        cout << "            Layer 1 of GCN "  << endl;
        cout << "=======================================" << endl;
    #endif
    std::vector<cl_mem_ext_ptr_t> inBuf_feats(num_cu);
    std::vector<cl_mem_ext_ptr_t> inBuf_weight(num_cu);

    for(int i=0;i<num_cu;i++)
    {
        if(i<num_cu/2) inBuf_feats[i].flags  = XCL_MEM_DDR_BANK0;
        else inBuf_feats[i].flags  = XCL_MEM_DDR_BANK3;
        inBuf_feats[i].obj = padded_dense_matrix.data(); 
        inBuf_feats[i].param = 0 ;

        if(i<num_cu/2) inBuf_weight[i].flags  = XCL_MEM_DDR_BANK0;
        else inBuf_weight[i].flags = XCL_MEM_DDR_BANK3;
        inBuf_weight[i].obj = padded_weight_matrix.data(); 
        inBuf_weight[i].param = 0 ;

    }

    for (int i = 0; i < num_cu; i++)
    {
      OCL_CHECK(err,
                    buffer_in_feats[i] = 
                        cl::Buffer (context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                                    sizeof(data_type)*(row_size*featurePn), &inBuf_feats[i], &err));

      OCL_CHECK(err, err = spdmm_krnls[i].setArg(0, buffer_in_feats[i]));

      OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_feats[i]}, 0 ));


      OCL_CHECK(err,
                buffer_in_weight[i] = cl::Buffer (context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX,
                                sizeof(data_type) * featurePn*hiddenPn, &inBuf_weight[i], &err)); 

      OCL_CHECK(err, err = mmult_krnls[i].setArg(1, buffer_in_weight[i]));
    
      OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in_weight[i]}, 0 )); 
    }

    
  

    for(int iter = 0; iter < calc_iter; iter++)
    {
        #ifdef DEBUG
        cout << "--------------------------------------" << endl;
        cout << "         starting iteration " << iter << endl;
        cout << "--------------------------------------" << endl;
        #endif

        //Allocate Buffer in Global Memory
        #ifdef DEBUG
        cout << "Allocating buffers..." << endl;
        #endif
        for (int i = 0; i < num_cu; i++) //input and output address will change every iteration
        {
        
            OCL_CHECK(err,
                    buffer_out_AX[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY ,
                                    sizeof(data_type)*(partition_size * featurePn), AX_results[iter * num_cu + i].data(), &err));
            
            OCL_CHECK(err,
                    buffer_in_mm_AX[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,
                                    sizeof(data_type)*(partition_size * featurePn), AX_results[iter * num_cu + i].data(), &err)); 

            OCL_CHECK(err,
                    buffer_in_row_ptr[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,
                                    sizeof(index_type)*(partition_size), row_ptr_part[iter * num_cu + i].data(), &err)); 
        
            OCL_CHECK(err,
                    buffer_in_CSRidxAndVal[i] = 
                        cl::Buffer (context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,
                                    sizeof(indexandvalue)*(nnz[iter * num_cu + i]), IndexAndValue[iter * num_cu + i].data(), &err));

            OCL_CHECK(err,
                    buffer_out_H[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY ,
                                    sizeof(data_type)*(partition_size*hiddenPn), AXW_results[iter * num_cu + i].data(), &err));
        }


            for (int i = 0; i < num_cu; i++)
            {
                #ifdef DEBUG
                cout << "setting kernel arguments for CU #" << i <<  endl;
                #endif
                //Setting kernel arguments
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(1, buffer_in_row_ptr[i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(2, buffer_in_CSRidxAndVal[i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(3, buffer_out_AX[i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(4, (partition_size-1))); // partition size k
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(5, nnz[iter * num_cu + i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(6,featurePn/VDATA_SIZE )); //featurePn/VDATA_SIZE
                //Copy input data to device global memory
                OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_row_ptr[i],buffer_in_CSRidxAndVal[i]}, 0 ));
            }
            //Launch the kernel
            #ifdef DEBUG
                cout << "kernel launch" <<  endl;
            #endif


            for(int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err, err = q.enqueueTask(spdmm_krnls[i], NULL, &event));
            }        
            #ifdef DEBUG
                cout << "enqueue task" <<  endl;
            #endif
            OCL_CHECK(err, err = q.finish());
            #ifdef DEBUG
                cout << "finish" <<  endl;
            #endif
           OCL_CHECK(err,
                            err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
                                                     &nstimestart));
            OCL_CHECK(err,
                        err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
                                                    &nstimeend));
            

            auto spdmm_time = nstimeend - nstimestart;
            total_spdmm += spdmm_time;

            cout << "========== spdmm timing l1 ===========" << endl;
            cout << spdmm_time/1e9 << endl;
            cout << "====================================" << endl ; 

            //Copy result from device global memory to host local memory
            for(int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err,  err = q.enqueueMigrateMemObjects({buffer_out_AX[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
            }
            OCL_CHECK(err, err = q.finish());

            //DENSE MM START
  
                for (int i = 0; i < num_cu; i++)
                {
                    #ifdef DEBUG
                    cout << "setting kernel arguments for MM #" << i <<  endl;
                    #endif
                    //Setting kernel arguments
                    OCL_CHECK(err, err = mmult_krnls[i].setArg(0, buffer_in_mm_AX[i])); 
                    OCL_CHECK(err, err = mmult_krnls[i].setArg(2, buffer_out_H[i])); 
                    OCL_CHECK(err, err = mmult_krnls[i].setArg(3, partition_size/VDATA_SIZE)); //amount of block row
                    OCL_CHECK(err, err = mmult_krnls[i].setArg(4, featurePn/VDATA_SIZE)); //amount of block col
                    OCL_CHECK(err, err = mmult_krnls[i].setArg(5, hiddenPn/VDATA_SIZE)); //amount of weight col
                    OCL_CHECK(err, err = mmult_krnls[i].setArg(6, true)); //ReLU
                    //Copy input data to device global memory
                    OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_mm_AX[i]}, 0 ));
                }

 
                //Launch the kernel
                for(int i = 0; i < num_cu; i++)
                {
                    #ifdef DEBUG
                    cout << "enqueue dense MM task " << i <<  endl;
                    #endif
                    OCL_CHECK(err, err = q.enqueueTask(mmult_krnls[i], NULL, &event));
                }  
            
                OCL_CHECK(err, err = q.finish());



                inter_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
                mm_time += inter_time;


                OCL_CHECK(err,
                            err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
                                                     &nstimestart));
                OCL_CHECK(err,
                            err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
                                                     &nstimeend));
                
                auto matmul_time = nstimeend - nstimestart;
                total_mmult += matmul_time;

                cout << "========== matmul timing l1===========" << endl;
                cout << matmul_time/1e9 << endl;
                cout << total_mmult/1e9 << endl;
                cout << "====================================" << endl ;
                
                #ifdef DEBUG
                        cout << "finish computation" << endl;
                #endif
                //Copy result from device global memory to host local memory
                for(int i = 0; i < num_cu; i++)
                {
                    OCL_CHECK(err,  err = q.enqueueMigrateMemObjects({buffer_out_H[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
                }
                OCL_CHECK(err, err = q.finish());
   
    
    } //finish iteration  
    auto l1_time = total_spdmm + total_mmult;
   
    std::vector<data_type,aligned_allocator<ap_int<512>>> H1; //hidden layer 1
    for(int pt = 0; pt < number_of_partition; pt++)
    {
        for(int i = 0; i < chunk_row_size; i++)
        {
            int row_num = pt*chunk_row_size + i;
            if(row_num < row_size)
            {
                for(int j = 0; j < featurePn; j++)
                    H1.push_back(AXW_results[pt][i*hiddenPn+j]);
            }
        }
    } 

    #ifdef DEBUG
        cout << "=======================================" << endl;
        cout << "            Layer 2 of GCN "  << endl;
        cout << "=======================================" << endl;
    #endif


    for(int i=0;i<num_cu;i++)
    {
        if(i<num_cu/2) inBuf_feats[i].flags  = XCL_MEM_DDR_BANK0;
        else inBuf_feats[i].flags  = XCL_MEM_DDR_BANK3;
        inBuf_feats[i].obj = H1.data(); 
        inBuf_feats[i].param = 0 ;

        if(i<num_cu/2) inBuf_weight[i].flags  = XCL_MEM_DDR_BANK0;
        else inBuf_weight[i].flags = XCL_MEM_DDR_BANK3;
        inBuf_weight[i].obj = padded_weight_matrix2.data();
        inBuf_weight[i].param = 0;

    }

    for (int i = 0; i < num_cu; i++)
    {
      OCL_CHECK(err,
                    buffer_in_feats[i] = 
                        cl::Buffer (context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                                    sizeof(data_type)*(row_size*featurePn), &inBuf_feats[i], &err));

      OCL_CHECK(err, err = spdmm_krnls[i].setArg(0, buffer_in_feats[i]));

      OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_feats[i]}, 0 ));


      OCL_CHECK(err,
                buffer_in_weight[i] = cl::Buffer (context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR| CL_MEM_EXT_PTR_XILINX,
                                sizeof(data_type) * outputPn*hiddenPn, &inBuf_weight[i], &err)); 

      OCL_CHECK(err, err = mmult_krnls[i].setArg(1, buffer_in_weight[i]));
    
      OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in_weight[i]}, 0 )); 
    }
    
    
    for(int iter = 0; iter < calc_iter; iter++)
    {
        #ifdef DEBUG
        cout << "--------------------------------------" << endl;
        cout << "         starting iteration " << iter << endl;
        cout << "--------------------------------------" << endl;
        #endif

        //Allocate Buffer in Global Memory
        #ifdef DEBUG
        cout << "Allocating buffers..." << endl;
        #endif

        for (int i = 0; i < num_cu; i++) //input and output address will change every iteration
        {
            OCL_CHECK(err,
                    buffer_in_row_ptr[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,
                                    sizeof(index_type)*(partition_size), row_ptr_part[iter * num_cu + i].data(), &err));

            OCL_CHECK(err,
                    buffer_out_AX[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY ,
                                    sizeof(data_type)*(partition_size * hiddenPn), AH_results[iter * num_cu + i].data(), &err));
            
            OCL_CHECK(err,
                    buffer_in_mm_AX[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,
                                    sizeof(data_type)*(partition_size * hiddenPn), AH_results[iter * num_cu + i].data(), &err)); 

            OCL_CHECK(err,
                    buffer_out_result[i] =
                        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY ,
                                    sizeof(data_type)*(outputPn * partition_size), AHW_results[iter * num_cu + i].data(), &err)); 
            
            OCL_CHECK(err,
                    buffer_in_CSRidxAndVal[i] = 
                        cl::Buffer (context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,
                                    sizeof(indexandvalue)*(nnz[iter * num_cu + i]), IndexAndValue[iter * num_cu + i].data(), &err));
        
        }

            for (int i = 0; i < num_cu; i++)
            {
                #ifdef DEBUG
                cout << "setting kernel arguments for CU #" << i <<  endl;
                #endif
                //Setting kernel arguments
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(1, buffer_in_row_ptr[i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(2, buffer_in_CSRidxAndVal[i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(3, buffer_out_AX[i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(4, (partition_size-1))); // partition size k
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(5, nnz[iter * num_cu + i]));
                OCL_CHECK(err, err = spdmm_krnls[i].setArg(6, hiddenPn/VDATA_SIZE));
                //Copy input data to device global memory
                OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_row_ptr[i],buffer_in_CSRidxAndVal[i]}, 0 ));
            }
            //for(int i=0;i<4;i++) OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_feats[i]}, 0 ));
        
            //Launch the kernel

            for(int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err, err = q.enqueueTask(spdmm_krnls[i], NULL, &event));
            }        
            OCL_CHECK(err, err = q.finish());


            OCL_CHECK(err,
                            err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
                                                     &nstimestart));
            OCL_CHECK(err,
                        err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
                                                    &nstimeend));
            
            auto spdmm_time = nstimeend - nstimestart;
            total_spdmm += spdmm_time;

            cout << "========== spdmm timing l2 ===========" << endl;
            cout << spdmm_time/1e9 << endl;
            cout << "====================================" << endl ;

            //Copy result from device global memory to host local memory
            for(int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err,  err = q.enqueueMigrateMemObjects({buffer_out_AX[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
            }
            OCL_CHECK(err, err = q.finish());
        
            //DENSE MM START
            for (int i = 0; i < num_cu; i++)
            {
                #ifdef DEBUG
                cout << "setting kernel arguments for MM #" << i <<  endl;
                #endif
                //Setting kernel arguments
                OCL_CHECK(err, err = mmult_krnls[i].setArg(0, buffer_in_mm_AX[i])); 
                OCL_CHECK(err, err = mmult_krnls[i].setArg(2, buffer_out_result[i])); 
                OCL_CHECK(err, err = mmult_krnls[i].setArg(3, partition_size/VDATA_SIZE)); //amount of block row
                OCL_CHECK(err, err = mmult_krnls[i].setArg(4, hiddenPn/VDATA_SIZE)); //amount of block col
                OCL_CHECK(err, err = mmult_krnls[i].setArg(5, outputPn/VDATA_SIZE)); //amount of weight col
                OCL_CHECK(err, err = mmult_krnls[i].setArg(6, false));
                //Copy input data to device global memory
                OCL_CHECK(err, err =  q.enqueueMigrateMemObjects({buffer_in_mm_AX[i]}, 0 ));
            }

            //Launch the kernel
            for(int i = 0; i < num_cu; i++)
            {
                #ifdef DEBUG
                cout << "enqueue dense MM task " << i <<  endl;
                #endif
                OCL_CHECK(err, err = q.enqueueTask(mmult_krnls[i], NULL, &event));
            }    

            OCL_CHECK(err, err = q.finish());

            #ifdef DEBUG
                    cout << "finish computation" << endl;
            #endif

            OCL_CHECK(err,
                        err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
                                                    &nstimestart));
            OCL_CHECK(err,
                        err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
                                                    &nstimeend));
            
            auto matmul_time = nstimeend - nstimestart;
            total_mmult += matmul_time;
	

            cout << "========== matmul timing l2===========" << endl;
            cout << matmul_time/1e9 << endl;
            cout << "====================================" << endl ;
            
            //Copy result from device global memory to host local memory
            for(int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err,  err = q.enqueueMigrateMemObjects({buffer_out_result[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
            }
            OCL_CHECK(err, err = q.finish()); 
   
        
    } //finish iteration 


    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    inter_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    int err_cnt = 0;
    total_time = total_spdmm + total_mmult;
    cout << "layer1 exe time: " << l1_time/1e9 << endl;
    cout << "layer2 exe time: " << (total_time - l1_time)/1e9 << endl;
    cout << "total exe time: " << total_time/1e9 << endl;
    cout << "total program time: " << inter_time << endl;
    cout << "total spdmm time: " << total_spdmm/1e9 << endl;
    cout << "total mmult time: " << total_mmult/1e9 << endl;
 
    if(!err_cnt) cout << "TEST PASSED." << endl;
    else cout << "TEST FAILED" << endl;

//OPENCL HOST CODE AREA END
 
    delete[] fileBuf;
    
    //std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl; 
    //return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
}

