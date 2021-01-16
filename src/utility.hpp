
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "data_type.hpp"
//#include "cl_function.hpp"


template <class DType>
void read_file (
    std::string filePath, 
    std::vector<DType,aligned_allocator<ap_int<512>>> & data_array
)
{
    std::ifstream ifd(filePath, std::ios::binary | std::ios::ate );
    int data_size = ifd.tellg();
    ifd.seekg(0, std::ios::beg);
    if (ifd.fail()){std::cout <<"ERROR! " << filePath << " is not opened correctly." << std::endl;} 

    std::vector<DType> buffer;
    buffer.resize(data_size / sizeof(DType)); 
    ifd.read((char *)buffer.data(), data_size);
    for (int i=0; i<buffer.size(); i++) data_array.push_back(buffer[i]);
}

void weight_padding(
std::vector<data_type,aligned_allocator<ap_int<512>>> & weight_matrix, 
std::vector<data_type,aligned_allocator<ap_int<512>>> & padded_weight_matrix,
int dense_mtx_col,int hidden_dim, int & hiddenPn)
{   
    int weight_row = dense_mtx_col;
    int weight_col = hidden_dim;
    
    if(dense_mtx_col%16!=0) weight_row = (dense_mtx_col/ VDATA_SIZE +1) * VDATA_SIZE;
    if(hidden_dim%16!=0)    weight_col = (hidden_dim/ VDATA_SIZE +1) * VDATA_SIZE;
    
    hiddenPn = weight_col;
    padded_weight_matrix.resize(weight_row * weight_col);

    for(int i=0; i < weight_row; i++)
    {
        if(i < dense_mtx_col)
        {
            for(int j=0; j < weight_col; j++)
            {
                if(j < hidden_dim) padded_weight_matrix[i*weight_col+j] = weight_matrix[i*hidden_dim+j];
                else padded_weight_matrix[i*weight_col+j] = 0;
            }
        } else
        {
            for(int j=0; j < weight_col; j++) padded_weight_matrix[i*weight_col+j] = 0;
        }    
    }
}

void padding(std::vector<data_type,aligned_allocator<ap_int<512>>> & dense_matrix, 
int dense_mtx_col, int featurePn, int row_size, 
std::vector<data_type,aligned_allocator<ap_int<512>>> & padded_dense_matrix )
{
    for(int i=0; i < row_size; i++)
    {
        for(int j=0; j < featurePn; j++)
        {
            if(j < dense_mtx_col) padded_dense_matrix[i*featurePn+j] = dense_matrix[i*dense_mtx_col+j];
            else padded_dense_matrix[i*featurePn+j] = 0;
        }
    }
}