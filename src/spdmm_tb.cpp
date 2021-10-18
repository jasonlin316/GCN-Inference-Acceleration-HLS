#include <hls_stream.h>
#include "ap_int.h"
#include "data_type.hpp"
#include <vector>
#define NNZ 123
// ********************************************************
int main() {
	// input, output and golden output arrays
    std::vector<data_type,aligned_allocator<ap_int<512>>>     value      ;// (nnz_size);
    std::vector<index_type,aligned_allocator<ap_int<512>>>    col_idx    ;//(nnz_size);
    std::vector<index_type,aligned_allocator<ap_int<512>>>    row_ptr    ;// (row_size+1); //+1 is to include the last element
    std::vector<data_type,aligned_allocator<ap_int<512>>>     dense_matrix;
    std::vector<indexandvalue,aligned_allocator<ap_int<512>>>  IndexAndValue;

    read_file <data_type> ("../data/data.bin", value);
    read_file <index_type> ("../data/indices.bin", col_idx);
    read_file <index_type> ("../data/indptr.bin", row_ptr);
    read_file <data_type> ("../data/feats.bin", dense_matrix);

    for(int i = 0; i < NNZ; i++)
    {
        indexandvalue element;
        element.data = value[i];
        element.index = col_idx[i];
        IndexAndValue.push_back(element);
    }
    
    //Call the kernel function 
    spdmm(dense_matrix, row_prt, IndexAndValue , output, (partition_size-1), NNZ, featurePn/VDATA_SIZE);
    

	// Compare the results file with the golden results
	int flag=0;
	for (i=0; i<SIZE; i++){
		if (golden_output[i] != kernel_output[i] ){
			flag=1;
			printf("Test data %d failed  !!!\n", i);
			return 1;
		}
	}

	if (flag==0)	printf("Test passed !\n");
   return 0;
}
