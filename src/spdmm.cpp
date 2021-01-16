#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>

#define VDATA_SIZE 16
#define DENSE_BUFFER_LENGTH 4096
#define INDEX_BUFFER_LENGTH 16
#define OUTPUTBUFFER_LENGTH 256





typedef float data_type;
typedef int index_type;


struct v_datatype {
    data_type data[VDATA_SIZE];
};

struct v_indextype {
	index_type data[VDATA_SIZE];
};

struct indexandvalue {
      data_type data;
      index_type index;
};

struct v_indexandvalue{
      indexandvalue data[8];
}; 
extern "C"{

static void readindptr( const v_indextype * CSRindptr, hls::stream<int> &additionNum1, hls::stream<int> &additionNum2, const int  k  ){

    v_indextype tmpindptr;
#pragma HLS aggregate variable = tmpindptr

    
    index_type last = 0;
    int counter = 0;
    
    int numberofit = int( (k + 1)  / VDATA_SIZE);
    if(int( (k + 1) % VDATA_SIZE)!= 0 ){
        numberofit = numberofit + 1;
    }
    
    
    for(int i = 0; i < numberofit; i = i + 1){
        tmpindptr = CSRindptr[i];
        
        for(int j = 0; j < VDATA_SIZE; j++){
#pragma HLS PIPELINE rewind
            if(counter < k){
              if(j == 0){
                if(i != 0){
                  additionNum1 << (tmpindptr.data[j] - last);
                  additionNum2 << (tmpindptr.data[j] - last);
                  counter = counter + 1;
                }
              }
              else{
                additionNum1 << (tmpindptr.data[j] - tmpindptr.data[j - 1]);
                additionNum2 << (tmpindptr.data[j] - tmpindptr.data[j - 1]);
                counter = counter + 1;
              }
            }
        }
        
        last = tmpindptr.data[VDATA_SIZE - 1];
    }
}



static void readindexandvalue( const v_indexandvalue * CSRindexandvalue,  hls::stream<index_type> &streamCSRindex, hls::stream<data_type> &streamCSRvalue, const int nnz){

    v_indexandvalue tmpindexandvalue;
#pragma HLS aggregate variable = tmpindexandvalue

    int counter = 0;
    for(int i = 0 ; i < nnz; i = i + 8 ){
      tmpindexandvalue = CSRindexandvalue[int(i / 8)];
      for(int j = 0; j < 8; j++ ){
#pragma HLS PIPELINE II=1 rewind
        if (counter < nnz){
          indexandvalue tmpiv = tmpindexandvalue.data[j];
#pragma HLS aggregate variable = tmpiv          
          streamCSRindex << tmpiv.index;
          streamCSRvalue << tmpiv.data;
          counter = counter + 1;
        }
      }
    }
}






static void readdense(const v_datatype * dense, hls::stream<index_type> &streamCSRindex, hls::stream<v_datatype> &streamdense,  const int  featurepn, const int  nnz){

    
    for (int i = 0; i < nnz ; i++){
        index_type iindex = streamCSRindex.read();
        for (int j = 0 ; j < featurepn; j++){
#pragma HLS PIPELINE II=1 rewind
          streamdense <<  dense[iindex*featurepn + j];
        }
    }
    
    
     
}


static void spdmm_compute(hls::stream<v_datatype> & streamdense, hls::stream<int> &additionNum, hls::stream<data_type> &streamCSRvalue,  hls::stream<v_datatype> &streamResult, const int  featurepn, const int  nnz, const int k){

    for (int i = 0; i < k ; i++){
      int itr_number = additionNum.read();
      v_datatype tmpoutbuffer[OUTPUTBUFFER_LENGTH];
#pragma HLS resource variable=tmpoutbuffer core=RAM_2P_LUTRAM 
#pragma HLS aggregate variable = tmpoutbuffer
//#pragma HLS aggregate variable = tmpoutbuffer

      
      
      for(int j = 0; j < itr_number; j++){
          data_type tmpvalue = streamCSRvalue.read();

          for(int ii = 0; ii < featurepn; ii = ii + 1){
#pragma HLS PIPELINE II=1 rewind

            v_datatype tmpdense= streamdense.read();
#pragma HLS aggregate variable = tmpdense

            if(j == 0){
              for(int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                tmpoutbuffer[ii].data[jj] = tmpdense.data[jj]*tmpvalue;
              }
            }
            else { 
              for(int jj = 0; jj < VDATA_SIZE; jj = jj + 1){
                data_type last =  tmpoutbuffer[ii].data[jj];
                tmpoutbuffer[ii].data[jj] = last + tmpdense.data[jj]*tmpvalue;

              } 
            }
                
            if(j == itr_number - 1 ){
                streamResult << tmpoutbuffer[ii];}
          }     
      }   
    }
}


static void readResult( hls::stream<v_datatype> &streamResult, hls::stream<int> &additionNum, v_datatype * output, const int k, const int  featurepn){

    v_datatype zeros;
#pragma HLS aggregate variable = zeros
    for(int i = 0; i < VDATA_SIZE; i++)
    {
#pragma HLS PIPELINE II=1
        zeros.data[i] = 0;
    }
    

    for (int i = 0; i < k ; i++){
       int itr_number = additionNum.read();
       
       if(itr_number > 0){
         for(int j = 0; j < featurepn; j = j + 1){
#pragma HLS PIPELINE II=1 rewind
           output[i*featurepn + j] = streamResult.read();
         }
       }
       else{
         for(int j = 0; j < featurepn; j = j + 1){
#pragma HLS PIPELINE II=1 rewind
           output[i*featurepn + j] = zeros;
         }
       
       }
    }


}





void spdmm(
    const v_datatype * dense,  // dense matrix
    const v_indextype * CSRindptr,   // sparse martix is stored in CSR format, column value
    const v_indexandvalue * CSRindexandvalue,      // row value
	  v_datatype * output,
    int k,   // partition size k
    int nnz,
    int featurePn // size of the sparse matrix
)
{
#pragma HLS aggregate variable = dense
#pragma HLS aggregate variable = CSRindptr
#pragma HLS aggregate variable = CSRindexandvalue
#pragma HLS aggregate variable = output


#pragma HLS INTERFACE m_axi port = dense offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = CSRindptr offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = CSRindexandvalue offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port=dense bundle=control
#pragma HLS INTERFACE s_axilite port=CSRindptr bundle=control
#pragma HLS INTERFACE s_axilite port=CSRindexandvalue bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control

#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=nnz bundle=control
#pragma HLS INTERFACE s_axilite port=featurePn bundle=control

#pragma HLS dataflow

  static hls::stream<int> additionNum1;
  #pragma HLS resource variable=additionNum1 core=RAM_2P_LUTRAM 
  #pragma HLS STREAM variable=additionNum1 depth=8
  
  
  static hls::stream<int> additionNum2;
  #pragma HLS resource variable=additionNum2 core=RAM_2P_LUTRAM 
  #pragma HLS STREAM variable=additionNum1 depth=8
  
  
  static hls::stream<index_type> streamCSRindex;
  #pragma HLS resource variable=streamCSRindex core=RAM_2P_LUTRAM 
  #pragma HLS STREAM variable=streamCSRindex depth=8
  

  static hls::stream<data_type> streamCSRvalue;
  #pragma HLS resource variable=streamCSRvalue core=RAM_2P_LUTRAM 
  #pragma HLS STREAM variable=streamCSRvalue depth=8
  
  
  static hls::stream<v_datatype> streamdense;
  #pragma HLS resource variable=streamdense core=RAM_2P_URAM
  #pragma HLS STREAM variable=streamdense depth=8
  
  
  static hls::stream<v_datatype> streamResult;
  #pragma HLS resource variable=streamResult core=RAM_2P_URAM
  #pragma HLS STREAM variable=streamResult depth=8
  
  
  readindptr(CSRindptr, additionNum1, additionNum2, k);
  readindexandvalue(CSRindexandvalue, streamCSRindex, streamCSRvalue, nnz);
  readdense(dense, streamCSRindex, streamdense, featurePn, nnz);
  spdmm_compute( streamdense, additionNum1, streamCSRvalue,  streamResult, featurePn,   nnz, k);
  readResult( streamResult, additionNum2,  output,k, featurePn);
  

}
}