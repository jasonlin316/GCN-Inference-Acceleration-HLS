#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "ap_int.h"
#include "data_type.hpp"

#define VDATA_SIZE 16
#define vSize 16
#define LAYER 8 //WEIGHT_SIZE/MAX_SIZE
//#define DEBUG
using namespace std;
//TRIPCOUNT indentifier
const unsigned int c_dt_size = VDATA_SIZE;

struct v_dt {
    float data[VDATA_SIZE];
};

//Maximum Array Size
#define MAX_SIZE 16
#define WEIGHT_SIZE 128
//TRIPCOUNT identifier



const unsigned int c_size = MAX_SIZE;
extern "C"{

void read_block_matrix(const v_dt *AX, hls::stream<v_dt> &AX_stream,   int block_row_num, int block_col_num){


      float localAXblock[16*8*MAX_SIZE][MAX_SIZE];
#pragma HLS resource variable=localAXblock core=RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable=localAXblock dim=2 complete

      for (int i = 0 ; i < block_row_num; i = i + 16){

            for (int j = 0; j < 16; j++){
                     for (int  k = 0; k < 16; k = k + 1){
                         for (int  kk = 0; kk < block_col_num; kk = kk + 1){
#pragma HLS PIPELINE II=1 rewind
                             v_dt tmpIn;
#pragma HLS aggregate variable = tmpIn
                             int index =  ((i + j ) * 16 + k ) * block_col_num + kk;
                             tmpIn = AX[index];
                             for (int v = 0; v < MAX_SIZE; v = v + 1)
                             {
                                 if (i + j < block_row_num) {localAXblock[j*128 + kk*16 + k][v] = tmpIn.data[v];}  
                                 else {localAXblock[j*128 + kk*16 + k][v] = 0;}
                             }
                         }
                     }                        
            }
            
            // send the data block
            for (int kk = 0; kk < block_col_num; kk = kk + 1){
                  for (int j = 0; j < 16; j++){
                        for (int k = 0; k < 16; k = k + 1){
#pragma HLS PIPELINE II=1 rewind
                            v_dt tmpIn;
#pragma HLS aggregate variable = tmpIn
                            for (int v = 0; v < MAX_SIZE; v = v + 1)
                            {
                                 tmpIn.data[v] = localAXblock[j*128 + kk*16 + k][v];    
                            }
                            
                            AX_stream << tmpIn;
                        }
                  }
            } 
            
            // send the block to compute unit
                 
      }
}





void compute(hls::stream<v_dt> &AX_stream, v_dt * c, const float localW[MAX_SIZE][MAX_SIZE][8][8], int block_row_num, int block_col_num, int weight_col, bool ReLU){

        float localAX[MAX_SIZE][MAX_SIZE][16];
#pragma HLS resource variable=localAX core=RAM_2P_LUTRAM
#pragma HLS ARRAY_PARTITION variable=localAX dim=1 complete
#pragma HLS ARRAY_PARTITION variable=localAX dim=2 complete  
        
        float localC[MAX_SIZE][MAX_SIZE][16][8];
#pragma HLS resource variable=localC core=RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable=localC dim=1 complete
#pragma HLS ARRAY_PARTITION variable=localC dim=2 complete
         
        for (int i = 0 ; i < block_row_num; i = i + 16){      
                

                
                for (int  kk = 0; kk < block_col_num; kk = kk + 1){
                               
                      // load localAX
                        for (int j = 0; j < 16; j = j + 1){
                        for (int  k = 0; k < 16; k = k + 1){
#pragma HLS PIPELINE II=1 rewind
                            v_dt tmpIn;
#pragma HLS aggregate variable = tmpIn
                            tmpIn = AX_stream.read();
                            
                            for (int v = 0; v < MAX_SIZE; v = v + 1)
                            {
                                 localAX[k][v][j] = tmpIn.data[v];    
                            }        
                            
                        
                        }
                      }
                        
                        
                        
                        systolic0: for(int  axCol = 0; axCol < vSize; axCol++)
                        {
                              systolic1: for(int  ww = 0; ww < weight_col; ww++) //trasverse by weight col
                              {
                              
#pragma HLS dependence variable=localC inter false
                                    systolic1inner: for (int  j = 0; j < 16; j++){
#pragma HLS PIPELINE II=1 rewind
                                      systolic2: for(int  ii = 0; ii < MAX_SIZE; ii++) 
                                      {
                                              systolic3: for(int  jj = 0; jj < MAX_SIZE; jj++) 
                                              {
                                                        float last;
                                                        if (axCol==0 && kk ==0) {last = 0;}
                                                        else {last = localC[ii][jj][j][ww];}
                                
                                                        //float result =  last + localAX[ii][axCol] * localW[axCol+col*VDATA_SIZE][jj + ww*VDATA_SIZE];
                                                        float result =  last + localAX[ii][axCol][j] * localW[axCol][jj][kk][ww];
                                                        
                                                        localC[ii][jj][j][ww] = result; //result; // Write back results                                              
                                              }
                                              
                                      }
                                    }
                                      
                                }
                                
                         }                                                                                          
                }
                
                
                
                // store localc
                for (int j = 0; j < 16 ; j = j + 1){
                      if (i + j < block_row_num) {
                            for (int k = 0; k < 16; k = k + 1){
                                for (int kk = 0; kk < weight_col; kk = kk + 1){
#pragma HLS PIPELINE II=1 rewind
                                   v_dt tmpIn;
#pragma HLS aggregate variable = tmpIn
                                    for (int v = 0 ; v < 16; v++){
                                        data_type tmpC;
                                        tmpC = localC[k][v][j][kk];
                                        if (tmpC < 0 && ReLU) tmpIn.data[v] = 0;
                                        else tmpIn.data[v] = tmpC;
                                        //tmpIn.data[v] = localC[k][v][j][kk];
                                    }
                                    int index = ((i + j)* 16 + k )*weight_col + kk;
                                    c[index] = tmpIn;                                      
                                }
                            
                            }
                      }
                } 
                   
        }
}


void loadweight(const v_dt *W, float localW[MAX_SIZE][MAX_SIZE][8][8], int block_col_num, int weight_col)
{

  for (int i = 0; i < block_col_num; i= i + 1){
            for (int j = 0; j < 16; j = j + 1){
                for (int k = 0; k < weight_col; k = k + 1){
#pragma HLS PIPELINE II=1 rewind
                    v_dt tmpIn;
#pragma HLS aggregate variable = tmpIn
                    int index = (i * 16 + j) * weight_col + k;
                    tmpIn = W[index];
                    for (int kk = 0; kk < 16; kk = kk + 1){
                        localW[j][kk][i][k] = tmpIn.data[kk];
                    }               
                }
            
            }
        }
}


  void mmult(
                const v_dt *AX,   // Read-Only Matrix AX
                const v_dt *W,   // Read-Only Matrix W
                v_dt *c,         // Output Result
                int block_row_num,        //amount of block row
                int block_col_num,        //amount of block col
                int weight_col,      //amount of weight col
                bool ReLU
            )
    {
#pragma HLS aggregate variable = AX
#pragma HLS aggregate variable = W
#pragma HLS aggregate variable = c
        
        
#pragma HLS dataflow  
        #pragma HLS INTERFACE m_axi port = AX offset = slave bundle = gmem0
        #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmem1
        
        #pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem0

        #pragma HLS INTERFACE s_axilite port = AX bundle = control
        #pragma HLS INTERFACE s_axilite port = W bundle = control
        #pragma HLS INTERFACE s_axilite port = c bundle = control
        #pragma HLS INTERFACE s_axilite port = block_row_num bundle = control
        #pragma HLS INTERFACE s_axilite port = block_col_num bundle = control
        #pragma HLS INTERFACE s_axilite port = weight_col bundle = control

        
        
        static hls::stream<v_dt> AX_stream;
#pragma HLS STREAM variable=AX_stream depth=32
        

         
        
        
        
        float localW[MAX_SIZE][MAX_SIZE][8][8];
#pragma HLS resource variable=localW core=RAM_2P_URAM 
#pragma HLS ARRAY_PARTITION variable=localW dim=2 complete
               
        
        // read GCN weight
        loadweight(W, localW, block_col_num, weight_col);

        read_block_matrix(AX, AX_stream,  block_row_num, block_col_num);
        compute(AX_stream, c, localW, block_row_num, block_col_num, weight_col, ReLU);
        
        
        
        

}
}