# I/O specification of GCN APIs

## User specified inputs and meta-data

### Meta-data

- Number of layers: in this design I explicit unroll everything into two layers, but if we want to keep it flexible we will need this variable.
- row_size: number of rows in the adjacency matrix (which is amount of vertices in the graph)
- nnz_size: number of non-zero elements in the adjacency matrix (which is the amount of edges in the graph)
- layer_dim: dimension for each layer. In the current host code, I used "hidden_dim" for the first layer and "output_dim" for the second layer currently. We might need to replaced it with array ```layer_dim[Number of layers]```  for generality.
- file path: file path to read data.

line 235-247 and 455-467: this line is for specifying memory bank on FPGA. We can place our array on different DDR_banks to get higher performance, but I am not sure how to wrapped this neatly.

```c
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
```

## Layer dependent variables and functions

line 64-67: each layer will has its own pre-trained weights.

Maybe consider declaring this in a 2D vector to support any layers

```c
std::vector<data_type,aligned_allocator<ap_int<512>>>     weight_matrix;
std::vector<data_type,aligned_allocator<ap_int<512>>>     weight_matrix2;
std::vector<data_type,aligned_allocator<ap_int<512>>>     padded_weight_matrix;
std::vector<data_type,aligned_allocator<ap_int<512>>>     padded_weight_matrix2;

//if our GCN models has three layers, we will need below:
std::vector<data_type,aligned_allocator<ap_int<512>>>     weight_matrix3;
std::vector<data_type,aligned_allocator<ap_int<512>>>     padded_weight_matrix3;
```

line 101-107: padding the weight matrices, the below code pad weight1 and weight2.

```c
int hiddenPn = hidden_dim;
if(dense_mtx_col%VDATA_SIZE!=0 || hidden_dim%VDATA_SIZE!=0) 
weight_padding(weight_matrix,padded_weight_matrix, dense_mtx_col,hidden_dim, hiddenPn);

int outputPn = output_dim;
if(hiddenPn%VDATA_SIZE!=0 || output_dim%VDATA_SIZE!=0) 
weight_padding(weight_matrix2,padded_weight_matrix2, hiddenPn, output_dim, outputPn);
```

line 169-172: this is for storing the intermediate and final results. 

Recall the slide for GCN, for the first layer we have A*X = AX, and then AX*W = H (hidden layer)

The output H of first layer will serve as input for next layer, so we have A*H = AH and AH*W.

We will need 2 vector for each layer (AHx and AHWx where "x" is the layer number)

```c
std::vector<data_type,aligned_allocator<ap_int<512>>>     AX_results [number_of_partition];
std::vector<data_type,aligned_allocator<ap_int<512>>>     AXW_results       [number_of_partition];
std::vector<data_type,aligned_allocator<ap_int<512>>>     AH_results        [number_of_partition];
std::vector<data_type,aligned_allocator<ap_int<512>>>     AHW_results       [number_of_partition];
```

The above can also be used as input/output for our APIs. For example:

```c
SpDMM(A,X,AX_results, ReLU = true);
mmult(AX_results, weight1, AXW_results);
SpDMM(A,AXW_results,AH_results, ReLU = False);
mmult(AH_results, weight2, AHW_results);
```

line 174-177: declaring the size of these vectors, basically = partition_size*layer_dimension

```c
for(int i=0; i < number_of_partition; i++) AX_results[i].resize(partition_size * featurePn);
for(int i=0; i < number_of_partition; i++) AXW_results[i].resize(partition_size*hiddenPn);
for(int i=0; i < number_of_partition; i++) AH_results[i].resize(partition_size*hiddenPn);
for(int i=0; i < number_of_partition; i++) AHW_results[i].resize(partition_size*outputPn);
```

NOTE: although it looks like layer dependent variables, line 210-217 can be reused in different layers. We just need to declare the correct size for different layer. For example, line 263 and line 483 declare different size for the weight buffer in different layer.

line 434-446: concatenation of multiple arrays, I am actually wondering if there is a way to avoid memory copying for this function. Anyway, this is used between each layer. Current design has only 2 layer so we only call this function once.

```c
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
```