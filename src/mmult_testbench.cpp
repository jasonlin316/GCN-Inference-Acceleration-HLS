#include <hls_stream.h>
#include "ap_int.h"
#include "data_type.hpp"

// ********************************************************
int main() {
	// input, output and golden output arrays
   int kernel_input[SIZE], kernel_output[SIZE], golden_output[SIZE];
   int retval = 0, i, j, z;
   FILE *fp;

   fp=fopen("./in.dat","r");
   //Read 256 entries from the input File
   for (i=0; i<SIZE; i++){
      int tmp;
      fscanf(fp, "%d", &tmp);
      kernel_input[i] = tmp;
      kernel_output[i] = 0;
   }
   fclose(fp);

   //Call the kernel function 
	fir(kernel_input, kernel_output);

	//Check 256 outputs
	fp=fopen("./out.golden.dat","r");
	for (i=0;i<SIZE;i++) {
		int tmp;
		fscanf(fp, "%d", &tmp);
		golden_output[i] = tmp;
	}
	fclose(fp);

	// Compare the results file with the golden results
//	retval = system("diff --brief -w ./out.dat ./out.golden.dat");
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
