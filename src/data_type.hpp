#include <stdio.h>
#include <ap_int.h>
#include <ostream>

#define VDATA_SIZE 16
#define DENSE_BUFFER_LENGTH 4096
#define INDEX_BUFFER_LENGTH 16
#define OUTPUTBUFFER_LENGTH 32

#define datawidth 512

typedef float data_type;
typedef int index_type;
typedef ap_int<512> wid_data_type;
typedef union{unsigned int intdata; float floatdata;} convertion;

const int indexbufferlength = INDEX_BUFFER_LENGTH * VDATA_SIZE;

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