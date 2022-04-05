# GCN Inference Acceleration using High-Level Synthesis

Source code for ["GCN Inference Acceleration using High-Level Synthesis"](https://ieeexplore.ieee.org/document/9622801), HPEC 2021.
Details of hardware design can be found in the paper.

### End-to-end GCN inference using HLS programming model
![end-to-end](https://github.com/jasonlin316/GCN-Inference-Acceleration-HLS/blob/main/pic/e2e.png)

File structure:
```
GCN Inference Acceleration HLS/
│   README.md
│
└───/data #input data stored in CSR format and a data generator
│   │   indptr.bin
│   │   indices.bin
│   │   data_generator.py # a python script to generate input matrices based on the size you specified
│   │   ...
└───/run #files and scripts for compilation and execution
│   │   makefile
│   │   design.cfg
│   │   ...
└───/src #kernel designs and some supportive functions
│   │   spdmm.cpp
│   │   mmult.cpp
│   │   ...
└─
```

Citation
```
@INPROCEEDINGS{9622801,
  author={Lin, Yi Chien and Zhang, Bingyi and Prasanna, Viktor},
  booktitle={2021 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={GCN Inference Acceleration using High-Level Synthesis}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/HPEC49654.2021.9622801}}

```
