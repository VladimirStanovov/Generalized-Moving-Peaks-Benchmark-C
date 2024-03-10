# Generalized-Moving-Peaks-Benchmark-C

C++ implementation of the Generalized Moving Peaks Benchmark (GMPB)

Includes implementation of the mQSO algorithm (T. Blackwell and J. Branke, “Multiswarms, exclusion, and anticonvergence in dynamic environments,” IEEE Transactions on Evolutionary Computation, vol. 10, no. 4, pp. 459–472, 2006.)

This benchmark is a part of the CEC 2022 and 2024 competitions: 

https://danialyazdani.com/CEC-2022.php 

https://competition-hub.github.io/GMPB-Competition/

Reference:

D. Yazdani, M. N. Omidvar, R. Cheng, J. Branke, T. T. Nguyen, and X. Yao, “Benchmarking continuous dynamic optimization: Survey and generalized test suite,” IEEE Transactions on Cybernetics, vol. 52(5), pp. 3380-3393, 2020.

M. Peng, Z. She, D. Yazdani, D. Yazdani, W. Luo, C. Li, J. Branke, T. T. Nguyen, A. H. Gandomi, Y. Jin, and X. Yao, “Evolutionary dynamic optimization laboratory: A matlab optimization platform for education and experimentation in dynamic environments,” arXiv preprint arXiv:2308.12644, 2023.

Original Evolutionary Dynamic Optimization Laboratory (EDOLAB) repository: https://github.com/Danial-Yazdani/EDOLAB-MATLAB

# Compilation and usage

Compile the gnbg-c++.cpp file with any compiler (e.g. GCC):

g++ -std=c++11 -O3 gmpb.cpp -o gmpb

There are three operation modes:

0: Creates "res_e#.txt" files to visualize each environment with a python script. Also outputs current error.

1: Runs mQSO algorithm 31 times and saves current error, current performance, best error before change. The included jupyter notebook contains the code for graphs and statistics.
