# Bilateral-Filter-CUDA
C++/CUDA code to perform bilateral filtering on OpenCV Mat inputs

# Requirements
CUDA

OpenCV 3+

# Code
Instead of using global memory, use CUDA's texture memory to access the input image pixels. The main and kernel code have provisions
to check run time of OpenCV's CPU implementation vs my GPU implementation
