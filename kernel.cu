// Reference - http://ecee.colorado.edu/~siewerts/extra/code/example_code_archive/a490dmis_code/CUDA/cuda_work/samples/3_Imaging/bilateralFilter/bilateral_kernel.cu
#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>

#define M_PI           3.14159265358979323846
#define TILE_X 16
#define TILE_Y 16


using namespace std;
using namespace cv;
// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
__constant__ float cGaussian[64];
// Initialize texture memory to store the input
texture<unsigned char, 2, cudaReadModeElementType> inTexture;

/* 
   GAUSSIAN IN 1D FOR SPATIAL DIFFERENCE

   Here, exp(-[(x_centre - x_curr)^2 + (y_centre - y_curr)^2]/(2*sigma*sigma)) can be broken down into ...
   exp[-(x_centre - x_curr)^2 / (2*sigma*sigma)] * exp[-(y_centre - y_curr)^2 / (2*sigma*sigma)] 
   i.e, 2D gaussian -> product of two 1D Gaussian

   A constant Gaussian 1D array can be initialzed to store the gaussian values
   Eg: For a kernel size 5, the pixel difference array will be ...
   [-2, -1, 0, 1 , 2] for which the gaussian kernel is applied

*/
void updateGaussian(int r,double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*r + 1));
}

// Gaussian function for range difference
__device__ inline double gaussian(float x, double sigma)
{
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2))) ;
}

// Bilateral filter kernel
__global__ void gpuCalculation(unsigned char* input, unsigned char* output,
	int width, int height,
	int r, double sI, double sS)
{
	// Initialize global Tile indices along x,y and xy
	int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
	int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

	// If within image size
	if ((txIndex < width) && (tyIndex < height))
	{
		double iFiltered = 0;
		double wP = 0;
		// Get the centre pixel value
		unsigned char centrePx = tex2D(inTexture, txIndex, tyIndex);
		// Iterate through filter size from centre pixel
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				// Get the current pixe; value
				unsigned char currPx = tex2D(inTexture, txIndex + dx, tyIndex + dy);
				// Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sI);
				iFiltered += w * currPx;
				wP += w;				
			}
		}
		output[tyIndex*width + txIndex] = iFiltered / wP;
	}
}

void bilateralFilter(const Mat & input, Mat & output, int r, double sI, double sS)
{
	// Events to calculate gpu run time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Size of image
	int gray_size = input.step*input.rows;

	// Variables to allocate space for input and output GPU variables
	size_t pitch;                                                      // Avoids bank conflicts (Read documentation for further info)
	unsigned char *d_input = NULL;
	unsigned char *d_output;

	// Create gaussain 1d array
	updateGaussian(r,sS);

	//Allocate device memory
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char)*input.step, input.rows); // Find pitch
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char)*input.step, sizeof(unsigned char)*input.step, input.rows, cudaMemcpyHostToDevice); // create input padded with pitch
	cudaBindTexture2D(0, inTexture, d_input, input.step, input.rows, pitch); // bind the new padded input to texture memory
	cudaMalloc<unsigned char>(&d_output, gray_size); // output variable

	//Copy data from OpenCV input image to device memory
	//cudaMemcpy(d_input, input.ptr(), gray_size, cudaMemcpyHostToDevice);

	//Creating the block size
	dim3 block(TILE_X, TILE_Y);

	//Calculate grid size to cover the whole image
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0); // start timer
	// Kernel call
	gpuCalculation << <grid, block >> > (d_input, d_output, input.cols, input.rows, r, sI, sS);
	cudaEventRecord(stop, 0); // stop timer
	cudaEventSynchronize(stop);

	// Copy output from device to host
	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	// Free GPU variables
	cudaFree(d_input);
	cudaFree(d_output);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the GPU: %f ms\n", time);
}
