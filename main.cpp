#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void bilateralFilter(const Mat & input, Mat & output, int r,double sI, double sS);

int main() {

	Mat input = imread("image-13.jpg", IMREAD_GRAYSCALE);
	// Pad image
	//Mat input_pd;
	//copyMakeBorder(input, input_pd, r, r, r, r, BORDER_REPLICATE);
	// Create output Mat
	Mat output_own(input.rows, input.cols, CV_8UC1);
	Mat output_cv;
	// Own bilateral filter (input,output,filter_half_size,sigmaI,sigmaS)
	bilateralFilter(input, output_own, 4, 75.0, 75.0);
	// OpenCV bilateral filter
	clock_t start_s = clock();
	cv::bilateralFilter(input, output_cv, 9, 75, 75);
	clock_t stop_s = clock();
	cout << "Time for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << endl;
	// Display own bf image
	imshow("Image1", output_own);
	// Display opencv bf image
	imshow("Image2", output_cv);
	cv::waitKey();
}