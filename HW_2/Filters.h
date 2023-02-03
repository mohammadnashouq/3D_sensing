#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat CreateGaussianKernel_s(float Spacial_sigma);

cv::Mat CreateGaussianKernel_w_s(int window_size, float sigma = 1);

cv::Mat CreateGaussianKernel_w(int window_size);



void OurFiler_Box(const cv::Mat& input, cv::Mat& output, const int window_size = 5);

void OurFiler_Gaussian(const cv::Mat& input, cv::Mat& output, const int window_size = 5);

void OurFilter_Bilateral(const cv::Mat& input, cv::Mat& output, float Spectral_Segma , float Spatial_Segma ,  const int window_size = 5);





void Joint_Bilateral(const cv::Mat& input_rgb, const cv::Mat& input_depth, cv::Mat& output, const int window_size = 5, float sigma = 5);
