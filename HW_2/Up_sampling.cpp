#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "Filters.h"

cv::Mat IterativeUpsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth) {
	// applying the joint bilateral filter to upsample a depth image, guided by an RGB image -- iterative upsampling
	int uf = log2(input_rgb.rows / input_depth.rows); // upsample factor
	cv::Mat D = input_depth.clone(); // lowres depth image
	cv::Mat I = input_rgb.clone(); // highres rgb image
	for (int i = 0; i < uf; ++i)
	{
		cv::resize(D, D, D.size() * 2); // doubling the size of the depth image
		Mat image_lowres;
		
		cv::resize(I, image_lowres, D.size());		// resizing the rgb image to depth image size
		Joint_Bilateral(image_lowres, D, D, 5, 0.1); // applying the joint bilateral filter with changed size depth and rbg images
	}
	cv::resize(D, D, input_rgb.size()); // in the end resizing the depth image to rgb image size
	Joint_Bilateral(input_rgb, D, D, 5, 0.1); // applying the joint bilateral filter with full res. size images
	return D;
}


cv::Mat Joint_BilateralUpsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth) {
	// applying the joint bilateral filter to upsample a depth image, guided by an RGB image -- iterative upsampling
	int uf = log2(input_rgb.rows / input_depth.rows); // upsample factor
	cv::Mat D = input_depth.clone(); // lowres depth image
	cv::Mat I = input_rgb.clone(); // highres rgb image

	cv::resize(D, D, D.size() * 2); // doubling the size of the depth image
	cv::resize(I, I, D.size());		// resizing the rgb image to depth image size
	Joint_Bilateral(I, D, D, 5, 0.1); // applying the joint bilateral filter with changed size depth and rbg images


	
	return D;
}