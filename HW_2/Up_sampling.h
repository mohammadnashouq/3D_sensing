#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "Filters.h"

cv::Mat Joint_BilateralUpsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth);
	
cv::Mat IterativeUpsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth);

