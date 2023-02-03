#pragma once



using namespace std;
using namespace cv;


void WritePLY(const char*, vector<Point3f>, vector<Point3f>);


void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, const double& scale);

void quality_matrices(cv::Mat original_image, cv::Mat output_gaussian, cv::Mat output_bilateral, cv::Mat output_median) ;
void StereoEstimation_Naive_1(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities);

void StereoEstimation_DP(const int& window_size,
    int height, int width, const int lambda, cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities);

void StereoEstimation_Dynamic(
  const int& window_size,
  int height,
  int width,
  int weight,
  cv::Mat& image1, cv::Mat& image2,
  cv::Mat& dynamic_disparities, const double& scale);

void StereoEstimation_Dynamic_new(
    const int& window_size,
    int height,
    int width,
    int weight,
    cv::Mat& image1, cv::Mat& image2,
    cv::Mat& dynamic_disparities, const double& scale);
void stereoMatchingDP(const Mat& left, const Mat& right, Mat& disparity);
    
int DisparitySpaceImage(
  cv::Mat& image1, cv::Mat& image2,
  int half_window_size,
  int r, int x, int y);

void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length, std::vector<cv::Point3f>& word_3d);
	