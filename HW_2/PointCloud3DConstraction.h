#include <string> 
#include <opencv2/opencv.hpp>

void Disparity2PointCloud(
    const std::string& output_file,
    const std::string& output_file_w_norm,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length , std::ofstream& logFile );