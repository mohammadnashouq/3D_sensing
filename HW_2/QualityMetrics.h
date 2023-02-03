#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


double SSD(const cv::Mat& img1, const cv::Mat& img2);


double RMSE(const cv::Mat& img1, const cv::Mat& img2);


double MSE(const cv::Mat& img1, const cv::Mat& img2);


double PSNR(const cv::Mat& img1, const cv::Mat& img2);


long double mean(const cv::Mat& img);


long double variance(const cv::Mat& img);


double covariance(const cv::Mat& img1, const cv::Mat& img2);


long double SSIM(const cv::Mat& img1, const cv::Mat& img2);
