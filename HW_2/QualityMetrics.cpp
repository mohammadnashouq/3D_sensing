#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


double SSD(const cv::Mat& img1, const cv::Mat& img2)
{
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	return ssd;
}

double RMSE(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return sqrt(mse);
}

double MSE(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return mse;
}

double PSNR(const cv::Mat& img1, const cv::Mat& img2)
{

	double max = 255;
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	double psnr = 10 * log10((max * max) / mse);
	return psnr;
}

long double mean(const cv::Mat& img)
{
	long double sum = 0;
	int size = img.rows * img.cols;
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols; ++c) {
			sum += img.at<uchar>(r, c);
		}
	}
	return sum / size;

}

long double variance(const cv::Mat& img)
{
	cv::Mat var_matrix = img.clone();
	long double sum = 0;
	int size = var_matrix.rows * var_matrix.cols;
	long double mean_ = mean(var_matrix);

	for (int r = 0; r < var_matrix.rows; ++r) {
		for (int c = 0; c < var_matrix.cols; ++c) {
			var_matrix.at<uchar>(r, c) -= mean_;
			var_matrix.at<uchar>(r, c) *= var_matrix.at<uchar>(r, c);
		}
	}

	for (int r = 0; r < var_matrix.rows; ++r) {
		for (int c = 0; c < var_matrix.cols; ++c) {
			sum += var_matrix.at<uchar>(r, c);
		}
	}
	return sum / size;
}

double covariance(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	long double sum = 0;
	long double mean1 = mean(img1);
	long double mean2 = mean(img2);
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			sum = sum + ((img1.at<uchar>(r, c) - mean1) * (img2.at<uchar>(r, c) - mean2));
		}
	}
	return sum / size;
}

long double SSIM(const cv::Mat& img1, const cv::Mat& img2)
{
	long double ssim = 0;
	long double k1 = 0.01, k2 = 0.03, L = 255;
	long double C1 = (k1 * L) * (k1 * L);
	long double C2 = (k2 * L) * (k2 * L);

	long double mu_x = mean(img1);
	long double mu_y = mean(img2);
	long double variance_x = variance(img1);
	long double variance_y = variance(img2);
	long double covariance_xy = covariance(img1, img2);

	ssim = ((2 * mu_x * mu_y + C1) * (2 * covariance_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (variance_x * variance_x + variance_y * variance_y + C2));
	return ssim;
}