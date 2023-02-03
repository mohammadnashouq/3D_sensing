#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>


cv::Mat CreateGaussianKernel_s(float Spacial_sigma) {
  

   

    const double k = 2.5;
    double r_max = Spacial_sigma * k;

    int window_size = (int)std::ceil(std::sqrt((r_max * r_max / 2)));
    if (window_size % 2 == 0) {
        window_size = window_size + 1;
    }

    int half_window_size = window_size / 2;

    cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);
    // see: lecture_03_slides.pdf, Slide 13
   
   // const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
   // const double sigma = r_max / k;




    // sum is for normalization 
    float sum = 0.0;

    for (int x = -window_size / 2; x <= window_size / 2; x++) {
        for (int y = -window_size / 2; y <= window_size / 2; y++) {
            float val = exp(-(x * x + y * y) / (2 * Spacial_sigma * Spacial_sigma));
            kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
            sum += val;
        }
    }

    // normalising the Kernel 
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            kernel.at<float>(i, j) /= sum;

    // note that this is a naive implementation
    // there are alternative (better) ways
    // e.g. 
    // - perform analytic normalisation (what's the integral of the gaussian? :))
    // - you could store and compute values as uchar directly in stead of float
    // - computing it as a separable kernel [ exp(x + y) = exp(x) * exp(y) ] ...
    // - ...

    return kernel;
}


cv::Mat CreateGaussianKernel_w_s(int window_size, float sigma = 1) // 0.1 ... 3
{
    cv::Mat kernel(window_size, window_size, CV_32FC1);
    double sum = 0.0;
    double i, j;
    for (i = 0; i < window_size; i++) {
        for (j = 0; j < window_size; j++) {
            kernel.at<float>(i, j) = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel.at<float>(i, j);
        }
    }
    for (i = 0; i < window_size; i++) {
        for (j = 0; j < window_size; j++) {
            kernel.at<float>(i, j) /= sum;
        }
    }
    return kernel;
}


cv::Mat CreateGaussianKernel_w(int window_size) {
    cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);

    int half_window_size = window_size / 2;

    // see: lecture_03_slides.pdf, Slide 13
    const double k = 2.5;
    const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
    const double sigma = r_max / k;

    // sum is for normalization 
    float sum = 0.0;

    for (int x = -window_size / 2; x <= window_size / 2; x++) {
        for (int y = -window_size / 2; y <= window_size / 2; y++) {
            float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
            sum += val;
        }
    }

    // normalising the Kernel 
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            kernel.at<float>(i, j) /= sum;

    // note that this is a naive implementation
    // there are alternative (better) ways
    // e.g. 
    // - perform analytic normalisation (what's the integral of the gaussian? :))
    // - you could store and compute values as uchar directly in stead of float
    // - computing it as a separable kernel [ exp(x + y) = exp(x) * exp(y) ] ...
    // - ...

    return kernel;
}

void OurFiler_Box(const cv::Mat& input, cv::Mat& output, const int window_size = 5) {

    const auto width = input.cols;
    const auto height = input.rows;

    // TEMPORARY CODE
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            output.at<uchar>(r, c) = 0;
        }
    }

    for (int r = window_size / 2; r < height - window_size / 2; ++r) {
        for (int c = window_size / 2; c < width - window_size / 2; ++c) {

            // box filter
            int sum = 0;
            for (int i = -window_size / 2; i <= window_size / 2; ++i) {
                for (int j = -window_size / 2; j <= window_size / 2; ++j) {
                    sum += input.at<uchar>(r + i, c + j);
                }
            }
            output.at<uchar>(r, c) = sum / (window_size * window_size);

        }
    }
}

void OurFiler_Gaussian(const cv::Mat& input, cv::Mat& output, const int window_size = 5) {

    const auto width = input.cols;
    const auto height = input.rows;

    cv::Mat gaussianKernel = CreateGaussianKernel_w(window_size);

    // TEMPORARY CODE
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            output.at<uchar>(r, c) = 0;
        }
    }

    for (int r = window_size / 2; r < height - window_size / 2; ++r) {
        for (int c = window_size / 2; c < width - window_size / 2; ++c) {

            int sum = 0;
            for (int i = -window_size / 2; i <= window_size / 2; ++i) {
                for (int j = -window_size / 2; j <= window_size / 2; ++j) {
                    sum
                        += input.at<uchar>(r + i, c + j)
                        * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);
                }
            }
            output.at<uchar>(r, c) = sum;

        }
    }
}

void OurFilter_Bilateral(const cv::Mat& input, cv::Mat& output, float Spectral_Segma, float Spatial_Segma, const int window_size = 5) {
    const auto width = input.cols;
    const auto height = input.rows;

    cv::Mat gaussianKernel = CreateGaussianKernel_s(Spatial_Segma);

    // TEMPORARY CODE
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            output.at<uchar>(r, c) = 0;
        }
    }

    auto d = [](float a, float b) {
        return std::abs(a - b);
    };

    auto p = [](float val , float sigma) {
       // const float sigma = 5;
        const float sigmaSq = sigma * sigma;
        const float normalization = std::sqrt(2 * M_PI) * sigma;
        return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
    };
#pragma omp parallel for 
    for (int r = window_size / 2; r < height - window_size / 2; ++r) {
#pragma omp critical   
        for (int c = window_size / 2; c < width - window_size / 2; ++c) {

            float sum_w = 0;
            float sum = 0;

            for (int i = -window_size / 2; i <= window_size / 2; ++i) {
                for (int j = -window_size / 2; j <= window_size / 2; ++j) {

                    float range_difference
                        = d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

                    float w
                        = p(range_difference, Spectral_Segma)
                        * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

                    sum
                        += input.at<uchar>(r + i, c + j) * w;
                    sum_w
                        += w;
                }
            }

            output.at<uchar>(r, c) = sum / sum_w;

        }
    }
}



void Joint_Bilateral(const cv::Mat& input_rgb, const cv::Mat& input_depth, cv::Mat& output, const int window_size = 5, float sigma = 5) {
    // converting the bilateral filter to Guided Joint bilateral filter for guided image upsampling 
    // upsampling a low resolution depth image, guided by an RGB image
    // weights formed using colors (image input_rgb), filtering happens by modifying depth (image input_depth)
    const auto width = input_rgb.cols;
    const auto height = input_rgb.rows;

    cv::Mat gaussianKernel = CreateGaussianKernel_w_s(window_size, 0.5); // sigma for the spatial filter (Gaussian, \(w_G\) kernel)
    auto d = [](float a, float b) {
        return std::abs(a - b);
    };

    auto p = [](float val, float sigma) {	// use of weighting function p : dissimilar pixels get lower weights, preserves strong edges, smooths other regions
        const float sigmaSq = sigma * sigma;
        const float normalization = std::sqrt(2 * M_PI) * sigma;
        return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
    };

#pragma omp parallel for 
    for (int r = window_size / 2; r < height - window_size / 2; ++r)
#pragma omp critical   
    {

        for (int c = window_size / 2; c < width - window_size / 2; ++c) {

            float sum_w = 0;
            float sum = 0;

            for (int i = -window_size / 2; i <= window_size / 2; ++i) {
                for (int j = -window_size / 2; j <= window_size / 2; ++j) {

                    float range_difference
                        = d(input_rgb.at<uchar>(r, c), input_rgb.at<uchar>(r + i, c + j)); // using the rgb image with the spectral filter

                    float w
                        = p(range_difference, sigma) // sigma for the spectral filter (\(f\) in the slides
                        * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

                    sum
                        += input_depth.at<uchar>(r + i, c + j) * w; // using the depth image with the spatial filter
                    sum_w
                        += w;
                }
            }

            output.at<uchar>(r, c) = sum / sum_w;

        }
    }
}


