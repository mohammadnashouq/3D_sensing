#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "main.h"
#include "ForwardPassing.h"
#include "BackwardPassing.h"

using namespace std::chrono;
using namespace std;

int main(int argc, char** argv) {

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 160;

  // stereo estimation parameters
  const int dmin = 200;
  const double scale = 3;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 8) {
    std::cerr << "Usage: " << argv[0] <<
    " IMAGE1 IMAGE2 OUTPUT_FILE WINDOW_SIZE_NAIVE WINDOW_SIZE_DYNAMIC WEIGHT ground_truth" << std::endl;
    return 1;
  }

  cv::Mat ground_truth = cv::imread(argv[7], cv::IMREAD_GRAYSCALE);

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  const std::string output_file = argv[3];
  const int window_size_naive = atoi(argv[4]);
  const int window_size_dynamic = atoi(argv[5]);
  const int weight = atoi(argv[6]);

  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size_naive = " << window_size_naive << std::endl;
  std::cout << "window_size_dynamic = " << window_size_dynamic << std::endl;
  std::cout << "occlusion weights = " << weight << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "scaling of disparity images to show = " << scale << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  ///////////////////////////////
  // Reconstruction and output //
  ///////////////////////////////

  /// Naive approach

  // naive disparity image
  cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  
  // To get the value of duration use the count()
  // member function on the duration object
  // stereo estimation
  auto start = high_resolution_clock::now();
  
  StereoEstimation_Naive_1(
    window_size_naive, dmin, height, width,
    image1, image2,
    naive_disparities);


  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  cout << "naive_disparities execution time" << duration.count() << endl;
  
  std::vector<cv::Point3f> word_3d;
  // reconstruction
  
  //Disparity2PointCloud(
  //   output_file + "_naive",
  //   height, width, naive_disparities,
  //   window_size_naive, dmin, baseline, focal_length, word_3d);

  // save / display images
  std::stringstream out1;
  out1 << output_file << "_naive.png";
  cv::imwrite(out1.str(), naive_disparities);

  cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
  cv::imshow("Naive", naive_disparities);
  
  /// Dynamic programming approach

  // dynamic disparity image
  cv::Mat dynamic_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  // stereo estimation
  start = high_resolution_clock::now();

  //StereoEstimation_Dynamic_new(
   //   window_size_dynamic, height, width, weight,
    //  image1, image2,
     // dynamic_disparities, scale);


  StereoEstimation_DP(  
    window_size_dynamic, height, width, weight,
    image1, image2,
    dynamic_disparities);

  
  // stereoMatchingDP(image1, image2, dynamic_disparities);
  imshow("disparity", dynamic_disparities);
  waitKey();
  
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);


cout << "dynamic_disparities execution time" << duration.count() << endl;

  // reconstruction
  //Disparity2PointCloud(
//    output_file + "_dynamic",
//    height, width, dynamic_disparities,
//    window_size_dynamic, dmin, baseline, focal_length,word_3d);

  // save / display images
  std::stringstream out2;
  out2 << output_file << "_dynamic.png";

  cv::Mat dynamic_disparities_image;
  cv::normalize(dynamic_disparities, dynamic_disparities_image, 255, 0, cv::NORM_MINMAX);

  cv::imwrite(out2.str(), dynamic_disparities_image);

  cv::namedWindow("Dynamic", cv::WINDOW_AUTOSIZE);
  cv::imshow("Dynamic", dynamic_disparities_image);

  /// OpenCV implementation

  cv::Mat opencv_disparities;
  cv::Ptr<cv::StereoBM > match = cv::StereoBM::create(16, 9);


  start = high_resolution_clock::now();

  match->compute(image1, image2, opencv_disparities);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);


  cout << "opencv_disparities execution time" << duration.count() << endl;

  cv::imshow("OpenCV result",opencv_disparities*1000);

  std::stringstream out3;
  out3 << output_file << "_opencv.png";
  cv::imwrite(out3.str(), opencv_disparities);

  cv::waitKey(0);


  quality_matrices(ground_truth, naive_disparities, dynamic_disparities, dynamic_disparities);
  return 0;
}

void StereoEstimation_Naive_1(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
    int half_window_size = window_size / 2;
    int progress = 0;
#pragma omp parallel for 
    for (int i = half_window_size; i < height - half_window_size; ++i)
#pragma omp critical   
    {
        ++progress;
        std::cout
            << "Calculating disparities for the naive approach... "
            //<< std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::ceil(((progress) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::flush;

        for (int j = half_window_size; j < width - half_window_size; ++j) {
            int min_ssd = INT_MAX;
            int disparity = 0;
            int dd = 0;
            for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
                int ssd = 0;

                // TODO: sum up matching cost (ssd) in a window
                //std::cout << "width,length" << height << width << std::endl;

                for (int k = -half_window_size; k < +half_window_size; ++k) {
                    for (int m = -half_window_size; m < +half_window_size; ++m) {


                        float image1_val = image1.at<uchar>(i + k, j + m);
                        //std::cout << "i + k, j + m, d + m" << i + k << j + m << d + m << std::endl;
                        float image2_val = image2.at<uchar>(i + k, j + d + m);

                        float dif = image1_val - image2_val;

                        ssd += dif * dif;


                    }
                }

                if (ssd < min_ssd) {
                    min_ssd = ssd;
                    disparity = d;
                }
            }

            naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity);
        }
    }

    std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
    std::cout << std::endl;
}


void StereoEstimation_Dynamic(
        const int& window_size,
        int height,
        int width,
        int weight,
        cv::Mat& image1, cv::Mat& image2,
        cv::Mat& dynamic_disparities, const double& scale)
{
    int half_window_size = window_size / 2;

    for (int r = half_window_size; r < height - half_window_size; ++r) {

        std::cout
                << "Calculating disparities for the dynamic approach... "
                << std::ceil(((r - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
                << std::flush;

        cv::Mat C = cv::Mat::zeros(width, width, CV_32F);
        cv::Mat M = cv::Mat::zeros(width, width, CV_8UC1);

        for (int x = 1; x < width; ++x) {
            C.at<float>(x, 0) = x * weight;
            M.at<uchar>(x, 0) = 3;
        }

        for (int y = 1; y < width; ++y) {
            C.at<float>(0, y) = y * weight;
            M.at<uchar>(0, y) = 2;
        }
        double dd = 0;
        for (int x = 1; x < width; ++x) {
            for (int y = 1; y < width; ++y) {
                try {
                dd = DisparitySpaceImage(image1, image2, half_window_size, r, x, y);
                }
                catch (...) {
                   
                    cout << "------------------- errer 1 ----------";
                }

                double match_cost = C.at<float>(x-1, y-1) + dd;
                double left_occl_cost = C.at<float>(x-1, y) + weight;
                double right_occl_cost = C.at<float>(x, y-1) + weight;

                if (match_cost < std::min(left_occl_cost, right_occl_cost)) {
                    C.at<float>(x, y) = match_cost;
                    M.at<uchar>(x, y) = 1;
                }
                else if (left_occl_cost < std::min(match_cost, right_occl_cost)) {
                    C.at<float>(x, y) = left_occl_cost;
                    M.at<uchar>(x, y) = 2;
                }
                else { // (right_occl_cost < std::min(match_cost, left_occl_cost))
                    C.at<float>(x, y) = right_occl_cost;
                    M.at<uchar>(x, y) = 3;
                }

            }
        }

        int x = width - 1;
        int y = width - 1;
        int c = width;
        int d = 0;
        while (x != 0 && y != 0) {
            switch (M.at<uchar>(x, y)) {
                case 1:
                    d = abs(x - y);
                    x--;
                    y--;
                    c--;
                    break;
                case 2:
                    x--;
                    break;
                case 3:
                    d = 0;
                    y--;
                    break;
            }
            //cout << "r - half_window_size " << r - half_window_size<<endl;
            //cout << "c" << c << endl;
            if (c > width - 1) {
                c = width - 1;
            }
            if (r - half_window_size < 0) {
                dynamic_disparities.at<uchar>(0, c) = d;
            }
            else
            dynamic_disparities.at<uchar>(r - half_window_size, c) = d;
        }
    }

    std::cout << "Calculating disparities for the dynamic approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

int cost(int x1, int x2, int y, int d) {
    const int PENALTY = 10;

    return abs(x1 - x2) + PENALTY * abs(y - d);
}

void stereoMatchingDP(const Mat& left, const Mat& right, Mat& disparity) {

    const int MAX_DISPARITY = 128;
    
    int height = left.rows;
    int width = left.cols;
    disparity = Mat(height, width, CV_16S, Scalar(0));

    Mat dp(height, width * MAX_DISPARITY, CV_32S, Scalar(0));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int d = 0; d < MAX_DISPARITY; d++) {
                if (x - d < 0) continue;
                int x2 = x - d;
                int cost_value = cost(left.at<uchar>(y, x), right.at<uchar>(y, x2), y, d);
                int min_cost = cost_value;
                if (y > 0) {
                    min_cost = min(min_cost, dp.at<int>(y - 1, x * MAX_DISPARITY + d) + cost_value);
                }
                if (x > 0) {
                    min_cost = min(min_cost, dp.at<int>(y, (x - 1) * MAX_DISPARITY + d) + cost_value);
                }
                dp.at<int>(y, x * MAX_DISPARITY + d) = min_cost;
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int min_cost = INT_MAX;
            int best_d = 0;
            for (int d = 0; d < MAX_DISPARITY; d++) {
                int cost_value = dp.at<int>(y, x * MAX_DISPARITY + d);
                if (cost_value < min_cost) {
                    min_cost = cost_value;
                    best_d = d;
                }
            }
            disparity.at<short>(y, x) = best_d;
        }
    }
}

void StereoEstimation_Dynamic_new(
    const int& window_size,
    int height,
    int width,
    int weight,
    cv::Mat& image1, cv::Mat& image2,
    cv::Mat& dynamic_disparities, const double& scale)
{

    // forward passing
    ForwardPassing forwardPassing(image1, image2, weight);
    double*** forwardPassedMatched = forwardPassing.FordwardPass();

    // backward passing1
    BackwardPassing backwardPassing1(image1, image2);
    double** backwardPassed1 = backwardPassing1.BackwardPassRight(forwardPassedMatched);

    // backward passing2
    BackwardPassing backwardPassing2(image1, image2);
    double** backwardPassed2 = backwardPassing2.BackwardPassLeft(forwardPassedMatched);


    std::cout << "Calculating disparities for the dynamic approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

int DisparitySpaceImage(
        cv::Mat& image1, cv::Mat& image2,
        int half_window_size, int r, int x, int y)
{
    int ssd = 0;
    for (int u = -half_window_size; u <= half_window_size; ++u) {
        for (int v = -half_window_size; v <= half_window_size; ++v) {
            //std::cout << "r + u " << r + u << std::endl;
            //std::cout << "x + v " << x + v << std::endl;
            //std::cout << "y + v" << y + v << std::endl;
            int t = x + v;
            int l = y + v;
            int o = r + u;
            
            int height = image1.size().height;
            int width = image1.size().width;

            if (o < 0) {
                o = 0;
            }
            if (o > height -1) {

                o = height -1;
            }


            if (t < 0) {
                t = 0;
            }
            if (t > height - 1) {
                t = height - 1;
            }

            if (l < 0) {
                l = 0;
            }
            if (l > height - 1) {
                l = height - 1;
            }

            int val_left = image1.at<uchar>(o, t);
            int val_right = image2.at<uchar>(o, l);
            ssd += (val_left - val_right) * (val_left - val_right);
        }
    }
    return ssd;
}


void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length, std::vector<cv::Point3f>& word_3d)
{
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());

    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;

            const float Z = focal_length * baseline / (disparities.at<uchar>(i, j) + dmin);
            const float X = (i - width / 2) * Z / focal_length;
            const float Y = (j - height / 2) * Z / focal_length;

            word_3d.push_back(cv::Point3f(X, Y, Z));

            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }


    cv::Mat_<float> features(0, 3);

    for (auto&& point : word_3d) {

        //Fill matrix
        cv::Mat row = (cv::Mat_<float>(1, 3) << point.x, point.y, point.z);
        features.push_back(row);
    }
    //std::cout << features << std::endl;

    cv::flann::Index flann_index(features, cv::flann::KDTreeIndexParams(1));


    unsigned int max_neighbours = 3;

    for (int i = 0; i < word_3d.size(); i++) {
        cv::Point3f p = word_3d[i];

        cv::Mat query = (cv::Mat_<float>(1, 3) << p.x, p.y, p.z);

        cv::Mat indices, dists; //neither assume type nor size here !
        double radius = 2.0;

        flann_index.radiusSearch(query, indices, dists, radius, max_neighbours,
            cv::flann::SearchParams(32));

        int index1 = indices.at<uchar>(0, 0);
        int index2 = indices.at<uchar>(0, 1);
        int index3 = indices.at<uchar>(0, 2);

        cv::Point3f np1 = word_3d[index1];
        cv::Point3f np2 = word_3d[index2];
        cv::Point3f np3 = word_3d[index3];
        int u5 = 5;

    }









    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}



double distance_(double x1, double y1, double z1, double x2, double y2, double z2) {
    
    double x1_x2 = x1 - x2;
    double y1_y2 = y1 - y2;
    double z1_z2 = z1 - z2;

    double dist_sqr = x1_x2* x1_x2 + y1_y2 * y1_y2 + z1_z2 * z1_z2;
    return std::sqrt(dist_sqr);
}

void StereoEstimation_DP(
    const int& window_size,
    int height, int width, const int lambda, cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities)
{
    Size imageSize = image1.size();
    Mat disparityMap = Mat::zeros(imageSize, CV_16UC1);

#pragma omp parallel for
    for (int y_0 = window_size; y_0 < imageSize.height - window_size; ++y_0)
    {
        Mat C = Mat::zeros(Size(imageSize.width - 2 * window_size, imageSize.width - 2 * window_size), CV_16UC1);
        Mat M = Mat::zeros(Size(imageSize.width - 2 * window_size, imageSize.width - 2 * window_size), CV_8UC1);
        C.at<unsigned short>(0, 0) = 0;
        M.at<unsigned char>(0, 0) = 0;
        for (int i = 1; i < C.size().height; i++)
        {
            C.at<unsigned short>(i, 0) = i * lambda;
            M.at<unsigned char>(i, 0) = 1;
        }
        for (int j = 1; j < C.size().width; j++)
        {
            C.at<unsigned short>(0, j) = j * lambda;
            M.at<unsigned char>(0, j) = 2;
        }
        for (int r = 1; r < C.size().height; r++)
        {
            for (int l = 1; l < C.size().width; l++)
            {
                Mat window_left = image1(Rect(l, y_0 - window_size, 2 * window_size + 1, 2 * window_size + 1));
                Mat window_right = image2(Rect(r, y_0 - window_size, 2 * window_size + 1, 2 * window_size + 1));
                Mat diff;
                absdiff(window_left, window_right, diff);

                int SAD = sum(diff)[0];
                int c_m = C.at<unsigned short>(r - 1, l - 1) + SAD;
                int c_l = C.at<unsigned short>(r - 1, l) + lambda;
                int c_r = C.at<unsigned short>(r, l - 1) + lambda;

                // Minimizing cost
                int c = c_m;
                int m = 0;
                if (c_l < c) // Occluded from left
                {
                    c = c_l;
                    m = 1;
                    if (c_r < c) // Occluded from right
                    {
                        c = c_r;
                        m = 2;
                    }
                }
                C.at<unsigned short>(r, l) = c;
                M.at<unsigned char>(r, l) = m;
            }
        }
        // Create disparity map
        int i = M.size().height - 1;
        int j = M.size().width - 1;
        while (j > 0)
        {
            if (M.at<unsigned char>(i, j) == 0)
            {
                disparityMap.at<unsigned short>(y_0, j) = abs(i - j);
                i--;
                j--;
            }
            else if (M.at<unsigned char>(i, j) == 1)
            {
                i--;
            }
            else if (M.at<unsigned char>(i, j) == 2)
            {
                disparityMap.at<unsigned short>(y_0, j) = 0;
                j--;
            }
        }
#pragma omp critical
        cout << y_0 - window_size + 1 << "/" << imageSize.height - 2 * window_size << "\r" << flush; // Progress
    }
    Mat disparityMap_CV_8UC1;
    disparityMap.convertTo(disparityMap_CV_8UC1, CV_8UC1);
    dp_disparities = disparityMap_CV_8UC1;
}

void get_closest_two_points(double x, double y, double z, std::vector<cv::Point3f> list_of_points,
    double& selected_x,
    double& selected_y,
    double& selected_z,
    double& selected_x_2,
    double& selected_y_2,
    double& selected_z_2) {
    double distance = 99999;
    double distance_2 = 99999;

    int size = list_of_points.size();
    

    for (int i = 0; i < size; i++) {
        double x_1 = list_of_points[i].x;
        double y_1 = list_of_points[i].y;
        double z_1 = list_of_points[i].z;
        double calculated_dist = distance_(x, y, z, x_1, y_1, z_1);
        if (calculated_dist < distance){

            distance = calculated_dist;
            selected_x = x_1;
            selected_y = y_1;
            selected_z = z_1;
        }
        if ((calculated_dist > distance)  && (calculated_dist < distance_2)) {


            distance_2 = calculated_dist;
            selected_x_2 = x_1;
            selected_y_2 = y_1;
            selected_z_2 = z_1;
        }

    }

    
}

void get_normal(cv::Point3f a, cv::Point3f b, cv::Point3f c) {
    cv::Mat A(3, 3, CV_32F);
    A.at<float>(0, 0) = a.x;
    A.at<float>(0, 1) = a.y;
    A.at<float>(0, 2) = a.z;
    A.at<float>(1, 0) = b.x;
    A.at<float>(1, 1) = b.y;
    A.at<float>(1, 2) = b.z;

    A.at<float>(2, 0) = c.x;
    A.at<float>(2, 1) = c.y;
    A.at<float>(2, 2) = c.z;


    cv::Mat eVecs(3, 3, CV_32F), eVals(3, 3, CV_32F);
    std::cout << A << std::endl;
    eigen(A.t() * A, eVals, eVecs);

    std::cout << eVals << std::endl;
    std::cout << eVecs << std::endl;


}

void quality_matrices(cv::Mat ground_truth, cv::Mat naive_disparities, cv::Mat dynamic_disparities, cv::Mat open_cv_img) {

    float SAD_naive = 0;
    float SAD_dynamic = 0;
    float SAD_open_cv = 0;
    
    float MSE_naive = 0;
    float MSE_dynamic = 0;
    float MSE_opencv = 0;
    
    float RMSE_naive = 0;
    float RMSE_dynamic = 0;
    float RMSE_opencv = 0;
    
    float SSD_naive = 0;
    float SSD_dynamic = 0;
    float SSD_opencv = 0;

    float PSNR_naive = 0;
    float PSNR_dynamic = 0;
    float PSNR_opencv = 0;
    

    const auto  height = ground_truth.rows;
    const auto width = ground_truth.cols;


    for (int i = 0; i < height; i++)
    {

        for (int j = 0; j < width; j++)
        {
            float i1 = ground_truth.at<uchar>(i, j);
            float i2 = naive_disparities.at<uchar>(i, j);
            float i3 = dynamic_disparities.at<uchar>(i, j);
            float i4 = open_cv_img.at<uchar>(i, j);
            //float i5 = output_ourfilter.at<uchar>(i, j);

            SAD_naive += std::abs(i1 - i2);
            SAD_dynamic += std::abs(i1 - i3);
            SAD_open_cv += std::abs(i1 - i4);
            //SAD_ourfilter += std::abs(i1 - i5);

            SSD_naive += (i1 - i2) * (i1 - i2);
            SSD_dynamic += (i1 - i3) * (i1 - i3);
            SSD_opencv += (i1 - i4) * (i1 - i4);
            //SSD_ourfilter += (i1 - i5) * (i1 - i5);



        }
    }


    MSE_naive = SSD_naive / (width * height);
    MSE_dynamic = SSD_dynamic / (width * height);
    MSE_opencv = SSD_opencv / (width * height);
    //MSE_ourfilter = SSD_ourfilter / (width * height);


    RMSE_naive = sqrt(MSE_naive);
    RMSE_dynamic = sqrt(MSE_dynamic);
    RMSE_opencv = sqrt(MSE_opencv);
    //RMSE_ourfilter = sqrt(MSE_ourfilter);

    int max1 = 255;


    PSNR_naive = 20 * log10(max1) - 10 * log10(MSE_naive);
    PSNR_dynamic = 20 * log10(max1) - 10 * log10(MSE_dynamic);
    PSNR_opencv = 20 * log10(max1) - 10 * log10(MSE_opencv);
    //PSNR_ourfilter = 20 * log10(max1) - 10 * log10(MSE_ourfilter);



    std::cout << "SAD between original and naive: " << SAD_naive << std::endl;
    std::cout << "SAD between original and dynamic: " << SAD_dynamic << std::endl;
    std::cout << "SAD between original and opencv: " << SAD_open_cv << std::endl;
    std::cout << " -------------------------------------- ";


    std::cout << "MSE between original and naive: " << MSE_naive << std::endl;
    std::cout << "MSE between original and dynamic: " << MSE_dynamic << std::endl;
    std::cout << "MSE between original and opencv: " << MSE_opencv << std::endl;

    std::cout << " -------------------------------------- ";

    std::cout << "SSD between original and naive: " << SSD_naive << std::endl;
    std::cout << "SSD between original and dynamic: " << SSD_dynamic << std::endl;
    std::cout << "SSD between original and opencv: " << SSD_opencv << std::endl;

    std::cout << " -------------------------------------- ";

    std::cout << "RMSE between original and naive: " << RMSE_naive << std::endl;
    std::cout << "RMSE between original and dynamic: " << RMSE_dynamic << std::endl;
    std::cout << "RMSE between original and opencv: " << RMSE_opencv << std::endl;

    std::cout << " -------------------------------------- ";

    std::cout << "PSNR between original and naive: " << PSNR_naive << std::endl;
    std::cout << "PSNR between original and dynamic: " << PSNR_dynamic << std::endl;
    std::cout << "PSNR between original and opencv: " << PSNR_opencv << std::endl;
    
    std::cout << " -------------------------------------- ";

}
