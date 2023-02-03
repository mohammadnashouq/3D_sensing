#include <string> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include "PLYWriter.h"

using namespace std::chrono;

void Disparity2PointCloud(
    const std::string& output_file,
    const std::string& output_file_w_norm,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length , std::ofstream& logFile)
{
   
    std::ofstream outfile(output_file);

    std::vector<cv::Point3f> word_3d;
    std::vector<cv::Point3f>   word_3d_norm;



    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point stop;
    std::chrono::microseconds duration;

    // Starting JB Upsampling for the Ground Truth Image


    std::cout << "Starting 3D Constraction --------------------" <<  std::endl;
    logFile << "Starting 3D Constraction --------------------" << std::endl;

    start = high_resolution_clock::now();

   

 


    int count = 0;

    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;

            const float Z = focal_length * baseline / (disparities.at<uchar>(i, j) + dmin);
            const float X = (i - width / 2) * Z / focal_length;
            const float Y = (j - height / 2) * Z / focal_length;
            outfile << X << " " << Y << " " << Z << std::endl;

            if (count % 30 == 0) {
                word_3d.push_back(cv::Point3f(X, Y, Z));
            }
           
            count++;
        }
    }




    stop = high_resolution_clock::now();

    duration = duration_cast<microseconds>(stop - start);

    std::cout << "3D Constraction execution time : " << duration.count() << std::endl;
    logFile << "3D Constraction execution time : " << duration.count() << std::endl;



    std::cout << "Starting 3D Constraction with Norm --------------------" << std::endl;
    logFile << "Starting 3D Constraction with Norm --------------------" << std::endl;
    start = high_resolution_clock::now();

    cv::Mat_<float> features(0, 3);

#pragma omp parallel for 
    for (int i = 0; i < word_3d.size(); i++)
#pragma omp critical 
    {

        cv::Point3f point = word_3d[i];

        //Fill matrix
        cv::Mat row = (cv::Mat_<float>(1, 3) << point.x, point.y, point.z);
        features.push_back(row);
    }
    //std::cout << features << std::endl;

    cv::flann::Index flann_index(features, cv::flann::KDTreeIndexParams(1));


    unsigned int max_neighbours = 3;


    std::cout << "word_3d size : " << word_3d.size() << std::endl;
    int c = 0;
    double size = word_3d.size();

#pragma omp parallel for 
    for (int i = 0; i < word_3d.size(); i++)
#pragma omp critical 
    {
        std::cout
            << "Calculating KDTree ... "
            << std::ceil((static_cast<double>(i) / size) * 100) << "%\r"
            << std::flush;


        // std::cout << "Processing point num : " << i << std::endl;


        /* c++;
         if (c == 1000)
         {
             std::cout << "Processing point num : " << i << std::endl;
             c = 0;
         }*/

        cv::Point3f p = word_3d[i];

        cv::Mat query = (cv::Mat_<float>(1, 3) << p.x, p.y, p.z);

        cv::Mat indices, dists; //neither assume type nor size here !
        double radius = 1000000000;

        flann_index.radiusSearch(query, indices, dists, radius, max_neighbours,
            cv::flann::SearchParams(32));

        int index1 = indices.at<int>(0, 0);
        int index2 = indices.at<int>(0, 1);
        int index3 = indices.at<int>(0, 2);

        cv::Point3f np1 = word_3d[index1];


        cv::Point3f np2 = word_3d[index2];
        cv::Point3f np3 = word_3d[index3];



        cv::Point3f PQ = cv::Point3f(np2.x - p.x, np2.y - p.y, np2.z - p.z);
        cv::Point3f PR = cv::Point3f(np3.x - p.x, np3.y - p.y, np3.z - p.z);

        float a1 = np2.x - p.x;
        float b1 = np2.y - p.y;
        float c1 = np2.z - p.z;

        float a2 = np3.x - p.x;
        float b2 = np3.y - p.y;
        float c2 = np3.z - p.z;

        int x = b1 * c2 - b2 * c1;
        int y = a2 * c1 - a1 * c2;
        int z = a1 * b2 - b1 * a2;



        cv::Point3f Normal = cv::Point3f(x, y, z);

        word_3d_norm.push_back(Normal);




    }

    stop = high_resolution_clock::now();

    duration = duration_cast<microseconds>(stop - start);

    std::cout << "3D Constraction with Norm execution time : " << duration.count() << std::endl;
    logFile << "3D Constraction with Norm execution time : " << duration.count() << std::endl;



    WritePLY(output_file_w_norm.c_str(), word_3d, word_3d_norm);

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    logFile << "Reconstructing 3D point cloud from disparities... Done.\r" << std::endl;
    std::cout << std::endl;
}
