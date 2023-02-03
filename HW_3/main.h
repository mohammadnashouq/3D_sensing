//=================[ Made by Daniel Kuknyo ]=================//
#ifndef MAIN_H
#define MAIN_H

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "happly.h"
#include "nanoflann.hpp"
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace nanoflann;
using namespace happly;
using namespace cv;

//============[ Parameters set by user ]=============

//-----[ Initial transformation for the clouds ]-----
const bool apply_init_transformation = true; // Set to true to apply transformation defined below
const double lvl_noise = 0; // Level of noise to add to the original point cloud (Gaussian) [0.1, 0.5, 1.0]
const double lvl_rotation = 15; // Rotation to add to the original point cloud (in degrees) [5, 10, 15, 20]
const double lvl_translation = 5; // Translation to add to the original point cloud (y axis) [0.5, 1, 5]

//--------[ Thresholds and other parameters ]--------
const bool write_log = true; // Set to true to write log into log file
const int max_leaf = 10; // Maximum leaf size for KD-tree search
const int num_result = 1; // Number of results for KNN search
const int icp_iter = 100; // Number of iteration for the ICP algorithm
const int tricp_iter = 100; // Number of iterations for the TR-ICP algorithm
const double icp_error_t = 5; // ICP error threshold
const double tricp_error_t = 0.01; // TR-ICP error threshold
const double icp_error_change_t = 0.05; // ICP error change threshold
const double tricp_error_change_t = 0.005; // TR-ICP error change threshold
const double phi = (1 + sqrt(5)) / 2; // Golden ratio by definition
const int lambda = 2; // Tolerance parameter for golden section objective function

//-----------[ Folders and other constants ]----------
const string output_dir = "./results/"; // The folder to save results to
const string log_file = "./runlogs.csv"; // Log file path
const string ply_header = "./data/ply_header.txt"; // Header to add to all output ply files
const vector<Point3i> colors = {Point3i(174, 4, 33), Point3i(172, 255, 36)}; // RGB Colors for point clouds

//============[ Parameters set by program ]===========
extern long n_rows;
extern string cloud_name;
extern double xi;
extern float timenow;

// Types defined for the run
typedef KDTreeEigenMatrixAdaptor<MatrixXd> kd_tree;

//----------------[ Methods for main ]-----------------

MatrixXd apply_init_transform(MatrixXd cloud);

MatrixXd nn_search(const MatrixXd& cloud_1, const MatrixXd& cloud_2);

pair<Matrix3d, Vector3d> estimate_transformation(MatrixXd cloud_1, MatrixXd cloud_2);

MatrixXd reorder(const MatrixXd& cloud, const MatrixXd& indices);

double calc_error(const MatrixXd& cloud_1, const MatrixXd& cloud_2, bool mean);

void transform_cloud(MatrixXd& cloud, const Matrix3d& R, const Vector3d& t);

Matrix4d icp(MatrixXd cloud_1, const MatrixXd& cloud_2);

MatrixXd sort_matrix(MatrixXd mat);

void reorder_trim(const MatrixXd& cloud_1, const MatrixXd& cloud_2, MatrixXd& cloud_1_new, MatrixXd& cloud_2_new,
                  const MatrixXd& nn, const long& trimmed_len);

Matrix4d tr_icp(MatrixXd cloud_1, const MatrixXd& cloud_2);

//----------------[ Methods for suppl ]-----------------

vector<Point3d> read_pointcloud(char* filename);

MatrixXd vector2mat(vector<Point3d> vec, const long& len);

double obj_func(double x, MatrixXd nn);

double golden_section_search(double a, double b, const double& eps, const MatrixXd& nn);

void output_clouds(const MatrixXd& cloud_1, const MatrixXd& cloud_2, const string& method);

Matrix4d estimate_T_true(const MatrixXd& cloud_1, const MatrixXd& cloud_2);

double rotation_error(MatrixXd cloud_1, MatrixXd cloud_2);

double translation_error(MatrixXd cloud_1, MatrixXd cloud_2);

void log_execution(const MatrixXd& cloud_1, const MatrixXd& cloud_2, const string& method, const double& error,
                   const double& rotation_error, const double& translation_error, const int& n_iter,
                   const bool& converged, const Matrix4d& T_true, const Matrix4d& T_pred);

string print_mat(const MatrixXd& mat, const string& sep);

#endif // MAIN_H