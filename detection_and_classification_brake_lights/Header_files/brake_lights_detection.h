#pragma once 
#include <opencv2/opencv.hpp>  
#include <vector>
using namespace cv;
using namespace std; 

Mat detector(const vector<Mat>& lab_channels, Mat& img, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S, float tao_tb); 

struct Parameters {
    float lambda_S = 0.3;
    float lambda_D = 0.5;
    float lambda_U = 0.2;
    float tao_tb = 0.85;
    float tao_S = 0.45;
    int tao_v = 28;
}; 

