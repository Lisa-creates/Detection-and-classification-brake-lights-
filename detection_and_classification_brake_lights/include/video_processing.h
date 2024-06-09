#pragma once 
#include <opencv2/opencv.hpp> 
#include <iostream>  
#include <fstream> 
#include <filesystem> 

#include"brake_lights_detection.h" 
#include"brake_lights_classification.h" 

using namespace cv; 
using namespace std; 
namespace fs = std::filesystem; 

int get_video(const string& video_path, const string& label_path);
Mat convert_to_Lab(const Mat& image); 
void img_preprocessing(Mat& image, vector<Mat>& lab_channels, const int weight, const int height); 