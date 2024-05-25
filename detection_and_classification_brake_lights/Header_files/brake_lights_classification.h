#pragma once

#include <opencv2/opencv.hpp> 

using namespace cv;
using namespace cv::ml;
using namespace std;

void classifier_get_features(Mat& data_l, Mat& data_r, Mat& data_third, Mat& stats, const vector<Mat> channels, Mat& img); 
void SVM_classifier_LR_light(Mat data_l, Mat data_r, Mat trainLabels, Mat data_l_test, Mat data_r_test, Mat testLabels, Mat& predictedLabels, Mat& predictedLabels_train); 
void SVM_classifier_third_light(Mat data_third, Mat trainLabels, Mat data_third_test, Mat testLabels, Mat& predictedLabels, Mat& predictedLabels_train); 
void main_classifier(Mat predict_LR, Mat predict_third, Mat labels); 