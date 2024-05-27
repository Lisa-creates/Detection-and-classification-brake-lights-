#pragma once

#include <opencv2/opencv.hpp> 

using namespace cv;
using namespace cv::ml;
using namespace std;

void classifier_get_features(Mat& data_l, Mat& data_r, Mat& data_third, const Mat& stats, const vector<Mat> channels, Mat& img);
void SVM_classifier_LR_light(const Mat& data_l, const Mat& data_r, const Mat& trainLabels, const Mat& data_l_test, const Mat& data_r_test, const Mat& testLabels, Mat& predictedLabels, Mat& predictedLabels_train); 
void SVM_classifier_third_light(const Mat& data_third, const Mat& trainLabels, const Mat& data_third_test, const Mat& testLabels, Mat& predictedLabels, Mat& predictedLabels_train);
void main_classifier(const Mat& predict_LR, const Mat& predict_third, const Mat& labels);