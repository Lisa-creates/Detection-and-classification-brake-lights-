#include <opencv2/opencv.hpp> 
#include <iostream>  
#include <fstream>
#include <vector> 

#include"Header_files/brake_lights_classification.h"
#include "Header_files/test_detector.h" 
#include "Header_files/brake_lights_detection.h" 
#include"Header_files/video_processing.h"

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;
#include <experimental/filesystem>  

namespace fs = std::experimental::filesystem;

using namespace cv;
using namespace std; 


void drawBoundingRectangles(const cv::Mat image, const cv::Mat stats) {
    for (int i = 0; i < stats.rows; ++i) {
        cv::rectangle(image, cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP)),
            cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT)),
            cv::Scalar(0, 255, 0), 2);
    }
}

void get_features_from_dataset(const vector<string>& input_folders_test, int new_weight, int new_height, Mat& data_l, Mat& data_r, Mat& data_third, Mat& labels_test, Mat& labels_test_classifier) {

    const int L = 0;
    int a = 1;
    const int B = 2;

    int detecting_light_on = 0;
    int total = 0;
    int detecting_light_off = 0;

    vector<string> img_name; 

    for (const string& folder : input_folders_test) {
        for (const auto& entry : fs::directory_iterator{ folder }) {
            string image_path = entry.path().string();
            Mat img = imread(image_path);

            vector<Mat> lab_channels(3);

            img_preprocessing(img, lab_channels, new_weight, new_height); 

            Parameters parameters; 

            Mat lateral_stats = detector(lab_channels, img, parameters.lambda_S, parameters.lambda_D, parameters.lambda_U, parameters.tao_v, parameters.tao_S, parameters.tao_tb);

            total++; 

            if (lateral_stats.rows >= 2) {
                classifier_get_features(data_l, data_r, data_third, lateral_stats, lab_channels, img);
                drawBoundingRectangles(img, lateral_stats);
            }


            if (folder.substr(folder.size() - 3) == "OFF") {
                if (lateral_stats.rows >= 2) {
                    labels_test_classifier.push_back(0);
                    detecting_light_off ++ ; 
                    img_name.push_back(image_path);
                }
                labels_test.push_back(0);

            }
            else {
                if (lateral_stats.rows >= 2) {
                    labels_test_classifier.push_back(1);
                    detecting_light_on++;
                    img_name.push_back(image_path);
                }
                labels_test.push_back(1);
            }


        }
    }

    cout << img_name[0]; 

    std::cout << endl << "detecting_light_on " << detecting_light_on << "detecting_light_off" << detecting_light_off << " total " << total << endl;
}



int main(int argc, char** argv)
{
    setlocale(LC_ALL, "Russian"); 

    int action; 

    cout << "Choose an action: "<< endl <<
        "1 - for video" << endl <<
        "2 - for photos and model training" << endl <<
        "3 - for testing the detector" << endl;

    cin >> action; 

    if (action == 1)
    {
        const string video_path = "videoplayback.mp4"; // your path 
        const string input_label = "default_video/label_2"; // your path 

        get_video(video_path, input_label);
        waitKey(0);

        return 0; 
    }
    else if (action == 3) {
        get_test_for_detector();
        waitKey(0);

        return 0; 
    }

    const int new_weight = 416;
    const int new_height = 416;

    vector<string> input_folders_train = { "TRAIN_LR/brake_TRAIN_OFF", "TRAIN_LR/brake_ON_TRAIN" }; // your path 
    vector<string> input_folders_test = { "TEST_LR/brake_TEST_OFF", "TEST_LR/brake_ON_TEST" }; // your path 

    vector<vector<int>> features_test;
    Mat labels_train;
    Mat labels_train_classifier;

    int detecting_light_on = 0;
    int detecting_light_HSV = 0;
    int total = 0;

    Mat data_l, data_r, data_third;

    get_features_from_dataset(input_folders_train, new_weight, new_height, data_l, data_r, data_third, labels_train, labels_train_classifier);

    cout << endl << labels_train_classifier.size() << " data " << data_third.size() << endl;


    Mat labels_test;
    Mat labels_test_classifier;

    Mat data_l_test, data_r_test, data_third_test;

    get_features_from_dataset(input_folders_test, new_weight, new_height, data_l_test, data_r_test, data_third_test, labels_test, labels_test_classifier);

    Mat predict_th, predict_LR;
    Mat predict_th_train, predict_LR_train;

    SVM_classifier_third_light(data_third, labels_train_classifier, data_third_test, labels_test_classifier, predict_th, predict_th_train);
    SVM_classifier_LR_light(data_l, data_r, labels_train_classifier, data_l_test, data_r_test, labels_test_classifier, predict_LR, predict_LR_train);

    main_classifier(predict_LR_train, predict_th_train, labels_train_classifier);
    main_classifier(predict_LR, predict_th, labels_test_classifier);

    waitKey(0);
    return 0;
} 
