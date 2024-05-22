#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  
#include <fstream> 

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;
#include <filesystem> 
#include <experimental/filesystem> 
#include <regex> 

// #include "brake_lights_classification.cpp" 

namespace fs = std::experimental::filesystem; 

Mat convertToLab2(const Mat& image) {
    Mat Lab_image;
    cvtColor(image, Lab_image, 45);
    return Lab_image;
}


void img_preprocessing2(Mat& image, vector<Mat>& lab_channels, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR);

    Mat Lab_image = convertToLab2(image);

    split(Lab_image, lab_channels);
}

int get_video() {

    //VideoCapture cap("videoplayback.mp4");
    //vector<string> input_folders = { "default_video/label_2" };

    VideoCapture cap("car_black.mp4");
    vector<string> input_folders = { "default_car_black/label_2" };

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Ptr<SVM> svm_lat = SVM::load("svm_model_lateral.yml");
    Ptr<SVM> svm_th = SVM::load("svm_model_third.yml"); 

    // while (1) {
    for (const string& folder : input_folders) {
        for (const auto& entry : fs::directory_iterator{ folder }) {
            Mat frame;

            cap >> frame;

            if (frame.empty())
                break;

            string label_path = entry.path().string();
            cout << label_path << endl;

            std::ifstream file(label_path);
            std::string line;
            string label__;
            float x, y, x2, y2, n, n1, n2;

            while (std::getline(file, line)) {

                std::istringstream iss(line);
                iss >> label__;
                iss >> n >> n1 >> n2 >> x >> y >> x2 >> y2;

            }
            cv::Rect car(x, y, x2 - x, y2 - y);

            // cout << rect; 

            // frame = frame(rect); 

            Mat frame_orig = frame;

            int new_weight = 416;
            int new_height = 416;

            Mat img_ROI = frame(car);

        //    resize(img_ROI, img_ROI, Size(new_weight, new_height), INTER_LINEAR);

            vector<Mat> lab_channels(3);

            img_preprocessing2(img_ROI, lab_channels, new_weight, new_height);
            
            Mat orig_ROI = img_ROI;

            double lambda_S = 0.45, lambda_D = 0.35, lambda_U = 0.2;
            float tao_tb = 0.67;
            float tao_S = 0.45;
            int tao_v = 32;

            Mat lateral_stats = detector_new(new_weight / 2, lab_channels, img_ROI, orig_ROI, lambda_S, lambda_D, lambda_U, tao_v, tao_S, tao_tb, 1);
            Mat frame_out = frame.clone();

            // double thresh = cv::threshold(lab_channels[1], img_ROI, 0, 255, THRESH_TRIANGLE);

            Mat data_l, data_r, data_third; 

            int label_class = 0; 

            if (lateral_stats.rows >= 2)
            {
                classifier_get_features(data_l, data_r, data_third, lateral_stats, lab_channels, frame);  

                Mat predict_data_l, predict_data_r, predict_data_third;

                cout << data_l << endl << data_r << endl << data_third << endl; 

                Mat dataMat_test(data_l.rows + data_r.rows, data_l.cols, CV_32F);

                cv::vconcat(data_l, data_r, dataMat_test);
                
                

                cout << "Here 1" << predict_data_third << endl; 

                svm_lat->predict(dataMat_test, predict_data_l); 
              // svm_lat->predict(data_r, predict_data_r); 
                if (lateral_stats.rows < 3) {
                    svm_th->predict(data_third, predict_data_third);

                    cout << "Here 2 --- " << predict_data_l.at<float>(0) << " " << predict_data_l.at<float>(1) << endl;

                    if (predict_data_l.at<float>(0) + predict_data_l.at<float>(1) + predict_data_third.at<float>(0) >= 3)
                        label_class = 1;
                }
                else {
                    if (predict_data_l.at<float>(0) + predict_data_l.at<float>(1) >= 2) 
                        label_class = 1;
                }
            }

            cout << label_class << endl; 
            

            for (int i = 0; i < lateral_stats.rows; ++i) { 

                cv::Rect r(lateral_stats.row(i).at<int>(cv::CC_STAT_LEFT), lateral_stats.row(i).at<int>(cv::CC_STAT_TOP), lateral_stats.row(i).at<int>(cv::CC_STAT_WIDTH), lateral_stats.row(i).at<int>(cv::CC_STAT_HEIGHT));
                cv::rectangle(img_ROI, r, cv::Scalar(0, 0, 200), 3); 

                string text; 
                if (label_class == 0)
                    text = "OFF";
                else
                    text = "ON";
                cv::putText(img_ROI, text, cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            }

            imshow("Frame", img_ROI);

            // Press  ESC on keyboard to exit
            char c = (char)waitKey(25);
            if (c == 27)
                break;
        }
    }
} 


