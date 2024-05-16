#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  

#include <vector>
#include <stdio.h>

#include "brake_lights_detection.cpp" 
#include "brake_lights_classification.cpp" 
// #include "video_processing.cpp" 
//#include "Source1.cpp"

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;
#include <filesystem> 
#include <experimental/filesystem> 

namespace fs = std::experimental::filesystem;




using namespace cv;
using namespace std; 

Mat convertToLab(const Mat& image) {
    Mat Lab_image; 
    cvtColor(image, Lab_image, 45); 
    return Lab_image;
} 

Mat convertToHSV(const Mat& image) {
    Mat Lab_image = image.clone();
    // cvtColor(image, Lab_image, 41); 
    return Lab_image;
} 

void img_preprocessing(Mat& image, vector<Mat>& lab_channels, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR); 

  //  imwrite(std::string("Resize_img.png").c_str(), image);

    Mat Lab_image = convertToLab(image);

    split(Lab_image, lab_channels);
}

void img_preprocessing_HSV(Mat& image, vector<Mat>& HSV_channels, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR); 

    Mat HSV_image = convertToHSV(image); 

    split(HSV_image, HSV_channels);
}

void drawBoundingRectangles(const cv::Mat image, const cv::Mat stats) {
    for (int i = 2; i < stats.rows; ++i) {
        cv::rectangle(image, cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP)),
            cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT)),
            cv::Scalar(0, 255, 0), 2);
    }

   // imwrite(std::string("Rectangles.png").c_str(), image);
}

void filterStatsAndCentroids(const cv::Mat& stats, const cv::Mat& centroids, cv::Mat& filteredStats, cv::Mat& filteredCentroids, cv::Mat& resizedImage) {
      
    int wight_img = resizedImage.cols; 
    int height_img = resizedImage.rows; 
    
    for (int r = 1; r < stats.rows; ++r) { // с 1, потому что 0 - это фон 
        int wight = stats.at<int>(r, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(r, cv::CC_STAT_HEIGHT);

      

        if (!(wight <= 3 && height <= 3) && !(wight == wight_img && height == height_img) && stats.at<int>(r, cv::CC_STAT_AREA) >= 80 && stats.at<int>(r, cv::CC_STAT_AREA) <= wight_img * height_img / 3
            && stats.row(r).at<int>(1) + height <= (5 * height_img) / 6) { // 
            filteredStats.push_back(stats.row(r));
            filteredCentroids.push_back(centroids.row(r));
        }
    }
} 

void get_rectangle_for_detector(const Mat channel, cv::Mat& filteredStats, cv::Mat& filteredCentroids, cv::Mat resized_img) {
    Mat otsu_img;
    //  int thresh = 0;
    int maxValue = 255;
    double thresh = cv::threshold(channel, otsu_img, 0, maxValue, THRESH_OTSU); // THRESH_TRIANGLE

    // cout << "Otsu Threshold: " << thresh << endl;
   // imshow("Image after Otsu Threshold", otsu_img);

    Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(otsu_img, labels, stats, centroids);

   // imwrite(std::string("Otsu_img.png").c_str(), otsu_img);

    /* Mat otsu_img2;

    int thresh2 = 0;
    int maxValue2 = 255;

    long double thres2 = cv::threshold(channel, otsu_img2, thresh2, maxValue2, THRESH_OTSU); // THRESH_TRIANGLE

    // cout << "Otsu Threshold: " << thresh << endl;
    //imshow("Image after Otsu Threshold", otsu_img);

   Mat labels2, stats2, centroids2;
    int numLabels2 = cv::connectedComponentsWithStats(otsu_img2, labels2, stats2, centroids2);

    stats.push_back(stats2);
    centroids.push_back(centroids2);

    //  cout << "Befor filtr" << stats << endl; */

    // drawBoundingRectangles(resized_img, stats);
    // imwrite(std::string("with_rectangels.png").c_str(), resized_img);

    filterStatsAndCentroids(stats, centroids, filteredStats, filteredCentroids, resized_img);
    //  cout << "After filtr" << filteredStats << endl; 

 //   drawBoundingRectangles(resized_img, filteredStats);
 //   imwrite(std::string("with_rectangels.png").c_str(), resized_img);
}

void get_features_from_dataset(const vector<string>& input_folders_test, int new_weight, int new_height, Mat& data_l, Mat& data_r, Mat& data_third, Mat& labels_test, Mat& labels_test_classifier) {

    const int L = 0;
    int a = 1;
    const int B = 2;

    const int h = 0;
    const int s = 1;
    const int v = 2;

    int detecting_light_on = 0;
    int detecting_light_HSV = 0;
    int total = 0;
    int detecting_light_off = 0;

    vector<string> img_name; 

    for (const string& folder : input_folders_test) {
        for (const auto& entry : fs::directory_iterator{ folder }) {
            string image_path = entry.path().string();
            Mat img = imread(image_path);

            vector<Mat> lab_channels(3);

            img_preprocessing(img, lab_channels, new_weight, new_height);
            /*
                        if (folder.substr(folder.size() - 3) == "OFF")
                            imwrite(std::string("CVAT_OFF_TRAIN\\" + std::to_string(total) + ".png").c_str(), img);
                        else
                            imwrite(std::string("CVAT_ON_TRAIN\\" + std::to_string(total) + ".png").c_str(), img);
                            */
            Mat filteredStats, filteredCentroids;

            get_rectangle_for_detector(lab_channels[a], filteredStats, filteredCentroids, img);

            Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weight / 2);

            total++;

            Mat lateral_stats_HSV;

            if (lateral_stats.rows < 2) {

                vector<Mat> HSV_channels(3);

                const int B = 2;

                img_preprocessing_HSV(img, HSV_channels, new_weight, new_height);

                Mat filteredStatsHSV, filteredCentroidsHSV;

                get_rectangle_for_detector(HSV_channels[B], filteredStatsHSV, filteredCentroidsHSV, img);

                lateral_stats_HSV = detector_new(filteredStatsHSV, filteredCentroidsHSV, new_weight / 2);

            }

            if (lateral_stats.rows >= 2) {
                classifier_get_features(data_l, data_r, data_third, lateral_stats, lab_channels, img);
                //  detecting_light_on++;
                  //  imwrite(std::string("LAB\\" + std::to_string(total) + "r.png").c_str(), img);
                drawBoundingRectangles(img, lateral_stats);
                imwrite(std::string("LAB\\" + std::to_string(total) + ".png").c_str(), img);
            }
            else if (lateral_stats_HSV.rows >= 2) {
                classifier_get_features(data_l, data_r, data_third, lateral_stats_HSV, lab_channels, img);
                // detecting_light_HSV++;
                drawBoundingRectangles(img, lateral_stats_HSV);
                imwrite(std::string("RGB\\" + std::to_string(detecting_light_HSV) + ".png").c_str(), img);
            }

            if (folder.substr(folder.size() - 3) == "OFF") {
                if (lateral_stats.rows >= 2 || lateral_stats_HSV.rows >= 2) {
                    labels_test_classifier.push_back(0);
                    detecting_light_off ++ ; 
                    img_name.push_back(image_path);
                }
                labels_test.push_back(0);

            }
            else {
                if (lateral_stats.rows >= 2 || lateral_stats_HSV.rows >= 2) {
                    labels_test_classifier.push_back(1);
                    detecting_light_on++;
                    img_name.push_back(image_path);
                }
                labels_test.push_back(1);
            }


        }
    }

    cout << img_name[0]; 

    std::cout << endl << "detecting_light_on " << detecting_light_on + detecting_light_HSV << "detecting_light_off" << detecting_light_off << " total " << total << endl;
}

int main(int argc, char** argv)
{

    // Mat image = imread("C:\\testdrive1583149193.4818494.png"); 
   // Mat image = imread("C:\\testdrive1583149150.6489925.png"); // красная 
  //  Mat image = imread("C:\\testdrive1583149169.625285.png"); // красная с включёнными 
    Mat image = imread("C:\\testdrive1583149191.58733.png"); // белая 
    //  Mat image = imread("C:\\testdrive1580902052.9363065.png");
     // Mat image = imread("C:\\11r.png"); 
    // Mat image = imread("C:\\18r.png");
  //  Mat image = imread("C:\\testdrive1583149188.5726807.png"); 
     //  Mat image = imread("C:\\red_off.png"); 

  //  get_video(); 

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    }

    int new_weight = 416;
    int new_height = 416;

    vector<string> input_folders_train = { "TRAIN/brake_TRAIN_LR_OFF", "TRAIN/brake_ON_TRAIN" };
    vector<string> input_folders_test = { "TEST/brake_TEST_LR_OFF", "TEST/brake_ON_TEST" }; 
    
    vector<vector<int>> features_test;
    Mat labels_train; 
    Mat labels_train_classifier;

    int detecting_light_on = 0;
    int detecting_light_HSV = 0;
    int total = 0; 

    string xmlPath = "dataset_rect/annotations.xml"; 
   /* vector<Rect> rectangles = parseXmlFile(xmlPath);

    for (auto& rect : rectangles) {
        cout << "Rect: (" << rect << endl;
    }
*/
    Mat data_l, data_r, data_third; 

     get_features_from_dataset(input_folders_train, new_weight, new_height, data_l, data_r, data_third, labels_train, labels_train_classifier);
     
     cout << endl << labels_train_classifier.size() << " data " << data_third.size() << endl;

     

     Mat labels_test; 
     Mat labels_test_classifier; 

     Mat data_l_test, data_r_test, data_third_test; 

     get_features_from_dataset(input_folders_test, new_weight, new_height, data_l_test, data_r_test, data_third_test, labels_test, labels_test_classifier);

  //   cout << endl << labels_test_classifier.size() << " data " << data_third_test.size() << endl; 

  //   cout << endl << labels_test_classifier; 

     Mat predict_th, predict_LR; 
     Mat predict_th_train, predict_LR_train; 

     SVM_classifier_third_light(data_third, labels_train_classifier, data_third_test, labels_test_classifier, predict_th, predict_th_train);

     SVM_classifier_LR_light2(data_l, data_r, labels_train_classifier, data_l_test, data_r_test, labels_test_classifier, predict_LR, predict_LR_train);

     main_classifier(predict_LR_train, predict_th_train, labels_train_classifier);

     main_classifier(predict_LR, predict_th, labels_test_classifier);

  //  cout << "Hey" << endl;

    Mat resized_img = image.clone();
    vector<Mat> lab_channels(3);

    img_preprocessing(image, lab_channels, new_weight, new_height);


    int a = 1;

    imshow("a channel", lab_channels[a]);

    /*
        Применение порогового значения Оцу
     */


    Mat filteredStats, filteredCentroids;

    get_rectangle_for_detector(lab_channels[a], filteredStats, filteredCentroids, image);

    cv::imshow("Исходное изображение с метками", image);


    Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weight / 2);
    /*
        if (lateral_stats.rows < 2) {

            vector<Mat> HSV_channels(3);

            img_preprocessing_HSV(image, HSV_channels, new_weight, new_height);

            Mat filteredStatsHSV, filteredCentroidsHSV;

            get_rectangle_for_detector(HSV_channels[2], filteredStatsHSV, filteredCentroidsHSV, image);

            Mat stats_ = detector_new(filteredStatsHSV, filteredCentroidsHSV, new_weight / 2);
            //  drawBoundingRectangles(image, stats_);
            //  cv::imshow("Исходное изображение с метками", image);
            //  cout << "ststs_" << stats_ << endl;

        }
    */
  //  cout << "lateral " << lateral_stats << endl;


    Mat new_i = image.clone();

    drawBoundingRectangles(resized_img, lateral_stats);
    imwrite("third_light.png", resized_img);

    //  Mat labels_test_, labels_test_classifier_;

    classifier_get_features(data_l, data_r, data_third, filteredStats, lab_channels, image);

    //  labels_test_classifier_ = { 1, 0 }; 

    //  cout << "data_l = " << data_l << "labels_test_classifier = " << labels_test_classifier_ << endl; 

    //SVM_classifier_third_light(data_l_, labels_test_classifier_);


    /*for (int i = 0; i < data_l.size(); i++)
     {
         for (int j = 0; j < data_l[i].size(); j++)
         {
             cout << data_l[i][j] << "   ";
         }
     }
     */

    // Wait for any keystroke in the window
    waitKey(0);
    return 0;
}

