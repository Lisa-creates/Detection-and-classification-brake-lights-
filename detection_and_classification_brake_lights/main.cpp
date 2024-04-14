#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  

#include <vector>
#include <opencv2/opencv.hpp>


#include <stdio.h>

#include "brake_lights_detection.cpp" 
// #include "brake_lights_classifcation.cpp" 

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;
#include <filesystem> 
//namespace fs = std::filesystem;
#include <experimental/filesystem> 

namespace fs = std::experimental::filesystem;

//std::experimental::filesystemstd::filesystem 



using namespace cv;
using namespace std; 

Mat convertToLab(const Mat& image) {
    Mat Lab_image;
    cvtColor(image, Lab_image, 45); 
    return Lab_image;
} 

Mat convertToHSV(const Mat& image) {
    Mat Lab_image;
    cvtColor(image, Lab_image, 41); 
    return Lab_image;
} 

void img_preprocessing(Mat& image, Mat& channel, const int num_channel, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR);

    Mat Lab_image = convertToLab(image); 

    // Выделение каналов 
    vector<Mat> lab_channels(3);

    split(Lab_image, lab_channels); 

    channel = lab_channels[num_channel];
}



void calculateFeaturesWithoutMean(const cv::Mat& imagePart, double threshold,
    double& maxValue, double& minValue, double& pixelPercentage) {

    cv::minMaxLoc(imagePart, &minValue, &maxValue);

    cv::Mat mask = imagePart > threshold;
    pixelPercentage = 100.0 * cv::countNonZero(mask) / (imagePart.rows * imagePart.cols);
} 

void calculateMean(const cv::Mat& imagePart, double& L_means, double& a_means) {
    cv::Scalar Arr_means, Arr_stddev;
    meanStdDev(imagePart, Arr_means, Arr_stddev);
    L_means = Arr_means[0];
    a_means = Arr_means[1];
} 

void drawBoundingRectangles(cv::Mat& image, cv::Mat stats) {
    for (int i = 0; i < stats.rows; ++i) {
        cv::rectangle(image, cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP)),
            cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT)),
            cv::Scalar(0, 255, 0), 2);
    }
}

void filterStatsAndCentroids(const cv::Mat& stats, const cv::Mat& centroids, cv::Mat& filteredStats, cv::Mat& filteredCentroids, cv::Mat& resizedImage) {
    for (int r = 0; r < stats.rows; ++r) {
        int wight = stats.at<int>(r, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(r, cv::CC_STAT_HEIGHT);

        int wight_img = resizedImage.cols; 
        int height_img = resizedImage.rows; 

        cout << wight_img << " " << height_img << endl; 

        if (!(wight >= 3 && height >= 3) && stats.at<int>(r, cv::CC_STAT_AREA) >= 80 && stats.at<int>(r, cv::CC_STAT_AREA) <= wight_img * height_img / 3) { // 
            filteredStats.push_back(stats.row(r));
            filteredCentroids.push_back(centroids.row(r));
        }
    }
} 

void get_rectangle_for_detector(const Mat channel,cv::Mat& filteredStats,cv::Mat& filteredCentroids, cv::Mat& resized_img) {
    Mat otsu_img;
    int thresh = 0;
    int maxValue = 255;
    long double thres = cv::threshold(channel, otsu_img, thresh, maxValue, THRESH_OTSU);

  // cout << "Otsu Threshold: " << thresh << endl;
    imshow("Image after Otsu Threshold", otsu_img);

    Mat labels, stats, centroids; 
    int numLabels = cv::connectedComponentsWithStats(otsu_img, labels, stats, centroids);

    // drawBoundingRectangles(resized_img, stats); 
    //  cv::imshow("Исходное изображение с метками", resized_img); 
    cout << "Befor filtr" << stats << endl;

    filterStatsAndCentroids(stats, centroids, filteredStats, filteredCentroids, resized_img);
    cout << "Befor filtr" << filteredStats << endl; 

    drawBoundingRectangles(resized_img, filteredStats);
}

int main(int argc, char** argv)
{ 

   // Mat image = imread("C:\\testdrive1583149193.4818494.png"); 
  //  Mat image = imread("C:\\testdrive1583149150.6489925.png"); // красная 
  // Mat image = imread("C:\\testdrive1583149169.625285.png"); // красная с включёнными 
  //  Mat image = imread("C:\\testdrive1583149177.9194167.png"); // белая 
   
    Mat image = imread("C:\\red_on.png");

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    }

    int new_weight = 416; 
    int new_height = 416; 
    const int L = 0;
    const int a = 1;
    const int B = 2;

  /*  vector<string> input_folders_test = {"TRAIN/brake_TRAIN_OFF", "TRAIN/brake_ON_TRAIN"};
    vector<vector<int>> features_test;
    vector<int> labels_test; 



    for (const string& folder : input_folders_test) {
        for (const auto& entry : fs::directory_iterator{ folder }) {
            string image_path = entry.path().string(); 
            Mat img = imread(image_path);
            
            Mat channel; 

            img_preprocessing(img, channel, a, new_weight, new_height); 

            Mat filteredStats, filteredCentroids; 

            get_rectangle_for_detector(channel, filteredStats, filteredCentroids, image); 

            Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weight / 2);

         //   imshow("Resized Down by defining height and width", img);

           
            if (folder.substr(folder.size() - 3) == "OFF") {
                labels_test.push_back(0);
            }
            else {
                labels_test.push_back(1);
            }
        }
    } */ 

    Mat resized_img = image.clone();
    Mat channel;

    img_preprocessing(image, channel, a, new_weight, new_height);

    imshow("a", image);
    imshow("a channel", channel); 

       /* 
    Применение порогового значения Оцу
       */


    Mat filteredStats, filteredCentroids; 
    get_rectangle_for_detector(channel, filteredStats, filteredCentroids, image);


    Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weight / 2); 

    drawBoundingRectangles(image, lateral_stats); 
    cv::imshow("Исходное изображение с метками", image);

    cout << "lateral " << lateral_stats; 



    // Features 

   /* for (int i = 0; i < lateral_stats.rows; ++i) { 

        Mat current_rectangle = lateral_stats.row(i);
        
        cv::Rect R_i(current_rectangle.at<int>(cv::CC_STAT_LEFT), current_rectangle.at<int>(cv::CC_STAT_TOP), current_rectangle.at<int>(cv::CC_STAT_WIDTH), current_rectangle.at<int>(cv::CC_STAT_HEIGHT));
        // Область фар 

        cv::Mat croppedL = lab_channels[L](R_i).clone();
        cv::Mat croppedA = lab_channels[a](R_i).clone();

        double maxValueL, minValueL, meanValueL, pixelPercentageL, tao_L = 128, meanValueTotalL;
        double maxValuea, minValuea, meanValuea, pixelPercentagea, tao_A = 128, meanValueTotala;

        calculateFeaturesWithoutMean(croppedL, tao_L, maxValueL, minValueL, pixelPercentageL);
        calculateFeaturesWithoutMean(croppedA, tao_A, maxValuea, minValuea, pixelPercentagea);
        calculateMean(Lab_image(R_i), meanValueL, meanValuea);
        calculateMean(Lab_image, meanValueTotalL, meanValueTotala);

        cout << "Max from function " << maxValueL << " Min from function " << minValueL << " PercentageL from function " << pixelPercentageL << '\n';
        cout << "Mean_L " << meanValueL << "Mean_a " << meanValuea << '\n';
    } */
    // Wait for any keystroke in the window  
    waitKey(0);
    return 0;
}
