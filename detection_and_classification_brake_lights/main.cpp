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
// #include "brake_lights_classification.cpp"

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

void img_preprocessing(Mat& image, vector<Mat>& lab_channels, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR);

    Mat Lab_image = convertToLab(image); 

    // Выделение каналов 
  //  vector<Mat> lab_channels(3);

    split(Lab_image, lab_channels); 

  //  channel = lab_channels[num_channel];
} 

void img_preprocessing_HSV(Mat& image, vector<Mat>& lab_channels, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR);

    Mat Lab_image = convertToHSV(image);

    // Выделение каналов 
  //  vector<Mat> lab_channels(3);

    split(Lab_image, lab_channels);

    //  channel = lab_channels[num_channel];
}



/* void calculateFeaturesWithoutMean(const cv::Mat& imagePart, double threshold,
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
} */

void drawBoundingRectangles(const cv::Mat image, const cv::Mat stats) {
    for (int i = 0; i < stats.rows; ++i) {
        cv::rectangle(image, cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP)),
            cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT)),
            cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("with rectangels", image);
}

void filterStatsAndCentroids(const cv::Mat& stats, const cv::Mat& centroids, cv::Mat& filteredStats, cv::Mat& filteredCentroids, cv::Mat& resizedImage) {
    for (int r = 0; r < stats.rows; ++r) {
        int wight = stats.at<int>(r, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(r, cv::CC_STAT_HEIGHT);

        int wight_img = resizedImage.cols; 
        int height_img = resizedImage.rows; 

        if (!(wight <= 3 && height <= 3) && !(wight == wight_img && height == height_img) && stats.at<int>(r, cv::CC_STAT_AREA) >= 80 && stats.at<int>(r, cv::CC_STAT_AREA) <= wight_img * height_img / 3) { // 
            filteredStats.push_back(stats.row(r));
            filteredCentroids.push_back(centroids.row(r));
        }
    }
} 

void get_rectangle_for_detector(const Mat channel,cv::Mat& filteredStats,cv::Mat& filteredCentroids, cv::Mat resized_img) {
    Mat otsu_img;
    int thresh = 0;
    int maxValue = 255;
    long double thres = cv::threshold(channel, otsu_img, thresh, maxValue, THRESH_OTSU);

  // cout << "Otsu Threshold: " << thresh << endl;
  //  imshow("Image after Otsu Threshold", otsu_img);

    Mat labels, stats, centroids; 
    int numLabels = cv::connectedComponentsWithStats(otsu_img, labels, stats, centroids);

  //  cout << "Befor filtr" << stats << endl;

    filterStatsAndCentroids(stats, centroids, filteredStats, filteredCentroids, resized_img);
  //  cout << "After filtr" << filteredStats << endl; 

    //drawBoundingRectangles(resized_img, filteredStats);
 //   cv::imshow("with rectangels", resized_img);
}

int main(int argc, char** argv)
{

    // Mat image = imread("C:\\testdrive1583149193.4818494.png"); 
    Mat image = imread("C:\\testdrive1583149150.6489925.png"); // красная 
   // Mat image = imread("C:\\testdrive1583149169.625285.png"); // красная с включёнными 
   //  Mat image = imread("C:\\testdrive1583149177.9194167.png"); // белая 

  //  Mat image = imread("C:\\red_off.png");

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    }

    int new_weight = 416;
    int new_height = 416;
    const int L = 0;
    int a = 1;
    const int B = 2; 

    const int h = 0;
    const int s = 1;
    const int v = 2; 

    vector<string> input_folders_test = { "TRAIN/brake_TRAIN_OFF", "TRAIN/brake_ON_TRAIN" };
    vector<vector<int>> features_test;
    vector<int> labels_test;

    int detecting_light_on = 0; 
    int detecting_light_HSV = 0; 
    int total = 0;
    
    for (const string& folder : input_folders_test) {
        for (const auto& entry : fs::directory_iterator{ folder }) {
            string image_path = entry.path().string();
            Mat img = imread(image_path);

            vector<Mat> lab_channels(3);

            img_preprocessing(img, lab_channels, new_weight, new_height);

            Mat filteredStats, filteredCentroids;

            get_rectangle_for_detector(lab_channels[a], filteredStats, filteredCentroids, image);

            Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weight / 2);

         //   imshow("Resized Down by defining height and width", img); 

            Mat lateral_stats_HSV; 

            if (lateral_stats.rows < 2) {

                vector<Mat> HSV_channels(3);

                img_preprocessing_HSV(img, HSV_channels, new_weight, new_height);

                Mat filteredStatsHSV, filteredCentroidsHSV;

                get_rectangle_for_detector(HSV_channels[v], filteredStatsHSV, filteredCentroidsHSV, image);

                lateral_stats_HSV = detector_new(filteredStatsHSV, filteredCentroidsHSV, new_weight / 2); 

            } 

            if (folder.substr(folder.size() - 3) == "OFF") {
                labels_test.push_back(0);
            }
            else {
                labels_test.push_back(1); 
                if (lateral_stats.rows >= 2)
                    detecting_light_on++; 
                if (lateral_stats_HSV.rows >= 2)
                    detecting_light_HSV++; 
                total++;
            }
        }
    } 

    cout << endl << "detecting_light_on " << detecting_light_on << " detecting_light_HSV " << detecting_light_HSV << " total " << total << endl; 

    vector<vector<double>> data_l, data_r, data_third;

    Mat resized_img = image.clone();
    vector<Mat> lab_channels(3);

    img_preprocessing(image, lab_channels, new_weight, new_height); 

    imshow("a channel", lab_channels[a]);

    /*
 Применение порогового значения Оцу
    */


    Mat filteredStats, filteredCentroids;

    get_rectangle_for_detector(lab_channels[a], filteredStats, filteredCentroids, image);

  //  cv::imshow("Исходное изображение с метками", image);


    Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weight / 2);

    if (lateral_stats.rows < 2) {

        vector<Mat> HSV_channels(3);

        img_preprocessing_HSV(image, HSV_channels, new_weight, new_height);

        Mat filteredStatsHSV, filteredCentroidsHSV;

        get_rectangle_for_detector(HSV_channels[2], filteredStatsHSV, filteredCentroidsHSV, image);

        Mat stats_ = detector_new(filteredStatsHSV, filteredCentroidsHSV, new_weight / 2);
        drawBoundingRectangles(image, stats_);
        cv::imshow("Исходное изображение с метками", image);
        cout << "ststs_" << stats_ << endl;

    }



    cout << "lateral " << lateral_stats << endl; 

    drawBoundingRectangles(image, lateral_stats);
 //   cv::imshow("lateral_stats", image);

    //  classifier_get_features(data_l, data_r, data_third, filteredStats, lab_channels, image); 

    cout << data_l.size() << endl;

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