#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  

#include <stdio.h>

#include "brake_lights_detection.cpp"

using namespace cv;
using namespace std; 

Mat convertToLab(const Mat& image) {
    Mat Lab_image;
    cvtColor(image, Lab_image, 45);
    return Lab_image;
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

void filterStatsAndCentroids(const cv::Mat& stats, const cv::Mat& centroids, cv::Mat& filteredStats, cv::Mat& filteredCentroids, const cv::Mat& resizedImage) {
    for (int r = 0; r < stats.rows; ++r) {
        int width = stats.at<int>(r, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(r, cv::CC_STAT_HEIGHT);

        if (!(width == resizedImage.cols && height == resizedImage.rows) && !(width == 1 && height == 1) && stats.at<int>(r, cv::CC_STAT_AREA) >= 0) { // 
            filteredStats.push_back(stats.row(r));
            filteredCentroids.push_back(centroids.row(r));
        }
    }
}

int main(int argc, char** argv)
{ 

   // Mat image = imread("C:\\testdrive1583149193.4818494.png"); 
   // Mat image = imread("C:\\testdrive1583149150.6489925.png"); // красная 
    Mat image = imread("C:\\testdrive1583149177.9194167.png");

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    }

    int new_weidth = 416;
    int new_height = 416;
    Mat resized_img;
    //resize down
    resize(image, resized_img, Size(new_weidth, new_height), INTER_LINEAR);
    // let's upscale the image using new  width and height
    imshow("Resized Down by defining height and width", resized_img);

    Mat Lab_image = convertToLab(resized_img);

    // namedWindow("RGB2Lab image", cv::WINDOW_NORMAL); //WINDOW_AUTOSIZE
  //   imshow("RGB2Lab image", Lab_image); 

   // detector(Lab_image);

    // Выделение каналов 
    vector<Mat> lab_channels(3);

    split(Lab_image, lab_channels);


    const int L = 0;
    const int a = 1;
    const int B = 2;

    //   imshow("a channel", lab_channels[a]); 

       /*
    Применение порогового значения Оцу
       */

    Mat otsu_img;
    int thresh = 0;
    int maxValue = 255;
    long double thres = cv::threshold(lab_channels[a], otsu_img, thresh, maxValue, THRESH_OTSU);

    cout << "Otsu Threshold: " << thresh << endl;
    imshow("Image after Otsu Threshold", otsu_img);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(otsu_img, labels, stats, centroids);

   // drawBoundingRectangles(resized_img, stats); 

  //  cv::imshow("Исходное изображение с метками", resized_img);
    //  int k[] = { stats.row(0).col(3) };
    cout << "stats 3: " << stats.row(0).col(0) << " stats: " << stats << "centroids: " << centroids;
    // cout << "stats.row(0).col(3) " << k[0] << "resized_img.cols " << resized_img.cols;

    Mat filteredStats, filteredCentroids;
    filterStatsAndCentroids(stats, centroids, filteredStats, filteredCentroids, resized_img);
    cout << filteredStats; 

 //   drawBoundingRectangles(resized_img, filteredStats);

    cv::Mat mm = stats.row(0); 

    cout << "stats 0: " << mm.col(0) << " stats: " << centroids << endl; 

 //   Point offset = centroids.row(0) - centroids.row(1); 

  //  cout << offset

    Mat lateral_stats = detector_new(filteredStats, filteredCentroids, new_weidth / 2); 

    drawBoundingRectangles(resized_img, lateral_stats); 
    cv::imshow("Исходное изображение с метками", resized_img); 

    cout << "lateral" << lateral_stats;

    // Features 

    cv::Rect rect(0, 0, Lab_image.cols / 2, Lab_image.rows / 2); // Область фар
    cv::Mat croppedL = lab_channels[L](rect).clone();
    cv::Mat croppedA = lab_channels[a](rect).clone();

    double maxValueL, minValueL, meanValueL, pixelPercentageL, tao_L = 128, meanValueTotalL;
    double maxValuea, minValuea, meanValuea, pixelPercentagea, tao_A = 128, meanValueTotala;

    calculateFeaturesWithoutMean(croppedL, tao_L, maxValueL, minValueL, pixelPercentageL);
    calculateFeaturesWithoutMean(croppedA, tao_A, maxValuea, minValuea, pixelPercentagea);
    calculateMean(Lab_image(rect), meanValueL, meanValuea);
    calculateMean(Lab_image, meanValueTotalL, meanValueTotala);

    cout << "Max from function " << maxValueL << " Min from function " << minValueL << " PercentageL from function " << pixelPercentageL << '\n';
    cout << "Mean_L " << meanValueL << "Mean_a " << meanValuea << '\n';

    // Wait for any keystroke in the window  
    waitKey(0);
    return 0;
}
