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

int main(int argc, char** argv)
{

 //   Mat image = imread("C:\\IMG_1675.PNG"); 
    Mat image = imread("C:\\testdrive1583149193.4818494.png"); 

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    } 

    int new_width = 416;
    int new_height = 416;
    Mat resized_img;
    //resize down
    resize(image, resized_img, Size(new_width, new_height), INTER_LINEAR);
    // let's upscale the image using new  width and height
    imshow("Resized Down by defining height and width", resized_img);

    Mat Lab_image = convertToLab(resized_img);

   // namedWindow("RGB2Lab image", cv::WINDOW_NORMAL); //WINDOW_AUTOSIZE
 //   imshow("RGB2Lab image", Lab_image); 
    
    detector(Lab_image); 

    // Выделение каналов 
    vector<Mat> lab_channels(3);

    split(Lab_image, lab_channels);


    const int L = 0;
    const int a = 1;
    const int B = 2; 

 //   imshow("a channel", lab_channels[a]); 

    Mat otsu_img;
    double thresh = 0;
    double maxValue = 255;

    long double thres = cv::threshold(lab_channels[a], otsu_img, thresh, maxValue, THRESH_OTSU);

    cout << "Otsu Threshold: " << thres << endl; 

    imshow("Image after Otsu Threshold", otsu_img); 

    // Применение анализа связанных компонентов
    vector<double> labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(otsu_img, labels, stats, centroids);

 /*   for (int i = 1; i < numLabels; ++i) {
        cv::rectangle(resized_img, cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP)),
            cv::Point(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT)),
            cv::Scalar(0, 255, 0), 2);
    }*/

    // Отображение исходного изображения с метками
    cv::imshow("Исходное изображение с метками", resized_img); 

    cout << "labels: "  << " stats: " << stats << "centroids: " << centroids;

    // Отображение исходного и выделенного изображения
   // cv::imshow("Изображение с метками", outputImage); 

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
