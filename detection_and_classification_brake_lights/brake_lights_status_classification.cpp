#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  

#include <stdio.h>

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

    Mat image = imread("C:\\IMG_1675.PNG");

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    } 

    Mat Lab_image = convertToLab(image);

    namedWindow("RGB2Lab image", cv::WINDOW_NORMAL); //WINDOW_AUTOSIZE
    imshow("RGB2Lab image", Lab_image); 

    // Выделение каналов 
    vector<Mat> lab_channels(3);

    split(Lab_image, lab_channels);

    const int L = 0;
    const int a = 1;
    const int B = 2; 
    
    // Features 

    cv::Rect rect(0, 0, image.cols / 2, image.rows / 2); // Область фар
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
