#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  

#include <stdio.h>


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    Mat image = imread("_.jpg");

    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press 
        return -1;
    }

    // Перевод в Lab 
    Mat RGB2Lab_image; 
    cvtColor(image, RGB2Lab_image, 45); // RGB2Lab

    namedWindow("RGB2Lab image", WINDOW_AUTOSIZE);
    imshow("RGB2Lab image", RGB2Lab_image); 

    // Выделение каналов 
    vector<Mat> lab_channels(3);

    split(RGB2Lab_image, lab_channels);

    const int L = 0;
    const int a = 1;
    const int B = 2; 
    
    // Features 

    double L_min, L_max, a_min, a_max; // maximum value of pixels in the L and a channel 
    Point L_minIdx, L_maxIdx, a_minIdx, a_maxIdx; // (u_i, v_i)
    minMaxLoc(lab_channels[L], &L_max, &L_min, &L_minIdx, &L_minIdx); 
    minMaxLoc(lab_channels[a], &a_max, &a_min, &a_minIdx, &a_maxIdx); 
    
    // Wait for any keystroke in the window  
    waitKey(0);
    return 0; 
}
