#include <opencv2/opencv.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  
#include <vector> 

#include <stdio.h>

using namespace cv;
using namespace std;



double calculate_I_S(Rect R_i, Rect R_j) {
    Rect intersection = R_i & R_j;
    Rect unionRect = R_i | R_j;

    double I_S = float(intersection.area()) / float(unionRect.area());

    /*   if (intersection.area() > 0) {
            std::cout << "Intersection found: " << intersection << std::endl;
            cout << intersection.area() << std::endl;
        }
        else {
            std::cout << "No intersection found 1" << std::endl;
        }

        if (unionRect.area() > 0) {
            std::cout << "Union found: " << unionRect << std::endl;
            cout << unionRect.area() << std::endl;
        }
        else {
            std::cout << "No unionRect found 1" << std::endl;
        }

        cout << "I_S " << I_S << std::endl; */

    return I_S;
} 

double calculate_I_D(Rect R_i, Rect R_j, vector<Rect> R) {
    double total_sum = 0;
    for (auto& rectangle : R)
        total_sum += rectangle.area(); 
    double I_D = (R_i.area() + R_j.area()) / total_sum; 
    cout << "I_D " << I_D << endl; 
    return I_D; 
} 

double calculate_I_U(double u_R_i, double u_R_j, double half_img_weight) {
    double numerator = min(half_img_weight - u_R_i, u_R_j - half_img_weight);
    double denominator = max(half_img_weight - u_R_i, u_R_j - half_img_weight); 
   // cout << "numerator " << numerator << "denominator " << denominator << endl; 
    double I_U = max(0.0, numerator / denominator);
    return I_U;
} 

double calculate_I_lb(Rect Ri, Rect Rj, double half_img_weight, double lambda_S, double lambda_D, double lambda_U, double tao_S, vector<cv::Rect> rectangles) {
    if (lambda_S + lambda_D + lambda_U != 1 || lambda_U < 0) {
        cout << "WrongLambdaValue"; 
        return 1.0;
    }
   
    else {
        cv::Rect Ci(0, 0, 1, 10); // Центр прямоугольников  
        cv::Rect Cj(1, 0, 2, 30); // !!!!!!! 
        double I_S = calculate_I_S(Ri, Rj);
        double I_D = calculate_I_D(Ri, Rj, rectangles);
        double I_U = 0;
        if (Ri.x < Rj.x) {
            double I_U = calculate_I_U(Ci.x, Cj.x, half_img_weight);
        }
        double I_lb = lambda_S * I_S + lambda_D * I_D + lambda_U * I_U;
      //  cout << "I_lb " << I_lb << endl;
        return I_lb;
    }
} 



void detector(const Mat& image) {
    cv::Rect Ri(0, 0, image.cols / 3, image.rows / 3);
    cv::Rect Rj(1, 0, image.cols / 2, image.rows / 2); // Области фар 
    double tao_S = 0; // Добавить проверку на  >= tao_S
    double lambda_S = 0.2; 
    double lambda_D = 0.3; 
    double lambda_U = 1 - lambda_S - lambda_D; 
    double half_img_weight = 600; 
    std::vector<cv::Rect> rectangles = {
     cv::Rect(10, 10, 100, 50),
     cv::Rect(50, 30, 80, 70),
     cv::Rect(20, 40, 60, 30)
    };
    double I_lb = calculate_I_lb(Ri,Rj, half_img_weight, lambda_S, lambda_D, lambda_U, tao_S, rectangles);  // может сгенерировать исключение
} 



