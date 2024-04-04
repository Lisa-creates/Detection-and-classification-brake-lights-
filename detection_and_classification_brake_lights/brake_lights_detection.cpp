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



double calculate_I_S(cv::Mat stats_i, cv::Mat stats_j, double tao_S) {
    cv::Rect R_i(stats_i.at<int>(cv::CC_STAT_LEFT), stats_i.at<int>(cv::CC_STAT_TOP), stats_i.at<int>(cv::CC_STAT_WIDTH), stats_i.at<int>(cv::CC_STAT_HEIGHT));
    cv::Rect R_j(stats_j.at<int>(cv::CC_STAT_LEFT), stats_j.at<int>(cv::CC_STAT_TOP), stats_j.at<int>(cv::CC_STAT_WIDTH), stats_j.at<int>(cv::CC_STAT_HEIGHT));
    Rect intersection = R_i & R_j;
    Rect unionRect = R_i | R_j;
    if (unionRect.area() != 0) {
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

        if (I_S >= tao_S)
            return I_S;
        else
            return 0; 
    }
    else
        return 0;
} 

double calculate_I_D(int area_i, int area_j, int total_sum) {
    double I_D = (area_i + area_j) / total_sum;
    cout << "I_D " << I_D << endl; 
    return I_D; 
} 

double calculate_I_U(int u_R_i, int u_R_j, double half_img_weight) {
    double numerator = min(half_img_weight - u_R_i, u_R_j - half_img_weight);
    double denominator = max(half_img_weight - u_R_i, u_R_j - half_img_weight); 
   // cout << "numerator " << numerator << "denominator " << denominator << endl; 
    double I_U = max(0.0, numerator / denominator);
    return I_U;
} 

double calculate_I_lb(const cv::Mat& stats_i, const cv::Mat& centroids_i, const cv::Mat& stats_j, const cv::Mat& centroids_j, bool& right_and_left, float half_img_weight, double lambda_S, double lambda_D, double lambda_U, double tao_S, int total_sum_rectangles) {
    const double EPSILON = 1e-6; // Погрешность для сравнения суммы с 1.0

    if (abs(lambda_S + lambda_D + lambda_U - 1.0) > EPSILON || lambda_U < 0) {
 //   if (lambda_S + lambda_D + lambda_U != 1.0 || lambda_U < 0) {// 
        cout << "WrongLambdaValue sum: " << lambda_S + lambda_D + lambda_U << endl << lambda_U;
        return -1.0;
    }
    else { 
        double I_S = calculate_I_S(stats_i, stats_j, tao_S);
        double I_D = calculate_I_D(stats_i.at<int>(4), stats_j.at<int>(4), total_sum_rectangles);
        double I_U = 0;
        if (stats_i.at<int>(0) < stats_i.at<int>(0)) {
            double I_U = calculate_I_U(centroids_i.at<double>(0), centroids_j.at<double>(0), half_img_weight);
        }
        else {
       //     right_and_left = true; 
            double I_U = calculate_I_U(centroids_j.at<double>(0), centroids_i.at<double>(0), half_img_weight);
        }
        double I_lb = lambda_S * I_S + lambda_D * I_D + lambda_U * I_U; 
        cout << "I_lb " << I_lb << endl;
        return I_lb;
    }
} 

Mat detector_new(const cv::Mat& stats, const cv::Mat& centroids, float half_img_weight) {

    double max_ij = -1.0;
    int tao_v = 60;
    double tao_S = 0.3; // Добавить проверку на  >= tao_S
    double lambda_S = 0.3;
    double lambda_D = 0.4;
    double lambda_U = 1 - lambda_S - lambda_D;

    vector<int> vector_max = { 0, 0 };

    int total_sum = 0;

    for (int r = 1; r < stats.rows; ++r) {
        total_sum += stats.row(r).at<int>(4); // Площадь прямоугольника из структуры stats
    }

    bool right_and_left = false;

    for (int i = 0; i < stats.rows; ++i) {
        for (int j = i + 1; j < stats.rows; ++j) {
            if (i != j) {
                if (abs(centroids.row(i).at<double>(1) - centroids.row(j).at<double>(1)) < tao_v) {
                    //  c0out << "I'm here"; 
                    double I_lb = calculate_I_lb(stats.row(i), centroids.row(i), stats.row(j), centroids.row(j), right_and_left, half_img_weight, lambda_S, lambda_D, lambda_U, tao_S, total_sum); // может сгенерировать исключение
                    if (I_lb > max_ij || max_ij == -1) {
                        max_ij = I_lb;
                        vector_max[0] = i;
                        vector_max[1] = j;
                    }
                }
            }
        }
    }

    Mat lateral_stats;

    if (max_ij != -1) {
        lateral_stats.push_back(stats.row(vector_max[0]));
        lateral_stats.push_back(stats.row(vector_max[1]));
    }
    return lateral_stats;

}