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


int calculateTotalArea(const cv::Mat& stats) {
    int total_sum = 0;

    for (int r = 0; r < stats.rows; ++r) {
        total_sum += stats.row(r).at<int>(4); // Площадь прямоугольника из структуры stats
    }

    return total_sum;
} 



void rect_in_center(cv::Rect& rect1, cv::Rect& rect2, const cv::Mat& centroids_i, const cv::Mat& centroids_j) {

    cv::Point center1(centroids_i.at<double>(0), centroids_j.at<double>(1));
    cv::Point center2(centroids_j.at<double>(0), centroids_j.at<double>(1));

    cv::Point diff(center1 - center2);

    rect2.x += diff.x;
    rect2.y += diff.y;

}



double calculate_I_S(cv::Mat stats_i, cv::Mat stats_j, double tao_S, const cv::Mat& centroids_i, const cv::Mat& centroids_j) {
    cv::Rect R_i(stats_i.at<int>(cv::CC_STAT_LEFT), stats_i.at<int>(cv::CC_STAT_TOP), stats_i.at<int>(cv::CC_STAT_WIDTH), stats_i.at<int>(cv::CC_STAT_HEIGHT));
    cv::Rect R_j(stats_j.at<int>(cv::CC_STAT_LEFT), stats_j.at<int>(cv::CC_STAT_TOP), stats_j.at<int>(cv::CC_STAT_WIDTH), stats_j.at<int>(cv::CC_STAT_HEIGHT));
    
    rect_in_center(R_i, R_j, centroids_i, centroids_j);
    
    Rect intersection = R_i & R_j;
    Rect unionRect = R_i | R_j; 
    // cout << "intersection " << intersection << "unionRect " << unionRect << endl;
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
            return -1; 
    }
    else
        return -1;
}

double calculate_I_D(int area_i, int area_j, float total_sum) {
    double I_D = (area_i + area_j) / total_sum;

  //  cout << "I_D " << (area_i + area_j) / total_sum << endl;  

    return I_D; 
} 

double calculate_I_U(double u_R_i, double u_R_j, double half_img_weight) {
    double numerator = min((half_img_weight - u_R_i), (u_R_j - half_img_weight));
    double denominator = max((half_img_weight - u_R_i), (u_R_j - half_img_weight)); 

  //  cout << "half_img_weight " << half_img_weight << "  u_R_i " << u_R_i << " u_R_j " << u_R_j;
  //  cout << "numerator " << numerator << " denominator " << denominator << " numerator / denominator " << numerator / denominator << endl; 
    
    if (denominator == 0.0 || numerator < 0.0 || denominator < 0.0)
        return -1; 

  //  << " numerator / denominator " << numerator / denominator << endl;
  //  cout << "b " << b; 

    double I_U = max(0.0, numerator / denominator);
    return I_U; 
} 

double calculate_I_lb(const cv::Mat& stats_i, const cv::Mat& centroids_i, const cv::Mat& stats_j, const cv::Mat& centroids_j, bool& right_and_left, float half_img_weight, double lambda_S, double lambda_D, double lambda_U, float tao_S, int total_sum_rectangles) {
    const double EPSILON = 1e-6; // Погрешность для сравнения суммы с 1.0

    if (abs(lambda_S + lambda_D + lambda_U - 1.0) > EPSILON || lambda_U < 0) {
        cout << "WrongLambdaValue sum: " << lambda_S + lambda_D + lambda_U << endl << lambda_U;
        return -1.0;
    }
    else {
        double I_S = calculate_I_S(stats_i, stats_j, tao_S, centroids_i, centroids_j);

        if (I_S == -1)
            return -1;

        double I_D = calculate_I_D(stats_i.at<int>(4), stats_j.at<int>(4), float(total_sum_rectangles));
      //  cout << endl << stats_i.at<int>(4) << "  " << stats_j.at<int>(4) <<"   " << float(total_sum_rectangles) << endl;
        double I_U = 0; 
     //   cout << endl << stats_j << "  " << stats_i << "   " << endl;
     //   cout << endl << " centroids_j.at<double>(0) " << centroids_j.at<double>(0) << "centroids_i.at<double>(0) " << centroids_i.at<double>(0); 
        if (stats_i.at<int>(0) < stats_j.at<int>(0)) {
            I_U = calculate_I_U(centroids_i.at<double>(0), centroids_j.at<double>(0), half_img_weight);
        }
        else {
            //    right_and_left = true; 
            I_U = calculate_I_U(centroids_j.at<double>(0), centroids_i.at<double>(0), half_img_weight);
        } 

        if (I_U == -1)
            return -1; 
    //    cout << "I_S " << I_S << " I_D " << I_D << " I_U " << I_U << endl; 

        double I_lb = lambda_S * I_S + lambda_D * I_D + lambda_U * I_U;
     //   cout << "I_lb " << I_lb << endl;
        return I_lb;
    }
} 


double calculate_I_tb(double u_k, double u_l, double u_r) {
    double numerator = min(u_k, (u_l + u_r) / 2);
    double denominator = max(u_k, (u_l + u_r) / 2);
    // cout << "numerator " << numerator << "denominator " << denominator << endl; 
    double I_tb = numerator / denominator; 
    return I_tb;
} 



int findThirdBrakeLight(const cv::Mat& stats, const cv::Mat& centroids, const std::vector<int>& vector_max, double tao_tb) {
    int index_third_light = -1; 
    double max_I_tb = -1.0; 
    int index_l_light = vector_max[0], index_r_light = vector_max[1];
     
    for (int i = 0; i < stats.rows; ++i) {
        if (i != index_l_light && i != index_r_light) {
            if ((centroids.row(i).at<double>(1) <= (centroids.row(index_l_light).at<double>(1) + centroids.row(index_r_light).at<double>(1)) / 2) && ((stats.row(index_l_light).at<int>(4) + stats.row(index_r_light).at<int>(4)) >= stats.row(i).at<int>(4))) {
                double I_tb = calculate_I_tb(centroids.row(i).at<double>(0), centroids.row(index_l_light).at<double>(0), centroids.row(index_r_light).at<double>(0));
             //   cout << "I_tb" << I_tb << endl;
                if (I_tb > tao_tb && (I_tb > max_I_tb || index_third_light == -1)) {
                    max_I_tb = I_tb;
                    index_third_light = i; 
                }
            }
        }
    }
  //  cout << "I_tb" << max_I_tb << endl;
    return index_third_light;
} 


Mat detector_new(const cv::Mat& stats, const cv::Mat& centroids, float half_img_weight) {

    double max_ij = -1.0;
    int tao_v = 60;
    float tao_S = 0.3;
    float tao_tb = 0.7;
    double lambda_S = 0.4;
    double lambda_D = 0.3;
    double lambda_U = 1 - lambda_S - lambda_D;

    vector<int> vector_max = { 0, 0 };

    float img_height = half_img_weight * 2; 

    int total_sum = calculateTotalArea(stats);
   // cout << "total_sum" << total_sum;
   //  cout << "centroids " << centroids; 
    bool right_and_left = false;

 

    for (int i = 0; i < stats.rows; ++i) {
        for (int j = i + 1; j < stats.rows; ++j) {
            if (i != j) {
                if (abs(centroids.row(i).at<double>(1) - centroids.row(j).at<double>(1)) < tao_v) {
                    double I_lb = calculate_I_lb(stats.row(i), centroids.row(i), stats.row(j), centroids.row(j), right_and_left, half_img_weight, lambda_S, lambda_D, lambda_U, tao_S, total_sum); // может сгенерировать исключение
                    if ((I_lb > max_ij || max_ij == -1) && (stats.row(i).at<int>(1) >= (img_height / 5) && stats.row(i).at<int>(1) <= (4 *img_height) / 5) && (stats.row(j).at<int>(1) >= (img_height / 5) && stats.row(j).at<int>(1) <= (4 * img_height) / 5)) {
                        max_ij = I_lb;
                        vector_max[0] = i;
                        vector_max[1] = j;
                    }
                }
            }
        }
    }

    int index_third_light = findThirdBrakeLight(stats, centroids, vector_max, tao_tb);

    Mat brake_light;

    if (max_ij != -1) { 
        if (stats.row(vector_max[0]).at<int>(0) < stats.row(vector_max[1]).at<int>(0)) {
            brake_light.push_back(stats.row(vector_max[0]));
            brake_light.push_back(stats.row(vector_max[1]));
        }
        else {
            brake_light.push_back(stats.row(vector_max[1])); 
            brake_light.push_back(stats.row(vector_max[0])); 
        }
    }
    if (index_third_light != -1)
        brake_light.push_back(stats.row(index_third_light));

    return brake_light; 

} 

