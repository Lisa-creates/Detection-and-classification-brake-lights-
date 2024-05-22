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


const int TAO_V = 60;
const float TAO_S = 0.3;
const float TAO_TB = 0.7;
const double LAMBDA_S = 0.3;
const double LAMBDA_D = 0.4;


int calculateTotalArea(const cv::Mat stats, const cv::Mat centroids, int tao_v) { 

    int total_sum = 0;

    for (int i = 0; i < stats.rows; ++i) {
        for (int j = 0; j < stats.rows; ++j) {
            if (i != j && abs(centroids.row(i).at<double>(1) - centroids.row(j).at<double>(1)) < tao_v) {
                total_sum += stats.row(i).at<int>(4); // Площадь прямоугольника из структуры stats 
                break; 
            } 
        }
    } 

   // cout << " total_sum " << total_sum << endl;

    return total_sum;
} 



void rect_in_center(Rect& rect2, Mat centroids_i, Mat centroids_j) {
    rect2.x += (centroids_i.at<double>(0) - centroids_j.at<double>(0));
    rect2.y += (centroids_i.at<double>(1) - centroids_j.at<double>(1));
}



double calculate_I_S(cv::Mat stats_i, cv::Mat stats_j, double tao_S, cv::Mat centroids_i, cv::Mat centroids_j) {
    cv::Rect R_i(stats_i.at<int>(cv::CC_STAT_LEFT), stats_i.at<int>(cv::CC_STAT_TOP), stats_i.at<int>(cv::CC_STAT_WIDTH), stats_i.at<int>(cv::CC_STAT_HEIGHT));
    cv::Rect R_j(stats_j.at<int>(cv::CC_STAT_LEFT), stats_j.at<int>(cv::CC_STAT_TOP), stats_j.at<int>(cv::CC_STAT_WIDTH), stats_j.at<int>(cv::CC_STAT_HEIGHT));
    
    rect_in_center(R_j, centroids_i, centroids_j);
    
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
  //  cout << "(area_i + area_j) " << (area_i + area_j) << " total_sum " << total_sum << endl;

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
    if (I_U == 0)
        return - 1; 
    return I_U; 
} 

double calculate_I_lb(const cv::Mat& stats_i, const cv::Mat& centroids_i, const cv::Mat& stats_j, const cv::Mat& centroids_j, float half_img_weight, double lambda_S, double lambda_D, double lambda_U, float tao_S, int total_sum_rectangles) {
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

        cout  << "I_S " << I_S << " I_D " << I_D << " I_U " << I_U << endl; 

        double I_lb = lambda_S * I_S + 2 * lambda_D * I_D + lambda_U * I_U;
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

// Поиск индекса боковых фар 

vector<int> FindLateralBrakeLight(int half_img_weight, const cv::Mat& stats, const cv::Mat& centroids, double lambda_S, double lambda_D, double lambda_U, int tao_v, float tao_S) {

    double max_ij = -1.0;
    /*int tao_v = 60;
    float tao_S = 0.3;
    float tao_tb = 0.7;
    double lambda_S = 0.3;
    double lambda_D = 0.4;
    double lambda_U = 1 - lambda_S - lambda_D;
*/
    vector<int> vector_max = { 0, 0 };

    float img_height = half_img_weight * 2;

    int total_sum = calculateTotalArea(stats, centroids, tao_v);
    // cout << "total_sum" << total_sum;
    //  cout << "centroids " << centroids; 


    for (int i = 0; i < stats.rows; ++i) { 
        for (int j = i + 1; j < stats.rows; ++j) {
            if (i != j) {
                if (abs(centroids.row(i).at<double>(1) - centroids.row(j).at<double>(1)) < tao_v && abs(stats.row(i).at<int>(0) - stats.row(j).at<int>(0)) >= max(stats.row(i).at<int>(2), stats.row(j).at<int>(2))
                    && (stats.row(i).at<int>(1) >= (img_height / 5) && stats.row(i).at<int>(1) <= (4 * img_height) / 5) && (stats.row(j).at<int>(1) >= (img_height / 5) && stats.row(j).at<int>(1) <= (4 * img_height) / 5)) {
                    double I_lb = calculate_I_lb(stats.row(i), centroids.row(i), stats.row(j), centroids.row(j), half_img_weight, lambda_S, lambda_D, lambda_U, tao_S, total_sum); // может сгенерировать исключение
                    if ((I_lb > max_ij || max_ij == -1) ) {
                        max_ij = I_lb;
                        vector_max[0] = i;
                        vector_max[1] = j;
                    }
                }
            }
        }
    }

    return vector_max;
}

// Поиск индекса третьей фар 

int findThirdBrakeLight(const cv::Mat& stats, const cv::Mat& centroids, const std::vector<int>& vector_max, float tao_tb) {
    int index_third_light = -1; 
    double max_I_tb = -1.0; 
    int index_l_light = vector_max[0], index_r_light = vector_max[1];
     
    for (int i = 0; i < stats.rows; ++i) {
        if (i != index_l_light && i != index_r_light) {
            if ((centroids.row(i).at<double>(1) <= (centroids.row(index_l_light).at<double>(1) + centroids.row(index_r_light).at<double>(1)) / 2) && ((stats.row(index_l_light).at<int>(4) + stats.row(index_r_light).at<int>(4)) >= stats.row(i).at<int>(4))
                && (stats.row(i).at<int>(2) / stats.row(i).at<int>(3)) > 1) 
                {
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


void filterStatsAndCentroids(const cv::Mat& stats, const cv::Mat& centroids, cv::Mat& filteredStats, cv::Mat& filteredCentroids, cv::Mat& resizedImage) {

    int wight_img = resizedImage.cols;
    int height_img = resizedImage.rows;

    for (int r = 1; r < stats.rows; ++r) { // с 1, потому что 0 - это фон 
        int wight = stats.at<int>(r, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(r, cv::CC_STAT_HEIGHT);



        if (!(wight <= 16 && height <= 16) && !(wight == wight_img && height == height_img) && stats.at<int>(r, cv::CC_STAT_AREA) >= 110 && stats.at<int>(r, cv::CC_STAT_AREA) <= height_img * 24
            && stats.row(r).at<int>(1) + height <= (5 * height_img) / 6) { // 
            filteredStats.push_back(stats.row(r));
            filteredCentroids.push_back(centroids.row(r));
        }
    }
}

void get_rectangle_for_detector(const Mat channel, cv::Mat& filteredStats, cv::Mat& filteredCentroids, cv::Mat resized_img) {
    Mat otsu_img;
    Mat bl;
    int maxValue = 255;
    double thresh = cv::threshold(channel, otsu_img, 0, maxValue, THRESH_TRIANGLE); //  THRESH_OTSU

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


Mat get_brake_light(Mat stats, vector<int> vector_max, int index_third_light) {
    Mat brake_light;

    if (vector_max[0] != vector_max[1]) {
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

Mat convertToHSV__(const Mat& image) {
    Mat Lab_image = image.clone();
    cvtColor(image, Lab_image, 41);
    return Lab_image;
}

void img_preprocessing_HSV__(Mat& image, vector<Mat>& HSV_channels) {

    resize(image, image, Size(416, 416), INTER_LINEAR);
    Mat HSV_image = convertToHSV__(image);

    split(HSV_image, HSV_channels);
}

Mat detector_new(float half_img_weight, vector<Mat> lab_channels, Mat img, Mat orig_img, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S, float tao_tb, int channel) {

    double max_ij = -1.0;
    /*int tao_v = 60;
    float tao_S = 0.3;
    float tao_tb = 0.7;
    double lambda_S = 0.3;
    double lambda_D = 0.4;
    double lambda_U = 1 - lambda_S - lambda_D;
*/
// vector<int> vector_max = { 0, 0 };

 // const int a = 1;

    Mat stats, centroids;

    get_rectangle_for_detector(lab_channels[channel], stats, centroids, img);

    float img_height = half_img_weight * 2;

    int total_sum = calculateTotalArea(stats, centroids, tao_v);
    // cout << "total_sum" << total_sum;
    //  cout << "centroids " << centroids; 


    vector<int> vector_max = FindLateralBrakeLight(half_img_weight, stats, centroids, lambda_S, lambda_D, lambda_U, tao_v, tao_S);
    
    Mat brake_light; 
    
    if (vector_max[0] != vector_max[1]) {
        int index_third_light = findThirdBrakeLight(stats, centroids, vector_max, tao_tb);

        brake_light = get_brake_light(stats, vector_max, index_third_light); 
    }
    
    if (brake_light.rows < 3) {

        vector<Mat> HSV_channels(3);

        const int v = 2; 

        img_preprocessing_HSV__(orig_img, HSV_channels);
        get_rectangle_for_detector(HSV_channels[v], stats, centroids, orig_img);

        if (brake_light.rows < 2)
            vector_max = FindLateralBrakeLight(half_img_weight, stats, centroids, lambda_S, lambda_D, lambda_U, tao_v, tao_S);
        
        if (vector_max[0] != vector_max[1]) {
            int index_third_light = findThirdBrakeLight(stats, centroids, vector_max, tao_tb);

            brake_light = get_brake_light(stats, vector_max, index_third_light);
        }
    }

    return brake_light;
}

