#include "Header_files/brake_lights_detection.h"

int calculate_total_area(const Mat& stats, const Mat& centroids, int tao_v); 
void rect_in_center(Rect& rect2, const Mat& centroids_i, const Mat& centroids_j);
double calculate_I_S(const Mat& stats_i, const Mat& stats_j, double tao_S, const Mat& centroids_i, const Mat& centroids_j); 
double calculate_I_D(int area_i, int area_j, float total_sum);  

/**
 * \brief Finds lateral brake lights (left and right)
 *
 * \param half_img_weight Half of the image weight
 * \param stats Candidates for brake lights 
 * \param centroids Coordinates of the centers of candidate rectangles 
 * \param lambda_S Weight for I_S
 * \param lambda_D Weight for I_D
 * \param lambda_U Weight for I_U
 * \param tao_v Threshold 
 * \param tao_S Threshold 
 *
 * \return Vector with indices of found lateral brake lights, {0, 0}, if not found 
 */
vector<int> find_lateral_brake_light(float half_img_weight, const Mat& stats, const Mat& centroids, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S); 

/**
 * \brief Finds third brake light
 *
 * \param stats Candidates for brake lights 
 * \param centroids  Coordinates of the centers of candidate rectangles 
 * \param vector_max Indices of found lateral brake lights
 * \param tao_tb Threshold for third brake light
 *
 * \return Index of found third brake light, -1 if not found
 */
int find_third_brake_light(const Mat& stats, const Mat& centroids, const vector<int>& vector_max, float tao_tb); 

/**
 * \brief Filters out rectangles that are not potential brake lights
 *
 * \param stats Candidates for brake lights 
 * \param centroids Coordinates of the centers of candidate rectangles 
 * \param filteredStats Filtered candidates for brake lights 
 * \param filteredCentroids Filtered coordinates of the centers of candidate rectangles 
 * \param resizedImage Resized image
 */
void filter_rectangels(const Mat& stats, const Mat& centroids, Mat& filteredStats, Mat& filteredCentroids, const Mat& resizedImage); 

/**
 * \brief Finds the brake light in an image
 *
* \param img The image.
* \param lambda_S The lambda value for the S channel.
* \param lambda_D The lambda value for the D channel.
* \param lambda_U The lambda value for the U channel.
* \param tao_v The threshold for the V channel.
* \param tao_S The threshold for the S channel.
* \param tao_tb The threshold for the third brake light.
 *
 * \return A Mat containing the brake light
 */
Mat detector(const vector<Mat>& lab_channels, Mat& img, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S, float tao_tb); 


int calculate_total_area(const Mat& stats, const Mat& centroids, int tao_v) { 

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



void rect_in_center(Rect& rect2, const Mat& centroids_i, const Mat& centroids_j) {
    rect2.x += (centroids_i.at<double>(0) - centroids_j.at<double>(0));
    rect2.y += (centroids_i.at<double>(1) - centroids_j.at<double>(1));
}  



double calculate_I_S(const Mat& stats_i, const Mat& stats_j, double tao_S, const Mat& centroids_i, const Mat& centroids_j) {
    Rect R_i(stats_i.at<int>(cv::CC_STAT_LEFT), stats_i.at<int>(cv::CC_STAT_TOP), stats_i.at<int>(cv::CC_STAT_WIDTH), stats_i.at<int>(cv::CC_STAT_HEIGHT));
    Rect R_j(stats_j.at<int>(cv::CC_STAT_LEFT), stats_j.at<int>(cv::CC_STAT_TOP), stats_j.at<int>(cv::CC_STAT_WIDTH), stats_j.at<int>(cv::CC_STAT_HEIGHT));
    
    rect_in_center(R_j, centroids_i, centroids_j); 
    Rect union_R = R_i | R_j; 
    // cout << "intersection " << intersection << "unionRect " << unionRect << endl;
    if (union_R.area() != 0) {
        Rect intersection = R_i & R_j;
        double I_S = float(intersection.area()) / float(union_R.area());

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

double calculate_I_U(double u_R_i, double u_R_j, float half_img_weight) {
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

double calculate_I_lb(const Mat& stats_i, const Mat& centroids_i, const Mat& stats_j, const Mat& centroids_j, float half_img_weight, double lambda_S, double lambda_D, double lambda_U, float tao_S, int total_sum_rectangles) {
    const double EPSILON = 1e-6; // Погрешность для сравнения суммы с 1.0

    if (abs(lambda_S + lambda_D + lambda_U - 1.0) > EPSILON || lambda_U < 0) {
        cout << "WrongLambdaValue sum: " << lambda_S + lambda_D + lambda_U << endl << lambda_U;
        return -1;
    }
    else {

        double I_S = calculate_I_S(stats_i, stats_j, tao_S, centroids_i, centroids_j);
        if (I_S == -1)
            return -1;

        double I_D = calculate_I_D(stats_i.at<int>(4), stats_j.at<int>(4), float(total_sum_rectangles)); 
        
        if (I_D == 1)
            return -2; 
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

        // cout  << "I_S " << I_S << " I_D " << I_D << " I_U " << I_U << endl; 

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



vector<int> find_lateral_brake_light(float half_img_weight, const Mat& stats, const Mat& centroids, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S) {

    double max_ij = -1.0;

    vector<int> vector_max = { 0, 0 };

    float img_height = half_img_weight * 2;

    int total_sum = calculate_total_area(stats, centroids, tao_v);
    // cout << "total_sum" << total_sum;
    //  cout << "centroids " << centroids; 


    for (int i = 0; i < stats.rows; ++i) { 
        for (int j = i + 1; j < stats.rows; ++j) {  
            if (abs(centroids.row(i).at<double>(1) - centroids.row(j).at<double>(1)) < tao_v && abs(stats.row(i).at<int>(0) - stats.row(j).at<int>(0)) >= max(stats.row(i).at<int>(2), stats.row(j).at<int>(2))
                && (stats.row(i).at<int>(1) >= (img_height / 5) && stats.row(i).at<int>(1) <= (4 * img_height) / 5) && (stats.row(j).at<int>(1) >= (img_height / 5) && stats.row(j).at<int>(1) <= (4 * img_height) / 5)) {
                double I_lb = calculate_I_lb(stats.row(i), centroids.row(i), stats.row(j), centroids.row(j), half_img_weight, lambda_S, lambda_D, lambda_U, tao_S, total_sum); // может сгенерировать исключение
                if (I_lb == -2)
                    return { i, j };
                if ((I_lb > max_ij || max_ij == -1)) {
                    max_ij = I_lb;
                    vector_max = { i, j };
                }
            }
            
        }
    }

    return vector_max;
}

// Поиск индекса третьей фар 

int find_third_brake_light(const Mat& stats, const Mat& centroids, const vector<int>& vector_max, float tao_tb) {
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


void filter_rectangels(const Mat& stats, const Mat& centroids, Mat& filteredStats, Mat& filteredCentroids, const Mat& resizedImage) {

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

void get_rectangle_for_detector(const Mat channel, Mat& filteredStats, Mat& filteredCentroids, const Mat resized_img) {
    Mat bin_img;
    Mat bl;
    int maxValue = 255;
    double thresh = threshold(channel, bin_img, 0, maxValue, THRESH_TRIANGLE); //  THRESH_OTSU

    // cout << "Otsu Threshold: " << thresh << endl;
   // imshow("Image after Otsu Threshold", bin_img);

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(bin_img, labels, stats, centroids);

    // imwrite(std::string("Otsu_img.png").c_str(), bin_img);

    filter_rectangels(stats, centroids, filteredStats, filteredCentroids, resized_img);
    //  cout << "After filtr" << filteredStats << endl; 

 //   drawBoundingRectangles(resized_img, filteredStats);
 //   imwrite(std::string("with_rectangels.png").c_str(), resized_img);
} 


Mat get_brake_light(const Mat& stats, const vector<int>& vector_max, int index_third_light) {
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



void img_preprocessing_HSV(Mat& image, vector<Mat>& HSV_channels) {

    resize(image, image, Size(416, 416), INTER_LINEAR);
    Mat HSV_image = image.clone();
    cvtColor(image, HSV_image, 41);
    split(HSV_image, HSV_channels);

}

Mat detector(const vector<Mat>& lab_channels, Mat& img, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S, float tao_tb) {

    double max_ij = -1.0;

    const int A_CHANNEL = 1;
    const int img_height = img.rows;
    const float half_img_weight = img_height / 2;

    Mat stats, centroids;

    get_rectangle_for_detector(lab_channels[A_CHANNEL], stats, centroids, img);

    // cout << "total_sum" << total_sum;
    //  cout << "centroids " << centroids; 


    vector<int> vector_max = find_lateral_brake_light(half_img_weight, stats, centroids, lambda_S, lambda_D, lambda_U, tao_v, tao_S);

    Mat brake_light;

    if (vector_max[0] != vector_max[1]) {
        int index_third_light = find_third_brake_light(stats, centroids, vector_max, tao_tb);
        brake_light = get_brake_light(stats, vector_max, index_third_light);
    }

   if (brake_light.rows < 3) {

        vector<Mat> HSV_channels(3);

        const int v = 2;

        img_preprocessing_HSV(img, HSV_channels);
        get_rectangle_for_detector(HSV_channels[v], stats, centroids, img);

        if (brake_light.rows < 2)
            vector_max = find_lateral_brake_light(half_img_weight, stats, centroids, lambda_S, lambda_D, lambda_U, tao_v, tao_S);

        if (vector_max[0] != vector_max[1]) {

            int index_third_light = find_third_brake_light(stats, centroids, vector_max, tao_tb);
            brake_light = get_brake_light(stats, vector_max, index_third_light);

        }
    }  

    return brake_light; 
}