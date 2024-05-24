#include <iostream>
#include <sstream>
#include <vector>
#include <string> 

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;
#include <filesystem> 
#include <experimental/filesystem> 
#include <regex> 
#include "brake_lights_detecthion.h" 

namespace fs = std::experimental::filesystem;


Mat convertToLab_(const Mat& image) {
    Mat Lab_image;
    cvtColor(image, Lab_image, 45);
    return Lab_image;
}


void img_preprocessing_(Mat& image, vector<Mat>& lab_channels, const int weight, const int height) {

   // resize(image, image, Size(weight, height), INTER_LINEAR);

    //  imwrite(std::string("Resize_img.png").c_str(), image);

    Mat Lab_image = convertToLab_(image);

    split(Lab_image, lab_channels);
}

void test_detector(float& IoU, int& total_zero, int& positive, double lambda_S, double lambda_D, double lambda_U, const vector<string>&  input_folders, int tao_v, float tao_S, float tao_tb) {

    // Итерация по файлам 
    for (const string& folder : input_folders) {
        for (const auto& entry : fs::directory_iterator{ folder }) {
            string label_path = entry.path().string();
            string full_path = label_path;
            string label = label_path.replace(0, folder.size() + 1, "");
            string num = label.replace(label_path.size() - 4, label_path.size(), "");
            // erase(label, '.txt'); 
            //  string l = 
           // cout << label_path << endl;

            vector<Rect> rectangles, rectangels_th;

            ifstream file(full_path);
            string line, label_th;
            float  x_th, y_th, x2_th, y2_th; 

            while (getline(file, line)) {
                istringstream iss(line);
                string label, label2;
                float x, y, x2, y2, n, n1, n2;
                iss >> label;
                //  cout << label; 

                if (label == "BRAKE")
                {
                    iss >> label2 >> n >> n1 >> n2 >> x >> y >> x2 >> y2;
                    Rect rect(x, y, x2 - x, y2 - y); 
                    rectangles.push_back(rect);
                }
                else
                {
                    iss >> label2 >> label_th >> n >> n1 >> n2 >> x_th >> y_th >> x2_th >> y2_th;;
                }

            }

            if (label_th.size() > 0)
            {
                Rect rect_third(x_th, y_th, x2_th - x_th, y2_th - y_th);
                rectangles.push_back(rect_third);
            }

            Mat image = imread(std::string("default/image_2/" + num + ".png").c_str());
            
            const int new_weight = 416;
            const int  new_height = 416; 
            const int A = 1; 

            vector<Mat> lab_channels(3); 

            Mat orig_img = image.clone();

            img_preprocessing_(image, lab_channels, new_weight, new_height);

            
            Mat lateral_stats = detector(lab_channels, image, lambda_S, lambda_D, lambda_U, tao_v, tao_S, tao_tb); 

            vector<Rect> rectangles_from_detector;

            for (int i = 0; i < lateral_stats.rows; i++)
            {
                cv::Rect R(lateral_stats.row(i).at<int>(cv::CC_STAT_LEFT), lateral_stats.row(i).at<int>(cv::CC_STAT_TOP), lateral_stats.row(i).at<int>(cv::CC_STAT_WIDTH), lateral_stats.row(i).at<int>(cv::CC_STAT_HEIGHT));
                rectangles_from_detector.push_back(R);
            }

            for (const auto& rect : rectangles_from_detector) {
                // std::cout << "Rect: (" << rect_third.x << ", " << rect_third.y << ", " << rect_third.width << ", " << rect_third.height << ")\n";
                cv::rectangle(image, rect, cv::Scalar(0, 0, 200), 3);
            }

            imwrite(std::string("try\\" + num + ".png").c_str(), image);

            float IoU_tmp = 0;

            if (rectangles[0].x > rectangles[1].x) {
                Rect tmp = rectangles[0];
                rectangles[0] = rectangles[1]; // !!!
                rectangles[1] = tmp;
            }

            int sz = 0;


            for (int k = 0; k < lateral_stats.rows; k++)
            {
                Rect intersection = rectangles[k] & rectangles_from_detector[k];
                Rect unionRect = rectangles[k] | rectangles_from_detector[k];
                IoU_tmp += float(intersection.area()) / float(unionRect.area());
                if (IoU_tmp == 0)
                    total_zero += 1;
                if (IoU_tmp > 0.4)
                    positive += 1;
            }

            IoU += (IoU_tmp);

        }
    }
} 



vector <double> choice_lambda(const vector<string> input_folders, int tao_v, float tao_S, float tao_tb) {
    float maxIoU = 0;
    int total_zero_m = 0;
    int positive_m = 0;

    // Итерация по файлам 

    double lambda_S, lambda_D, lambda_U, lambda_S_m, lambda_D_m;

    // Границы для переменных
    double min_value = 0.1; // Минимальное значение, чтобы избежать деления на ноль
    double max_value = 0.9; // Максимальное значение, чтобы сумма не превышала 1

    // Шаг изменения переменных
    double step = 0.1; 

    // Перебор значений
    for (lambda_S = min_value; lambda_S <= max_value; lambda_S += step) {
        for (lambda_D = min_value; lambda_D <= max_value - lambda_S; lambda_D += step) {
            lambda_U = 1 - lambda_S - lambda_D;
            if (lambda_U >= min_value && lambda_U <= max_value) {
                float IoU = 0;
                int total_zero = 0;
                int positive = 0;
                test_detector(IoU, total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb);
                //  cout "mIoU: " << maxIoU / 100 << endl;
                if (positive >= positive_m)
                {
                    maxIoU = IoU;
                    lambda_S_m = lambda_S;
                    lambda_D_m = lambda_D;
                    int total_zero_m = total_zero;
                    int positive_m = positive; 

                    cout << "lambda_S: " << lambda_S << endl; 
                    cout << "lambda_D: " << lambda_D << endl;
                    cout << "IoU: " << IoU << endl;
                    cout << "total_zero: " << total_zero << endl;
                    cout << "positive: " << positive << endl;
                }
            }
        }
    }

    return { lambda_S, lambda_D, lambda_U };
} 



vector <double> choice_tao(const vector<string>& input_folders, double lambda_S, double lambda_D, double lambda_U) {
    float maxIoU = 0;
    int total_zero_m = 0;
    int positive_m = 0;

    // Границы для переменных
    double min_value = 0.1; // Минимальное значение 
    double max_value = 0.95; // Максимальное значение

    // Шаг изменения переменных
    double step = 0.05; 

    float tao_tb = 0.67;
    float tao_S = 0.45;
    int tao_v = 35;


    float tao_tb_m;
    float tao_S_m; 
    float tao_v_m; 

   // Перебор значений
    for (tao_S = min_value; tao_S <= max_value; tao_S += step) {
        float IoU = 0;
        int total_zero = 0;
        int positive = 0;
        test_detector(IoU, total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb);
        //  cout "mIoU: " << maxIoU / 100 << endl;
        if (positive >= positive_m)
        {
            maxIoU = IoU;
            tao_S_m = tao_S;
            total_zero_m = total_zero;
            positive_m = positive; 

            cout << "tao_S: " << tao_S << endl;
            cout << "IoU: " << IoU << endl;
            cout << "total_zero: " << total_zero << endl;
            cout << "positive: " << positive << endl;
        }
    }

    for (tao_tb = min_value; tao_tb <= max_value; tao_tb += step) {
        float IoU = 0;
        int total_zero = 0;
        int positive = 0;
        test_detector(IoU, total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb);
        //  cout "mIoU: " << maxIoU / 100 << endl;
        if (positive >= positive_m)
        {
            maxIoU = IoU;
            tao_tb_m = tao_S;
            total_zero_m = total_zero;
            positive_m = positive;
        }

    }

    

    for (int tao_v = 15; tao_v <= 150; tao_v += 1) {
        float IoU = 0;
        int total_zero = 0;
        int positive = 0;
        test_detector(IoU, total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb);
        //  cout "mIoU: " << maxIoU / 100 << endl;
        if (positive >= positive_m)
        {
            maxIoU = IoU;
            tao_v_m = tao_v;
            total_zero_m = total_zero;
            positive_m = positive; 
            maxIoU = IoU; 

            cout << "tao_V: " << tao_v << endl;
            cout << "IoU: " << IoU << endl;
            cout << "total_zero: " << total_zero << endl;
            cout << "positive: " << positive << endl;
        }
        cout << "tao_V: " << tao_v << endl;
    }

    vector <double> tao = { tao_S_m, tao_tb_m, tao_v_m };

    return tao;

}


void get_test_for_detector() {
    vector<Rect> rectangles;

    // Чтение данных из нескольких файлов txt
  //  vector<string> fileNames = { "148.txt"};

    vector<string> input_folders = { "default/label_2" };
    double lambda_S = 0.45, lambda_D = 0.35, lambda_U = 0.2; 
    float IoU = 0;
    int total_zero = 0;
    int positive = 0;
    float tao_tb = 0.67;
    float tao_S = 0.45; 
    int tao_v = 35;

   // choice_tao(input_folders, lambda_S, lambda_D, lambda_U); 

    // choice_lambda(input_folders, tao_v, tao_S, tao_tb); 

    test_detector(IoU, total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb); 
    
    cout << "total_zero: " << total_zero << endl;
    cout << "positive: " << positive << endl; 
}


