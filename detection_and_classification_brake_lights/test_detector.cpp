#include "test_detector.h" 


void test_detector(int& total_zero, int& positive, float lambda_S, float lambda_D, float lambda_U, const string& folder, int tao_v, float tao_S, float tao_tb) {

    for (const auto& entry : fs::directory_iterator{ folder }) {
        string label_path = entry.path().string();
        string full_path = label_path;
        string label = label_path.replace(0, folder.size() + 1, "");
        string num = label.replace(label_path.size() - 4, label_path.size(), "");

        vector<Rect> rectangles;

        ifstream file(full_path);
        string line, label_th;
        float  x_th, y_th, x2_th, y2_th;

        while (getline(file, line)) {
            istringstream iss(line);
            string label, label2;
            float x, y, x2, y2, n, n1, n2;
            iss >> label;
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

        Mat image = imread(string("default/image_2/" + num + ".png").c_str());

        const int new_weight = 416;
        const int  new_height = 416;
        const int A = 1;

        vector<Mat> lab_channels(3);

        img_preprocessing(image, lab_channels, new_weight, new_height);

        Mat lateral_stats = detector(lab_channels, image, lambda_S, lambda_D, lambda_U, tao_v, tao_S, tao_tb);
        vector<Rect> rectangles_from_detector;

        for (int i = 0; i < lateral_stats.rows; i++)
        {
            Rect R(lateral_stats.row(i).at<int>(CC_STAT_LEFT), lateral_stats.row(i).at<int>(CC_STAT_TOP), lateral_stats.row(i).at<int>(CC_STAT_WIDTH), lateral_stats.row(i).at<int>(CC_STAT_HEIGHT)); 
            rectangles_from_detector.push_back(R);
        }

        for (const auto& rect : rectangles_from_detector) {
            rectangle(image, rect, Scalar(0, 0, 200), 3);
        }

        imwrite(std::string("try\\" + num + ".png").c_str(), image); 

        if (rectangles[0].x > rectangles[1].x) {
            swap(rectangles[0], rectangles[1]);
        }

        for (int k = 0; k < lateral_stats.rows; k++)
        {
            Rect intersection = rectangles[k] & rectangles_from_detector[k];
            Rect unionRect = rectangles[k] | rectangles_from_detector[k];
            float IoU = float(intersection.area()) / float(unionRect.area());
            if (IoU > 0.4)
                positive += 1;
            else if (IoU == 0)
                total_zero += 1;
        }
    }
    cout << "total_zero: " << total_zero << endl;
    cout << "positive: " << positive << endl;
}

vector <float> choice_lambda(const string& folders, int tao_v, float tao_S, float tao_tb) {

    int total_zero_m = 40; 
    int positive_m = 0;
 
    float lambda_S, lambda_D, lambda_U, lambda_S_m, lambda_D_m;
    float min_value = 0.1; 
    float max_value = 0.9; 
    float step = 0.1; 

    for (lambda_S = min_value; lambda_S <= max_value; lambda_S += step) {
        for (lambda_D = min_value; lambda_D <= max_value - lambda_S; lambda_D += step) {
            lambda_U = 1 - lambda_S - lambda_D;
            if (lambda_U >= min_value && lambda_U <= max_value) {
                int total_zero = 0;
                int positive = 0; 
               
                test_detector(total_zero, positive, lambda_S, lambda_D, lambda_U, folders, tao_v, tao_S, tao_tb);
                if (positive >= positive_m && total_zero <= total_zero_m)
                {
                    lambda_S_m = lambda_S;
                    lambda_D_m = lambda_D;
                    total_zero_m = total_zero;
                    positive_m = positive; 

                    cout << "lambda_S: " << lambda_S << endl; 
                    cout << "lambda_D: " << lambda_D << endl;
                    cout << "total_zero: " << total_zero << endl;
                    cout << "positive: " << positive << endl;
                }
            }
        }
    }
    return { lambda_S, lambda_D, lambda_U };
} 

void choice_tao(const string& input_folders, float lambda_S, float lambda_D, float lambda_U, int tao_v, float tao_S, float tao_tb, int& tao_v_opt, float& tao_S_opt, float& tao_tb_opt) {

    int total_zero_m = 0;
    int positive_m = 0;

    const float min_value = 0.1;
    const float max_value = 0.95;
    const float step = 0.05; 

     for (tao_S = min_value; tao_S <= max_value; tao_S += step) {
         int total_zero = 0;
         int positive = 0;
         test_detector(total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb);
         if (positive >= positive_m)
         {
             tao_S_opt = tao_S;
             total_zero_m = total_zero;
             positive_m = positive; 

             cout << "tao_S: " << tao_S << endl;
             cout << "total_zero: " << total_zero << endl;
             cout << "positive: " << positive << endl;
         }
     }

     for (tao_tb = min_value; tao_tb <= max_value; tao_tb += step) {
         int total_zero = 0;
         int positive = 0;
         test_detector(total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb);
         if (positive >= positive_m)
         {
             tao_tb_opt = tao_tb;
             total_zero_m = total_zero;
             positive_m = positive;
             cout << "tao_tb " << tao_tb << endl;
             cout << "total_zero: " << total_zero << endl;
             cout << "positive: " << positive << endl;
         }
     }

    for (int tao_v = 15; tao_v <= 150; tao_v += 1) {
        float IoU = 0;
        int total_zero = 0;
        int positive = 0;
        test_detector(total_zero, positive, lambda_S, lambda_D, lambda_U, input_folders, tao_v, tao_S, tao_tb); 
        if (positive >= positive_m)
        {
            tao_v_opt = tao_v;
            total_zero_m = total_zero;
            positive_m = positive;                        
            cout << "tao_V: " << tao_v << endl;
            cout << "total_zero: " << total_zero << endl;
            cout << "positive: " << positive << endl;
        }
        cout << "tao_V: " << tao_v << endl;
    }
}

void get_test_for_detector() {

    vector<Rect> rectangles;

    const string folder = "default/label_2";
    Parameters parameters;

    int total_zero = 0;
    int positive = 0; 

    test_detector(total_zero, positive, parameters.lambda_S, parameters.lambda_D, parameters.lambda_U, folder, parameters.tao_v, parameters.tao_S, parameters.tao_tb);

}


