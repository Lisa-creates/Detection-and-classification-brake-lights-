#include"video_processing.h"

Mat convert_to_Lab(const Mat& image) {
    Mat lab_image;
    cvtColor(image, lab_image, 45);
    return lab_image;
}

void img_preprocessing(Mat& image, vector<Mat>& lab_channels, const int weight, const int height) {

    resize(image, image, Size(weight, height), INTER_LINEAR);
    Mat lab_image = convert_to_Lab(image);
    split(lab_image, lab_channels);
}

int get_video(const string& video_path, const string& label_path) { 

    Ptr<SVM> svm_lat = SVM::load("svm_model_lateral.yml"); 
    Ptr<SVM> svm_th = SVM::load("svm_model_third.yml");

    VideoCapture cap("car_black.mp4");
    vector<string> input_folders = { "default_car_black/label_2" };

    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    int orig_height = cap.get(CAP_PROP_FRAME_WIDTH);
    int orig_width = cap.get(CAP_PROP_FRAME_HEIGHT);

    const string folder = "default_car_black/label_2";

    for (const auto& entry : fs::directory_iterator{ folder }) {

        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        string label_path = entry.path().string();

        ifstream file(label_path);
        string line;

        float x, y, x2, y2;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            string label_str;
            float  n, n1, n2;
            iss >> label_str;
            iss >> n >> n1 >> n2 >> x >> y >> x2 >> y2;

        }

        Rect car(x, y, x2 - x, y2 - y);
        Mat frame_orig = frame;

        int new_width = 416;
        int new_height = 416;

        Mat img_ROI = frame(car);
        vector<Mat> lab_channels(3);

        img_preprocessing(img_ROI, lab_channels, new_width, new_height);

        Parameters parameters;
        Mat lateral_stats = detector(lab_channels, img_ROI, parameters.lambda_S, parameters.lambda_D, parameters.lambda_U, parameters.tao_v, parameters.tao_S, parameters.tao_tb);
        int label_class = 0;

        if (lateral_stats.rows >= 2)
        {
            Mat predict_data_l, predict_data_r, predict_data_third;
            Mat data_l, data_r, data_third;
            classifier_get_features(data_l, data_r, data_third, lateral_stats, lab_channels, frame);
            //  cout << data_l << endl << data_r << endl << data_third << endl; 

            Mat data_mat_test(data_l.rows + data_r.rows, data_l.cols, CV_32F);
            cv::vconcat(data_l, data_r, data_mat_test);

            svm_lat->predict(data_mat_test, predict_data_l);

            if (lateral_stats.rows == 3) {
                svm_th->predict(data_third, predict_data_third);
                if (predict_data_l.at<float>(0) + predict_data_l.at<float>(1) + predict_data_third.at<float>(0) >= 3)
                    label_class = 1;
            }
            else {
                if (predict_data_l.at<float>(0) + predict_data_l.at<float>(1) >= 2)
                    label_class = 1;
            }
        }

        for (int i = 0; i < lateral_stats.rows; ++i) {

            Rect r(lateral_stats.row(i).at<int>(CC_STAT_LEFT), lateral_stats.row(i).at<int>(CC_STAT_TOP), lateral_stats.row(i).at<int>(CC_STAT_WIDTH), lateral_stats.row(i).at<int>(CC_STAT_HEIGHT));
            rectangle(img_ROI, r, Scalar(0, 0, 200), 3);
            string text;

            if (label_class == 0)
                text = "OFF";
            else
                text = "ON";
            putText(img_ROI, text, Point(r.x, r.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

        for (int i = 0; i < lateral_stats.rows; ++i) {

            int x_lights = (lateral_stats.row(i).at<int>(CC_STAT_LEFT)) * (orig_width / new_width) + car.x;
            int y_lights = (lateral_stats.row(i).at<int>(CC_STAT_TOP)) * (orig_height / new_height) + car.y;
            int width_rect = lateral_stats.row(i).at<int>(CC_STAT_WIDTH) * (orig_width / new_width);
            int height_rect = lateral_stats.row(i).at<int>(CC_STAT_HEIGHT) * (orig_height / new_height);  

            Rect r(x_lights, y_lights, x_lights + width_rect, y_lights + height_rect);
            rectangle(frame_orig, r, Scalar(0, 0, 200), 3);

            string text;
            if (label_class == 0)
                text = "OFF";
            else
                text = "ON";
            putText(frame_orig, text, Point(r.x, r.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

        imshow("Frame", img_ROI);

        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;
    cap.release();

    return 0; 

}


