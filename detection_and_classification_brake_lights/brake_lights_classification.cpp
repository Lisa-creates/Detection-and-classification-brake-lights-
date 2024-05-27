#include"Header_files/brake_lights_classification.h"

/*
*\brief Computes features for a rectangular image area
*
* \param current_rectangle Rectangular image area
* \param lab_channels Vector of LAB color space channels
* \param tao_L Threshold for L channel
* \param tao_A Threshold for a channel
* \param lab_image Image in LaB color space
*
* \return Matrix with computed features.
*/
Mat calculate_features(const Mat & current_rectangle, const vector<Mat>&lab_channels, const double tao_L, const double tao_A, const Mat & lab_image);

/*
*\brief Fills input matrices data_l, data_r and data_third with features for left, right and third light respectively,
* obtained features from calculate_features
*
* \param data_l Data matrix for left brake light
* \param data_r Data matrix for right brake light
* \param data_third Data matrix for third brake light(if it exists, if not, filled by left brake light features)
* \param stats Coordinates of left, right and third brake light rect
* \param channels Vector of image channels
* \param img Image
*/
void classifier_get_features(Mat & data_l, Mat & data_r, Mat & data_third, const Mat & stats, const vector<Mat> channels, Mat & img);

/*
*\brief Prints metrics(accuracy, recall, precision F2 - measure) based on predicted and true class labels.
*
*\param test_labels Class labels for testing.
* \param predicted_labels Predicted class labels.
* \param type_sample Sample type(e.g. "test for third light" or "train for side lights").
*/
void print_metrics(const Mat & test_labels, const Mat & predicted_labels, const char* type_sample);

/*
*\brief Trains and tests SVM classifier for left and right light
*
* \param data_l Data matrix for left stop light
* \param data_r Data matrix for right stop light
* \param train_labels Class labels for training sample.
* \param data_l_test Data matrix for testing left stop light
* \param data_r_test Data matrix for testing right stop light
* \param test_labels Class labels for testing sample
* \param predicted_labels Array for saving predicted labels on test sample.
* \param predicted_labels_train Array for saving predicted labels on train sample.
*/
void SVM_classifier_LR_light(const Mat & data_l, const Mat & data_r, const Mat & trainLabels, const Mat & data_l_test, const Mat & data_r_test, const Mat & test_labels, Mat & predicted_labels, Mat & predicted_labels_train);

/*
*\brief Trains SVM model on data_third for third light
*
* \param data_third Data matrix(for third light) for training SVM model.
* \param data_third_test Data matrix for testing.
* \param train_labels Class labels for training model.
* \param test_labels Class labels for testing model.
*/
void SVM_classifier_third_light(const Mat & data_third, const Mat & train_labels, const Mat & data_third_test, const Mat & test_labels, Mat & predicted_labels, Mat & predicted_labels_train);


/*
*\brief Main classification function, which combines results of two predictions, and makes the final prediction.
*
* \param predict_LR Prediction results for lateral lights
* \param predict_third Prediction results for third light
* \param labels Class labels for comparison of predictions
*/
void main_classifier(const Mat & predict_LR, const Mat & predict_third, const Mat & labels);

void calculate_features_without_mean(const Mat& image_part, double threshold,
    float& max_float, float& min_float, float& pixelPercentage) {

    double max_value, min_value; 
    minMaxLoc(image_part, &min_value, &max_value);

    max_float = static_cast<float>(max_value); 
    min_float = static_cast<float>(min_value); 

    Mat mask = image_part > threshold;
    pixelPercentage = 100.0 * countNonZero(mask) / (image_part.rows * image_part.cols);
}

void calculate_mean(const Mat& imagePart, float& L_means, float& a_means) {
    Scalar Arr_means, Arr_stddev;
    meanStdDev(imagePart, Arr_means, Arr_stddev);
    L_means = Arr_means[0];
    a_means = Arr_means[1];
} 

Mat calculate_features(const Mat& current_rectangle, const vector<Mat>& lab_channels, const double tao_L, const double tao_A, const Mat& Lab_image) {
    const int L = 0;
    const int a = 1;
    const int B = 2;

    Rect R_i(current_rectangle.at<int>(CC_STAT_LEFT), current_rectangle.at<int>(CC_STAT_TOP), current_rectangle.at<int>(CC_STAT_WIDTH), current_rectangle.at<int>(CC_STAT_HEIGHT));
    
    // ќбласть фар 
    Mat cropped_L = lab_channels[L](R_i).clone();
    Mat cropped_a = lab_channels[a](R_i).clone();

    float max_value_L, min_value_l, mean_value_L, pixel_percentage_l, mean_value_total_l;
    float max_value_a, min_value_a, meanValuea, pixel_percentage_a, mean_value_total_a;

    calculate_features_without_mean(cropped_L, tao_L, max_value_L, min_value_l, pixel_percentage_l);
    calculate_features_without_mean(cropped_a, tao_A, max_value_a, min_value_a, pixel_percentage_a);
    calculate_mean(Lab_image(R_i), mean_value_L, meanValuea);
    calculate_mean(Lab_image, mean_value_total_l, mean_value_total_a);

    Mat new_row = (Mat_<float>(1, 10) << max_value_L, min_value_l, pixel_percentage_l, max_value_a, min_value_a, pixel_percentage_a, mean_value_L, meanValuea, mean_value_total_l, mean_value_total_a);
    return new_row; 
} 

void classifier_get_features(Mat& data_l, Mat& data_r, Mat& data_third, const Mat& stats, const vector<Mat> channels, Mat& img) {
    const int l = 0, r = 1, third = 2;
    const double tao_L = 150, tao_A = 155; 

    if (stats.rows >= 2) {
        data_l.push_back(calculate_features(stats.row(l), channels, tao_L, tao_A, img));
        data_r.push_back(calculate_features(stats.row(r), channels, tao_L, tao_A, img)); 
        if (stats.rows == 3) {
            data_third.push_back({ calculate_features(stats.row(third), channels, tao_L, tao_A, img) });
        } 
        else {
            data_third.push_back({ calculate_features(stats.row(l), channels, tao_L, tao_A, img) }); 
        }
    }
} 

void print_metrics(const Mat& test_labels, const Mat& predicted_labels, const char* type_sample) {
    Mat confusion_matrix = Mat::zeros(2, 2, CV_32S); 
    for (int i = 0; i < test_labels.rows; ++i) {
        int true_label = test_labels.at<int>(i, 0);
        int predicted_label = predicted_labels.at<float>(i, 0);
        confusion_matrix.at<int>(true_label, predicted_label)++;
    }

    cout << type_sample << endl; 

    for (int i = 0; i < confusion_matrix.rows; ++i) {
        int true_positives = confusion_matrix.at<int>(i, i); 
        int true_negatives = confusion_matrix.at<int>(i, i);
        int false_positives = 0;
        int false_negatives = 0;
        for (int j = 0; j < confusion_matrix.cols; ++j) { 
            if (j != i) {
                false_negatives += confusion_matrix.at<int>(i, j);
                false_positives += confusion_matrix.at<int>(j, i);
            }
        }
        double precision = true_positives / static_cast<double>(true_positives + false_positives);
        double recall = true_positives / static_cast<double>(true_positives + false_negatives);
        double F2 = 5 * ((precision * recall) / (4 * precision + recall));
        double accuracy = (true_positives + true_negatives) / static_cast<double>(true_positives + false_negatives + true_negatives + false_positives); 
        cout << "Class " << i << " - Accuracy: " << accuracy << ", Precision: " << precision << ", Recall: " << recall << ", F2: " << F2 << endl;
    } 
}

void SVM_classifier_LR_light(const Mat& data_l, const Mat& data_r, const Mat& trainLabels, const Mat& data_l_test, const Mat& data_r_test, const Mat& testLabels, Mat& predictedLabels, Mat& predictedLabels_train) {

    Mat dataMat;
    cv::vconcat(data_l, data_r, dataMat); // первый массив записываетс€ после 2 
    
    Mat trainLabelsMat(trainLabels.rows * 2, trainLabels.cols, CV_32S);
    cv::vconcat(trainLabels, trainLabels, trainLabelsMat);
    
    Mat dataMat_test(data_l_test.rows + data_r_test.rows, data_l_test.cols, CV_32F);
    cv::vconcat(data_l_test, data_r_test, dataMat_test);

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    Ptr<TrainData> td = TrainData::create(dataMat, ROW_SAMPLE, trainLabelsMat);
    int kFold = 5; 

    svm->trainAuto(td, kFold); 
    svm->predict(dataMat, predictedLabels_train);
    print_metrics(trainLabelsMat, predictedLabels_train, "train lateral light");

    svm->save("svm_model_lateral.yml");
    svm->predict(dataMat_test, predictedLabels);
    print_metrics(testLabels, predictedLabels, "test lateral light"); 
}

void SVM_classifier_third_light(const Mat& data_third, const Mat& train_labels, const Mat& data_third_test, const Mat& testLabels, Mat& predictedLabels, Mat& predictedLabels_train) {

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    Ptr<TrainData> td = TrainData::create(data_third, ROW_SAMPLE, train_labels);
    svm->trainAuto(td); 

    svm->predict(data_third, predictedLabels_train); 
    print_metrics(train_labels, predictedLabels_train, "train th");
    svm->save("svm_model_third.yml");

    svm->predict(data_third_test, predictedLabels);
    print_metrics(testLabels, predictedLabels, "test th");
} 

void main_classifier(const Mat& predict_LR, const Mat& predict_third, const Mat& labels) {

    Mat predict(labels.rows, labels.cols, CV_32S);

    for (int i = 0; i < labels.rows; ++i) { 
        if (predict_LR.rows > labels.rows) {
            if (predict_LR.at<float>(i) + predict_third.at<float>(i) + predict_LR.at<float>(i + labels.rows) >= 2)
                predict.at<float>(i) = 1;
            else
                predict.at<float>(i) = 0;
        } 
        else {
            if (predict_LR.at<float>(i) + predict_third.at<float>(i) >= 2)
                predict.at<float>(i) = 1;
            else
                predict.at<float>(i) = 0;
        }
    }
   print_metrics(labels, predict, "test all");
} 

