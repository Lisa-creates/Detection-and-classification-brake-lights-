#include"Header_files/brake_lights_classification.h"

/*
*\brief Computes features for a rectangular image area
*
* \param current_rectangle Rectangular image area
* \param lab_channels Vector of LAB color space channels
* \param tao_L Threshold for L channel
* \param tao_A Threshold for a channel
* \param Lab_image Image in LaB color space
*
* \return Matrix with computed features.
*/
Mat calculate_features(const Mat & current_rectangle, const vector<Mat>&lab_channels, const double tao_L, const double tao_A, const Mat & Lab_image);

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
*\param testLabels Class labels for testing.
* \param predictedLabels Predicted class labels.
* \param type_sample Sample type(e.g. "test for third light" or "train for side lights").
*/
void print_metrics(const Mat & testLabels, const Mat & predictedLabels, const char* type_sample);

/*
*\brief Trains and tests SVM classifier for left and right light
*
* \param data_l Data matrix for left stop light
* \param data_r Data matrix for right stop light
* \param trainLabels Class labels for training sample.
* \param data_l_test Data matrix for testing left stop light
* \param data_r_test Data matrix for testing right stop light
* \param testLabels Class labels for testing sample
* \param predictedLabels Array for saving predicted labels on test sample.
* \param predictedLabels_train Array for saving predicted labels on train sample.
*/
void SVM_classifier_LR_light(const Mat & data_l, const Mat & data_r, const Mat & trainLabels, const Mat & data_l_test, const Mat & data_r_test, const Mat & testLabels, Mat & predictedLabels, Mat & predictedLabels_train);

/*
*\brief Trains SVM model on data_third for third light
*
* \param data_third Data matrix(for third light) for training SVM model.
* \param data_third_test Data matrix for testing.
* \param trainLabels Class labels for training model.
* \param testLabels Class labels for testing model.
*/
void SVM_classifier_third_light(const Mat & data_third, const Mat & trainLabels, const Mat & data_third_test, const Mat & testLabels, Mat & predictedLabels, Mat & predictedLabels_train);


/*
*\brief Main classification function, which combines results of two predictions, and makes the final prediction.
*
* \param predict_LR Prediction results for lateral lights
* \param predict_third Prediction results for third light
* \param labels Class labels for comparison of predictions
*/
void main_classifier(const Mat & predict_LR, const Mat & predict_third, const Mat & labels);



void calculate_features_without_mean(const Mat& imagePart, double threshold,
    float& max_, float& min_, float& pixelPercentage) {

    double maxValue, minValue; 

    minMaxLoc(imagePart, &minValue, &maxValue);

    max_ = static_cast<float>(maxValue); 
    min_ = static_cast<float>(minValue); 

    Mat mask = imagePart > threshold;
    pixelPercentage = 100.0 * countNonZero(mask) / (imagePart.rows * imagePart.cols);
}

void calculate_mean(const Mat& imagePart, float& L_means, float& a_means) {
    Scalar Arr_means, Arr_stddev;
    meanStdDev(imagePart, Arr_means, Arr_stddev);
    L_means = Arr_means[0];
    a_means = Arr_means[1];
} 

Mat calculate_features(const Mat& current_rectangle, const vector<Mat>& lab_channels, const double tao_L, const double tao_A, const Mat& Lab_image) {
    // Mat current_rectangle = lateral_stats.row(i);
    const int L = 0;
    const int a = 1;
    const int B = 2;

    Rect R_i(current_rectangle.at<int>(CC_STAT_LEFT), current_rectangle.at<int>(CC_STAT_TOP), current_rectangle.at<int>(CC_STAT_WIDTH), current_rectangle.at<int>(CC_STAT_HEIGHT));
    // ќбласть фар 

    Mat croppedL = lab_channels[L](R_i).clone();
    Mat croppedA = lab_channels[a](R_i).clone();

    float maxValueL, minValueL, meanValueL, pixelPercentageL, meanValueTotalL;
    float maxValuea, minValuea, meanValuea, pixelPercentagea, meanValueTotala;

    calculate_features_without_mean(croppedL, tao_L, maxValueL, minValueL, pixelPercentageL);
    calculate_features_without_mean(croppedA, tao_A, maxValuea, minValuea, pixelPercentagea);
    calculate_mean(Lab_image(R_i), meanValueL, meanValuea);
    calculate_mean(Lab_image, meanValueTotalL, meanValueTotala);

   // vector<float> feat = { maxValueL, minValueL,  pixelPercentageL, maxValuea, minValuea, pixelPercentagea,  meanValueL, meanValuea, meanValueTotalL, meanValueTotala };

    Mat new_row = (Mat_<float>(1, 10) << maxValueL, minValueL, pixelPercentageL, maxValuea, minValuea, pixelPercentagea, meanValueL, meanValuea, meanValueTotalL, meanValueTotala);

   // Mat feat_mat = Mat(feat).reshape(10, 1);

    // cout << "Max from function " << maxValueL << " Min from function " << minValueL << " PercentageL from function " << pixelPercentageL << '\n';
   //  cout << "Mean_L " << meanValueL << "Mean_a " << meanValuea << '\n'; 

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



void print_metrics(const Mat& testLabels, const Mat& predictedLabels, const char* type_sample) {
    Mat confusionMatrix = Mat::zeros(2, 2, CV_32S); 
    for (int i = 0; i < testLabels.rows; ++i) {
        int trueLabel = testLabels.at<int>(i, 0);
        int predictedLabel = predictedLabels.at<float>(i, 0);
        confusionMatrix.at<int>(trueLabel, predictedLabel)++;
    }

    cout << type_sample << endl; 

    for (int i = 0; i < confusionMatrix.rows; ++i) {
        int truePositives = confusionMatrix.at<int>(i, i); 
        int trueNegatives = confusionMatrix.at<int>(i, i);
        int falsePositives = 0;
        int falseNegatives = 0;
        for (int j = 0; j < confusionMatrix.cols; ++j) { 
            if (j != i) {
                falseNegatives += confusionMatrix.at<int>(i, j);
                falsePositives += confusionMatrix.at<int>(j, i);
            }
        }
        double precision = truePositives / static_cast<double>(truePositives + falsePositives);
        double recall = truePositives / static_cast<double>(truePositives + falseNegatives);
        double F2 = 5 * ((precision * recall) / (4 * precision + recall));
        double accuracy = (truePositives + trueNegatives) / static_cast<double>(truePositives + falseNegatives + trueNegatives + falsePositives); 
        cout << "Class " << i << " - Accuracy: " << accuracy << ", Precision: " << precision << ", Recall: " << recall << ", F2: " << F2 << endl;
    } 
}



void SVM_classifier_LR_light(const Mat& data_l, const Mat& data_r, const Mat& trainLabels, const Mat& data_l_test, const Mat& data_r_test, const Mat& testLabels, Mat& predictedLabels, Mat& predictedLabels_train) {

 /*   Mat dataMat(data_l.rows + data_r.rows, data_l.cols, CV_32F);
    for (int i = 0; i < dataMat.rows; ++i) {
        for (int j = 0; j < dataMat.cols; ++j) {
            if (i < data_l.rows)
                dataMat.row(i).at<float>(j) = data_l.row(i).at<float>(j);
            else
                dataMat.row(i).at<float>(j) = data_r.row(i - data_r.rows).at<float>(j);
        }
    }
    for (int i = 0; i < trainLabelsMat.rows; ++i) {
        if (i < trainLabels.rows)
            trainLabelsMat.at<int>(i, 0) = trainLabels.at<int>(i, 0);
        else
            trainLabelsMat.at<int>(i, 0) = trainLabels.at<int>(i - trainLabels.rows, 0);
    }
*/
    Mat dataMat;
    cv::vconcat(data_l, data_r, dataMat); // первый массив записываетс€ после 2 

    cout << "data_l.rows " << data_l.rows << " dataMat.rows " << dataMat.rows << endl;



    Mat trainLabelsMat(trainLabels.rows * 2, trainLabels.cols, CV_32S);
    /*   for (int i = 0; i < trainLabelsMat.rows; ++i) {
           if (i < trainLabels.rows)
               trainLabelsMat.at<int>(i, 0) = trainLabels.at<int>(i, 0);
           else
               trainLabelsMat.at<int>(i, 0) = trainLabels.at<int>(i - trainLabels.rows, 0);
       }*/

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


void SVM_classifier_third_light(const Mat& data_third, const Mat& trainLabels, const Mat& data_third_test, const Mat& testLabels, Mat& predictedLabels, Mat& predictedLabels_train) {

    // cout << data_third;

  /* Mat trainLabelsMat(trainLabels.rows, trainLabels.cols, CV_32S);
    for (int i = 0; i < trainLabelsMat.rows; ++i) {
        trainLabelsMat.at<int>(i, 0) = trainLabels.at<int>(i, 0);
    }

    // cout << "Here" << trainLabelsMat;

    Mat dataMat(data_third.rows, data_third.cols, CV_32F);
    for (int i = 0; i < dataMat.rows; ++i) {
        for (int j = 0; j < dataMat.cols; ++j) {
            dataMat.row(i).at<float>(j) = data_third.row(i).at<float>(j);
        }
    }


    Mat testLabelsMat(testLabels.rows, testLabels.cols, CV_32S);
    for (int i = 0; i < testLabelsMat.rows; ++i) {
        testLabelsMat.at<int>(i, 0) = testLabels.at<int>(i, 0);
    }

    Mat dataMat_test(data_third_test.rows, data_third_test.cols, CV_32F);
    for (int i = 0; i < dataMat_test.rows; ++i) {
        for (int j = 0; j < dataMat_test.cols; ++j) {
            dataMat_test.row(i).at<float>(j) = data_third_test.row(i).at<float>(j);
        }
    }
*/ 
    Ptr<SVM> svm = SVM::create();

    svm->setType(SVM::C_SVC);

    svm->setKernel(SVM::RBF);

    Ptr<TrainData> td = TrainData::create(data_third, ROW_SAMPLE, trainLabels);

    svm->trainAuto(td); 

    svm->predict(data_third, predictedLabels_train); 

    print_metrics(trainLabels, predictedLabels_train, "train th");
    svm->save("svm_model_third.yml");


    svm->predict(data_third_test, predictedLabels);

    print_metrics(testLabels, predictedLabels, "test th");
} 


void main_classifier(const Mat& predict_LR, const Mat& predict_third, const Mat& labels) {

    Mat predict(labels.rows, labels.cols, CV_32S);

    for (int i = 0; i < labels.rows; ++i) { // + predict_LR.at<int>(i + labels.rows, 0) 
        if (predict_LR.rows > labels.rows) {
         //   cout << i << " " << predict_LR.at<float>(i) + predict_third.at<float>(i) + predict_LR.at<float>(i + labels.rows) << endl;
            if (predict_LR.at<float>(i) + predict_third.at<float>(i) + predict_LR.at<float>(i + labels.rows) >= 2)
                predict.at<float>(i) = 1;
            else
                predict.at<float>(i) = 0;
        } 
        else {
         ///   cout << i << " " << predict_LR.at<float>(i) + predict_third.at<float>(i) + predict_LR.at<float>(i + labels.rows) << endl;
            if (predict_LR.at<float>(i) + predict_third.at<float>(i) >= 2)
                predict.at<float>(i) = 1;
            else
                predict.at<float>(i) = 0;
        }
    }

   print_metrics(labels, predict, "test all");
} 

