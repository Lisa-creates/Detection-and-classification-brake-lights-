#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


using namespace cv; 
using namespace cv::ml;
using namespace std; 



struct Features {
    double maxValueL;
    double minValueL;
    double pixelPercentageL;
    double meanValueL;
    double maxValuea;
    double minValuea;
    double pixelPercentagea;
    double meanValuea;
};

void calculateFeaturesWithoutMean(const cv::Mat& imagePart, double threshold,
    float& max_, float& min_, float& pixelPercentage) {

    double maxValue, minValue; 

    cv::minMaxLoc(imagePart, &minValue, &maxValue);

    max_ = static_cast<float>(maxValue); 
    min_ = static_cast<float>(minValue); 

    cv::Mat mask = imagePart > threshold;
    pixelPercentage = 100.0 * cv::countNonZero(mask) / (imagePart.rows * imagePart.cols);
}

void calculateMean(const cv::Mat& imagePart, float& L_means, float& a_means) {
    cv::Scalar Arr_means, Arr_stddev;
    meanStdDev(imagePart, Arr_means, Arr_stddev);
    L_means = Arr_means[0];
    a_means = Arr_means[1];
} 

Mat features(const Mat current_rectangle, const vector<Mat> lab_channels, const double tao_L, const double tao_A, const Mat Lab_image) {
    // Mat current_rectangle = lateral_stats.row(i);
    const int L = 0;
    const int a = 1;
    const int B = 2;

    cv::Rect R_i(current_rectangle.at<int>(cv::CC_STAT_LEFT), current_rectangle.at<int>(cv::CC_STAT_TOP), current_rectangle.at<int>(cv::CC_STAT_WIDTH), current_rectangle.at<int>(cv::CC_STAT_HEIGHT));
    // Область фар 

    cv::Mat croppedL = lab_channels[L](R_i).clone();
    cv::Mat croppedA = lab_channels[a](R_i).clone();

    float maxValueL, minValueL, meanValueL, pixelPercentageL, meanValueTotalL;
    float maxValuea, minValuea, meanValuea, pixelPercentagea, meanValueTotala;

    calculateFeaturesWithoutMean(croppedL, tao_L, maxValueL, minValueL, pixelPercentageL);
    calculateFeaturesWithoutMean(croppedA, tao_A, maxValuea, minValuea, pixelPercentagea);
    calculateMean(Lab_image(R_i), meanValueL, meanValuea);
    calculateMean(Lab_image, meanValueTotalL, meanValueTotala);

    vector<float> feat = { maxValueL, minValueL,  pixelPercentageL, maxValuea, minValuea, pixelPercentagea,  meanValueL, meanValuea, meanValueTotalL, meanValueTotala };

    cv::Mat new_row = (cv::Mat_<float>(1, 10) << maxValueL, minValueL, pixelPercentageL, maxValuea, minValuea, pixelPercentagea, meanValueL, meanValuea, meanValueTotalL, meanValueTotala);

    Mat feat_mat = Mat(feat).reshape(10, 1);

    // cout << "Max from function " << maxValueL << " Min from function " << minValueL << " PercentageL from function " << pixelPercentageL << '\n';
   //  cout << "Mean_L " << meanValueL << "Mean_a " << meanValuea << '\n'; 

    return new_row; 
} 

void classifier_get_features(Mat& data_l, Mat& data_r, Mat& data_third, Mat& stats, const vector<Mat> channels, Mat& img) {

    const int l = 0, r = 1, third = 2;
    const double tao_L = 150, tao_A = 155; 

    if (stats.rows >= 2) {
        data_l.push_back(features(stats.row(l), channels, tao_L, tao_A, img));
        data_r.push_back(features(stats.row(r), channels, tao_L, tao_A, img)); 

        
        if (stats.rows == 3) {
            data_third.push_back({ features(stats.row(third), channels, tao_L, tao_A, img) });
        } 
        else {
            data_third.push_back({ features(stats.row(l), channels, tao_L, tao_A, img) }); 
        }
    }
} 


void SVM_classifier_third_light(Mat data_third, Mat trainLabels) {
     
   // cout << data_third;

    Mat trainLabelsMat(trainLabels.rows, trainLabels.cols, CV_32S);
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

    cout << dataMat << endl; 
   

    // Set up SVM for OpenCV 3
    Ptr<SVM> svm = SVM::create();
    // Set SVM type
    svm->setType(SVM::C_SVC);
    // Set SVM Kernel to Radial Basis Function (RBF)
    svm->setKernel(SVM::RBF);
    // Set parameter C
    svm->setC(12.5);
    // Set parameter Gamma
    svm->setGamma(0.50625);

    // Train SVM on training data
    Ptr<TrainData> td = TrainData::create(data_third, ROW_SAMPLE, trainLabels); 
    svm->trainAuto(td); 

    // Save trained model
    svm->save("digits_svm_model.yml"); 


    // Predict on training data
    Mat predictedLabels; 
    svm->predict(data_third, predictedLabels);

    // Calculate precision and recall
    Mat confusionMatrix = Mat::zeros(10, 10, CV_32S); // Assuming 10 classes
    for (int i = 0; i < data_third.rows; ++i) {
        int trueLabel = trainLabels.at<int>(i, 0);
        int predictedLabel = predictedLabels.at<float>(i, 0);
        confusionMatrix.at<int>(trueLabel, predictedLabel)++;
    }

    for (int i = 0; i < confusionMatrix.rows; ++i) {
        int truePositives = confusionMatrix.at<int>(i, i);
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
        cout << "Class " << i << " - Precision: " << precision << ", Recall: " << recall << endl;
    }

    // Test on a held out test set
 //   svm->predict(testMat, testResponse);
} 
