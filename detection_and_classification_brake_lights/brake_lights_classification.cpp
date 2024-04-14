using namespace cv; 
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
    double& maxValue, double& minValue, double& pixelPercentage) {

    cv::minMaxLoc(imagePart, &minValue, &maxValue);

    cv::Mat mask = imagePart > threshold;
    pixelPercentage = 100.0 * cv::countNonZero(mask) / (imagePart.rows * imagePart.cols);
}

void calculateMean(const cv::Mat& imagePart, double& L_means, double& a_means) {
    cv::Scalar Arr_means, Arr_stddev;
    meanStdDev(imagePart, Arr_means, Arr_stddev);
    L_means = Arr_means[0];
    a_means = Arr_means[1];
} 

void features() { 



    for (int i = 0; i < lateral_stats.rows; ++i) {

        Mat current_rectangle = lateral_stats.row(i);

        cv::Rect R_i(current_rectangle.at<int>(cv::CC_STAT_LEFT), current_rectangle.at<int>(cv::CC_STAT_TOP), current_rectangle.at<int>(cv::CC_STAT_WIDTH), current_rectangle.at<int>(cv::CC_STAT_HEIGHT));
        // Область фар 

        cv::Mat croppedL = lab_channels[L](R_i).clone();
        cv::Mat croppedA = lab_channels[a](R_i).clone();

        double maxValueL, minValueL, meanValueL, pixelPercentageL, tao_L = 128, meanValueTotalL;
        double maxValuea, minValuea, meanValuea, pixelPercentagea, tao_A = 128, meanValueTotala;

        calculateFeaturesWithoutMean(croppedL, tao_L, maxValueL, minValueL, pixelPercentageL);
        calculateFeaturesWithoutMean(croppedA, tao_A, maxValuea, minValuea, pixelPercentagea);
        calculateMean(Lab_image(R_i), meanValueL, meanValuea);
        calculateMean(Lab_image, meanValueTotalL, meanValueTotala); 
        
        vector<double> feat = { maxValueL, minValueL,  pixelPercentageL, maxValuea, minValuea, pixelPercentagea,  meanValueL, meanValuea, meanValueTotalL, meanValueTotala }; 


        cout << "Max from function " << maxValueL << " Min from function " << minValueL << " PercentageL from function " << pixelPercentageL << '\n';
        cout << "Mean_L " << meanValueL << "Mean_a " << meanValuea << '\n';
    }
}