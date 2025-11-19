#ifndef YOLOV11_H
#define YOLOV11_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct Detection {
    float confidence;
    int class_id;
    cv::Rect box;
    std::string class_name;
};

class YOLOv11 {
public:
    YOLOv11(const std::string& model_path);

    // Configuration
    void setConf(float conf);      // Set confidence threshold
    void setNMS(float nms);        // Set NMS threshold
    void setModel(const std::string& model_path);  // Set ONNX model path

    // Core processing
    void preprocessImage(const cv::Mat& image);
    void runInference();
    void postprocessResults();
    void drawBoundingBoxes(cv::Mat& image);

    // Convenience method
    std::vector<Detection> detect(const cv::Mat& image);

    // Getters
    std::vector<Detection> getDetections() const;

private:
    cv::dnn::Net net;
    cv::Mat blob;
    cv::Mat output_blob;
    cv::Size original_size;
    std::vector<Detection> detections;
    std::vector<std::string> class_names;
    std::string model_path;  // Store model path

    float conf_threshold = 0.5f;
    float nms_threshold = 0.4f;
    cv::Size input_size = cv::Size(640, 640);

    void loadClassNames();
    std::string getClassName(int class_id) const;
};

#endif // YOLOV11_H