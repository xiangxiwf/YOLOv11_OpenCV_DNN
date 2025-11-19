#include "YOLOv11.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

int main() {
    try {
        // Configuration
        std::string model_path = "/Users/wufeng/C++Projects/OpenCV_DNN/yolo11n.onnx";
        std::string image_path = "/Users/wufeng/C++Projects/OpenCV_DNN/test_image.jpg";

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return -1;
        }

        // Initialize YOLO detector
        YOLOv11 detector("");  // Empty path initially
        detector.setModel(model_path);  // Set model path using interface
        detector.setConf(0.25f);  // Set confidence threshold
        detector.setNMS(0.35f);   // Set NMS threshold

        // Method 2: Step-by-step processing
        auto start = std::chrono::high_resolution_clock::now();

        detector.preprocessImage(image);    // Preprocess
        detector.runInference();            // Run inference
        detector.postprocessResults();      // Postprocess
        std::vector<Detection> detections = detector.getDetections(); // Get results

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Method 1: All-in-one detection (alternative approach)
        /*
        std::vector<Detection> detections_method1 = detector.detect(image);
        // The detect() method internally calls:
        // - preprocessImage(image)
        // - runInference()
        // - postprocessResults()
        // - and returns the detections directly
        */

        // Draw and save results
        cv::Mat result_image = image.clone();
        detector.drawBoundingBoxes(result_image);

        // Display result
        cv::imshow("YOLO Detection Results", result_image);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // Save result
        cv::imwrite("/Users/wufeng/C++Projects/OpenCV_DNN/build/detection_result.jpg", result_image);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}