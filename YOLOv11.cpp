#include "YOLOv11.h"
#include <iostream>
#include <algorithm>

YOLOv11::YOLOv11(const std::string& model_path) {
    this->model_path = model_path;
    loadClassNames();

    // Only load model if path is not empty
    if (!model_path.empty()) {
        try {
            net = cv::dnn::readNetFromONNX(model_path);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "Model loaded successfully: " << model_path << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    } else {
        std::cout << "YOLOv11 initialized without model. Use setModel() to load a model." << std::endl;
    }
}

void YOLOv11::setConf(float conf) {
    conf_threshold = conf;
}

void YOLOv11::setNMS(float nms) {
    nms_threshold = nms;
}

void YOLOv11::setModel(const std::string& model_path) {
    this->model_path = model_path;

    // Only load model if path is not empty
    if (!model_path.empty()) {
        try {
            // Load new model
            net = cv::dnn::readNetFromONNX(model_path);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            std::cout << "Model loaded successfully: " << model_path << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    } else {
        std::cout << "Warning: Empty model path provided. No model loaded." << std::endl;
    }
}

void YOLOv11::preprocessImage(const cv::Mat& image) {
    original_size = image.size();
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, input_size, cv::Scalar(0,0,0), true, false, CV_32F);
}

void YOLOv11::runInference() {
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    output_blob = outputs[0];
}

void YOLOv11::postprocessResults() {
    detections.clear();

    // Transpose [1, 84, 8400] to [8400, 84]
    cv::Mat output;
    cv::transpose(output_blob.reshape(1, output_blob.size[1]), output);

    for (int i = 0; i < output.rows; ++i) {
        float* data = output.ptr<float>(i);

        float x_center = data[0];
        float y_center = data[1];
        float width = data[2];
        float height = data[3];

        // Get class scores (start from index 4)
        float* class_scores = data + 4;
        auto max_it = std::max_element(class_scores, class_scores + class_names.size());
        float confidence = *max_it;
        int class_id = static_cast<int>(max_it - class_scores);

        if (confidence >= conf_threshold && confidence <= 1.0f) {
            // Convert from normalized coordinates [0,1] to pixel coordinates
            float scale_x = static_cast<float>(original_size.width) / input_size.width;
            float scale_y = static_cast<float>(original_size.height) / input_size.height;

            int x = static_cast<int>((x_center - width / 2.0f) * scale_x);
            int y = static_cast<int>((y_center - height / 2.0f) * scale_y);
            int w = static_cast<int>(width * scale_x);
            int h = static_cast<int>(height * scale_y);

            // Clamp to image bounds
            x = std::max(0, std::min(x, original_size.width - 1));
            y = std::max(0, std::min(y, original_size.height - 1));
            w = std::max(1, std::min(w, original_size.width - x));
            h = std::max(1, std::min(h, original_size.height - y));

            cv::Rect box(x, y, w, h);

            Detection detection;
            detection.confidence = confidence;
            detection.class_id = class_id;
            detection.box = box;
            detection.class_name = getClassName(class_id);

            detections.push_back(detection);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (const auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    std::vector<Detection> filtered_detections;
    for (int idx : indices) {
        filtered_detections.push_back(detections[idx]);
    }
    detections = filtered_detections;
}

void YOLOv11::drawBoundingBoxes(cv::Mat& image) {
    for (const auto& det : detections) {

        // Line and text-box color (BGR)
        cv::Scalar box_color(39,254, 222);  // (225,255,40) RGB → BGR

        // Text color: black
        cv::Scalar text_color(0, 0, 0);

        // Draw bounding box
        cv::rectangle(image, det.box, box_color, 2);

        // Create label text
        std::string label = det.class_name + " " +
            std::to_string(det.confidence).substr(0, 4);

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.55;
        int thickness = 1;
        int baseline = 0;

        // Compute text size
        cv::Size text_size = cv::getTextSize(label, font_face,
                                             font_scale, thickness,
                                             &baseline);

        // ------------------------------------------
        // SHIFT requested:
        //   → move right +1 px
        //   → move up -2 px
        // ------------------------------------------
        int shift_x = 2;
        int shift_y = -4;

        // Base position (previous tight alignment)
        int text_x = det.box.x + 1 - 2 + shift_x;
        int text_y = det.box.y - 1 + shift_y;

        // If above screen, move to bottom
        if (text_y - text_size.height < 0) {
            text_y = det.box.y + det.box.height + text_size.height + 2;
        }

        cv::Point text_origin(text_x, text_y);

        // Background box (tight fit)
        cv::Rect bg_box(
            text_origin.x - 2,
            text_origin.y - text_size.height - 2,
            text_size.width + 4,
            text_size.height + baseline + 4
        );

        // Draw filled background
        cv::rectangle(image, bg_box, box_color, cv::FILLED);

        // Draw text
        cv::putText(image,
                    label,
                    cv::Point(text_origin.x, text_origin.y - 1),
                    font_face,
                    font_scale,
                    text_color,
                    thickness,
                    cv::LINE_AA);
    }
}

std::vector<Detection> YOLOv11::detect(const cv::Mat& image) {
    preprocessImage(image);
    runInference();
    postprocessResults();
    return detections;
}

std::vector<Detection> YOLOv11::getDetections() const {
    return detections;
}

void YOLOv11::loadClassNames() {
    class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}

std::string YOLOv11::getClassName(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(class_names.size())) {
        return class_names[class_id];
    }
    return "unknown";
}