#ifndef DEEPSORT_YOLO_TYPE_HPP
#define DEEPSORT_YOLO_TYPE_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <algorithm>

namespace Yolo_Type {
    struct Detection {
        int class_id; // 类别ID
        float confidence; // 置信度
        cv::Rect box; // 检测框
    };

    const std::vector<std::string> COCO_CLASSES = {
        "1", "2", "3", "4", "sentry"
    };

    constexpr int DEFAULT_INPUT_WIDTH = 640;
    constexpr int DEFAULT_INPUT_HEIGHT = 640;
    constexpr float DEFAULT_CONF_THRESHOLD = 0.5f;
    constexpr float DEFAULT_NMS_THRESHOLD = 0.45f;
    constexpr int DEFAULT_NUM_CLASSES = 80;

    template<int INPUT_WIDTH = DEFAULT_INPUT_WIDTH,
        int INPUT_HEIGHT = DEFAULT_INPUT_HEIGHT,
        float CONF_THRESH = DEFAULT_CONF_THRESHOLD,
        float NMS_THRESH = DEFAULT_NMS_THRESHOLD,
        int NUM_CLASSES = static_cast<int>(COCO_CLASSES.size())>
    class YoloDetector {
    public:
        explicit YoloDetector(const std::string &model_path = "../model/armor.onnx");

        template<typename MatType = cv::Mat>
        std::vector<Detection> detect(const MatType &image);

        template<typename MatType = cv::Mat>
        void draw_detections(MatType &img, const std::vector<Detection> &detections);

    private:
        template<typename MatType = cv::Mat>
        cv::Mat preprocess(const MatType &img);

        template<typename T = float>
        std::vector<Detection> postprocess(const cv::Mat &output, const cv::Mat &img);

        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_;
        std::vector<int64_t> input_shape_; // NCHW: [1,3,INPUT_HEIGHT,INPUT_WIDTH]
    };
}

namespace Yolo_Type {
    template<int W, int H, float C, float N, int NC>
    YoloDetector<W, H, C, N, NC>::YoloDetector(const std::string &model_path)
        : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU)) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11_Inference");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        input_shape_ = {1, 3, H, W}; // NCHW
    }

    template<int W, int H, float C, float N, int NC>
    template<typename MatType>
    std::vector<Detection> YoloDetector<W, H, C, N, NC>::detect(const MatType &image) {
        cv::Mat input_blob = preprocess(image);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        const char *input_name = input_name_ptr.get();
        Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        const char *output_name = output_name_ptr.get();

        // 创建输入张量
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, (float *) input_blob.data, input_blob.total(),
            input_shape_.data(), input_shape_.size()
        );

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1
        );

        auto *output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        cv::Mat output_mat(static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]), CV_32F, output_data);
        output_mat = output_mat.t();

        return postprocess(output_mat, cv::Mat(image));
    }

    template<int W, int H, float C, float N, int NC>
    template<typename MatType>
    void YoloDetector<W, H, C, N, NC>::draw_detections(MatType &img, const std::vector<Detection> &detections) {
        for (const auto &det: detections) {
            cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
            if (det.class_id >= 0 && det.class_id < NC) {
                std::string label = COCO_CLASSES[det.class_id] + " " + std::to_string(det.confidence).substr(0, 4);
                cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    template<int W, int H, float C, float N, int NC>
    template<typename MatType>
    cv::Mat YoloDetector<W, H, C, N, NC>::preprocess(const MatType &img) {
        float scale_x = static_cast<float>(W) / img.cols;
        float scale_y = static_cast<float>(H) / img.rows;
        float scale = std::min(scale_x, scale_y);
        int new_w = static_cast<int>(img.cols * scale);
        int new_h = static_cast<int>(img.rows * scale);

        cv::Mat resized;
        cv::resize(cv::Mat(img), resized, cv::Size(new_w, new_h)); // 兼容UMat

        cv::Mat input_img(H, W, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(input_img(cv::Rect((W - new_w) / 2, (H - new_h) / 2, new_w, new_h)));

        input_img.convertTo(input_img, CV_32F);
        input_img /= 255.0f;
        cv::dnn::blobFromImage(input_img, input_img, 1.0, cv::Size(), cv::Scalar(), true, false);

        return input_img;
    }

    template<int W, int H, float C, float N, int NC>
    template<typename T>
    std::vector<Detection> YoloDetector<W, H, C, N, NC>::postprocess(const cv::Mat &output, const cv::Mat &img) {
        std::vector<Detection> detections;

        int num_anchors = output.rows;
        int num_classes = NC;

        T *data = reinterpret_cast<T *>(output.data); // 模板化数据类型

        auto img_h = static_cast<float>(img.rows);
        auto img_w = static_cast<float>(img.cols);

        float scale = std::min(static_cast<float>(W) / img_w, static_cast<float>(H) / img_h);
        float pad_w = (W - img_w * scale) / 2.0f;
        float pad_h = (H - img_h * scale) / 2.0f;

        for (int i = 0; i < num_anchors; ++i) {
            // 提取置信度（利用模板参数C作为阈值）
            T *cls_ptr = data + 4;
            T conf = *std::max_element(cls_ptr, cls_ptr + num_classes);
            if (conf < static_cast<T>(C)) {
                data += num_classes + 4;
                continue;
            }

            // 提取类别ID和坐标
            int class_id = std::max_element(cls_ptr, cls_ptr + num_classes) - cls_ptr;
            float cx = static_cast<float>(data[0]);
            float cy = static_cast<float>(data[1]);
            float bw = static_cast<float>(data[2]);
            float bh = static_cast<float>(data[3]);

            // 转换xyxy并还原到原图尺寸
            float x1 = (cx - bw / 2.0f - pad_w) / scale;
            float y1 = (cy - bh / 2.0f - pad_h) / scale;
            float x2 = (cx + bw / 2.0f - pad_w) / scale;
            float y2 = (cy + bh / 2.0f - pad_h) / scale;

            // 边界裁剪
            x1 = std::max(0.0f, std::min(x1, img_w - 1));
            y1 = std::max(0.0f, std::min(y1, img_h - 1));
            x2 = std::max(0.0f, std::min(x2, img_w - 1));
            y2 = std::max(0.0f, std::min(y2, img_h - 1));

            detections.push_back({class_id, static_cast<float>(conf), cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2))});
            data += num_classes + 4;
        }

        // NMS
        std::vector<int> indices;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        for (const auto &det: detections) {
            confidences.push_back(det.confidence);
            boxes.push_back(det.box);
        }
        cv::dnn::NMSBoxes(boxes, confidences, C, N, indices);

        // 筛选NMS结果
        std::vector<Detection> result;
        for (int idx: indices) {
            result.push_back(detections[idx]);
        }
        return result;
    }
}


#endif //DEEPSORT_YOLO_TYPE_HPP
