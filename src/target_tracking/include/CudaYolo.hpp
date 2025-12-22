#ifndef TARGET_TRACKING_CUDAYOLO_HPP
#define TARGET_TRACKING_CUDAYOLO_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <numeric>

namespace TensorRT_Yolo_Type {
    struct Detection {
        int class_id; // 类别ID
        float confidence; // 置信度
        cv::Rect box; // 检测框
    };

    const std::vector<std::string> DEFAULT_CLASSES = {
        "1", "2", "3", "4", "sentry"
    };

    constexpr int DEFAULT_INPUT_WIDTH = 640;
    constexpr int DEFAULT_INPUT_HEIGHT = 640;
    constexpr float DEFAULT_CONF_THRESHOLD = 0.5f;
    constexpr float DEFAULT_NMS_THRESHOLD = 0.45f;

    /**
     *  日志类
     */
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char *msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };

    /**
     * TensorRT引擎封装类
     */
    class TensorRTEngine {
    public:
        TensorRTEngine() = default;

        ~TensorRTEngine();

        bool loadEngine(const std::string &enginePath);

        bool convertOnnxToEngine(const std::string &onnxPath,
                                 const std::string &enginePath,
                                 int maxBatchSize = 1,
                                 bool fp16 = true);

        void *getInputBuffer() { return buffers[inputIndex]; }
        void *getOutputBuffer() { return buffers[outputIndex]; }
        float *getOutput() { return prob; }
        [[nodiscard]] int getOutputSize() const { return outputSize; }
        [[nodiscard]] int getInputWidth() const { return inputWidth; }
        [[nodiscard]] int getInputHeight() const { return inputHeight; }

        bool inference();

    private:
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime{nullptr};
        std::unique_ptr<nvinfer1::ICudaEngine> engine{nullptr};
        std::unique_ptr<nvinfer1::IExecutionContext> context{nullptr};

        void *buffers[2];
        int inputIndex;
        int outputIndex;

        float *prob{nullptr};
        int outputSize;
        int inputWidth;
        int inputHeight;

        cudaStream_t stream{};
    };

    /**
     * 主要的接口类
     * @tparam INPUT_WIDTH 输入图像宽度
     * @tparam INPUT_HEIGHT 输入图像高度
     * @tparam CONF_THRESH 置信度
     * @tparam NMS_THRESH NMS值
     */
    template<int INPUT_WIDTH = DEFAULT_INPUT_WIDTH,
        int INPUT_HEIGHT = DEFAULT_INPUT_HEIGHT,
        float CONF_THRESH = DEFAULT_CONF_THRESHOLD,
        float NMS_THRESH = DEFAULT_NMS_THRESHOLD>
    class YoloDetectorTRT {
    public:
        explicit YoloDetectorTRT(const std::string &model_path = "../model/armor.onnx",
                                 std::string engine_path = "../model/armor.engine",
                                 const std::vector<std::string> &CLASSES = DEFAULT_CLASSES,
                                 bool force_rebuild = false);


        ~YoloDetectorTRT();

        std::vector<Detection> detect(const cv::Mat &image);

        void draw_detections(cv::Mat &img, const std::vector<Detection> &detections);

        float get_inference_time();

    private:
        const std::vector<std::string> classes_;
        std::string engine_path_;
        TensorRTEngine engine_;
        cv::Mat input_blob_;
        cudaEvent_t start_, stop_;

        void preprocess(const cv::Mat &img);

        std::vector<Detection> postprocess(float *output, const cv::Size &img_size);

        bool convert_onnx_to_trt(const std::string &onnx_path,
                                 const std::string &engine_path,
                                 bool fp16 = true);
    };
}

/**
 * TensorRTEngine实现方法
 */
namespace TensorRT_Yolo_Type {
    inline TensorRTEngine::~TensorRTEngine() {
        cudaFree(buffers[inputIndex]);
        cudaFree(buffers[outputIndex]);
        delete[] prob;
    }

    inline bool TensorRTEngine::loadEngine(const std::string &enginePath) {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile) {
            std::cerr << "Failed to open engine file: " << enginePath << std::endl;
            return false;
        }
        assert(engineFile.is_open());

        engineFile.seekg(0, std::ios::end);
        size_t engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(engineSize);
        engineFile.read(engineData.data(), static_cast<long>(engineSize));

        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "Failed to create runtime" << std::endl;
            return false;
        }

        engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }

        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }

        // 获取输入输出信息
        inputIndex = engine->getBindingIndex("images");
        outputIndex = engine->getBindingIndex("output0");

        auto inputDims = engine->getBindingDimensions(inputIndex);
        auto outputDims = engine->getBindingDimensions(outputIndex);

        inputHeight = inputDims.d[2];
        inputWidth = inputDims.d[3];
        outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; ++i) {
            outputSize *= outputDims.d[i];
        }

        // 分配GPU内存
        cudaMalloc(&buffers[inputIndex], inputWidth * inputHeight * 3 * sizeof(float));
        cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));

        // 分配CPU内存用于结果
        prob = new float[outputSize];

        // 创建CUDA流
        cudaStreamCreate(&stream);

        return true;
    }

    inline bool TensorRTEngine::convertOnnxToEngine(const std::string &onnxPath, const std::string &enginePath,
                                                    int maxBatchSize, bool fp16) {
    }

    inline bool TensorRTEngine::inference() {
    }
}

/**
 * YoloDetectorTRT实现方法
 */
namespace TensorRT_Yolo_Type {
    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::YoloDetectorTRT(
        const std::string &model_path, std::string engine_path, const std::vector<std::string> &CLASSES,
        bool force_rebuild) : classes_(CLASSES), engine_path_(std::move(engine_path)), start_(nullptr), stop_(nullptr) {
        // 检查是否存在引擎文件，如果不存在则从ONNX转换
        std::ifstream engineFile(engine_path);
        if (force_rebuild || !engineFile.good()) {
            std::cout << "Building TensorRT engine from ONNX..." << std::endl;
            if (!engine_.convertOnnxToEngine(model_path, engine_path)) {
                std::cerr << "Failed to convert ONNX to TensorRT engine" << std::endl;
                return;
            }
            std::cout << "Engine built successfully." << std::endl;
        }

        // 加载引擎
        if (!engine_.loadEngine(engine_path)) {
            std::cerr << "Failed to load TensorRT engine" << std::endl;
            return;
        }

        // 预分配内存
        input_blob_.create(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC3);

        // 创建CUDA事件用于计时
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::~YoloDetectorTRT() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    std::vector<Detection> YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::detect(
        const cv::Mat &image) {
        std::vector<Detection> detections;

        // 预处理
        cudaEventRecord(start_);
        preprocess(image);
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);

        // 推理
        cudaEventRecord(start_);
        if (!engine_.inference()) {
            std::cerr << "Inference failed" << std::endl;
            return detections;
        }
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);

        // 后处理
        cudaEventRecord(start_);
        detections = postprocess(engine_.getOutput(), image.size());
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);

        return detections;
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    void YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::draw_detections(
        cv::Mat &img, const std::vector<Detection> &detections) {
        for (const auto &det: detections) {
            cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
            if (det.class_id >= 0 && det.class_id < classes_.size()) {
                std::string label = classes_[det.class_id] + " " +
                                    std::to_string(det.confidence).substr(0, 4);
                cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    float YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::get_inference_time() {
        float time_ms = 0;
        cudaEventElapsedTime(&time_ms, start_, stop_);
        return time_ms;
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    void YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::preprocess(const cv::Mat &img) {
        // 计算缩放比例
        float scale_x = static_cast<float>(INPUT_WIDTH) / img.cols;
        float scale_y = static_cast<float>(INPUT_HEIGHT) / img.rows;
        float scale = std::min(scale_x, scale_y);

        int new_w = static_cast<int>(img.cols * scale);
        int new_h = static_cast<int>(img.rows * scale);

        // 调整大小并填充
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h));

        cv::Mat input_img(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(input_img(cv::Rect((INPUT_WIDTH - new_w) / 2,
                                          (INPUT_HEIGHT - new_h) / 2,
                                          new_w, new_h)));

        // 转换为float并归一化
        input_img.convertTo(input_img, CV_32FC3);
        input_img /= 255.0f;

        // 转换为CHW格式
        std::vector<cv::Mat> channels;
        cv::split(input_img, channels);

        // 将数据复制到GPU
        float *gpu_input = static_cast<float *>(engine_.getInputBuffer());
        size_t channel_size = INPUT_WIDTH * INPUT_HEIGHT;

        for (int i = 0; i < 3; ++i) {
            cudaMemcpyAsync(gpu_input + i * channel_size,
                            channels[i].data,
                            channel_size * sizeof(float),
                            cudaMemcpyHostToDevice);
        }
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    std::vector<Detection> YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::postprocess(
        float *output, const cv::Size &img_size) {
        std::vector<Detection> detections;
        const int DEFAULT_MAX_OUTPUT_BBOX_COUNT = 16; //这里可能有一个限制

        int num_detections = static_cast<int>(output[0]);
        if (num_detections > DEFAULT_MAX_OUTPUT_BBOX_COUNT) {
            num_detections = DEFAULT_MAX_OUTPUT_BBOX_COUNT;
        }

        auto img_w = static_cast<float>(img_size.width);
        auto img_h = static_cast<float>(img_size.height);

        float scale = std::min(static_cast<float>(INPUT_WIDTH) / img_w,
                               static_cast<float>(INPUT_HEIGHT) / img_h);
        float pad_w = (INPUT_WIDTH - img_w * scale) / 2.0f;
        float pad_h = (INPUT_HEIGHT - img_h * scale) / 2.0f;

        for (int i = 0; i < num_detections; ++i) {
            float *det_ptr = output + 1 + i * 7; // 7个值: [batch_id, class_id, confidence, x1, y1, x2, y2]

            int class_id = static_cast<int>(det_ptr[1]);
            float confidence = det_ptr[2];

            if (confidence < CONF_THRESH) continue;

            // 还原到原图坐标
            float x1 = (det_ptr[3] * INPUT_WIDTH - pad_w) / scale;
            float y1 = (det_ptr[4] * INPUT_HEIGHT - pad_h) / scale;
            float x2 = (det_ptr[5] * INPUT_WIDTH - pad_w) / scale;
            float y2 = (det_ptr[6] * INPUT_HEIGHT - pad_h) / scale;

            // 边界裁剪
            x1 = std::max(0.0f, std::min(x1, img_w - 1));
            y1 = std::max(0.0f, std::min(y1, img_h - 1));
            x2 = std::max(0.0f, std::min(x2, img_w - 1));
            y2 = std::max(0.0f, std::min(y2, img_h - 1));

            if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;

            detections.push_back({
                class_id,
                confidence,
                cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                         cv::Point(static_cast<int>(x2), static_cast<int>(y2)))
            });
        }

        // NMS
        if (detections.empty()) return detections;

        std::vector<int> indices;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (const auto &det: detections) {
            confidences.push_back(det.confidence);
            boxes.push_back(det.box);
        }

        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH, indices);

        std::vector<Detection> result;
        for (int idx: indices) {
            result.push_back(detections[idx]);
        }

        return result;
    }

    template<int INPUT_WIDTH, int INPUT_HEIGHT, float CONF_THRESH, float NMS_THRESH>
    bool YoloDetectorTRT<INPUT_WIDTH, INPUT_HEIGHT, CONF_THRESH, NMS_THRESH>::convert_onnx_to_trt(
        const std::string &onnx_path, const std::string &engine_path, bool fp16) {
        TensorRTEngine engine;
        return engine.convertOnnxToEngine(onnx_path, engine_path, 1, fp16);
    }
}

#endif //TARGET_TRACKING_CUDAYOLO_HPP
