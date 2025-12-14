#ifndef TARGET_TRACKING_BASIC_HPP
#define TARGET_TRACKING_BASIC_HPP

#include <chrono>
#include <memory>

#include "eigen3/Eigen/Eigen"
#include "Yolo.hpp"
#include "Deepsort/Deepsort.hpp"

namespace DeepSort {
    // 全局配置
    const float MAX_DISTANCE = 0.2; // 余弦距离阈值
    const int MAX_AGE = 30; // 最大未匹配帧数
    const int N_INIT = 3; // 初始匹配帧数
    const int FEATURE_DIM = 128; // 重识别特征维度
    const float IOU_THRESHOLD = 0.003; // IOU匹配阈值

    enum TrackeID {
        RED_1 = 1, //红方英雄
        RED_2 = 2, //红方工程
        RED_3 = 3, //红方3号步兵
        RED_4 = 4, //红方4号步兵
        RED_SENTRY = 5, //红方哨兵
        BLUE_1 = 6, //蓝方英雄
        BLUE_2 = 7, //蓝方工程
        BLUE_3 = 8, //蓝方3号步兵
        BLUE_4 = 9, //蓝方4号步兵
        BLUE_SENTRY = 10, //蓝方哨兵
    };

    struct BBox;

    // 装甲板结构体
    struct ArmorBBox {
        ArmorBBox() {
        }

        ArmorBBox(float x, float y, float x1_, float y1_, float confidence, Eigen::VectorXf matrix,
                  int id, TrackeID tracke_id) {
            x1 = x;
            y1 = y;
            x2 = x1_;
            y2 = y1_;
            score = confidence;
            feature = matrix;
            armor_number = id;
        }

        float x1, y1, x2, y2; // 左上角/右下角坐标
        float score; // 检测置信度
        Eigen::VectorXf feature; // 重识别特征
        //std::shared_ptr<BBox> father; //指向全车的结构体
        int armor_number; // 装甲板数字（对应TrackeID）
    };

    // 边界框结构体(这个是全车追踪的)
    struct BBox {
        BBox() : armor_bbox() {
        } ;

        BBox(float x, float y, float x1_, float y1_, float confidence, Eigen::VectorXf matrix,
             int id, TrackeID tracke_id) {
            x1 = x;
            y1 = y;
            x2 = x1_;
            y2 = y1_;
            score = confidence;
            feature = matrix;
        }

        float x1, y1, x2, y2; // 左上角/右下角坐标
        float score; // 检测置信度
        Eigen::VectorXf feature; // 重识别特征
        ArmorBBox armor_bbox; //所包含的装甲板
        int track_id = -1; // 跟踪ID
        TrackeID vehicle_id = static_cast<TrackeID>(-1); // 车辆类型ID（红/蓝方）
    };

    // 跟踪器状态
    enum TrackState {
        TENTATIVE, // 暂定
        CONFIRMED, // 确认
        DELETED // 删除
    };

    class Timer {
    private:
        std::chrono::steady_clock::time_point start;

    public:
        void startTimer() { start = std::chrono::steady_clock::now(); }

        float getElapsedMs() {
            auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<float, std::milli>(end - start).count();
        }
    };
}

#endif //TARGET_TRACKING_BASIC_HPP
