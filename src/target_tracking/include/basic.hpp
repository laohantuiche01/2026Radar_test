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

    enum TrackID {
        HERO_1 = 1, //英雄
        ENGINEERING_2 = 2, //工程
        INFANTRY_3 = 3, //3号步兵
        INFANTRY_4 = 4, //4号步兵
        SENTRY = 5, //哨兵
    };

    struct BBox;

    // 装甲板结构体
    struct ArmorBBox {
        ArmorBBox() {
            x1 = 0;
            y1 = 0;
            x2 = 0;
            y2 = 0;
            score = 0;
            id = -1;
            armor_number = -1;
        }

        ArmorBBox(float x, float y, float x1_, float y1_, float confidence,
                  int id_, TrackID tracke_id) {
            x1 = x;
            y1 = y;
            x2 = x1_;
            y2 = y1_;
            score = confidence;
            id = id_;
            armor_number = tracke_id;
        }

        float x1, y1, x2, y2; // 左上角/右下角坐标
        float score; // 检测置信度
        int id;
        int armor_number; // 装甲板数字（对应TrackeID）
    };

    // 边界框结构体(这个是全车追踪的)
    struct BBox {
        BBox() : armor_bbox() {
            x1 = 0;
            y1 = 0;
            x2 = 0;
            y2 = 0;
            score = 0;
            vehicle_id = static_cast<TrackID>(-1);
            track_id = -1;
        } ;

        BBox(float x, float y, float x1_, float y1_, float confidence,
             int id_, TrackID tracke_id_) {
            x1 = x;
            y1 = y;
            x2 = x1_;
            y2 = y1_;
            score = confidence;
            vehicle_id = tracke_id_;
            track_id = id_;
        }

        float x1, y1, x2, y2; // 左上角/右下角坐标
        float score; // 检测置信度
        ArmorBBox armor_bbox; //所包含的装甲板
        int track_id = -1; // 跟踪ID
        TrackID vehicle_id = static_cast<TrackID>(-1); // 车辆类型ID（红/蓝方）
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
