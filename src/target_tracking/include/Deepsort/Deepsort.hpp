#ifndef TARGET_TRACKING_DEEPSORT_HPP
#define TARGET_TRACKING_DEEPSORT_HPP

#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

#include "backdel/backdel.hpp"
#include "DeepsortHelper.hpp"
#include "Yolo.hpp"

namespace DeepSort {
    struct OutputDetection {
        int id; // 跟踪ID
        float confidence; // 置信度
        cv::Rect box; // 边界框
        int armor_number; // 装甲板数字(-1表示未知)
        TrackID vehicle_id; // 车辆类型ID
    };

    // 核心算法类：专注于车辆追踪
    class DeepSort {
    public:
        DeepSort();

        // 车辆追踪核心函数
        std::vector<OutputDetection> track(
            const std::vector<BBox> &vehicle_detections,
            const std::vector<ArmorBBox> &armor_detections
        );

    private:
        DeepSortData &data_manager_; // 数据管理器引用

        // 级联匹配(车辆)
        std::map<int, int> cascade_matching(const std::vector<BBox> &detections);

        // IOU匹配
        static float iou(const BBox &a, const BBox &b);

        // 余弦相似度
        static float cosine_distance(const Eigen::VectorXf &a, const Eigen::VectorXf &b);

        // 匈牙利算法
        static std::vector<int> hungarian(const Eigen::MatrixXf &cost);

        // 装甲板与车辆匹配(仅用于未知ID车辆)
        void match_armor_to_unknown_vehicles(
            const std::vector<ArmorBBox> &armor_detections,
            std::vector<OutputDetection> &track_results
        );
    };

    // 控制器类：对外交互接口
    class DeepSortControl {
    public:
        DeepSortControl();

        // 外部接入函数
        std::vector<OutputDetection> track(
            const std::vector<Yolo_Type::Detection> &armor_detections,
            const std::vector<Yolo_Type::Detection> &vehicle_detections
        );

    private:
        DeepSort deepsort_; // 核心算法实例
        std::vector<ArmorBBox> armor_bbox_; // 装甲板检测结果
        std::vector<BBox> vehicle_bbox_; // 车辆检测结果
    };
}

#endif //TARGET_TRACKING_DEEPSORT_HPP
