#ifndef TARGET_TRACKING_DEEPSORTHELPER_HPP
#define TARGET_TRACKING_DEEPSORTHELPER_HPP


#include "Deepsort.hpp"
#include "Yolo.hpp"
#include "Tracker.hpp"
#include "backdel/backdel.hpp"

namespace DeepSort {
    // 数据转换接口类
    class Interface_ {
    public:
        // 将装甲板检测结果转换为ArmorBBox
        static std::vector<ArmorBBox> convert_armor_detections(
            const std::vector<Yolo_Type::Detection> &raw_armor_detections);

        static std::vector<BBox> convert_bbox_detections(
            const std::vector<Yolo_Type::Detection> &raw_vehicle_detections);

    };

    class ArmorMatch {
    private:
        DeepSortData &data_manager_;

    public:
        //构造函数
        ArmorMatch() : data_manager_(DeepSortData::get_instance()) {
        }

        //计算两个BBox的中心距离
        static float calculate_center_distance(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox);

        // 计算IOU（装甲板与车辆）
        static float calculate_iou(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox);

        // 匹配装甲板到车辆
        static std::map<int, int> match_armor_to_vehicle(
            const std::vector<ArmorBBox> &armor_bboxes,
            const std::vector<BBox> &vehicle_bboxes);


        // 检查装甲板是否在车辆范围内
        static bool is_armor_in_vehicle(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox);

        // 核心匹配函数
        void Match(const std::vector<ArmorBBox> &armor_detect_result, const std::vector<BBox> &vehicle_detect_result,
                   const cv::Mat &frame);

        // 可视化匹配结果
        void visualize_matching(const cv::Mat &frame,
                                const std::vector<ArmorBBox> &armor_bboxes,
                                const std::vector<BBox> &vehicle_bboxes,
                                const std::map<int, int> &armor_vehicle_map);

        // 获取车辆对应的装甲板数字（核心接口）
        std::map<TrackeID, int> GetVehicleArmorNumber();

        //track_id获取
        std::map<int, int> GetTrackIdArmorNumber();
    };
}

#endif //TARGET_TRACKING_DEEPSORTHELPER_HPP
