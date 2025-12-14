#ifndef TARGET_TRACKING_DEEPSORTHELPER_HPP
#define TARGET_TRACKING_DEEPSORTHELPER_HPP


#include "Deepsort.hpp"
#include "Yolo.hpp"
#include "Tracker.hpp"
#include "ImageHandle/Image.hpp"
#include "backdel/backdel.hpp"

namespace DeepSort {
    // 数据转换接口类
    class Interface_ {
    public:
        // 将装甲板检测结果转换为ArmorBBox
        static std::vector<ArmorBBox> convert_armor_detections(
            const std::vector<ArmorBBox> &raw_armor_detections) {
            std::vector<ArmorBBox> armor_bboxes;
            for (const auto &det: raw_armor_detections) {
                ArmorBBox armor_bbox;
                armor_bbox.x1 = det.x1;
                armor_bbox.y1 = det.y1;
                armor_bbox.x2 = det.x2;
                armor_bbox.y2 = det.y2;
                armor_bbox.score = det.score;
                armor_bbox.feature = det.feature;
                armor_bbox.armor_number = det.armor_number;
                armor_bbox.father = nullptr; // 初始化为空，匹配后关联
                armor_bboxes.push_back(armor_bbox);
            }
            return armor_bboxes;
        }

        static std::vector<BBox> convert_bbox_detections(
            const std::vector<BBox> &raw_vehicle_detections) {
            std::vector<BBox> armor_bboxes;
            for (const auto &det: raw_vehicle_detections) {
                BBox vehicle_bbox;
                vehicle_bbox.x1 = det.x1;
                vehicle_bbox.y1 = det.y1;
                vehicle_bbox.x2 = det.x2;
                vehicle_bbox.y2 = det.y2;
                vehicle_bbox.score = det.score;
                vehicle_bbox.feature = det.feature;
                armor_bboxes.push_back(vehicle_bbox);
            }
            return armor_bboxes;
        }

        // 计算余弦距离（特征匹配）
        static float cosine_distance(const Eigen::VectorXf &feat1, const Eigen::VectorXf &feat2) {
            if (feat1.size() != feat2.size() || feat1.size() == 0) {
                return 1.0f; // 最大距离
            }
            float dot = feat1.dot(feat2);
            float norm1 = feat1.norm();
            float norm2 = feat2.norm();
            return 1.0f - (dot / (norm1 * norm2 + 1e-6f));
        }
    };


    class ArmorMatch {
    private:
        DeepSortData &data_manager_;

    public:
        //构造函数
        ArmorMatch(bool use_gmm = true, float min_contour_area = 500.0f)
            : data_manager_(DeepSortData::get_instance()) {
        }


        //计算两个BBox的中心距离
        static float calculate_center_distance(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
            cv::Point2f c1((armor_bbox.x1 + armor_bbox.x2) / 2, (armor_bbox.y1 + armor_bbox.y2) / 2);
            cv::Point2f c2((vehicle_bbox.x1 + vehicle_bbox.x2) / 2, (vehicle_bbox.y1 + vehicle_bbox.y2) / 2);
            return static_cast<float>(cv::norm(c1 - c2));
        }

        // 计算IOU（装甲板与车辆）
        static float calculate_iou(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
            float x1 = std::max(armor_bbox.x1, vehicle_bbox.x1);
            float y1 = std::max(armor_bbox.y1, vehicle_bbox.y1);
            float x2 = std::min(armor_bbox.x2, vehicle_bbox.x2);
            float y2 = std::min(armor_bbox.y2, vehicle_bbox.y2);

            float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area1 = (armor_bbox.x2 - armor_bbox.x1) * (armor_bbox.y2 - armor_bbox.y1);
            float area2 = (vehicle_bbox.x2 - vehicle_bbox.x1) * (vehicle_bbox.y2 - vehicle_bbox.y1);

            return inter_area / (area1 + area2 - inter_area + 1e-6f);
        }

        // 匹配装甲板到车辆
        static std::map<int, int> match_armor_to_vehicle(
            const std::vector<ArmorBBox> &armor_bboxes,
            const std::vector<BBox> &vehicle_bboxes) {
            std::map<int, int> armor_vehicle_map; // armor_idx -> vehicle_idx
            std::vector<bool> vehicle_matched(vehicle_bboxes.size(), false);

            // 对每个装甲板，找到最佳匹配的车辆
            for (size_t a_idx = 0; a_idx < armor_bboxes.size(); ++a_idx) {
                float best_score = 0.0f;
                int best_v_idx = -1;

                for (size_t v_idx = 0; v_idx < vehicle_bboxes.size(); ++v_idx) {
                    if (vehicle_matched[v_idx]) continue;

                    // 检查装甲板是否在车辆范围内
                    if (!is_armor_in_vehicle(armor_bboxes[a_idx], vehicle_bboxes[v_idx])) {
                        continue;
                    }

                    // 计算匹配分数（IOU + 距离 + 特征加权）
                    float iou = calculate_iou(armor_bboxes[a_idx], vehicle_bboxes[v_idx]);
                    float distance = calculate_center_distance(armor_bboxes[a_idx], vehicle_bboxes[v_idx]);
                    float distance_score = 1.0f / (1.0f + distance / 50.0f);
                    float cos_dist = Interface_::cosine_distance(armor_bboxes[a_idx].feature,
                                                                 vehicle_bboxes[v_idx].feature);
                    float feature_score = 1.0f - std::min(cos_dist, MAX_DISTANCE);

                    float match_score = 0.5f * iou + 0.3f * distance_score + 0.2f * feature_score;

                    if (match_score > best_score && iou > IOU_THRESHOLD) {
                        best_score = match_score;
                        best_v_idx = static_cast<int>(v_idx);
                    }
                }

                if (best_v_idx != -1) {
                    armor_vehicle_map[a_idx] = best_v_idx;
                    vehicle_matched[best_v_idx] = true;
                }
            }

            return armor_vehicle_map;
        }



        // 检查装甲板是否在车辆范围内
        static bool is_armor_in_vehicle(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
            const float expand_ratio = 1.2f; // 车辆边界框扩展比例
            BBox expanded_vehicle = vehicle_bbox;
            float w = vehicle_bbox.x2 - vehicle_bbox.x1;
            float h = vehicle_bbox.y2 - vehicle_bbox.y1;

            expanded_vehicle.x1 -= w * (expand_ratio - 1) / 2;
            expanded_vehicle.y1 -= h * (expand_ratio - 1) / 2;
            expanded_vehicle.x2 += w * (expand_ratio - 1) / 2;
            expanded_vehicle.y2 += h * (expand_ratio - 1) / 2;

            // 检查装甲板是否完全在扩展后的车辆框内
            return (armor_bbox.x1 >= expanded_vehicle.x1 &&
                    armor_bbox.y1 >= expanded_vehicle.y1 &&
                    armor_bbox.x2 <= expanded_vehicle.x2 &&
                    armor_bbox.y2 <= expanded_vehicle.y2);
        }

        // 核心匹配函数
        void Match(const std::vector<ArmorBBox> &armor_detect_result, const std::vector<BBox> &vehicle_detect_result,
                   const cv::Mat &frame) {
            //数据类型转换
            std::vector<BBox> vehicle_bboxes = Interface_::convert_bbox_detections(vehicle_detect_result);
            std::vector<ArmorBBox> armor_bboxes = Interface_::convert_armor_detections(armor_detect_result);
            data_manager_.predict_all_trackers();
            auto armor_vehicle_map = match_armor_to_vehicle(armor_bboxes, vehicle_bboxes);
            auto active_trackers = data_manager_.get_active_trackers();

            std::vector<bool> tracker_matched(active_trackers.size(), false);
            std::vector<bool> armor_matched(armor_bboxes.size(), false);

            for (size_t t_idx = 0; t_idx < active_trackers.size(); ++t_idx) {
                auto tracker = active_trackers[t_idx];
                float best_iou = 0.0f;
                int best_armor_idx = -1;
                // 寻找匹配的装甲板
                for (size_t a_idx = 0; a_idx < armor_bboxes.size(); ++a_idx) {
                    if (armor_matched[a_idx]) continue;
                    // 检查装甲板是否属于该车辆
                    if (armor_vehicle_map.count(a_idx) &&
                        tracker->is_in_vehicle(armor_bboxes[a_idx])) {
                        float iou = calculate_iou(armor_bboxes[a_idx], *tracker->get_vehicle_bbox());
                        if (iou > best_iou && iou > IOU_THRESHOLD) {
                            best_iou = iou;
                            best_armor_idx = static_cast<int>(a_idx);
                        }
                    }
                }

                // 更新匹配到的跟踪器
                if (best_armor_idx != -1) {
                    int vehicle_idx = armor_vehicle_map[best_armor_idx];
                    data_manager_.update_tracker(
                        tracker->get_track_id(),
                        armor_bboxes[best_armor_idx],
                        vehicle_bboxes[vehicle_idx]);

                    tracker_matched[t_idx] = true;
                    armor_matched[best_armor_idx] = true;
                } else {
                    // 无匹配装甲板，标记为遮挡
                    data_manager_.handle_occlusion(tracker->get_track_id());
                }
            }

            //创建新跟踪器
            for (size_t a_idx = 0; a_idx < armor_bboxes.size(); ++a_idx) {
                if (armor_matched[a_idx] || !armor_vehicle_map.count(a_idx)) continue;

                int vehicle_idx = armor_vehicle_map[a_idx];
                int armor_number = armor_bboxes[a_idx].armor_number;

                // 创建新跟踪器
                int track_id = data_manager_.create_tracker(
                    armor_bboxes[a_idx],
                    vehicle_bboxes[vehicle_idx],
                    armor_number);

                armor_matched[a_idx] = true;
            }

            data_manager_.clean_deleted_trackers();

            visualize_matching(frame, armor_bboxes, vehicle_bboxes, armor_vehicle_map);
        }

        // 可视化匹配结果
        void visualize_matching(const cv::Mat &frame,
                                const std::vector<ArmorBBox> &armor_bboxes,
                                const std::vector<BBox> &vehicle_bboxes,
                                const std::map<int, int> &armor_vehicle_map) {
            cv::Mat vis_image = frame.clone();

            for (const auto &v_bbox: vehicle_bboxes) {
                cv::rectangle(vis_image,
                              cv::Point(v_bbox.x1, v_bbox.y1),
                              cv::Point(v_bbox.x2, v_bbox.y2),
                              cv::Scalar(0, 255, 0), 2);
            }

            for (const auto &[a_idx, v_idx]: armor_vehicle_map) {
                const auto &a_bbox = armor_bboxes[a_idx];
                const auto &v_bbox = vehicle_bboxes[v_idx];

                cv::rectangle(vis_image,
                              cv::Point(a_bbox.x1, a_bbox.y1),
                              cv::Point(a_bbox.x2, a_bbox.y2),
                              cv::Scalar(0, 0, 255), 2);

                if (a_bbox.armor_number != -1) {
                    std::string num_str = std::to_string(a_bbox.armor_number);
                    std::string type_str;
                    switch (static_cast<TrackeID>(a_bbox.armor_number)) {
                        case RED_1: type_str = "RED_Hero";
                            break;
                        case RED_2: type_str = "RED_Engineer";
                            break;
                        case RED_3: type_str = "RED_Infantry3";
                            break;
                        case RED_4: type_str = "RED_Infantry4";
                            break;
                        case RED_SENTRY: type_str = "RED_Sentry";
                            break;
                        case BLUE_1: type_str = "BLUE_Hero";
                            break;
                        case BLUE_2: type_str = "BLUE_Engineer";
                            break;
                        case BLUE_3: type_str = "BLUE_Infantry3";
                            break;
                        case BLUE_4: type_str = "BLUE_Infantry4";
                            break;
                        case BLUE_SENTRY: type_str = "BLUE_Sentry";
                            break;
                        default: type_str = "UNKNOWN";
                    }
                    cv::putText(vis_image,
                                num_str + "(" + type_str + ")",
                                cv::Point(a_bbox.x1, a_bbox.y1 - 5),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.6, cv::Scalar(255, 255, 0), 2);
                }

                cv::Point2f a_center((a_bbox.x1 + a_bbox.x2) / 2, (a_bbox.y1 + a_bbox.y2) / 2);
                cv::Point2f v_center((v_bbox.x1 + v_bbox.x2) / 2, (v_bbox.y1 + v_bbox.y2) / 2);
                cv::line(vis_image, a_center, v_center, cv::Scalar(255, 0, 0), 2);
            }

            auto trackers = data_manager_.get_active_trackers();
            for (const auto &tracker: trackers) {
                auto v_bbox = tracker->get_vehicle_bbox();
                int armor_num = tracker->get_armor_number();
                std::string state_str;
                switch (tracker->get_state()) {
                    case TENTATIVE: state_str = "TENTATIVE";
                        break;
                    case CONFIRMED: state_str = "CONFIRMED";
                        break;
                    case DELETED: state_str = "DELETED";
                        break;
                }

                std::string text = "ID:" + std::to_string(tracker->get_track_id()) +
                                   " Num:" + std::to_string(armor_num) +
                                   " State:" + state_str;

                cv::putText(vis_image, text,
                            cv::Point(v_bbox->x1, v_bbox->y2 + 20),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.5, cv::Scalar(0, 255, 255), 2);
            }

            cv::imshow("Result", vis_image);
            cv::waitKey(1);
        }

        // 获取车辆对应的装甲板数字（核心接口）
        std::map<TrackeID, int> GetVehicleArmorNumber() {
            std::map<TrackeID, int> result;
            auto trackers = data_manager_.get_active_trackers();

            for (const auto &tracker: trackers) {
                result[tracker->get_vehicle_id()] = tracker->get_armor_number();
            }

            return result;
        }

        //track_id获取
        std::map<int, int> GetTrackIdArmorNumber() {
            std::map<int, int> result;
            auto trackers = data_manager_.get_active_trackers();

            for (const auto &tracker: trackers) {
                result[tracker->get_track_id()] = tracker->get_armor_number();
            }

            return result;
        }
    };
}

#endif //TARGET_TRACKING_DEEPSORTHELPER_HPP
