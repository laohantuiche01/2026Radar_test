#include "DeepsortHelper.hpp"

std::vector<DeepSort::ArmorBBox> DeepSort::Interface_::convert_armor_detections(
    const std::vector<Yolo_Type::Detection> &detections) {
    std::vector<ArmorBBox> result;
    for (const auto &det: detections) {
        result.emplace_back(
            static_cast<float>(det.box.x),
            static_cast<float>(det.box.y),
            static_cast<float>(det.box.x + det.box.width),
            static_cast<float>(det.box.y + det.box.height),
            static_cast<TrackID>(det.class_id)
        );
    }
    return result;
}

std::vector<DeepSort::BBox> DeepSort::Interface_::convert_bbox_detections(
    const std::vector<Yolo_Type::Detection> &detections) {
    std::vector<BBox> result;
    for (const auto &det: detections) {
        result.emplace_back(
            static_cast<float>(det.box.x),
            static_cast<float>(det.box.y),
            static_cast<float>(det.box.x + det.box.width),
            static_cast<float>(det.box.y + det.box.height),
            det.confidence,
            -1,
            static_cast<TrackID>(-1)
        );
    }
    return result;
}

//===========================================================================================================================

float DeepSort::ArmorMatch::calculate_center_distance(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
    cv::Point2f c1((armor_bbox.x1 + armor_bbox.x2) / 2, (armor_bbox.y1 + armor_bbox.y2) / 2);
    cv::Point2f c2((vehicle_bbox.x1 + vehicle_bbox.x2) / 2, (vehicle_bbox.y1 + vehicle_bbox.y2) / 2);
    return static_cast<float>(cv::norm(c1 - c2));
}

float DeepSort::ArmorMatch::calculate_iou(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
    float x1 = std::max(armor_bbox.x1, vehicle_bbox.x1);
    float y1 = std::max(armor_bbox.y1, vehicle_bbox.y1);
    float x2 = std::min(armor_bbox.x2, vehicle_bbox.x2);
    float y2 = std::min(armor_bbox.y2, vehicle_bbox.y2);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (armor_bbox.x2 - armor_bbox.x1) * (armor_bbox.y2 - armor_bbox.y1);
    float area2 = (vehicle_bbox.x2 - vehicle_bbox.x1) * (vehicle_bbox.y2 - vehicle_bbox.y1);

    return inter_area / (area1 + area2 - inter_area + 1e-6f);
}

std::map<int, int> DeepSort::ArmorMatch::match_armor_to_vehicle(const std::vector<ArmorBBox> &armor_bboxes,
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

            // 计算匹配分数（IOU + 距离）
            float iou = calculate_iou(armor_bboxes[a_idx], vehicle_bboxes[v_idx]);
            float distance = calculate_center_distance(armor_bboxes[a_idx], vehicle_bboxes[v_idx]);
            float distance_score = 1.0f / (1.0f + distance / 50.0f);
            // float cos_dist = Interface_::cosine_distance(armor_bboxes[a_idx].feature,
            //                                              vehicle_bboxes[v_idx].feature);
            // float feature_score = 1.0f - std::min(cos_dist, MAX_DISTANCE);
            //float match_score = 0.5f * iou + 0.3f * distance_score + 0.2f * feature_score;
            float match_score = 0.6f * iou + 0.4f * distance_score ;

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

bool DeepSort::ArmorMatch::is_armor_in_vehicle(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
    const float expand_ratio = 2.0f; // 车辆边界框扩展比例
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

void DeepSort::ArmorMatch::Match(const std::vector<ArmorBBox> &armor_bboxes,
                                 const std::vector<BBox> &vehicle_bboxes, const cv::Mat &frame) {
    //数据类型转换
    // std::vector<BBox> vehicle_bboxes = Interface_::convert_bbox_detections(vehicle_detect_result);
    // std::vector<ArmorBBox> armor_bboxes = Interface_::convert_armor_detections(armor_detect_result);
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
            if (armor_vehicle_map.count(a_idx)
              //  && tracker->is_in_vehicle(armor_bboxes[a_idx])
                ) {
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
        int track_id = data_manager_.create_tracker(vehicle_bboxes[vehicle_idx]);

        armor_matched[a_idx] = true;
    }

    data_manager_.clean_deleted_trackers();

    visualize_matching(frame, armor_bboxes, vehicle_bboxes, armor_vehicle_map);
}

void DeepSort::ArmorMatch::visualize_matching(const cv::Mat &frame, const std::vector<ArmorBBox> &armor_bboxes,
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
            switch (static_cast<TrackID>(a_bbox.armor_number)) {
                case HERO_1: type_str = "_Hero";
                    break;
                case ENGINEERING_2: type_str = "_Engineer";
                    break;
                case INFANTRY_3: type_str = "_Infantry3";
                    break;
                case INFANTRY_4: type_str = "_Infantry4";
                    break;
                case SENTRY: type_str = "_Sentry";
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

std::map<DeepSort::TrackID, int> DeepSort::ArmorMatch::GetVehicleArmorNumber() {
    std::map<TrackID, int> result;
    auto trackers = data_manager_.get_active_trackers();

    for (const auto &tracker: trackers) {
        result[tracker->get_vehicle_id()] = tracker->get_armor_number();
    }

    return result;
}

std::map<int, int> DeepSort::ArmorMatch::GetTrackIdArmorNumber() {
    std::map<int, int> result;
    auto trackers = data_manager_.get_active_trackers();

    for (const auto &tracker: trackers) {
        result[tracker->get_track_id()] = tracker->get_armor_number();
    }

    return result;
}


//=======================================================================================================================
