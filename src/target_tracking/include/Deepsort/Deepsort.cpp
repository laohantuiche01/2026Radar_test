#include "Deepsort.hpp"

namespace DeepSort {
    DeepSort::DeepSort() : data_manager_(DeepSortData::get_instance()) {}

    std::vector<OutputDetection> DeepSort::track(
        const std::vector<Yolo_Type::Detection>& vehicle_detections,
        const std::vector<ArmorBBox>& armor_detections
    ) {
        //转换车辆检测结果格式
        std::vector<BBox> vehicle_bboxes = convert_vehicle_detections(vehicle_detections);

        //预测所有跟踪器状态
        data_manager_.predict_all_trackers();

        //车辆级联匹配
        auto matches = cascade_matching(vehicle_bboxes);

        //更新跟踪器
        auto active_trackers = data_manager_.get_active_trackers();
        std::vector<bool> matched_trackers(active_trackers.size(), false);

        //更新已匹配的跟踪器
        for (const auto& [track_idx, det_idx] : matches) {
            auto tracker = active_trackers[track_idx];
            tracker->update(armor_bboxes[det_idx], vehicle_bboxes[det_idx]);
            matched_trackers[track_idx] = true;
        }

        //处理未匹配的跟踪器(遮挡处理)
        for (size_t i = 0; i < active_trackers.size(); ++i) {
            if (!matched_trackers[i]) {
                data_manager_.handle_occlusion(active_trackers[i]->get_track_id());
            }
        }

        //为未匹配的检测创建新跟踪器
        std::vector<bool> matched_dets(vehicle_bboxes.size(), false);
        for (const auto& [_, det_idx] : matches) {
            matched_dets[det_idx] = true;
        }
        for (size_t i = 0; i < vehicle_bboxes.size(); ++i) {
            if (!matched_dets[i]) {
                data_manager_.create_tracker(ArmorBBox(), vehicle_bboxes[i], -1);
            }
        }

        //清理已删除的跟踪器
        data_manager_.clean_deleted_trackers();

        //生成跟踪结果
        std::vector<OutputDetection> results;
        for (const auto& tracker : data_manager_.get_active_trackers()) {
            auto bbox = tracker->get_vehicle_bbox();
            results.push_back({
                tracker->get_track_id(),
                bbox->score,
                cv::Rect(bbox->x1, bbox->y1, bbox->x2 - bbox->x1, bbox->y2 - bbox->y1),
                tracker->get_armor_number(),
                tracker->get_vehicle_id()
            });
        }

        //对未知ID的车辆进行装甲板匹配
        match_armor_to_unknown_vehicles(armor_detections, results);

        return results;
    }

    std::map<int, int> DeepSort::cascade_matching(const std::vector<BBox>& detections) {
        std::map<int, int> matches;
        auto active_trackers = data_manager_.get_active_trackers();
        int n = active_trackers.size();
        int m = detections.size();

        if (n == 0 || m == 0) return matches;

        // 构建代价矩阵(余弦距离+IOU)
        Eigen::MatrixXf cost(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float cos_dist = cosine_distance(active_trackers[i]->get_vehicle_bbox()->feature, detections[j].feature);
                float iou_dist = 1.0f - iou(*active_trackers[i]->get_vehicle_bbox(), detections[j]);
                cost(i, j) = 0.7f * cos_dist + 0.3f * iou_dist;
            }
        }

        // 匈牙利算法匹配
        std::vector<int> match = hungarian(cost);
        for (int i = 0; i < n; ++i) {
            if (match[i] != -1 && cost(i, match[i]) < MAX_DISTANCE) {
                matches[i] = match[i];
            }
        }

        // 未匹配目标用IOU补充匹配
        std::vector<int> unmatched_tracks, unmatched_dets;
        for (int i = 0; i < n; ++i) {
            if (match[i] == -1) unmatched_tracks.push_back(i);
        }
        for (int j = 0; j < m; ++j) {
            if (std::find(match.begin(), match.end(), j) == match.end()) {
                unmatched_dets.push_back(j);
            }
        }

        Eigen::MatrixXf iou_cost(unmatched_tracks.size(), unmatched_dets.size());
        for (int i = 0; i < unmatched_tracks.size(); ++i) {
            for (int j = 0; j < unmatched_dets.size(); ++j) {
                iou_cost(i, j) = 1.0f - iou(
                    *active_trackers[unmatched_tracks[i]]->get_vehicle_bbox(),
                    detections[unmatched_dets[j]]
                );
            }
        }

        std::vector<int> iou_match = hungarian(iou_cost);
        for (int i = 0; i < unmatched_tracks.size(); ++i) {
            if (iou_match[i] != -1 && iou_cost(i, iou_match[i]) < (1.0f - IOU_THRESHOLD)) {
                matches[unmatched_tracks[i]] = unmatched_dets[iou_match[i]];
            }
        }

        return matches;
    }

    void DeepSort::match_armor_to_unknown_vehicles(
        const std::vector<ArmorBBox>& armor_detections,
        std::vector<OutputDetection>& track_results
    ) {
        // 筛选出未知ID的车辆跟踪结果
        std::vector<OutputDetection*> unknown_vehicles;
        for (auto& res : track_results) {
            if (res.vehicle_id == static_cast<TrackeID>(-1) || res.armor_number == -1) {
                unknown_vehicles.push_back(&res);
            }
        }

        if (unknown_vehicles.empty() || armor_detections.empty()) return;

        // 装甲板与未知车辆匹配
        for (auto* vehicle : unknown_vehicles) {
            float best_score = 0.0f;
            int best_armor_idx = -1;
            cv::Rect vehicle_rect = vehicle->box;
            BBox vehicle_bbox{
                static_cast<float>(vehicle_rect.x),
                static_cast<float>(vehicle_rect.y),
                static_cast<float>(vehicle_rect.x + vehicle_rect.width),
                static_cast<float>(vehicle_rect.y + vehicle_rect.height),
                vehicle->confidence,
                Eigen::VectorXf(),
                ArmorBBox(),
                vehicle->id,
                vehicle->vehicle_id
            };

            for (size_t i = 0; i < armor_detections.size(); ++i) {
                if (ArmorMatch::is_armor_in_vehicle(armor_detections[i], vehicle_bbox)) {
                    float iou = ArmorMatch::calculate_iou(armor_detections[i], vehicle_bbox);
                    float dist_score = 1.0f / (1.0f + ArmorMatch::calculate_center_distance(armor_detections[i], vehicle_bbox) / 50.0f);
                    float match_score = 0.6f * iou + 0.4f * dist_score;

                    if (match_score > best_score && iou > IOU_THRESHOLD) {
                        best_score = match_score;
                        best_armor_idx = i;
                    }
                }
            }

            // 更新车辆ID信息
            if (best_armor_idx != -1) {
                vehicle->armor_number = armor_detections[best_armor_idx].armor_number;
                vehicle->vehicle_id = static_cast<TrackeID>(vehicle->armor_number);
                auto tracker = data_manager_.get_tracker(vehicle->id);
                if (tracker) {
                    tracker->update(armor_detections[best_armor_idx], vehicle_bbox);
                }
            }
        }
    }

    // 辅助函数实现
    std::vector<BBox> DeepSort::convert_vehicle_detections(const std::vector<Yolo_Type::Detection>& detections) {
        std::vector<BBox> result;
        for (const auto& det : detections) {
            result.push_back({
                static_cast<float>(det.box.x),
                static_cast<float>(det.box.y),
                static_cast<float>(det.box.x + det.box.width),
                static_cast<float>(det.box.y + det.box.height),
                det.confidence,
                Eigen::VectorXf(),  // 假设特征后续会填充
                ArmorBBox(),
                -1,
                static_cast<TrackeID>(-1)
            });
        }
        return result;
    }

    float DeepSort::iou(const BBox& a, const BBox& b) {
        float x1 = std::max(a.x1, b.x1);
        float y1 = std::max(a.y1, b.y1);
        float x2 = std::min(a.x2, b.x2);
        float y2 = std::min(a.y2, b.y2);
        if (x2 < x1 || y2 < y1) return 0.0f;

        float inter = (x2 - x1) * (y2 - y1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (area_a + area_b - inter);
    }

    float DeepSort::cosine_distance(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
        if (a.size() != b.size() || a.size() == 0) return 1.0f;
        float dot = a.dot(b);
        float norm_a = a.norm();
        float norm_b = b.norm();
        return 1.0f - (dot / (norm_a * norm_b + 1e-6f));
    }

    std::vector<int> DeepSort::hungarian(const Eigen::MatrixXf& cost) {
        int n = cost.rows(), m = cost.cols();
        std::vector<int> match(n, -1);
        std::vector<bool> used(m, false);

        for (int i = 0; i < n; ++i) {
            float min_cost = 1e9;
            int best_j = -1;
            for (int j = 0; j < m; ++j) {
                if (!used[j] && cost(i, j) < min_cost) {
                    min_cost = cost(i, j);
                    best_j = j;
                }
            }
            if (best_j != -1) {
                match[i] = best_j;
                used[best_j] = true;
            }
        }
        return match;
    }
}

#include "Deepsort.hpp"

namespace DeepSort {
    DeepSortControl::DeepSortControl() {}

    std::vector<OutputDetection> DeepSortControl::track(
        const Yolo_Type::Detection& armor_detections,
        const Yolo_Type::Detection& vehicle_detections
    ) {
        // 转换装甲板检测结果
        armor_bbox_ = Interface_::convert_armor_detections(armor_detections);

        // 调用核心算法进行跟踪
        return deepsort_.track(vehicle_detections, armor_bbox_);
    }
}