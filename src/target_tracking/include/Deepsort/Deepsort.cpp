#include "Deepsort.hpp"

std::map<int, int> DeepSort::DeepSort::cascade_matching(const std::vector<BBox> &detections) {
    std::map<int, int> matches; // track_idx -> det_idx
    std::vector<Tracker> active_trackers;
    for (auto &tracker: trackers) {
        if (tracker.state != DELETED) active_trackers.push_back(tracker);
    }

    // 构建代价矩阵（余弦距离+IOU）
    int n = active_trackers.size(), m = detections.size();
    Eigen::MatrixXf cost(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float cos_dist = cosine_distance(active_trackers[i].feature, detections[j].feature);
            float iou_dist = 1.0f - iou(active_trackers[i].get_bbox(), detections[j]);
            cost(i, j) = 0.7f * cos_dist + 0.3f * iou_dist; // 加权融合
        }
    }

    // 匈牙利匹配
    std::vector<int> match = hungarian(cost);
    for (int i = 0; i < n; i++) {
        if (match[i] != -1 && cost(i, match[i]) < MAX_DISTANCE) {
            matches[i] = match[i];
        }
    }

    // 未匹配的检测框，用IOU补充匹配
    std::vector<int> unmatched_dets;
    for (int j = 0; j < m; j++) {
        if (std::find(match.begin(), match.end(), j) == match.end()) {
            unmatched_dets.push_back(j);
        }
    }

    std::vector<int> unmatched_tracks;
    for (int i = 0; i < n; i++) {
        if (match[i] == -1) unmatched_tracks.push_back(i);
    }

    // IOU匹配
    Eigen::MatrixXf iou_cost(unmatched_tracks.size(), unmatched_dets.size());
    for (int i = 0; i < unmatched_tracks.size(); i++) {
        for (int j = 0; j < unmatched_dets.size(); j++) {
            iou_cost(i, j) = 1.0f - iou(
                                 active_trackers[unmatched_tracks[i]].get_bbox(),
                                 detections[unmatched_dets[j]]
                             );
        }
    }

    std::vector<int> iou_match = hungarian(iou_cost);
    for (int i = 0; i < unmatched_tracks.size(); i++) {
        if (iou_match[i] != -1 && iou_cost(i, iou_match[i]) < (1.0f - IOU_THRESHOLD)) {
            matches[unmatched_tracks[i]] = unmatched_dets[iou_match[i]];
        }
    }

    return matches;
}

// 计算两个bbox的IOU
float DeepSort::DeepSort::iou(const BBox &a, const BBox &b) {
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

// 计算余弦距离（特征相似度）
float DeepSort::DeepSort::cosine_distance(const Eigen::VectorXf &a, const Eigen::VectorXf &b) {
    float dot = a.dot(b);
    float norm_a = a.norm();
    float norm_b = b.norm();
    if (norm_a == 0 || norm_b == 0) return 1.0f;
    return 1.0f - (dot / (norm_a * norm_b));
}

// 匈牙利算法
std::vector<int> DeepSort::DeepSort::hungarian(const Eigen::MatrixXf &cost) {
    int n = static_cast<int>(cost.rows()), m = static_cast<int>(cost.cols());
    std::vector<int> match(n, -1);
    // 简化版匈牙利实现（完整实现需引入KM算法，此处适配小矩阵）
    // 实际生产建议使用开源KM算法库
    for (int i = 0; i < n; i++) {
        float min_cost = 1e9;
        int best_j = -1;
        for (int j = 0; j < m; j++) {
            if (std::find(match.begin(), match.end(), j) != match.end()) continue;
            if (cost(i, j) < min_cost) {
                min_cost = cost(i, j);
                best_j = j;
            }
        }
        if (best_j != -1) match[i] = best_j;
    }
    return match;
}
