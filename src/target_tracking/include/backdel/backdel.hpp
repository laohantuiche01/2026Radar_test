//
// Created by zxk on 2025/12/8.
//

#ifndef TARGET_TRACKING_BACKDEL_HPP
#define TARGET_TRACKING_BACKDEL_HPP

#include "basic.hpp"
#include "Deepsort/Tracker.hpp"
#include "Kalman/Kalman.hpp"


namespace DeepSort {
    // 数据管理单例类
    class DeepSortData {
    private:
        std::map<int, std::shared_ptr<Tracker>> trackers_; // track_id -> Tracker
        std::map<TrackeID, int> vehicle_armor_map_;        // 车辆类型ID -> 装甲板数字
        int next_track_id_ = 0;
        std::mutex mutex_;

        DeepSortData() = default;
        ~DeepSortData() = default;

    public:
        // 禁用拷贝
        DeepSortData(const DeepSortData&) = delete;
        DeepSortData& operator=(const DeepSortData&) = delete;

        // 单例实例
        static DeepSortData& get_instance() {
            static DeepSortData instance;
            return instance;
        }

        // 创建新跟踪器
        int create_tracker(const ArmorBBox& armor_bbox, const BBox& vehicle_bbox, int armor_number) {
            std::lock_guard<std::mutex> lock(mutex_);
            int track_id = next_track_id_++;
            auto tracker = std::make_shared<Tracker>(track_id);
            tracker->init(armor_bbox, vehicle_bbox, armor_number);
            trackers_[track_id] = tracker;
            vehicle_armor_map_[static_cast<TrackeID>(armor_number)] = armor_number;
            return track_id;
        }

        // 获取跟踪器
        std::shared_ptr<Tracker> get_tracker(int track_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (trackers_.count(track_id)) {
                return trackers_[track_id];
            }
            return nullptr;
        }

        // 获取所有活跃跟踪器（非DELETED状态）
        std::vector<std::shared_ptr<Tracker>> get_active_trackers() {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<std::shared_ptr<Tracker>> active;
            for (auto& [id, tracker] : trackers_) {
                if (tracker->get_state() != DELETED) {
                    active.push_back(tracker);
                }
            }
            return active;
        }

        // 更新跟踪器
        void update_tracker(int track_id, const ArmorBBox& armor_bbox, const BBox& vehicle_bbox) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (trackers_.count(track_id)) {
                trackers_[track_id]->update(armor_bbox, vehicle_bbox);
                // 更新车辆-装甲板映射
                if (armor_bbox.armor_number != -1) {
                    vehicle_armor_map_[static_cast<TrackeID>(armor_bbox.armor_number)] = armor_bbox.armor_number;
                }
            }
        }

        // 处理跟踪器遮挡
        void handle_occlusion(int track_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (trackers_.count(track_id)) {
                trackers_[track_id]->handle_occlusion();
            }
        }

        // 预测所有跟踪器
        void predict_all_trackers() {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [id, tracker] : trackers_) {
                tracker->predict();
            }
        }

        // 获取车辆对应的装甲板数字
        int get_armor_number(TrackeID vehicle_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (vehicle_armor_map_.count(vehicle_id)) {
                return vehicle_armor_map_[vehicle_id];
            }
            return -1;
        }

        // 重载：通过track_id获取装甲板数字
        int get_armor_number(int track_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (trackers_.count(track_id)) {
                return trackers_[track_id]->get_armor_number();
            }
            return -1;
        }

        // 清理删除状态的跟踪器
        void clean_deleted_trackers() {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = trackers_.begin(); it != trackers_.end();) {
                if (it->second->get_state() == DELETED) {
                    // 移除车辆-装甲板映射
                    TrackeID vid = it->second->get_vehicle_id();
                    vehicle_armor_map_.erase(vid);
                    it = trackers_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    };
}


#endif //TARGET_TRACKING_BACKDEL_HPP
