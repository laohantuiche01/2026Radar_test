#ifndef TEST_ALGORITHM_TRACKER_HPP
#define TEST_ALGORITHM_TRACKER_HPP

#include "basic.hpp"
#include "Kalman/Kalman.hpp"

namespace DeepSort {
    class Tracker {
    private:
        KalmanFilter kf_;
        TrackState state_ = TENTATIVE; // 跟踪状态
        int track_id_ = -1; // 跟踪ID
        TrackeID vehicle_id_ = static_cast<TrackeID>(-1); // 车辆类型ID
        int armor_number_ = -1; // 绑定的装甲板数字
        int consecutive_occlusion_ = 0; // 连续遮挡帧数
        int hits_ = 0; // 连续匹配帧数
        int age_ = 0; // 跟踪器存活帧数

        // 全车边界框
        std::shared_ptr<BBox> vehicle_bbox_;

    public:
        explicit Tracker(int track_id) : track_id_(track_id) {
            vehicle_bbox_ = std::make_shared<BBox>();
        }

        // 初始化跟踪器
        void init(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox, int armor_number) {
            kf_.init(armor_bbox);
            *vehicle_bbox_ = vehicle_bbox;
            vehicle_bbox_->armor_bbox = armor_bbox;
            vehicle_bbox_->armor_bbox.father = vehicle_bbox_;
            armor_number_ = armor_number;
            vehicle_id_ = static_cast<TrackeID>(armor_number);
            hits_ = 1;
            age_ = 1;

            // 初始状态：满足N_INIT则确认
            if (hits_ >= N_INIT) {
                state_ = CONFIRMED;
            }
        }

        //只跟踪车辆,未识别到装甲板
        void init(const BBox &vehicle_bbox) {
            kf_.init(vehicle_bbox);

        }

        // 预测下一帧位置
        void predict() {
            if (state_ != DELETED) {
                kf_.predict();
                age_++;
                if (state_ == TENTATIVE && hits_ < N_INIT) {
                    state_ = DELETED; // 未达到初始匹配帧数，删除
                }
            }
        }

        // 更新跟踪器
        void update(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
            if (state_ == DELETED) return;

            // 更新卡尔曼滤波
            kf_.update(armor_bbox);

            // 更新车辆和装甲板信息
            *vehicle_bbox_ = vehicle_bbox;
            vehicle_bbox_->armor_bbox = armor_bbox;
            vehicle_bbox_->armor_bbox.father = vehicle_bbox_;
            vehicle_bbox_->track_id = track_id_;

            // 更新装甲板数字和车辆ID
            if (armor_bbox.armor_number != -1) {
                armor_number_ = armor_bbox.armor_number;
                vehicle_id_ = static_cast<TrackeID>(armor_number_);
            }

            // 更新状态
            hits_++;
            consecutive_occlusion_ = 0;
            if (state_ == TENTATIVE && hits_ >= N_INIT) {
                state_ = CONFIRMED;
            }
        }

        // 处理遮挡
        void handle_occlusion() {
            if (state_ == DELETED) return;

            consecutive_occlusion_++;
            age_++;
            if (consecutive_occlusion_ >= MAX_AGE) {
                state_ = DELETED;
            }
        }

        // 获取跟踪ID
        [[nodiscard]] int get_track_id() const { return track_id_; }

        // 获取装甲板数字
        [[nodiscard]] int get_armor_number() const { return armor_number_; }

        // 获取车辆类型ID
        [[nodiscard]] TrackeID get_vehicle_id() const { return vehicle_id_; }

        // 获取车辆边界框
        [[nodiscard]] std::shared_ptr<BBox> get_vehicle_bbox() const { return vehicle_bbox_; }

        // 获取装甲板预测边界框
        ArmorBBox get_armor_bbox() { return kf_.get_armor_bbox(); }

        // 获取跟踪状态
        [[nodiscard]] TrackState get_state() const { return state_; }

        // 检查装甲板是否在车辆范围内
        [[nodiscard]] bool is_in_vehicle(const ArmorBBox &armor_bbox) const {
            const float offset = 20.0f;
            return (armor_bbox.x1 >= vehicle_bbox_->x1 - offset &&
                    armor_bbox.y1 >= vehicle_bbox_->y1 - offset &&
                    armor_bbox.x2 <= vehicle_bbox_->x2 + offset &&
                    armor_bbox.y2 <= vehicle_bbox_->y2 + offset);
        }

        // 获取存活帧数
        [[nodiscard]] int get_age() const { return age_; }

        // 标记为删除
        void mark_deleted() { state_ = DELETED; }
    };
}


#endif //TEST_ALGORITHM_TRACKER_HPP
