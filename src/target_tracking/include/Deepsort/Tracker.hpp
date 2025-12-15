#ifndef TEST_ALGORITHM_TRACKER_HPP
#define TEST_ALGORITHM_TRACKER_HPP

#include "basic.hpp"
#include "Kalman/Kalman.hpp"
#include <opencv2/opencv.hpp>

namespace DeepSort {
    class Tracker {
    private:
        KalmanFilter kf_; // 卡尔曼滤波器
        TrackState state_ = TENTATIVE; // 跟踪状态
        const int track_id_; // 跟踪ID（不可修改）
        TrackID vehicle_id_ = static_cast<TrackID>(-1); // 车辆类型ID
        int armor_number_ = -1; // 装甲板数字
        int consecutive_occlusion_ = 0; // 连续遮挡帧数
        int hits_ = 0; // 连续匹配成功帧数
        int age_ = 0; // 跟踪器总存活帧数
        std::shared_ptr<BBox> vehicle_bbox_; // 车辆边界框
        cv::Rect2f last_known_vehicle_box_; // 最后已知的车辆框（用于遮挡时参考）

        // 状态转换逻辑
        void update_state() {
            if (state_ == DELETED) return;

            // 暂定状态 -> 确认状态：满足初始匹配帧数
            if (state_ == TENTATIVE && hits_ >= N_INIT) {
                state_ = CONFIRMED;
            }
            // 任何状态 -> 删除状态：超过最大遮挡帧数
            else if (consecutive_occlusion_ >= MAX_AGE) {
                state_ = DELETED;
            }
        }

    public:
        // 显式构造函数，强制指定track_id
        explicit Tracker(int track_id) : track_id_(track_id) {
            vehicle_bbox_ = std::make_shared<BBox>();
        }

        // 禁用拷贝
        Tracker(const Tracker &) = delete;

        Tracker &operator=(const Tracker &) = delete;

        // 移动构造
        Tracker(Tracker &&) = default;

        Tracker &operator=(Tracker &&) = default;

        // 初始化跟踪器
        void init(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox, int armor_number) {
            kf_.init(armor_bbox); // 用车辆初始化卡尔曼
            *vehicle_bbox_ = vehicle_bbox;
            bind_armor(armor_bbox, armor_number); // 绑定装甲板信息
            last_known_vehicle_box_ = cv::Rect2f(vehicle_bbox.x1, vehicle_bbox.y1,
                                                 vehicle_bbox.x2 - vehicle_bbox.x1,
                                                 vehicle_bbox.y2 - vehicle_bbox.y1);
            hits_ = 1;
            age_ = 1;
            update_state(); // 初始状态检查
        }

        // 仅初始化车辆跟踪
        void init(const BBox &vehicle_bbox) {
            kf_.init(vehicle_bbox); // 用车辆框初始化卡尔曼
            *vehicle_bbox_ = vehicle_bbox;
            last_known_vehicle_box_ = cv::Rect2f(vehicle_bbox.x1, vehicle_bbox.y1,
                                                 vehicle_bbox.x2 - vehicle_bbox.x1,
                                                 vehicle_bbox.y2 - vehicle_bbox.y1);
            hits_ = 1;
            age_ = 1;
            update_state(); // 初始状态检查
        }

        // 预测下一帧位置
        void predict() {
            if (state_ == DELETED) return;

            kf_.predict();
            age_++;

            // 未确认跟踪器若未达到匹配帧数，直接标记删除
            if (state_ == TENTATIVE && hits_ < N_INIT) {
                state_ = DELETED;
            }
        }

        // 更新跟踪器
        void update(const ArmorBBox &armor_bbox, const BBox &vehicle_bbox) {
            if (state_ == DELETED) return;

            // 更新卡尔曼滤波
            kf_.update(vehicle_bbox);
            *vehicle_bbox_ = vehicle_bbox;
            last_known_vehicle_box_ = cv::Rect2f(vehicle_bbox.x1, vehicle_bbox.y1,
                                                 vehicle_bbox.x2 - vehicle_bbox.x1,
                                                 vehicle_bbox.y2 - vehicle_bbox.y1);

            // 绑定装甲板信息（若存在）
            if (armor_bbox.armor_number != -1) {
                bind_armor(armor_bbox, armor_bbox.armor_number);
            }

            // 重置遮挡计数，更新匹配计数
            consecutive_occlusion_ = 0;
            hits_++;
            update_state(); // 检查状态转换
        }

        // 仅更新车辆框
        void update_vehicle(const BBox &vehicle_bbox) {
            if (state_ == DELETED) return;

            kf_.update(vehicle_bbox); // 用车辆框更新卡尔曼
            *vehicle_bbox_ = vehicle_bbox;
            last_known_vehicle_box_ = cv::Rect2f(vehicle_bbox.x1, vehicle_bbox.y1,
                                                 vehicle_bbox.x2 - vehicle_bbox.x1,
                                                 vehicle_bbox.y2 - vehicle_bbox.y1);

            consecutive_occlusion_ = 0;
            hits_++;
            update_state();
        }

        // 处理遮挡情况
        void handle_occlusion() {
            if (state_ == DELETED) return;

            consecutive_occlusion_++;
            age_++;
            update_state(); // 检查是否需要删除
        }

        // 绑定装甲板到车辆
        void bind_armor(const ArmorBBox &armor_bbox, int armor_number) {
            vehicle_bbox_->armor_bbox = armor_bbox;
            //vehicle_bbox_->armor_bbox.father = vehicle_bbox_;
            armor_number_ = armor_number;
            vehicle_id_ = static_cast<TrackID>(armor_number_); // 装甲板数字映射到车辆ID
        }

        // 检查装甲板是否在车辆范围内
        [[nodiscard]] bool is_armor_in_vehicle(const ArmorBBox &armor_bbox) const {
            if (!vehicle_bbox_) return false;

            const float offset = 20.0f;
            const auto &v = *vehicle_bbox_;
            return (armor_bbox.x1 >= v.x1 - offset &&
                    armor_bbox.y1 >= v.y1 - offset &&
                    armor_bbox.x2 <= v.x2 + offset &&
                    armor_bbox.y2 <= v.y2 + offset &&
                    armor_bbox.x1 < armor_bbox.x2 && // 确保装甲板框有效
                    armor_bbox.y1 < armor_bbox.y2);
        }

        // 获取最后已知的车辆框
        [[nodiscard]] cv::Rect2f get_last_known_vehicle_box() const {
            return last_known_vehicle_box_;
        }

        // 以下为只读属性获取方法
        [[nodiscard]] int get_track_id() const { return track_id_; }
        [[nodiscard]] int get_armor_number() const { return armor_number_; }
        [[nodiscard]] TrackID get_vehicle_id() const { return vehicle_id_; }
        [[nodiscard]] std::shared_ptr<BBox> get_vehicle_bbox() const { return vehicle_bbox_; }
        //[[nodiscard]] ArmorBBox get_armor_bbox() const { return kf_.get_armor_bbox(); }
        [[nodiscard]] TrackState get_state() const { return state_; }
        [[nodiscard]] int get_age() const { return age_; }
        [[nodiscard]] int get_consecutive_occlusion() const { return consecutive_occlusion_; }
        [[nodiscard]] int get_hits() const { return hits_; }

        // 强制标记为删除
        void mark_deleted() { state_ = DELETED; }
    };
}

#endif // TEST_ALGORITHM_TRACKER_HPP
