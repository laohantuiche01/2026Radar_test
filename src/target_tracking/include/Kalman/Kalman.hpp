#ifndef TEST_ALGORITHM_KALMAN_HPP
#define TEST_ALGORITHM_KALMAN_HPP

#include "basic.hpp"
#include "eigen3/Eigen/Eigen"

namespace DeepSort {
    class KalmanFilter {
    private:
        int dim_x = 8;  // 状态维度 [x,y,a,h,vx,vy,va,vh]
        int dim_z = 4;  // 观测维度 [x,y,a,h]

        Eigen::MatrixXf F; // 状态转移矩阵
        Eigen::MatrixXf H; // 观测矩阵
        Eigen::MatrixXf Q; // 过程噪声协方差
        Eigen::MatrixXf R; // 观测噪声协方差
        Eigen::MatrixXf P; // 状态协方差矩阵
        Eigen::VectorXf x; // 状态向量

        // 计算IOU
        static float calculate_iou(const BBox& bbox1, const BBox& bbox2) {
            float x1 = std::max(bbox1.x1, bbox2.x1);
            float y1 = std::max(bbox1.y1, bbox2.y1);
            float x2 = std::min(bbox1.x2, bbox2.x2);
            float y2 = std::min(bbox1.y2, bbox2.y2);

            float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
            float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);

            return inter_area / (area1 + area2 - inter_area + 1e-6f);
        }

    public:
        [[nodiscard]] Eigen::VectorXf bbox_to_state(const BBox& bbox) const  {
            Eigen::VectorXf state = Eigen::VectorXf::Zero(dim_x);
            float center_x = (bbox.x1 + bbox.x2) / 2.0f;
            float center_y = (bbox.y1 + bbox.y2) / 2.0f;
            float aspect_ratio = (bbox.x2 - bbox.x1) / std::max(1.0f, (bbox.y2 - bbox.y1)); // 宽高比
            float height = bbox.y2 - bbox.y1;

            state(0) = center_x;
            state(1) = center_y;
            state(2) = aspect_ratio;
            state(3) = height;
            // 速度项初始为0
            return state;
        }

        //ArmorBBox转卡尔曼状态
        [[nodiscard]] Eigen::VectorXf bbox_to_state(const ArmorBBox& armor_bbox) const {
            Eigen::VectorXf state = Eigen::VectorXf::Zero(dim_x);
            float center_x = (armor_bbox.x1 + armor_bbox.x2) / 2.0f;
            float center_y = (armor_bbox.y1 + armor_bbox.y2) / 2.0f;
            float aspect_ratio = (armor_bbox.x2 - armor_bbox.x1) / std::max(1.0f, (armor_bbox.y2 - armor_bbox.y1));
            float height = armor_bbox.y2 - armor_bbox.y1;

            state(0) = center_x;
            state(1) = center_y;
            state(2) = aspect_ratio;
            state(3) = height;
            return state;
        }

        // 将卡尔曼状态转换为BBox（全车）
        static BBox state_to_bbox(const Eigen::VectorXf& state) {
            BBox bbox;
            float center_x = state(0);
            float center_y = state(1);
            float aspect_ratio = state(2);
            float height = state(3);
            float width = aspect_ratio * height;

            bbox.x1 = center_x - width / 2.0f;
            bbox.y1 = center_y - height / 2.0f;
            bbox.x2 = center_x + width / 2.0f;
            bbox.y2 = center_y + height / 2.0f;

            return bbox;
        }

        // 卡尔曼状态转ArmorBBox
        static ArmorBBox state_to_armor_bbox(const Eigen::VectorXf& state) {
            ArmorBBox armor_bbox;
            float center_x = state(0);
            float center_y = state(1);
            float aspect_ratio = state(2);
            float height = state(3);
            float width = aspect_ratio * height;

            armor_bbox.x1 = center_x - width / 2.0f;
            armor_bbox.y1 = center_y - height / 2.0f;
            armor_bbox.x2 = center_x + width / 2.0f;
            armor_bbox.y2 = center_y + height / 2.0f;

            return armor_bbox;
        }

        KalmanFilter() {
            // 初始化状态转移矩阵F
            F = Eigen::MatrixXf::Identity(dim_x, dim_x);
            F(0, 4) = F(1, 5) = F(2, 6) = F(3, 7) = 1.0f; // 位置 += 速度

            // 初始化观测矩阵H
            H = Eigen::MatrixXf::Zero(dim_z, dim_x);
            H(0, 0) = H(1, 1) = H(2, 2) = H(3, 3) = 1.0f;

            // 过程噪声Q
            Q = Eigen::MatrixXf::Zero(dim_x, dim_x);
            Q.block<4, 4>(4, 4) = Eigen::Matrix4f::Identity() * 0.01f;

            // 观测噪声R
            R = Eigen::MatrixXf::Identity(dim_z, dim_z) * 0.1f;

            // 初始协方差P
            P = Eigen::MatrixXf::Identity(dim_x, dim_x) * 10.0f;
        }

        Eigen::VectorXf predict() {
            x = F * x;
            P = F * P * F.transpose() + Q;
            return x;
        }

        Eigen::VectorXf update(const BBox& bbox) {
            Eigen::VectorXf z = bbox_to_state(bbox);
            Eigen::VectorXf y = z - H * x;
            Eigen::MatrixXf S = H * P * H.transpose() + R;
            Eigen::MatrixXf K = P * H.transpose() * S.inverse();

            x = x + K * y;
            P = (Eigen::MatrixXf::Identity(dim_x, dim_x) - K * H) * P;
            return x;
        }

        Eigen::VectorXf update(const ArmorBBox& armor_bbox) {
            Eigen::VectorXf z = bbox_to_state(armor_bbox);
            Eigen::VectorXf y = z - H * x;
            Eigen::MatrixXf S = H * P * H.transpose() + R;
            Eigen::MatrixXf K = P * H.transpose() * S.inverse();

            x = x + K * y;
            P = (Eigen::MatrixXf::Identity(dim_x, dim_x) - K * H) * P;
            return x;
        }

        void init(const BBox& bbox) {
            x = bbox_to_state(bbox);
            x.tail(4).setZero(); // 初始速度为0
        }

        void init(const ArmorBBox& armor_bbox) {
            x = bbox_to_state(armor_bbox);
            x.tail(4).setZero(); // 初始速度为0
        }

        BBox get_bbox() {
            return state_to_bbox(x);
        }

        ArmorBBox get_armor_bbox() {
            return state_to_armor_bbox(x);
        }

        [[nodiscard]] Eigen::VectorXf get_state() const { return x; }

        [[nodiscard]] Eigen::MatrixXf get_covariance() const { return P; }
    };
}


#endif //TEST_ALGORITHM_KALMAN_HPP
