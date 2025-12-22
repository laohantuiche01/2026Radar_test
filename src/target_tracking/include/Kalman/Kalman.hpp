#ifndef TEST_ALGORITHM_KALMAN_HPP
#define TEST_ALGORITHM_KALMAN_HPP

#include "basic.hpp"
#include "eigen3/Eigen/Eigen"

namespace DeepSort {
    // class KalmanFilter {
    // private:
    //     int dim_x = 8; // 状态维度 [x,y,a,h,vx,vy,va,vh]
    //     int dim_z = 4; // 观测维度 [x,y,a,h]
    //
    //     Eigen::MatrixXf F; // 状态转移矩阵
    //     Eigen::MatrixXf H; // 观测矩阵（关键修复点）
    //     Eigen::MatrixXf Q; // 过程噪声协方差
    //     Eigen::MatrixXf R; // 观测噪声协方差
    //     Eigen::MatrixXf P; // 状态协方差矩阵
    //     Eigen::VectorXf x; // 修复：将RowVectorXf改为VectorXf（列向量）
    //
    //     // 计算IOU（保持不变）
    //     static float calculate_iou(const BBox &bbox1, const BBox &bbox2) {
    //         float x1 = std::max(bbox1.x1, bbox2.x1);
    //         float y1 = std::max(bbox1.y1, bbox2.y1);
    //         float x2 = std::min(bbox1.x2, bbox2.x2);
    //         float y2 = std::min(bbox1.y2, bbox2.y2);
    //
    //         float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    //         float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
    //         float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
    //
    //         return inter_area / (area1 + area2 - inter_area + 1e-6f);
    //     }
    //
    // public:
    //     [[nodiscard]] Eigen::VectorXf bbox_to_state(const BBox &bbox) const {
    //         Eigen::VectorXf state = Eigen::VectorXf::Zero(dim_x);
    //         float center_x = (bbox.x1 + bbox.x2) / 2.0f;
    //         float center_y = (bbox.y1 + bbox.y2) / 2.0f;
    //         float aspect_ratio = (bbox.x2 - bbox.x1) / std::max(1.0f, (bbox.y2 - bbox.y1));
    //         float height = bbox.y2 - bbox.y1;
    //
    //         state(0) = center_x;
    //         state(1) = center_y;
    //         state(2) = aspect_ratio;
    //         state(3) = height;
    //         return state;
    //     }
    //
    //     [[nodiscard]] Eigen::VectorXf bbox_to_state(const ArmorBBox &armor_bbox) const {
    //         Eigen::VectorXf state = Eigen::VectorXf::Zero(dim_x);
    //         float center_x = (armor_bbox.x1 + armor_bbox.x2) / 2.0f;
    //         float center_y = (armor_bbox.y1 + armor_bbox.y2) / 2.0f;
    //         float aspect_ratio = (armor_bbox.x2 - armor_bbox.x1) / std::max(1.0f, (armor_bbox.y2 - armor_bbox.y1));
    //         float height = armor_bbox.y2 - armor_bbox.y1;
    //
    //         state(0) = center_x;
    //         state(1) = center_y;
    //         state(2) = aspect_ratio;
    //         state(3) = height;
    //         return state;
    //     }
    //
    //     static BBox state_to_bbox(const Eigen::VectorXf &state) {
    //         BBox bbox;
    //         float center_x = state(0);
    //         float center_y = state(1);
    //         float aspect_ratio = state(2);
    //         float height = state(3);
    //         float width = aspect_ratio * height;
    //
    //         bbox.x1 = center_x - width / 2.0f;
    //         bbox.y1 = center_y - height / 2.0f;
    //         bbox.x2 = center_x + width / 2.0f;
    //         bbox.y2 = center_y + height / 2.0f;
    //
    //         return bbox;
    //     }
    //
    //     static ArmorBBox state_to_armor_bbox(const Eigen::VectorXf &state) {
    //         ArmorBBox armor_bbox;
    //         float center_x = state(0);
    //         float center_y = state(1);
    //         float aspect_ratio = state(2);
    //         float height = state(3);
    //         float width = aspect_ratio * height;
    //
    //         armor_bbox.x1 = center_x - width / 2.0f;
    //         armor_bbox.y1 = center_y - height / 2.0f;
    //         armor_bbox.x2 = center_x + width / 2.0f;
    //         armor_bbox.y2 = center_y + height / 2.0f;
    //
    //         return armor_bbox;
    //     }
    //
    //     KalmanFilter() {
    //         // 初始化状态转移矩阵F (8x8)
    //         F = Eigen::MatrixXf::Identity(dim_x, dim_x);
    //         F(0, 4) = F(1, 5) = F(2, 6) = F(3, 7) = 1.0f;
    //
    //         // 修复：观测矩阵H应为4x8（观测维度x状态维度）
    //         H = Eigen::MatrixXf::Zero(dim_z, dim_x);  // 关键修复：将(dim_x, dim_z)改为(dim_z, dim_x)
    //         H(0, 0) = H(1, 1) = H(2, 2) = H(3, 3) = 1.0f;
    //
    //         // 过程噪声Q (8x8)
    //         Q = Eigen::MatrixXf::Zero(dim_x, dim_x);
    //         Q.block<4, 4>(4, 4) = Eigen::Matrix4f::Identity() * 0.01f;
    //
    //         // 观测噪声R (4x4)
    //         R = Eigen::MatrixXf::Identity(dim_z, dim_z) * 0.1f;
    //
    //         // 初始协方差P (8x8)
    //         P = Eigen::MatrixXf::Identity(dim_x, dim_x) * 10.0f;
    //
    //         // 初始化状态向量为列向量
    //         x = Eigen::VectorXf::Zero(dim_x);
    //     }
    //
    //     Eigen::VectorXf predict() {
    //         x = F * x;
    //         P = F * P * F.transpose() + Q;
    //         return x;
    //     }
    //
    //     Eigen::VectorXf update(const BBox &bbox) {
    //         Eigen::VectorXf z = bbox_to_state(bbox);  // z是4x1列向量
    //         Eigen::VectorXf y = z - H * x;  // 修复后维度匹配：4x1 = 4x1 - (4x8 * 8x1)
    //         Eigen::MatrixXf S = H * P * H.transpose() + R;
    //         Eigen::MatrixXf K = P * H.transpose() * S.inverse();
    //
    //         x = x + K * y;
    //         P = (Eigen::MatrixXf::Identity(dim_x, dim_x) - K * H) * P;
    //         return x;
    //     }
    //
    //     // 修复：将RowVectorXf改为VectorXf，确保维度一致
    //     Eigen::VectorXf update(const ArmorBBox &armor_bbox) {
    //         Eigen::VectorXf z = bbox_to_state(armor_bbox);  // 修复：使用VectorXf而非RowVectorXf
    //         Eigen::VectorXf y = z - H * x;  // 维度匹配：4x1 = 4x1 - (4x8 * 8x1)
    //         Eigen::MatrixXf S = H * P * H.transpose() + R;
    //         Eigen::MatrixXf K = P * H.transpose() * S.inverse();
    //
    //         x = x + K * y;
    //         P = (Eigen::MatrixXf::Identity(dim_x, dim_x) - K * H) * P;
    //         return x;
    //     }
    //
    //     void init(const BBox &bbox) {
    //         x = bbox_to_state(bbox);
    //         x.tail(4).setZero();
    //     }
    //
    //     void init(const ArmorBBox &armor_bbox) {
    //         x = bbox_to_state(armor_bbox);
    //         x.tail(4).setZero();
    //     }
    //
    //     BBox get_bbox() {
    //         return state_to_bbox(x);
    //     }
    //
    //     ArmorBBox get_armor_bbox() {
    //         return state_to_armor_bbox(x);
    //     }
    //
    //     [[nodiscard]] Eigen::VectorXf get_state() const { return x; }
    //
    //     [[nodiscard]] Eigen::MatrixXf get_covariance() const { return P; }
    // };

    class KalmanFilter {
    private:
        int dim_x = 8; // 状态维度 [x,y,a,h,vx,vy,va,vh]
        int dim_z = 4; // 观测维度 [x,y,a,h]

        cv::KalmanFilter kf; // OpenCV卡尔曼滤波器实例
        cv::Mat state; // 状态向量 (dim_x x 1)
        cv::Mat measurement; // 观测向量 (dim_z x 1)

        // 计算IOU
        static float calculate_iou(const BBox &bbox1, const BBox &bbox2) {
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
        [[nodiscard]] Eigen::VectorXf bbox_to_state(const BBox &bbox) const {
            Eigen::VectorXf state = Eigen::VectorXf::Zero(dim_x);
            float center_x = (bbox.x1 + bbox.x2) / 2.0f;
            float center_y = (bbox.y1 + bbox.y2) / 2.0f;
            float aspect_ratio = (bbox.x2 - bbox.x1) / std::max(1.0f, (bbox.y2 - bbox.y1));
            float height = bbox.y2 - bbox.y1;

            state(0) = center_x;
            state(1) = center_y;
            state(2) = aspect_ratio;
            state(3) = height;
            return state;
        }

        [[nodiscard]] Eigen::VectorXf bbox_to_state(const ArmorBBox &armor_bbox) const {
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

        static BBox state_to_bbox(const Eigen::VectorXf &state) {
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

        static ArmorBBox state_to_armor_bbox(const Eigen::VectorXf &state) {
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
            // 初始化OpenCV卡尔曼滤波器
            kf.init(dim_x, dim_z, 0);

            // 状态转移矩阵F (8x8)
            kf.transitionMatrix = cv::Mat::eye(dim_x, dim_x, CV_32F);
            kf.transitionMatrix.at<float>(0, 4) = 1.0f; // x += vx
            kf.transitionMatrix.at<float>(1, 5) = 1.0f; // y += vy
            kf.transitionMatrix.at<float>(2, 6) = 1.0f; // a += va
            kf.transitionMatrix.at<float>(3, 7) = 1.0f; // h += vh

            // 观测矩阵H (4x8)
            kf.measurementMatrix = cv::Mat::zeros(dim_z, dim_x, CV_32F);
            kf.measurementMatrix.at<float>(0, 0) = 1.0f; // 观测x
            kf.measurementMatrix.at<float>(1, 1) = 1.0f; // 观测y
            kf.measurementMatrix.at<float>(2, 2) = 1.0f; // 观测a
            kf.measurementMatrix.at<float>(3, 3) = 1.0f; // 观测h

            // 过程噪声协方差Q
            kf.processNoiseCov = cv::Mat::zeros(dim_x, dim_x, CV_32F);
            cv::Mat q_block = 0.01f * cv::Mat::eye(4, 4, CV_32F);
            q_block.copyTo(kf.processNoiseCov(cv::Rect(4, 4, 4, 4))); // 速度项噪声

            // 观测噪声协方差R
            kf.measurementNoiseCov = 0.1f * cv::Mat::eye(dim_z, dim_z, CV_32F);

            // 初始状态协方差P
            kf.errorCovPost = 10.0f * cv::Mat::eye(dim_x, dim_x, CV_32F);

            // 初始化状态向量
            state = cv::Mat::zeros(dim_x, 1, CV_32F);
            measurement = cv::Mat::zeros(dim_z, 1, CV_32F);
            kf.statePost = state.clone();
        }

        Eigen::VectorXf predict() {
            cv::Mat prediction = kf.predict();
            state = prediction.clone();

            // 转换为Eigen向量返回
            Eigen::VectorXf eigen_pred(dim_x);
            for (int i = 0; i < dim_x; ++i) {
                eigen_pred(i) = prediction.at<float>(i);
            }
            return eigen_pred;
        }

        Eigen::VectorXf update(const BBox &bbox) {
            Eigen::VectorXf z_eigen = bbox_to_state(bbox);

            // 转换为OpenCV矩阵
            for (int i = 0; i < dim_z; ++i) {
                measurement.at<float>(i) = z_eigen(i);
            }

            // 执行更新
            cv::Mat updated = kf.correct(measurement);
            state = updated.clone();

            // 转换为Eigen向量返回
            Eigen::VectorXf eigen_updated(dim_x);
            for (int i = 0; i < dim_x; ++i) {
                eigen_updated(i) = updated.at<float>(i);
            }
            return eigen_updated;
        }

        Eigen::VectorXf update(const ArmorBBox &armor_bbox) {
            Eigen::VectorXf z_eigen = bbox_to_state(armor_bbox);

            // 转换为OpenCV矩阵
            for (int i = 0; i < dim_z; ++i) {
                measurement.at<float>(i) = z_eigen(i);
            }

            // 执行更新
            cv::Mat updated = kf.correct(measurement);
            state = updated.clone();

            // 转换为Eigen向量返回
            Eigen::VectorXf eigen_updated(dim_x);
            for (int i = 0; i < dim_x; ++i) {
                eigen_updated(i) = updated.at<float>(i);
            }
            return eigen_updated;
        }

        void init(const BBox &bbox) {
            Eigen::VectorXf init_state = bbox_to_state(bbox);

            // 初始化状态向量
            for (int i = 0; i < dim_x; ++i) {
                state.at<float>(i) = init_state(i);
            }
            // 速度项初始化为0
            for (int i = 4; i < dim_x; ++i) {
                state.at<float>(i) = 0.0f;
            }
            kf.statePost = state.clone();
        }

        void init(const ArmorBBox &armor_bbox) {
            Eigen::VectorXf init_state = bbox_to_state(armor_bbox);

            // 初始化状态向量
            for (int i = 0; i < dim_x; ++i) {
                state.at<float>(i) = init_state(i);
            }
            // 速度项初始化为0
            for (int i = 4; i < dim_x; ++i) {
                state.at<float>(i) = 0.0f;
            }
            kf.statePost = state.clone();
        }

        BBox get_bbox() {
            // 转换状态向量为Eigen格式
            Eigen::VectorXf eigen_state(dim_x);
            for (int i = 0; i < dim_x; ++i) {
                eigen_state(i) = state.at<float>(i);
            }
            return state_to_bbox(eigen_state);
        }

        ArmorBBox get_armor_bbox() {
            // 转换状态向量为Eigen格式
            Eigen::VectorXf eigen_state(dim_x);
            for (int i = 0; i < dim_x; ++i) {
                eigen_state(i) = state.at<float>(i);
            }
            return state_to_armor_bbox(eigen_state);
        }

        [[nodiscard]] Eigen::VectorXf get_state() const {
            Eigen::VectorXf eigen_state(dim_x);
            for (int i = 0; i < dim_x; ++i) {
                eigen_state(i) = state.at<float>(i);
            }
            return eigen_state;
        }

        [[nodiscard]] Eigen::MatrixXf get_covariance() const {
            Eigen::MatrixXf eigen_cov(dim_x, dim_x);
            for (int i = 0; i < dim_x; ++i) {
                for (int j = 0; j < dim_x; ++j) {
                    eigen_cov(i, j) = kf.errorCovPost.at<float>(i, j);
                }
            }
            return eigen_cov;
        }
    };
}

#endif //TEST_ALGORITHM_KALMAN_HPP
