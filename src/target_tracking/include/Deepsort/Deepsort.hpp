#ifndef TARGET_TRACKING_DEEPSORT_HPP
#define TARGET_TRACKING_DEEPSORT_HPP

#include <eigen3/Eigen/Eigen>

#include "backdel/backdel.hpp"
#include "DeepsortHelper.hpp"

namespace DeepSort {

    // 核心算法
    class DeepSort {
    public:


    private:

        //级联匹配
        std::map<int, int> cascade_matching(const std::vector<BBox> &detections);

        //iou匹配
        static float iou(const BBox &a, const BBox &b);

        //余弦相似度
        static float cosine_distance(const Eigen::VectorXf &a, const Eigen::VectorXf &b);

        //匈牙利算法
        static std::vector<int> hungarian(const Eigen::MatrixXf &cost);
    };

    //控制器
    class DeepSortControl {
    public:

    private:
    };
}

#endif //TARGET_TRACKING_DEEPSORT_HPP
