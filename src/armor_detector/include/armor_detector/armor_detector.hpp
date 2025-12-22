// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under Apache License 2.0.
//
// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ARMOR_DETECTOR_DETECTOR_HPP_
#define ARMOR_DETECTOR_DETECTOR_HPP_

// std
#include <cmath>
#include <string>
#include <vector>
// third party 
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
// project
#include "armor_detector/light_corner_corrector.hpp"
#include "armor_detector/types.hpp"
#include "armor_detector/number_classifier.hpp"

#include "interfaces/rm_msgs.h"

namespace auto_aim {
    class Detector {
    public:
        struct LightParams {
            // width / height
            double min_ratio;       //最小的宽高比
            double max_ratio;       //最大的宽高比
            // vertical angle
            double max_angle;       //竖直方向最大角度
            // judge color
            int color_diff_thresh;      //颜色差异阈值
            int min_h, max_h, min_s, max_s, min_v, max_v;   //hsv筛选

            double min_length;
            double max_length;
        };

        struct ArmorParams {
            double min_light_ratio;
            // light pairs distance
            double min_small_center_distance;
            double max_small_center_distance;
            double min_large_center_distance;
            double max_large_center_distance;
            //NOTE 两个灯条最大夹角（经过绝对值处理后的夹角）
            double max_relative_angle;
            // horizontal angle
            double max_angle;
        };

        Detector(const int &bin_thres, const EnemyColor &color, const LightParams &l,
                 const ArmorParams &a);

        std::vector<Armor> detect(const cv::Mat &input) noexcept;   //外部主要要的接受函数，                   输入图像，得到容器

        cv::Mat preprocessImage(const cv::Mat &input) noexcept;     //图像的预处理，                          输入图像，返回hsv筛选出的mask

        std::vector<Light> findLights(const cv::Mat &rbg_img,
                                      const cv::Mat &binary_img) noexcept;  //寻找灯条的函数                   输入原图像和预处理的mask图像，返回筛选出的灯条

        std::vector<Armor> matchLights(const std::vector<Light> &lights) noexcept;  //将灯条结合为机器人容器     输入为灯条的容器，返回一个敌人的容器

        // For debug usage
        cv::Mat getAllNumbersImage() const noexcept;    //返回框选到的Mat图像                                   debug使用，返回只有装甲板的图像

        void drawResults(cv::Mat &img) const noexcept;      //画出可视化目标                                   输入为图像，对图像操作，不返回值

        // Parameters
        int binary_thres;
        EnemyColor detect_color;
        LightParams light_params;
        ArmorParams armor_params;

        std::unique_ptr<NumberClassifier> classifier;
        std::unique_ptr<LightCornerCorrector> corner_corrector;

        // Debug msgs
        cv::Mat binary_img;
        std::vector<DebugLight> debug_lights;
        std::vector<DebugArmor> debug_armors;

    private:
        bool isLight(const Light &possible_light) noexcept; //判断是否是灯条                                   输入为可能的判断目标，返回为bool

        bool containLight(const int i, const int j, const std::vector<Light> &lights) noexcept; //判断两个灯条是否为同一个目标上的  输入为灯条索引与容器，返回bool

        ArmorType isArmor(const Light &light_1, const Light &light_2) noexcept; //判断是否为敌人               输入为两个灯条结构体，返回枚举元素

        cv::Mat gray_img_;

        std::vector<Light> lights_;
        std::vector<Armor> armors_;
    };
} // namespace fyt::auto_aim

#endif // ARMOR_DETECTOR_DETECTOR_HPP_
