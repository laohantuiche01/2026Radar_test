// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
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

#ifndef ARMOR_DETECTOR_DETECTOR_NODE_HPP_
#define ARMOR_DETECTOR_DETECTOR_NODE_HPP_

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// std
#include <memory>
#include <string>
#include <vector>
// project
#include "armor_detector/armor_detector.hpp"
#include "armor_detector/number_classifier.hpp"

#include "utils/utils.h"

namespace ckyf {
    using namespace auto_aim;
    // Armor Detector Node
    // Subscribe to the image topic, run the armor detection alogorithm and publish
    // the detected armors
    class DetectorNode : public rclcpp::Node {
    public:
        DetectorNode();

    private:
        std::string imageTopic;

        void imageCallback(sensor_msgs::msg::Image::SharedPtr img_msg);   //受到图像信息的回调函数

        // void targetCallback(const rm_interfaces::msg::Target::SharedPtr
        // target_msg);

        std::unique_ptr<Detector> initDetector();  //创建工厂类的函数  ->armor_detector.hpp->Detector

        std::vector<Armor>
        detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg); //在图像的回调函数中调用     输入图像信息，返回Armor的容器

        void createDebugPublishers() noexcept;  //创建debug发布

        void destroyDebugPublishers() noexcept;     //销毁debug发布

        int selfColor;

        // Dynamic Parameter
        std::vector<std::string> classNames;

        rcl_interfaces::msg::SetParametersResult
        onSetParameters(std::vector<rclcpp::Parameter> parameters); //设置参数回调（动态调整参数）

        rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr on_set_parameters_callback_handle_;

        // Armor Detector
        std::unique_ptr<Detector> detector_;

        // Visualization marker publisher
        visualization_msgs::msg::Marker armor_marker_;
        visualization_msgs::msg::Marker text_marker_;
        visualization_msgs::msg::MarkerArray marker_array_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

        // Image subscription
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
        rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr raw_robot_pub_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rect_pub_;

        // Debug information
        bool debug_;
        std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
        std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
        image_transport::Publisher binary_img_pub_;
        // image_transport::Publisher number_img_pub_;
        image_transport::Publisher result_img_pub_;
    };
} // namespace auto_aim

#endif // ARMOR_DETECTOR_DETECTOR_NODE_HPP_
