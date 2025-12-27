
// std
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <execution>
// ros2
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
// third party
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// project
#include "../include/armor_detector/armor_detector_node.hpp"

#include <fmt/format.h>

#include "/opt/ros/humble/include/rclcpp/rclcpp/logging.hpp"
#include "../include/armor_detector/types.hpp"
int num = 0;
int count = 0;

namespace ckyf {
    DetectorNode::DetectorNode()
        : Node("armor_detector") {
        RCLCPP_INFO(get_logger(), "Starting ArmorDetectorNode!");
        //BCN 自己车辆颜色 0是红色，1是蓝色
        selfColor = this->declare_parameter("self_color", 0);
        // Detector
        detector_ = initDetector();

        imageTopic = this->declare_parameter("image_topic", std::string("/sensor_far/raw/image"));
        //BCN img sub
        rect_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(imageTopic + "/rect", 1);
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(imageTopic, 1,
                                                                      std::bind(&DetectorNode::imageCallback, this,
                                                                          std::placeholders::_1));
        raw_robot_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(imageTopic + "/raw_robot", 5);

        // Debug Publishers
        debug_ = this->declare_parameter("debug", true);
        if (debug_) {
            createDebugPublishers();
        }
        // Debug param change moniter
        debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
        debug_cb_handle_ = debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter &p) {
            debug_ = p.as_bool();
            debug_ ? createDebugPublishers() : destroyDebugPublishers();
        });
    }

    void DetectorNode::imageCallback(sensor_msgs::msg::Image::SharedPtr img_msg) {
        geometry_msgs::msg::PointStamped robot_img_msg;
        robot_img_msg.header = img_msg->header;
        // Get the transform from odom to gimbal
        if (img_msg->data.empty()) {
            return;
        }
        // Detect armors
        img_msg->header.stamp = this->now();
        auto armors = detectArmors(img_msg); ///====================================================有bug

        std::for_each(armors.begin(), armors.end(), [this, &robot_img_msg](Armor &armor) {
            robot_img_msg.header.frame_id = imageTopic;
            robot_img_msg.point.x = armor.center.x;
            robot_img_msg.point.y = armor.center.y + (armor.left_light.length + armor.right_light.length) / 2.0 / 1.2;

            if (armor.number == "1") robot_img_msg.point.z = 1;
            if (armor.number == "2") robot_img_msg.point.z = 2;
            if (armor.number == "3") robot_img_msg.point.z = 3;
            if (armor.number == "4") robot_img_msg.point.z = 4;
            if (armor.number == "sentry") robot_img_msg.point.z = 0;

            //发送画框信息
            std_msgs::msg::Float64MultiArray robot_rect_msg;
            robot_rect_msg.data.push_back(armor.left_light.top.x);
            robot_rect_msg.data.push_back(armor.left_light.top.y);
            robot_rect_msg.data.push_back(armor.right_light.top.x - armor.left_light.top.x);
            robot_rect_msg.data.push_back(armor.right_light.bottom.y - armor.left_light.top.y);
            robot_rect_msg.data.push_back(robot_img_msg.point.z);

            rect_pub_->publish(robot_rect_msg);
            raw_robot_pub_->publish(robot_img_msg);
        });
        // count++;
        // if (count%10 == 0) return;f
        for (auto armor: armors) {
            cv::Mat robot_img;
            // auto name = fmt::format("/home/zhu/下载/datasets/9_{}.png", num);
            cv::resize(armor.number_img, robot_img, cv::Size(128, 128), cv::INTER_AREA);
            // cv::imwrite(name,robot_img);
            // num++;
            // std::cout<<name<<std::endl;
            cv::imshow("save_equal", robot_img);

            cv::waitKey(1);
        }
    }

    std::unique_ptr<Detector> DetectorNode::initDetector() {
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        param_desc.integer_range.resize(1);
        param_desc.integer_range[0].step = 1;
        param_desc.integer_range[0].from_value = 0;
        param_desc.integer_range[0].to_value = 255;
        int binary_thres = declare_parameter("binary_thres", 160, param_desc);

        Detector::LightParams l_params = {
            .min_ratio = declare_parameter("light.min_ratio", 0.08),
            .max_ratio = declare_parameter("light.max_ratio", 0.4),
            .max_angle = declare_parameter("light.max_angle", 40.0),
            .color_diff_thresh = static_cast<int>(declare_parameter("light.color_diff_thresh", 25)),
            .min_h = static_cast<int>(declare_parameter("light.min_h", 90)), //90
            .max_h = static_cast<int>(declare_parameter("light.max_h", 150)), //150
            .min_s = static_cast<int>(declare_parameter("light.min_s", 0)),
            .max_s = static_cast<int>(declare_parameter("light.max_s", 255)),
            .min_v = static_cast<int>(declare_parameter("light.min_v", 255)),
            .max_v = static_cast<int>(declare_parameter("light.max_v", 255)),
            .min_length = declare_parameter("light.min_length", 5.0),
            .max_length = declare_parameter("light.max_length", 25.0)

        };
        RCLCPP_INFO(this->get_logger(), "HSV : %d %d %d %d %d %d", l_params.min_h, l_params.min_s, l_params.min_v,
                    l_params.max_h, l_params.max_s, l_params.max_v);
        Detector::ArmorParams a_params = {
            .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.6),
            .min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8),
            .max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2),
            .min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2),
            .max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.0),
            .max_relative_angle = declare_parameter("max_relative_angle", 145.0),
            .max_angle = declare_parameter("armor.max_angle", 35.0)
        };
        //NOTE：默认检测蓝色装甲板
        std::unique_ptr<Detector> detector;

        detector = std::make_unique<Detector>(binary_thres, EnemyColor::RED, l_params, a_params);

        // Init classifier
        //NOTE 模型文件路径
        namespace fs = std::filesystem;
        fs::path model_path = URLResolver::getResolvedPath("package://armor_detector/model/armor.engine");
        RCLCPP_ERROR(this->get_logger(), "model_path: %s", model_path.string().c_str());
        fs::path label_path = URLResolver::getResolvedPath("package://armor_detector/model/label.txt");
        if (!(fs::exists(model_path) && fs::exists(label_path))) {
            RCLCPP_ERROR(get_logger(), "Model or Label file does not exist");
        }
        std::ifstream label_file(label_path);
        std::string line;
        while (std::getline(label_file, line)) {
            classNames.push_back(line);
        }
        label_file.close();

        double threshold = this->declare_parameter("classifier_threshold", 0.7);
        std::vector<std::string> ignore_classes = this->declare_parameter(
            "ignore_classes", std::vector<std::string>{"negative"});
        detector->classifier = std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);

        // Init Corrector
        bool use_pca = this->declare_parameter("use_pca", true);
        if (use_pca) {
            detector->corner_corrector = std::make_unique<LightCornerCorrector>();
        }

        // Set dynamic parameter callback
        on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&DetectorNode::onSetParameters, this, std::placeholders::_1));

        return detector;
    }

    std::vector<Armor> DetectorNode::detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
        // Convert ROS img to cv::Mat
        auto img = cv_bridge::toCvShare(img_msg, img_msg->encoding)->image;

        //这里由于模拟器接受的图像类型，需要进行修改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        // if (img_msg->encoding != "bgr8")
        //     cvtColor(img, img, cv::COLOR_BayerBG2BGR);

        // if (selfColor == 0)
        cvtColor(img, img, cv::COLOR_BGR2RGB);

        auto armors = detector_->detect(img);
        // auto armors = std::vector<Armor>();
        auto final_time = this->now();
        auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;

        // Publish debug info
        if (debug_) {
            binary_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img).toImageMsg());

            detector_->drawResults(img);
            // Draw latency
            std::stringstream latency_ss;
            latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
            auto latency_s = latency_ss.str();
            cv::putText(img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "bgr8", img).toImageMsg());
        }

        return armors;
    }

    rcl_interfaces::msg::SetParametersResult
    DetectorNode::onSetParameters(std::vector<rclcpp::Parameter> parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param: parameters) {
            if (param.get_name() == "binary_thres") {
                detector_->binary_thres = param.as_int();
            } else if (param.get_name() == "classifier_threshold") {
                detector_->classifier->threshold = param.as_double();
            } else if (param.get_name() == "light.min_ratio") {
                detector_->light_params.min_ratio = param.as_double();
            } else if (param.get_name() == "light.max_ratio") {
                detector_->light_params.max_ratio = param.as_double();
            } else if (param.get_name() == "light.max_angle") {
                detector_->light_params.max_angle = param.as_double();
            } else if (param.get_name() == "light.color_diff_thresh") {
                detector_->light_params.color_diff_thresh = param.as_int();
            } else if (param.get_name() == "armor.min_light_ratio") {
                detector_->armor_params.min_light_ratio = param.as_double();
            } else if (param.get_name() == "armor.min_small_center_distance") {
                detector_->armor_params.min_small_center_distance = param.as_double();
            } else if (param.get_name() == "armor.max_small_center_distance") {
                detector_->armor_params.max_small_center_distance = param.as_double();
            } else if (param.get_name() == "armor.min_large_center_distance") {
                detector_->armor_params.min_large_center_distance = param.as_double();
            } else if (param.get_name() == "armor.max_large_center_distance") {
                detector_->armor_params.max_large_center_distance = param.as_double();
            } else if (param.get_name() == "armor.max_angle") {
                detector_->armor_params.max_angle = param.as_double();
            }
        }
        return result;
    }

    void DetectorNode::createDebugPublishers() noexcept {
        //BCN debug pub
        this->declare_parameter("armor_detector.result_img.jpeg_quality", 50);
        this->declare_parameter("armor_detector.binary_img.jpeg_quality", 50);
        std::stringstream ss_b, ss_r;
        ss_b << imageTopic << "/binary_img";
        ss_r << imageTopic << "/result_img";
        binary_img_pub_ = image_transport::create_publisher(this, ss_b.str());
        // number_img_pub_ = image_transport::create_publisher(this, "armor_detector/number_img");
        result_img_pub_ = image_transport::create_publisher(this, ss_r.str());
    }

    void DetectorNode::destroyDebugPublishers() noexcept {
        binary_img_pub_.shutdown();
        // number_img_pub_.shutdown();
        result_img_pub_.shutdown();
    }
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ckyf::DetectorNode>());
    rclcpp::shutdown();
    return 0;
}
