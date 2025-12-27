#include "rclcpp/rclcpp.hpp"
#include "Deepsort/DeepsortHelper.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "cv_bridge/cv_bridge.h"

const std::vector<std::string> VEHICLE_CLASSES = {"car"};
const std::vector<std::string> NUM_CLASSES = {"1", "2", "3", "4"};

class DeepsortTest : public rclcpp::Node
{
public:
    DeepsortTest() : Node("deepsort_test"),
                     vehicle_detector(
                         Yolo_Type::YoloDetector<1088, 1088, 0.5f, 0.5f>("../model/vehicle.onnx", VEHICLE_CLASSES))
    {
        RCLCPP_INFO(this->get_logger(), "DeepsortTest::DeepsortTest");
        armor_results_.clear();

        armor_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("/sensor_far/raw/image/rect", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
            {
                auto armor_msg = msg->data;
                armor_results_.resize(armor_msg.size());
                for (int i = 0; i < static_cast<int>(armor_msg.size()); ++i)
                {
                    armor_results_[i] = armor_msg[i];
                }
            }
        );

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sensor_far/raw/image", 10,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr msg)
            {
                try
                {
                    if (msg->width <= 0 || msg->height <= 0)
                    {
                        RCLCPP_WARN(this->get_logger(), "Invalid image dimensions: %dx%d", msg->width, msg->height);
                        return;
                    }
                    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
                    cv::Mat temp_image = cv_ptr->image;
                    cv::Mat image;
                    cv::cvtColor(temp_image, image, cv::COLOR_BGR2RGB);

                    //模拟出数据
                    std::vector<Yolo_Type::Detection> armor_detections;
                    if (!armor_results_.empty())
                    {
                        Yolo_Type::Detection armor_detection;
                        armor_detection.box.x = static_cast<int>(armor_results_[0]);
                        armor_detection.box.y = static_cast<int>(armor_results_[1]);
                        armor_detection.box.width = static_cast<int>(armor_results_[2]);
                        armor_detection.box.height = static_cast<int>(armor_results_[3]);
                        armor_detection.class_id = static_cast<int>(armor_results_[4]);
                        armor_detection.confidence = 0.91;
                        armor_detections.push_back(armor_detection);
                    }

                    auto vehicle_detection = vehicle_detector.detect(image);
                    auto results = control.track(armor_detections, vehicle_detection);
                    for (auto& result : results)
                    {
                        cv::rectangle(image, cv::Point(result.box.x, result.box.y),
                                      cv::Point(result.box.x + result.box.width, result.box.y + result.box.height),
                                      cv::Scalar(0, 0, 255));
                    }

                    for (auto& vehicle : vehicle_detection)
                    {
                        cv::rectangle(image, cv::Point(vehicle.box.x, vehicle.box.y),
                                      cv::Point(vehicle.box.x + vehicle.box.width, vehicle.box.y + vehicle.box.height),
                                      cv::Scalar(0, 255, 0));
                    }

                    cv::imshow("result gmm", image);
                    cv::waitKey(30);
                }
                catch (cv_bridge::Exception& e)
                {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                }
                catch (const std::exception& e)
                {
                    RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
                }
                catch (...)
                {
                    RCLCPP_ERROR(this->get_logger(), "Unknown exception in image callback");
                }
            }
        );
    }

private:
    DeepSort::DeepSortControl control;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr armor_sub_; //接受装甲板信息
    std::vector<double> armor_results_;
    Yolo_Type::YoloDetector<1088, 1088, 0.5f, 0.5f> vehicle_detector;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DeepsortTest>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    cv::destroyAllWindows();
    return 0;
}
