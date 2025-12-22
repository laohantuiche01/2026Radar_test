#include "rclcpp/rclcpp.hpp"
#include "Deepsort/DeepsortHelper.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

const std::vector<std::string> VEHICLE_CLASSES = {"car"};
const std::vector<std::string> NUM_CLASSES = {"1", "2", "3", "4"};

class DeepsortTest : public rclcpp::Node {
public:
    DeepsortTest() : Node("deepsort_test"),
                     vehicle_detector(
                         Yolo_Type::YoloDetector<1088, 1088, 0.5f, 0.5f>("../model/vehicle.onnx", VEHICLE_CLASSES)) {
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sensor_far/raw/image", 10,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
                try {
                    if (msg->width <= 0 || msg->height <= 0) {
                        RCLCPP_WARN(this->get_logger(), "Invalid image dimensions: %dx%d", msg->width, msg->height);
                        return;
                    }
                    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
                    cv::Mat temp_image = cv_ptr->image;
                    cv::Mat image;
                    cv::cvtColor(temp_image, image, cv::COLOR_BGR2RGB);



                    auto armor_detection = detector.detect(image);
                    auto vehicle_detection = vehicle_detector.detect(image);

                    auto results = control.track(armor_detection, vehicle_detection);

                     for (auto &result: results) {
                         cv::rectangle(image, cv::Point(result.box.x, result.box.y),
                                       cv::Point(result.box.x + result.box.width, result.box.y + result.box.height),
                                       cv::Scalar(0, 0, 255));
                     }

                    for (auto &vehicle: vehicle_detection) {
                        cv::rectangle(image, cv::Point(vehicle.box.x, vehicle.box.y),
                                      cv::Point(vehicle.box.x + vehicle.box.width, vehicle.box.y + vehicle.box.height),
                                      cv::Scalar(0, 255, 0));
                    }

                    for (auto &vehicle: armor_detection) {
                        cv::rectangle(image, cv::Point(vehicle.box.x, vehicle.box.y),
                                      cv::Point(vehicle.box.x + vehicle.box.width, vehicle.box.y + vehicle.box.height),
                                      cv::Scalar(255, 0, 0));
                        std::cout<<vehicle.class_id<<std::endl;
                    }


                    cv::imshow("result gmm", image);
                    cv::waitKey(30);
                } catch (cv_bridge::Exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
                } catch (...) {
                    RCLCPP_ERROR(this->get_logger(), "Unknown exception in image callback");
                }
            }
        );
    }

private:
    DeepSort::DeepSortControl control;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    Yolo_Type::YoloDetector<1088, 1088, 0.5f, 0.5f> vehicle_detector;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DeepsortTest>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    cv::destroyAllWindows();
    return 0;
}
