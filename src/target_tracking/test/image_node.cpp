#include "rclcpp/rclcpp.hpp"
#include "Deepsort/DeepsortHelper.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

class DeepsortTest : public rclcpp::Node {
public:
    DeepsortTest() : Node("deepsort_test") {
        detector = Yolo_Type::YoloDetector<128, 128, 0.5f, 0.5f, 5>("../model/armor.onnx");
        matcher = std::make_shared<DeepSort::ArmorMatch>(false,1000);

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

                    auto detect = detector.detect(image);
                    auto detect_bbox = DeepSort::ArmorBBox::from_yolo_to_armorbbox(detect);

                    matcher->Match(detect_bbox, image);
                    auto vehicle_armor_map = matcher->GetVehicleArmorNumber();
                    for (const auto &[vehicle_id, armor_num]: vehicle_armor_map) {
                        std::cout << "车辆类型ID: " << static_cast<int>(vehicle_id)
                                << " 装甲板数字: " << armor_num << std::endl;
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
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    Yolo_Type::YoloDetector<128, 128, 0.5f, 0.5f, 5> detector;
    std::shared_ptr<DeepSort::ArmorMatch> matcher;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DeepsortTest>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    cv::destroyAllWindows();
    return 0;
}
