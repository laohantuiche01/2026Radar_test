#include "Yolo.hpp"
#include "Deepsort/Deepsort.hpp"
#include "backdel/backdel.hpp"

// int main() {
//     cv::VideoCapture cap(2);
//     if (!cap.isOpened()) {
//         std::cerr << "视频打开失败！" << std::endl;
//         return -1;
//     }
//     DeepSort::ArmorMatch match;
//     Yolo_Type::YoloDetector<128,128,0.5f,0.5f,5> yolo("../model/armor.onnx") ;
//
//     cv::Mat frame;
//     while (cap.read(frame)) {
//         auto detections=yolo.detect(frame);
//         cv::Mat frame2;
//         cv::cvtColor(frame,frame2,cv::COLOR_RGB2GRAY);
//         match.Match(detections, frame2);
//         cv::imshow("DeepSort Tracking", frame);
//         if (cv::waitKey(1) == 27) break;
//     }
//
//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }

int main() {
    // 初始化匹配器：使用GMM算法，最小车辆轮廓面积500
    DeepSort::ArmorMatch matcher(true, 500.0f);
    Yolo_Type::YoloDetector<128,128,0.5f,0.5f,5> detector;

    // 读取视频/摄像头
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    while (cap.read(frame)) {
        //装甲板数字识别
        std::vector<DeepSort::ArmorBBox> armor_detections;

        auto detection=detector.detect(frame);
        detector.draw_detections(frame,detection);
        auto armor_bbox=DeepSort::ArmorBBox::from_yolo_to_armorbbox(detection);

        matcher.Match(armor_detections, frame);

        auto vehicle_armor_map = matcher.GetVehicleArmorNumber();
        for (const auto& [vehicle_id, armor_num] : vehicle_armor_map) {
            std::cout << "车辆类型ID: " << static_cast<int>(vehicle_id)
                      << " 装甲板数字: " << armor_num << std::endl;
        }

        // 按ESC退出
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}