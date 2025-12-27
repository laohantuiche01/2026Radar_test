#include "Yolo.hpp"
#include "Deepsort/Deepsort.hpp"
#include "backdel/backdel.hpp"
#include "basic.hpp"
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
//
// int main() {
//     // 初始化匹配器：使用GMM算法，最小车辆轮廓面积500
//     DeepSort::ArmorMatch matcher(true, 500.0f);
//     Yolo_Type::YoloDetector<128,128,0.5f,0.5f,5> detector;
//
//     // 读取视频/摄像头
//     cv::VideoCapture cap(2);
//     if (!cap.isOpened()) return -1;
//
//     cv::Mat frame;
//     while (cap.read(frame)) {
//         //装甲板数字识别
//         std::vector<DeepSort::ArmorBBox> armor_detections;
//
//         auto detection=detector.detect(frame);
//         detector.draw_detections(frame,detection);
//         auto armor_bbox=DeepSort::ArmorBBox::from_yolo_to_armorbbox(detection);
//
//         matcher.Match(armor_detections, frame);
//
//         auto vehicle_armor_map = matcher.GetVehicleArmorNumber();
//         for (const auto& [vehicle_id, armor_num] : vehicle_armor_map) {
//             std::cout << "车辆类型ID: " << static_cast<int>(vehicle_id)
//                       << " 装甲板数字: " << armor_num << std::endl;
//         }
//
//         // 按ESC退出
//         if (cv::waitKey(1) == 27) break;
//     }
//
//     return 0;
// }

int main() {
    DeepSort::DeepSortControl control;
    std::vector<Yolo_Type::Detection> vehicle_detection;
    std::vector<Yolo_Type::Detection> vehicle_detection_;
    std::vector<Yolo_Type::Detection> armor_detection;
    std::vector<Yolo_Type::Detection> armor_detection_;

    Yolo_Type::Detection vehicle_detection1; {
        vehicle_detection1.box.x = 10;
        vehicle_detection1.box.y = 10;
        vehicle_detection1.box.width = 100;
        vehicle_detection1.box.height = 50;
        vehicle_detection1.class_id = 1;
        vehicle_detection1.confidence = 1.0f;
        vehicle_detection.push_back(vehicle_detection1);
    }
    Yolo_Type::Detection vehicle_detection2; {
        vehicle_detection2.box.x = 150;
        vehicle_detection2.box.y = 10;
        vehicle_detection2.box.width = 100;
        vehicle_detection2.box.height = 50;
        vehicle_detection2.class_id = 1;
        vehicle_detection2.confidence = 1.0f;
        vehicle_detection.push_back(vehicle_detection2);
    }

    Yolo_Type::Detection vehicle_detection3; {
        vehicle_detection3.box.x = 20;
        vehicle_detection3.box.y = 20;
        vehicle_detection3.box.width = 100;
        vehicle_detection3.box.height = 50;
        vehicle_detection3.class_id = 1;
        vehicle_detection3.confidence = 1.0f;
        vehicle_detection_.push_back(vehicle_detection3);
    }
    Yolo_Type::Detection vehicle_detection4; {
        vehicle_detection4.box.x = 160;
        vehicle_detection4.box.y = 20;
        vehicle_detection4.box.width = 100;
        vehicle_detection4.box.height = 50;
        vehicle_detection4.class_id = 1;
        vehicle_detection4.confidence = 1.0f;
        vehicle_detection_.push_back(vehicle_detection4);
    }

    Yolo_Type::Detection armor_detection1; {
        armor_detection1.box.x = 50;
        armor_detection1.box.y = 40;
        armor_detection1.box.width = 20;
        armor_detection1.box.height = 10;
        armor_detection1.class_id = 1;
        armor_detection1.confidence = 1.0f;
        armor_detection.push_back(armor_detection1);
    }

    control.track(armor_detection, vehicle_detection);
    control.track(armor_detection_, vehicle_detection_);
    std::cout<<111;
    return 0;
}
