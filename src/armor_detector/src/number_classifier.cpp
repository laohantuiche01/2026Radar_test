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

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
// std
#include <algorithm>
#include <cstddef>
#include <execution>
#include <fstream>
#include <future>
#include <map>
#include <string>
#include <vector>
// 3rd party
#include <fmt/format.h>
// project
#include "../include/armor_detector/number_classifier.hpp"
#include "../include/armor_detector/types.hpp"
inline void *func(const std::vector<det::Binding> &Shape, cv::Mat img) {
    cv::Mat tensor;
    cv::dnn::blobFromImage(img, tensor, 1 / 255.0, cv::Size(128, 128), cv::Scalar(), true);
    void *ptr = malloc(Shape[0].size * Shape[0].dsize);
    memcpy(ptr, tensor.ptr<float>(), Shape[0].size * Shape[0].dsize);
    return ptr;
}
namespace auto_aim {
    NumberClassifier::NumberClassifier(const std::string &model_path,
                                       const std::string &label_path,
                                       const double thre,
                                       const std::vector<std::string> &ignore_classes)
        : threshold(thre), ignore_classes_(ignore_classes) {
        Trt_ = std::make_shared<TrtInfer>(model_path);

        std::ifstream label_file(label_path);
        std::string line;
        while (std::getline(label_file, line)) {
            class_names_.push_back(line);
        }
    }

    cv::Mat NumberClassifier::extractNumber(const cv::Mat &src, const Armor &armor) const noexcept {
        cv::Point2f center = armor.center;
        cv::Point2f bias = cv::Point2f(armor.left_light.length, armor.left_light.length)*1.2;

        cv::Point2f l, r;
        l.x = center.x - bias.x < 0 ? 0 : center.x - bias.x;
        l.y = center.y - bias.y < 0 ? 0 : center.y - bias.y;
        r.x = center.x + bias.x > src.cols ? src.cols : center.x + bias.x;
        r.y = center.y + bias.y > src.rows ? src.rows : center.y + bias.y;

        // cv::Mat img, img_yuv;
        // cv::cvtColor(src(cv::Rect(l, r)), img_yuv, cv::COLOR_BGR2HSV);
        // std::vector<cv::Mat> channels;
        // cv::split(img_yuv, channels);
        //
        // // 均衡L通道
        // channels[2]*=4.0;
        // // cv::equalizeHist(channels[2], channels[2]);
        //
        // cv::merge(channels, img_yuv);
        // cv::cvtColor(img_yuv, img, cv::COLOR_HSV2BGR);

        // cv::Mat img = src(cv::Rect(l,r)).clone();
        // std::for_each(std::execution::par, img.begin<cv::Vec3b>(), img.end<cv::Vec3b>(), [](cv::Vec3b &color) {
        //     if (std::abs(color[0]-color[1])<5 && std::abs(color[0]-color[2])<5 && color[0]!=0) {
        //         color = cv::Vec3b(255, 255, 255);
        //     }
        // });

        // cv::Mat img = src(cv::Rect(l,r)).clone();
        // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        // // cv::equalizeHist(img, img);
        // img *= 10.0;
        // cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

//gray channel
        // cv::Mat img = src(cv::Rect(l,r)).clone();
        // std::vector<cv::Mat> imgs;
        // cv::split(img, imgs);
        // cv::Mat minchannel;
        // cv::min(imgs[0], imgs[1], minchannel);
        // cv::min(imgs[2], minchannel, minchannel);
        // // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        // // cv::Mat darkChannel;
        // // cv::erode(minchannel, img, kernel);
        // minchannel *= 100.0;
        // cv::cvtColor(minchannel, img, cv::COLOR_GRAY2BGR);

        cv::Mat img = src(cv::Rect(l,r)).clone();
        return std::move(img);
    }

    void NumberClassifier::classify(const cv::Mat &src, Armor &armor) noexcept {
        // Normalize
        cv::Mat input = armor.number_img;
        // Create blob from image
        cv::Mat blob;
        if (input.channels() != 3) {perror("single channel"); return;}
        void *ptr = func(Trt_->getInputBinding(), input);

        // Set the input blob for the neural network
        mutex_.lock();
        Trt_->copy_from_data(&ptr);
        Trt_->infer();
        // Forward pass the image blob through the model
        auto *f_ptr = static_cast< float *>(Trt_->getResult()[0]);
        mutex_.unlock();

        // Decode the output
        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::for_each(f_ptr, f_ptr + 336, [&f_ptr, &bboxes, &confidences, &class_ids](float &f) {
            float* p_ = &f;
            bboxes.emplace_back(*(p_), *(p_ + 336), *(p_ + 2 * 336), *(p_ + 3 * 336));
            float conf = 0;
            int class_id = 0;
            for (int i = 0; i < 5 ; ++i) {
                if (*(p_ + (4 + i) * 336) > conf) {
                    conf = *(p_ + (4 + i) * 336);
                    class_id = i;
                }
            }
            confidences.emplace_back(conf);
            class_ids.emplace_back(class_id);
        });
        int index = 0;
        std::vector<int> indexs;
        cv::dnn::NMSBoxes(bboxes, confidences, 0.7, 0.0, indexs);
        for_each(indexs.begin(), indexs.end(), [&confidences,&index](int index_) {
            confidences[index_] > confidences[index];
            index = index_;
        });
        int label_id = class_ids[index];
        float confidence = confidences[index];
        // long label_id = std::max_element(f_ptr, f_ptr + 4) - f_ptr;
        // float confidence = f_ptr[label_id];

        armor.confidence = confidence;
        armor.number = class_names_[label_id];

        armor.classfication_result = fmt::format("{}:{:.1f}%", armor.number, armor.confidence * 100.0);
    }

    void NumberClassifier::eraseIgnoreClasses(std::vector<Armor> &armors) noexcept {
        armors.erase(
            std::remove_if(armors.begin(),
                           armors.end(),
                           [this](const Armor &armor) {
                               if (armor.confidence < threshold) {
                                   return true;
                               }

                               for (const auto &ignore_class: ignore_classes_) {
                                   if (armor.number == ignore_class) {
                                       return true;
                                   }
                               }

                               bool mismatch_armor_type = false;
                               // if (armor.type == ArmorType::LARGE) {
                               //     mismatch_armor_type = armor.number == "outpost" || armor.number == "2" ||
                               //                           armor.number == "sentry";
                               // } else if (armor.type == ArmorType::SMALL) {
                               //     mismatch_armor_type = armor.number == "1" || armor.number == "base";
                               // }
                               return mismatch_armor_type;
                           }),
            armors.end());
    }
} // namespace fyt::auto_aim
