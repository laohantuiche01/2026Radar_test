//
// Created by mmz on 25-6-13.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <sstream>
#include <filesystem>
#include <string>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rcpputils/env.hpp"
#include "rcpputils/filesystem_helper.hpp"

namespace auto_aim {
#define RM_ASSERT(condition)                                   \
        do {                                                       \
            if (!(condition)) {                                    \
                std::ostringstream oss;                            \
                oss << "Assertion failed: (" << #condition << ")"; \
                std::cerr << oss.str() << std::endl;               \
                std::abort();                                      \
            }                                                      \
        } while (0)

    //统一资源调配

    /**
     *  作用：代码中写死 package://auto_aim/model.pt，URLResolver 会自动转换为该功能包在当前系统中的实际安装路径（如 /opt/ros/humble/share/auto_aim/model.pt 或 workspace 中的构建路径）。
        代码中使用 $ROS_HOME/logs，URLResolver 会自动替换为当前环境中 ROS_HOME 对应的实际路径，避免硬编码导致的兼容性问题。
     *
     *  本地文件系统（如 /home/user/data/model.onnx）
        ROS 2 功能包的共享目录（如 package://auto_aim/config/params.yaml，其中 auto_aim 是功能包名）
        依赖环境变量的路径（如 $ROS_HOME/cache/data.bin，ROS_HOME 通常对应 ~/.ros）
        URLResolver 类通过统一的接口（getResolvedPath）解析这些路径，无需手动区分来源，简化资源加载逻辑。
     */
    class URLResolver {
    public:
        //得到相应的文件路径    输入为其他形式，返回实际的绝对路径
        static std::filesystem::path getResolvedPath(const std::string &url) {
            const std::string resolved_url = resolveUrl(url);
            UrlType url_type = parseUrl(url);

            std::string res;

            switch (url_type) {
                case UrlType::EMPTY: {
                    break;
                }
                case UrlType::FILE: {
                    res = resolved_url.substr(7);
                    break;
                }
                case UrlType::PACKAGE: {
                    res = getPackageFileName(resolved_url);
                    break;
                }
                default: {
                    break;
                }
            }

            return std::filesystem::path(res);
        }

    private:
        enum class UrlType {
            EMPTY = 0, // empty string
            FILE, // file
            PACKAGE, // package
            INVALID, // anything >= is invalid
        };

        //处理 URL 中的环境变量替换（目前仅支持$ROS_HOME）
        static std::string resolveUrl(const std::string &url) {
            std::string resolved;
            size_t rest = 0;

            while (true) {
                // find the next '$' in the URL string
                size_t dollar = url.find('$', rest);

                if (dollar >= url.length()) {
                    // no more variables left in the URL
                    resolved += url.substr(rest);
                    break;
                }

                // copy characters up to the next '$'
                resolved += url.substr(rest, dollar - rest);

                if (url.substr(dollar + 1, 1) != "{") {
                    // no '{' follows, so keep the '$'
                    resolved += "$";
                } else if (url.substr(dollar + 1, 10) == "{ROS_HOME}") {
                    // substitute $ROS_HOME
                    std::string ros_home;
                    std::string ros_home_env = rcpputils::get_env_var("ROS_HOME");
                    std::string home_env = rcpputils::get_env_var("HOME");
                    if (!ros_home_env.empty()) {
                        // use environment variable
                        ros_home = ros_home_env;
                    } else if (!home_env.empty()) {
                        // use "$HOME/.ros"
                        ros_home = home_env + "/.ros";
                    }
                    resolved += ros_home;
                    dollar += 10;
                } else {
                    // not a valid substitution variable
                    resolved += "$"; // keep the bogus '$'
                }

                // look for next '$'
                rest = dollar + 1;
            }

            return resolved;
        }

        //判断 URL 的类型（基于前缀识别）
        static URLResolver::UrlType parseUrl(const std::string &url) {
            if (url == "") {
                return UrlType::EMPTY;
            }

            // Easy C++14 replacement for boost::iequals from :
            // https://stackoverflow.com/a/4119881
            auto iequals = [](const std::string &a, const std::string &b) {
                return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char a, char b) {
                    return tolower(a) == tolower(b);
                });
            };

            if (iequals(url.substr(0, 8), "file:///")) {
                return UrlType::FILE;
            }
            if (iequals(url.substr(0, 10), "package://")) {
                // look for a '/' following the package name, make sure it is
                // there, the name is not empty, and something follows it
                size_t rest = url.find('/', 10);
                if (rest < url.length() - 1 && rest > 10) {
                    return UrlType::PACKAGE;
                }
            }
            return UrlType::INVALID;
        }

        //解析package://类型的 URL，转换为实际文件路径
        static std::string getPackageFileName(const std::string &url) {
            // Scan URL from after "package://" until next '/' and extract
            // package name.  The parseURL() already checked that it's present.
            size_t prefix_len = std::string("package://").length();
            size_t rest = url.find('/', prefix_len);
            std::string package(url.substr(prefix_len, rest - prefix_len));

            // Look up the ROS package path name.
            std::string pkg_path = ament_index_cpp::get_package_share_directory(package);
            if (pkg_path.empty()) {
                // package not found?
                return pkg_path;
            } else {
                // Construct file name from package location and remainder of URL.
                return pkg_path + url.substr(rest);
            }
        }
    };
}
#endif //UTILS_H
