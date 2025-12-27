cmake_minimum_required(VERSION 3.10)
project(armor_detector)

### Set python
#set(Python3_EXECUTABLE "/usr/bin/python3")
#set(PYTHON_EXECUTABLE "/usr/bin/python3")

## Use C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
#add_definitions(-Wall -Werror)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

set(OPENVINO_ROOT "/usr/local/runtime")
set(OPENVINO_INCLUDE_DIR ${OPENVINO_ROOT}/include)
set(OPENVINO_LIB_DIR ${OPENVINO_ROOT}/lib/intel64)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(fmt REQUIRED)
find_package(async_frame REQUIRED)
find_package(OpenVINO
        PATHS ${OPENVINO_ROOT}
        REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########
add_executable(${PROJECT_NAME}
        src/armor_detector_node.cpp
        src/armor_detector.cpp
        src/light_corner_corrector.cpp
        src/number_classifier.cpp
)
#
#ament_auto_add_library(${PROJECT_NAME} SHARED
#        DIRECTORY src
#)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OpenCV_INCLUDE_DIRS}
        ${async_frame_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${OPENVINO_INCLUDE_DIR}
)
target_link_directories(${PROJECT_NAME} PRIVATE
        ${OPENVINO_LIB_DIR}
)
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        fmt::fmt
        tbb
        ${aysnc_frame_LIBRARIES}
        cv_bridge::cv_bridge
        rclcpp::rclcpp
)

ament_target_dependencies(${PROJECT_NAME}
        rclcpp
        cv_bridge
        async_frame
        image_transport
        visualization_msgs
)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
        DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(
        INSTALL_TO_SHARE
        docs
        model
)
