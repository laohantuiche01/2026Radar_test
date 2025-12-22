cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Set python
set(Python3_EXECUTABLE "/usr/bin/python3")
set(PYTHON_EXECUTABLE "/usr/bin/python3")

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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(fmt REQUIRED)
find_package(async_frame REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
        DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${async_frame_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        fmt::fmt
        tbb
        ${aysnc_frame_LIBRARIES}
)

#multi_camera
rclcpp_components_register_node(${PROJECT_NAME}
        PLUGIN ckyf::DetectorNode
        EXECUTABLE armor_detector_node
)

#############
## Install ##
#############

ament_auto_package(
        INSTALL_TO_SHARE
        docs
        model
)
