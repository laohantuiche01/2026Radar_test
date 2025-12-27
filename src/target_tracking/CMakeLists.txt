cmake_minimum_required(VERSION 3.8)
project(target_tracking)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

include(${CMAKE_SOURCE_DIR}/cmake/TensorRTConfig.cmake)

set(CMAKE_CXX_STANDARD 23)

#######################
## find dependencies ##
#######################

set(ONNXRUNTIME_ROOT "/home/laohantuiche/dependence/onnxruntime/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
ament_auto_find_build_dependencies()

# 增加测试文件
ament_auto_add_executable(model_debug
        test/image_receive.cpp
        include/Deepsort/DeepsortHelper.cpp
        include/Deepsort/Deepsort.cpp
)
ament_target_dependencies(model_debug
        rclcpp
        OpenCV
        geometry_msgs
        sensor_msgs
        cv_bridge
)
target_include_directories(model_debug PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_directories(model_debug PRIVATE
        ${ONNXRUNTIME_LIB_DIR}
)

# 增加测试文件
ament_auto_add_executable(node_debug
        test/image_node.cpp
        include/Deepsort/DeepsortHelper.cpp
        include/Deepsort/Deepsort.cpp
)
ament_target_dependencies(node_debug
        rclcpp
        OpenCV
        geometry_msgs
        sensor_msgs
        cv_bridge
        CUDAToolkit
)
target_include_directories(node_debug PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(node_debug
        ${TensorRT_LIBRARY}
        ${CUDA_LIBS}
        ${EIGEN3_LIBRARIES}
        ${ONNXRUNTIME_LIBS}
)
target_link_directories(node_debug PRIVATE
        ${ONNXRUNTIME_LIB_DIR}
)

# 编译为库
add_library(${PROJECT_NAME} SHARED
        test/image_receive.cpp
)

# 指定库的头文件路径
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# 链接依赖库
ament_target_dependencies(${PROJECT_NAME}
        rclcpp
        cv_bridge
)

# 安装库文件
install(TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"  # 只安装头文件
)
# 导出库信息
ament_export_targets(export_${PROJECT_NAME}
        HAS_LIBRARY_TARGET
        NAMESPACE ${PROJECT_NAME}::
)
ament_export_dependencies(rclcpp)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()