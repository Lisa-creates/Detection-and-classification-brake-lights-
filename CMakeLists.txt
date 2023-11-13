cmake_minimum_required(VERSION 3.8)

set(OpenCV_DIR "C:/opencv/build")
list(APPEND CMAKE_PREFIX_PATH ${OpenCV_DIR})

project(detection_and_classification_brake_lights)

add_executable(brake_lights_status_classification "brake_lights_status_classification.cpp")
find_package(OpenCV REQUIRED)

target_link_libraries(brake_lights_status_classification ${OpenCV_LIBS}) 