cmake_minimum_required(VERSION 3.8)

set(OpenCV_DIR "C:/opencv/build") # Change to your path to OpenCV 'build' directory

list(APPEND CMAKE_PREFIX_PATH ${OpenCV_DIR})

add_executable(brake_lights_status_classification "main.cpp" "brake_lights_detection.cpp" "brake_lights_classification.cpp" "video_processing.cpp" "test_detector.cpp")

find_package(OpenCV REQUIRED)

target_include_directories(brake_lights_status_classification PUBLIC include/)

target_link_libraries(brake_lights_status_classification ${OpenCV_LIBS}) 
