cmake_minimum_required(VERSION 3.8)

project(detection_and_classification_brake_lights)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

add_subdirectory(detection_and_classification_brake_lights)