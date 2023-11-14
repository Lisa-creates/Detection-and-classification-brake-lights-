cmake_minimum_required(VERSION 3.8)

project(detection_and_classification_brake_lights)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

add_subdirectory(detection_and_classification_brake_lights)
