#pragma once
#include <iostream>
#include <vector>
#include <fstream> 
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1; 
#include <experimental/filesystem> 
#include "Header_files/brake_lights_detection.h" 
#include"Header_files/video_processing.h"

using namespace cv;
using namespace std; 
namespace fs = std::experimental::filesystem;

void get_test_for_detector(); 