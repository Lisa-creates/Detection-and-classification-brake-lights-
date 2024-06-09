#pragma once
#include <iostream>
#include <vector>
#include <fstream> 
#include <filesystem> 
#include "brake_lights_detection.h" 
#include"video_processing.h"

using namespace cv;
using namespace std; 
namespace fs = std::filesystem;

void get_test_for_detector(); 