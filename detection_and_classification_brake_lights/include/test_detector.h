#pragma once 

#include <filesystem> 
#include <fstream> 
#include <iostream>
#include <vector> 

#include "brake_lights_detection.h" 
#include "video_processing.h"

using namespace cv;
using namespace std; 

namespace fs = std::filesystem;

void get_test_for_detector(); 