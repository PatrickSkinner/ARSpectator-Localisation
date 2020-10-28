//
//  lineDetection.hpp
//  lineDetector
//
//  Created by Patrick Skinner on 13/10/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#ifndef lineDetection_hpp
#define lineDetection_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

#endif /* lineDetection_hpp */

vector<Vec4f> getLines(Mat src, bool autoThresh, bool useMask);
int histThreshold(Mat inputImg, Mat& outputImg, int channel);
int getMaxAreaContourId(vector <vector<cv::Point>> contours);
