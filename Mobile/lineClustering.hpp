//
//  lineClustering.hpp
//  lineDetector
//
//  Created by Patrick Skinner on 13/10/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#ifndef lineClustering_hpp
#define lineClustering_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

#endif /* lineClustering_hpp */

vector<Vec4f> trimLines(vector<Vec4f> inputLines);
Mat clusterLines(Mat in, vector<Vec4f>& lines);
Vec4f fitBestLine( vector<Vec4f> inputLines, Vec2f center, bool isHori);
vector<Match> findMatches( vector<Vec4f> detectedLines, vector<Vec4f> templateLines, int selectedLine);
vector<Vec4f> cleanLines(vector<Vec4f> sortedLines, Mat labels);
