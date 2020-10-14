//
//  support.hpp
//  lineDetector
//
//  Created by Patrick Skinner on 13/10/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#ifndef support_hpp
#define support_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>

#endif /* support_hpp */

using namespace std;
using namespace cv;

class Match{
public:
    Vec4f l1;
    Vec4f l2;
    double dist;
    Match();
    Match(Vec4f line1, Vec4f line2, double distance);
};

double getAngle(Vec4f line1);

Vec3f intersect(Vec3f a, Vec3f b);

Vec3f intersect(Vec4f a, Vec4f b);

bool checkThreshold(float angle, float baseline, float threshold);

Vec2f getCenter(Vec4f line);

float getGradient(Vec4f v);

bool compareLinesByY(Vec4f v1, Vec4f v2);

float lineLength(Vec4f line);

bool compareVec(Vec4f v1, Vec4f v2);

float minimum_distance(Vec4f lineIn, Point2f p);

bool isMatched(int n, std::vector<Vec4f> lines, std::vector<Match> matches);
