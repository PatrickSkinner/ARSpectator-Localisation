//
//  support.cpp
//  lineDetector
//
//  Created by Patrick Skinner on 13/10/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#include "support.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

Match::Match(){
        l1 = NULL; // L1 is template
        l2 = NULL; // L2 is detected line
        dist = 99999;
    }
    
Match::Match(Vec4f line1, Vec4f line2, double distance){
        l1 = line1;
        l2 = line2;
        dist = distance;
    }

/** Get angle of given line */
double getAngle(Vec4f line1){
    double angle1 = atan2( ( line1[3] - line1[1] ), ( line1[2] - line1[0] ) );
    angle1 *= (180/ CV_PI);
    if(angle1 < 0) angle1 = 180 + angle1; // All angles should be in range of 0-180 degrees
    return angle1;
}

/** Convert homogenous vector to form (x, y, 1) */
Vec3f constrainVec(Vec3f in){
    if(in[2] != 0){
        return Vec3f( in[0]/in[2], in[1]/in[2], in[2]/in[2]);
    } else {
        return in;
    }
}

/** Find intersection point of two lines */
Vec3f intersect(Vec3f a, Vec3f b){
    return constrainVec( a.cross(b) );
}

/** Find intersection point of two lines */
Vec3f intersect(Vec4f a, Vec4f b){
    Vec3f aH = Vec3f(a[0], a[1], 1).cross( Vec3f(a[2], a[3], 1 ) );
    Vec3f bH = Vec3f(b[0], b[1], 1).cross( Vec3f(b[2], b[3], 1 ) );
    return intersect(aH, bH);
}

/** Check if two angles are within a given range of eachother */
bool checkThreshold(float angle, float baseline, float threshold){
    if( angle > (baseline-threshold) && angle < (baseline+threshold)) return true;
    if( (baseline-threshold) < 0 && angle > 180+(baseline-threshold)) return true;
    if( (baseline+threshold) > 180){
        if( angle > (baseline-threshold)){
            return true;
        }
        if( angle < (baseline+threshold)-180){
            return true;
        }
    }
    return false;
}

/** Get center point of given line */
Vec2f getCenter(Vec4f line){
    return Vec2f( ((line[0] + line[2] )/2) , ((line[1] + line[3] )/2) );
}

/** Get gradient of given line */
float getGradient(Vec4f line)
{
    float grad;
    
    if( (line[2] - line[0]) != 0 ){
        grad = ((line[3] - line[1] + 0.0) / (line[2] - line[0] + 0.0));
    } else {
        grad = 0.0;
    }
    
    return grad;
}

/** Compare lines by angle for sorting purposes*/
bool compareVec(Vec4f v1, Vec4f v2)
{
    return (getAngle(v1) < getAngle(v2));
}

/** Compare lines by y intercept at  screen width/2 */
bool compareLinesByY(Vec4f v1, Vec4f v2)
{
    Vec4f vert = Vec4f(1920/2, 0, 1920/2, 1080);
    return ( intersect(v1, vert)[1] < intersect(v2, vert)[1]);
}

/** Calculate the length of a given line */
float lineLength(Vec4f line){
    return sqrt( pow( abs(line[2] - line[0]), 2) + pow( abs(line[1] - line[3]), 2) ) ;
}

/** Find the minimum distance between a point and a line  */
float minimum_distance(Vec4f lineIn, Point2f p) {
    float x1 = lineIn[0];
    float y1 = lineIn[1];
    float x2 = lineIn[2];
    float y2 = lineIn[3];
    
    float a = abs( (y2-y1)*p.x - (x2-x1)*p.y + x2*y1 - y2*x1);
    float b = sqrt( ((y2-y1)*(y2-y1)) + ((x2-x1)*(x2-x1)) );
    
    return a/b;
}

bool isMatched(int n, vector<Vec4f> lines, vector<Match> matches){
    for(int i = 0; i < matches.size(); i++){
        if( lines[n] == matches[i].l2 ){
            //cout << lines[n] << "    is equal to    " << matches[i].l2 << endl;
            return true;
        }
        //cout << lines[n] << "    isn't equal to    " << matches[i].l2 << endl;
    }
    return false;
}
