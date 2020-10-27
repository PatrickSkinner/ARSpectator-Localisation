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


/** Compare lines by leftmost x coordinate for sorting purposes*/
bool compareVecByX(Vec4f v1, Vec4f v2)
{
    float leftmostV1 = v1[0];
    if( v1[2] < v1[0] ) leftmostV1 = v1[2];
    float leftmostV2 = v2[0];
    if( v2[2] < v2[0] ) leftmostV2 = v2[2];
    
    return (leftmostV1 < leftmostV2);
}

float getSpan( vector<Vec4f> lines){
    cout << "lines: " << lines.size() << endl;
    vector<Vec4f> sortedLines = lines;
    sort(sortedLines.begin(), sortedLines.end(), compareVecByX);
    
    float span = 0;// abs( sortedLines[0][2] - sortedLines[0][0] );
    float tempS = sortedLines[0][0];
    float tempE = sortedLines[0][2];
    
    bool flag = false; // flag = true when span was incremented with previous iteration.
    
    cout << "Line: " << tempS << "\t-\t" << tempE << endl;
    
    for(int i = 1; i < sortedLines.size(); i++){
        float start = sortedLines[i][0];
        float end = sortedLines[i][2];
        cout << "Line: " << start << "\t-\t" << end << endl;
        
        if( start < tempE && end > tempE){ // merge by setting new end point
            tempE = end;
            flag = false;
        } else if ( start > tempS && end > tempE){ // split clusters
            cout << " Span += " << tempE - tempS << "\t" << tempE << " - " << tempS << endl;
            span += ( tempE - tempS);
            tempE = end;
            tempS = start;
            if(i < sortedLines.size()-1){
                flag = true;
            } else {
                flag = false;
            }
        } else {
            flag = false;
        }

    }
    
    if(!flag){
        //cout << "final increment" << endl;
        cout << " Span += " << tempE - tempS << "\t" << tempE << " - " << tempS << endl;
        span += (tempE - tempS);
    }
    return span;
}


/*
 float getSpan( vector<Vec4f> lines){
     cout << "lines: " << lines.size() << endl;
     vector<Vec4f> sortedLines = lines;
     sort(sortedLines.begin(), sortedLines.end(), compareVecByX);
     
     float span = 0;// abs( sortedLines[0][2] - sortedLines[0][0] );
     float tempS = sortedLines[0][0];
     float tempE = sortedLines[0][2];
     
     bool flag = false; // flag = true when span was incremented with previous iteration.
     
     cout << "Line: " << tempS << "\t-\t" << tempE << endl;
     
     for(int i = 1; i < sortedLines.size(); i++){
         //cout << "\n----------------------\n";
         Vec4f lastLine = sortedLines[i-1];
         Vec4f currentLine = sortedLines[i];
         
         float s1 = lastLine[0];
         float e1 = lastLine[2];
         float s2 = currentLine[0];
         float e2 = currentLine[2];
         //cout << s1 << " - " << e1 << endl;
         //cout << s2 << " - " << e2 << endl;
         cout << "Line: " << s2 << "\t-\t" << e2 << endl;
         
         if ( s2 <= e1 && e2 >= e1){
             tempE = e2;
             cout << "merge 1 \n";
             flag = false;
         } else if (s2 <= s1 && e2 <= e1 ){
             tempE = e1;
             cout << "merge 2\n";
             flag = false;
         } else if ( s2 > e1 && e2 > e1 && s2 > tempE){
             // increment span
             cout << "tempE : " << tempE << "\t e1: " << e1 << endl;
             if( tempE < e1){
                 cout << "split1 \t +" << e1 - tempS << "\n";
                 span += ( e1 - tempS);
             } else {
                 cout << "split2 \t +" << tempE - tempS << "\n";
                 span += ( tempE - tempS);
             }
             tempS = s2;
             tempE = e2;
             if(i < sortedLines.size()-1){
                 flag = true;
             } else {
                 flag = false;
             }
         } else {
             //cout << "no action\n";
             flag = false;
         }
     }
     
     if(!flag){
         //cout << "final increment" << endl;
         span += (tempE - tempS);
     }
     return span;
 }

 
 
 */
 
