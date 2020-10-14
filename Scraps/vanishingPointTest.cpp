//
//  main.cpp
//  lineDetector
//
//  Created by Patrick Skinner on 24/05/19.
//  Copyright © 2019 Patrick Skinner. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include "fstream"
#include <time.h>

using namespace cv;
using namespace std;

bool useRectification = true;
bool manualSelection = false;
bool useMask = true;
bool mirrorInput = false;
bool rotateInput = false;

bool debugDisplay = true;
bool outputDisplay = true;
bool getPose = true;
bool getReprojection = true;

Mat src;
Mat HSV;
Mat thresh;
Mat finuks;
Mat boundary;

String filename = "0065.png";
String imageSet = "Set1.txt";
Point2f clickCoords = Point2f(640,900);
int selectedLine = 0; // 0 = leftmost, 1 = center, 2 = rightmost
bool guiMode = true;

float baseline = -1;
int blThresh = 14;
Vec4f divider;

class Match{
public:
    Vec4f l1;
    Vec4f l2;
    
    double dist;
    
    Match(){
        l1 = NULL; // L1 is template
        l2 = NULL; // L2 is detected line
        dist = 99999;
    }
    
    Match(Vec4f line1, Vec4f line2, double distance){
        l1 = line1;
        l2 = line2;
        dist = distance;
    }
};

/** compared matches by distance for sorting purposes */
bool compareMatches(Match m1, Match m2){
    return (m1.dist < m2.dist);
}

/** Get center point of given line */
Vec2f getCenter(Vec4f line){
    return Vec2f( ((line[0] + line[2] )/2) , ((line[1] + line[3] )/2) );
}

/** Convert homogenous vector to form (x, y, 1) */
Vec3f constrainVec(Vec3f in){
    if(in[2] != 0){
        return Vec3f( in[0]/in[2], in[1]/in[2], in[2]/in[2]);
    } else {
        return in;
    }
}

/** Get gradient of given line */
float getGradient(Vec4f v)
{
    float vGrad;
    
    if( (v[2] - v[0]) != 0 ){
        vGrad = ((v[3] - v[1] + 0.0) / (v[2] - v[0] + 0.0));
    } else {
        vGrad = 0.0;
    }
    
    return vGrad;
}

/** Get angle of given line */
double getAngle(Vec4f line1){
    double angle1 = atan2( ( line1[3] - line1[1] ), ( line1[2] - line1[0] ) );
    angle1 *= (180/ CV_PI);
    if(angle1 < 0) angle1 = 180 + angle1; // All angles should be in range of 0-180 degrees
    return angle1;
}

/** Return difference between angle of two given lines */
double getAngle(Vec4f line1, Vec4f line2){
    double angle1 = atan2( ( line1[3] - line1[1] ), ( line1[2] - line1[0] ) );
    double angle2 = atan2( ( line2[3] - line2[1] ), ( line2[2] - line2[0] ) );
    angle1 *= (180/ CV_PI);
    angle2 *= (180/ CV_PI);
    if(angle1 < 0) angle1 = 180 + angle1;
    if(angle2 < 0) angle2 = 180 + angle2;
    //cout << "A1: " << angle1 << " , A2: " << angle2 << endl;
    //cout << "Difference between : " << abs(angle1-angle2) << endl << endl;
    return abs(angle1-angle2);
}

/** Cotangent function */
float cotan(float i) {
    if( i < CV_PI/2 + 0.001 && i > CV_PI/2 - 0.001){
        return 0;
    }
    return(1 / tan(i));
}

/** Compare lines by angle */
bool compareVec(Vec4f v1, Vec4f v2)
{
    //return (getGradient(v1) < getGradient(v2));
    return (getAngle(v1) < getAngle(v2));
}

/** Calculate the length of a given line */
float lineLength(Vec4f line){
    return sqrt( pow( abs(line[2] - line[0]), 2) + pow( abs(line[1] - line[3]), 2) ) ;
}

/** Return the distance between the midpoints of two lines */
float midpointDistance(Vec4f line1, Vec4f line2){
    Vec2f mid1 = getCenter(line1);
    Vec2f mid2 = getCenter(line2);
    return abs( lineLength( Vec4f(mid1[0], mid1[1], mid2[0], mid2[1] )));
}

/** Calculate the Hausdorff distance between two lines */
float lineDistance(Vec4f line1, Vec4f line2){
    
    Vec4f ac, bd, ad, bc;
    ac = Vec4f( line1[0], line1[1], line2[0], line2[1] );
    bd = Vec4f( line1[2], line1[3], line2[2], line2[3] );
    ad = Vec4f( line1[0], line1[1], line2[2], line2[3] );
    bc = Vec4f( line1[2], line1[3], line2[0], line2[1] );
    
    return min(    max( lineLength(ac),lineLength(bd)),     max( lineLength(ad),lineLength(bc))       );
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

/** Calculate the total Hausdorff distance between two line sets */
float getSetDistance(vector<Vec4f> templateLines, vector<Vec4f> detectedLines){
    float totalDistance = 0.0;
    
    for(int i = 0; i < templateLines.size(); i++)
    {
        for(int j = 0; j < detectedLines.size(); j++)
        {
            // For lines AB and CD, distance is defined as min(max(|𝐴𝐶|,|𝐵𝐷|),max(|𝐴𝐷|,|𝐵𝐶|))
            Vec4f ac, bd, ad, bc;
            ac = Vec4f(templateLines[i][0], templateLines[i][1], detectedLines[j][0], detectedLines[j][1] );
            bd = Vec4f(templateLines[i][2], templateLines[i][3], detectedLines[j][2], detectedLines[j][3] );
            ad = Vec4f(templateLines[i][0], templateLines[i][1], detectedLines[j][2], detectedLines[j][3] );
            bc = Vec4f(templateLines[i][2], templateLines[i][3], detectedLines[j][0], detectedLines[j][1] );
            
            totalDistance += min(    max( lineLength(ac),lineLength(bd)),     max( lineLength(ad),lineLength(bc))       );
        }
    }
    
    return totalDistance;
}

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

/** Find intersection point of two circles, code from jupdike on Github */
Point2f intersectTwoCircles(float x1, float y1, float r1, float x2, float y2, float r2) {
    float centerdx = x1 - x2;
    float centerdy = y1 - y2;
    float R = sqrt(centerdx * centerdx + centerdy * centerdy);
    if (!(abs(r1 - r2) <= R && R <= r1 + r2)) { // no intersection
        return Point2f(0,0);
    }
    
    float R2 = R*R;
    float R4 = R2*R2;
    float a = (r1*r1 - r2*r2) / (2 * R2);
    float r2r2 = (r1*r1 - r2*r2);
    float c =sqrt(2 * (r1*r1 + r2*r2) / R2 - (r2r2 * r2r2) / R4 - 1);
    
    float fx = (x1+x2) / 2 + a * (x2 - x1);
    float gx = c * (y2 - y1) / 2;
    //float ix1 = fx + gx;
    float ix2 = fx - gx;
    
    float fy = (y1+y2) / 2 + a * (y2 - y1);
    float gy = c * (x1 - x2) / 2;
    //float iy1 = fy + gy;
    float iy2 = fy - gy;
    
    // note if gy == 0 and gx == 0 then the circles are tangent and there is only one solution
    // but that one solution will just be duplicated as the code is currently written
    
    return Point2f(ix2, iy2);
}

/** for each L1, find the best matching L2 */
// int whichLine enumerates 0 = left, 1 = center, 2 = right. This describes which line the users has selected.
vector<Match> getBestMatches(vector<Match> matches, vector<Vec4f> templateLines, Point2f clickCoords, int whichLine){
    vector<Match> bestMatches;
    //Mat debugMat = src.clone();
    
    int closest = -1;
    int closestDist = 9990;
    for(int i = 0; i < matches.size(); i++){

        if(minimum_distance(matches[i].l2, clickCoords) < closestDist){
            closest = i;
            closestDist = minimum_distance(matches[i].l2, clickCoords);
        }
    }
    
    //line(debugMat, Point2f(matches[closest].l2[0], matches[closest].l2[1]), Point2f(matches[closest].l2[2], matches[closest].l2[3]), Scalar(0,255,0), 5);
    //line(debugMat, clickCoords, Point2f(getCenter(matches[closest].l2)[0], getCenter(matches[closest].l2)[1]), Scalar(0,255,0), 5);
    //cout << "Closest Index: " << closest << ",     Distance: " << closestDist << endl;
    
    
    Match candidate;
    
    if(whichLine == 0){ // Matches start with leftmost line
        int templateLine = 0; // Index of the leftmost line in the template.
        
        for(int i = 0; i < matches.size(); i++){
            if(matches[i].dist < candidate.dist){
                if(matches[i].l1 == templateLines[templateLine] && matches[i].l2 == matches[closest].l2) candidate = matches[i]; // find best match for leftmost template line
            }
        }

        bestMatches.push_back(candidate);
        
        for(int i = templateLine+1; i < templateLines.size()-2; i++){
            candidate = Match();
            for(int j = 0; j < matches.size(); j++){
                if(matches[j].dist < candidate.dist){
                    if(matches[j].l1 == templateLines[i]){
                        long lastMatch = bestMatches.size()-1;
                        if( getCenter(matches[j].l2)[0] > getCenter(bestMatches[lastMatch].l2)[0] ){ // Candidate match midpoint is to the right of previous matched line
                            candidate = matches[j];
                        }
                    }
                }
            }
            bestMatches.push_back(candidate);
        }
    }
    
    if(whichLine == 1){ // Match from center to rightmost, then from leftmost back towards the center.
        int templateLine = 1; // Index of the center line in the template.
        
        for(int i = 0; i < matches.size(); i++){
            if(matches[i].dist < candidate.dist){
                if(matches[i].l1 == templateLines[templateLine] && matches[i].l2 == matches[closest].l2) candidate = matches[i]; // find best match for center template line
            }
        }

        bestMatches.push_back(candidate);
        
        for(int i = templateLine+1; i < templateLines.size()-2; i++){
            candidate = Match();
            for(int j = 0; j < matches.size(); j++){
                if(matches[j].dist < candidate.dist){
                    if(matches[j].l1 == templateLines[i]){
                        long lastMatch = bestMatches.size()-1;
                        if( getCenter(matches[j].l2)[0] > getCenter(bestMatches[lastMatch].l2)[0] ){ // Candidate match midpoint is to the right of previous matched line
                         candidate = matches[j];
                        }
                    }
                }
            }
            bestMatches.push_back(candidate);
        }
        
        // Match leftmost line.
        candidate = Match();
        for(int j = 0; j < matches.size(); j++){
            if(matches[j].dist < candidate.dist){
                if(matches[j].l1 == templateLines[0]){
                    if( getCenter(matches[j].l2)[0] < getCenter(bestMatches[0].l2)[0] ){ // Candidate match midpoint is to the left of center line
                        candidate = matches[j];
                    }
                }
            }
        }
        bestMatches.push_back(candidate);
        
        // Match lines between leftmost line and center line.
        for(int i = 1; i < templateLine; i++){
            candidate = Match();
            for(int j = 0; j < matches.size(); j++){
                if(matches[j].dist < candidate.dist){
                    if(matches[j].l1 == templateLines[i]){
                        long lastMatch = bestMatches.size()-1;
                        if( getCenter(matches[j].l2)[0] > getCenter(bestMatches[lastMatch].l2)[0] && getCenter(matches[j].l2)[0] < getCenter(bestMatches[0].l2)[0]){
                            // Candidate match midpoint is to the right of previous matched line and left of center line.
                            candidate = matches[j];
                        }
                    }
                }
            }
            bestMatches.push_back(candidate);
        }
    }
    
    
    if(whichLine == 2){ // Matches start with rightmost line
        int templateLine = 2; // Index of the rightmost line in the template.
        
        for(int i = 0; i < matches.size(); i++){
            if(matches[i].dist < candidate.dist){
                if(matches[i].l1 == templateLines[templateLine] && matches[i].l2 == matches[closest].l2) candidate = matches[i]; // find best match for rightmost template line
            }
        }

        bestMatches.push_back(candidate);
        
        for(int i = templateLine-1; i >= 0; i--){
            candidate = Match();
            for(int j = 0; j < matches.size(); j++){
                if(matches[j].dist < candidate.dist){
                    if(matches[j].l1 == templateLines[i]){
                        long lastMatch = bestMatches.size()-1;
                        if( getCenter(matches[j].l2)[0] < getCenter(bestMatches[lastMatch].l2)[0] ){ // Candidate match midpoint is to the left of previous matched line
                            candidate = matches[j];
                        }
                    }
                }
            }
            bestMatches.push_back(candidate);
        }

    }
    
    

    candidate = Match();
    for( int i = 0; i < matches.size(); i++){
        bool flag = false;
        if(matches[i].dist < candidate.dist){
            for(int j = 0; j < bestMatches.size(); j++){
                if( matches[i].l2 == bestMatches[j].l2){
                    flag = true;
                }
            }
            if(matches[i].l1 == templateLines[ templateLines.size()-2 ] && !flag) candidate = matches[i]; // find best match for top horizontal template line
        }
    }
    bestMatches.push_back(candidate);
    
    candidate = Match();
    for( int i = 0; i < matches.size(); i++){
        bool flag = false;
        if(matches[i].dist < candidate.dist){
            for(int j = 0; j < bestMatches.size(); j++){
                if( matches[i].l2 == bestMatches[j].l2){
                    flag = true;
                }
            }
            if(matches[i].l1 == templateLines[ templateLines.size()-1 ] && !flag) candidate = matches[i]; // find best match for bottom template line
        }
    }
    bestMatches.push_back(candidate);
    
    //cout << "HOW MANY MATCHES????     " << bestMatches.size() << endl;
    //imshow("DEEEEEBUUUUUG", debugMat);
    
    return bestMatches;
}

// Given a cluster of lines find one single line of best fit, biased twowards the center of the pitch to avoid outliers.
extern "C++" Vec4f fitBestLine( vector<Vec4f> inputLines, Vec2f center, bool isHori){
    if(inputLines.size() == 1 ) return inputLines[0];
    
    float avgX = 0.0;
    float avgY = 0.0;
    float avgAngle = 0.0;
    
    float x = 0;
    float y = 0;
    float ang = 0;
    float closestDist = 99999;
    
    float searchRange = 15;
    
    //line( src, Point( center[0]-4000, center[1]), Point( center[0]+4000, center[1]), Scalar(255,255,255), 5, 0);
    
    for(int i = 0; i < inputLines.size(); i++){
        avgX += getCenter(inputLines[i])[0];
        float midY = getCenter(inputLines[i])[1];
        avgY += midY;
        
        double angle = 0;
        angle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
        avgAngle += angle;
        
        float checkAng = angle*(180/CV_PI);
        if(checkAng < 0) checkAng += 180;
        float dist = -1;

        if( !isHori){// Line is vertical
            Vec4f horiz = Vec4f( center[0]-100, center[1], center[0]+100, center[1]);
            dist = abs(intersect(inputLines[i], horiz)[0] - center[0]);
        } else { //Line is horizontal
            if( (lineLength(inputLines[i]) > (1080/3 + 1080/2)/3) ||
                  (intersect(divider, Vec4f( getCenter(inputLines[i])[0], 0, getCenter(inputLines[i])[0], 1080 ))[1] > getCenter(inputLines[i])[1]) ) { // Awful hack to avoid those short horizontal lines being chosen as best fit.
                
                Vec4f vert = Vec4f( center[0], center[1]-1000, center[0], center[1]+1000);
                dist = abs(intersect(inputLines[i], vert)[1] - center[1]);
            } else {
                /*
                cout << lineLength(inputLines[i]) << " is too short " << endl;
                
                if( (lineLength(inputLines[i]) > 399 ) ) {
                    cout << inputLines[i] << endl;
                    cout << lineLength(inputLines[i]) << endl << endl;
                }
                 */
            }
        }

        if( dist < closestDist && dist != -1){
            closestDist = dist;
            x = getCenter(inputLines[i])[0];
            y = getCenter(inputLines[i])[1];
            //cout << "angle: " << ang << ",   X: "<< inputLines[i][3] - inputLines[i][1] << ",   Y: " << inputLines[i][3] - inputLines[i][1] << endl;
            ang = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
            ang *= (180/CV_PI);
            if(ang < 0) ang += 180;
            //ang = getAngle(inputLines[i]);
        }
        //cout << getAngle(inputLines[i]) << endl;
        //avgAngle += getAngle(inputLines[i]);
    }
    
    //cout << closestDist << "!\n";
    
    /*
    avgX /= inputLines.size();
    avgY /= inputLines.size();
    avgAngle /= inputLines.size();
    
    
    // Override averaging with closest line to center
    avgX = x;
    avgY = y;
    //avgAngle = ang;
    */
    
    avgX = 0;
    avgY = 0;
    avgAngle = 0;
    
    
    Point2f compare = Point2f(x, y);
    int count = 0;
    
    for( int i = 0; i < inputLines.size(); i++){
        //if(ang >= 45 && ang <= 160){
        // fix y axis, vertical line
        //if( !(ang > baseline - blThresh && ang < baseline + blThresh) && !(ang > (180 + (baseline - blThresh))) ){
        if( !isHori){ //  vertical
            //cout << ang << " is not within " << blThresh << " of " << baseline << endl;
            int distAtX = 0;
            float grad = getGradient(inputLines[i]);
            
            float yDiff = y - inputLines[i][1];
            float steps = yDiff / grad;
            if(grad == 0) steps = 0;
            
            /*
            cout << "Grad: " << grad << endl;
            cout << "yDiff: " << yDiff << endl;
            cout << "Steps: " << steps << endl;
            */
            
            Point2f p = Point2f(inputLines[i][0] + steps , y);
            //line( src, Point(inputLines[0][0], inputLines[0][1]), Point(inputLines[0][2], inputLines[0][3]), Scalar(200,0,255), 10, 0);
            //line( src, Point( p.x - searchRange, p.y), Point( p.x + searchRange, p.y), Scalar(255,255,255), 5, 0);
            
            //cout << "POINT P: " << p << endl << endl;
            
            distAtX = abs( lineLength( Vec4f(compare.x, compare.y, p.x, p.y )));
            if(distAtX < searchRange+50){ //increased search range for vertical lines, hacky af
                
                float thisAngle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
                //thisAngle *= (180/CV_PI);
                if(thisAngle < 0) thisAngle += CV_PI;
                //cout << " thisAngle :    " << thisAngle << endl;

                avgX += p.x;
                avgY += p.y;
                avgAngle += thisAngle;
                count++;
            }
        } else { // fix x axis, horizontal line
            
            //cout << ang << " is within " << blThresh << " of " << baseline << endl;
            int distAtX = 0;
            float grad = getGradient(inputLines[i]);
            
            float xDiff = x - inputLines[i][0];
            float steps = xDiff * grad;
            
            /*
            cout << "Grad: " << grad << endl;
            cout << "xDiff: " << xDiff << endl;
            cout << "Steps: " << steps << endl;
            */
            
            Point2f p = Point2f(x, inputLines[i][1] + steps);
            //line( src, Point(inputLines[0][0], inputLines[0][1]), Point(inputLines[0][2], inputLines[0][3]), Scalar(200,0,255), 10, 0);
            //line( src, Point( p.x, p.y-9), Point( p.x, p.y+9), Scalar(255,255,255), 5, 0);
            //line( src, Point( p.x, p.y-9), Point( p.x, p.y+9), Scalar(255,255,255), 5, 0);
            
            distAtX = abs( lineLength( Vec4f(compare.x, compare.y, p.x, p.y )));
            
            if(distAtX < searchRange){
                float thisAngle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
                //thisAngle *= (180/CV_PI);
                //if(thisAngle < 0) thisAngle += CV_PI;
                //cout << " thisAngle :    " << thisAngle << endl;
                
                
                
                avgX += p.x;
                avgY += p.y;
                avgAngle += thisAngle;
                count++;
                
            }
        }
    }
    
    if(count != 0){
        avgX /= count;
        avgY /= count;
        avgAngle /= count;
    }
    
    //cout << "Average angle for cluster: " << avgAngle*(180/CV_PI) << endl << endl;
    
    //cout << "avgAng: " << avgAngle*(180/CV_PI) << ",   grad: " << tan(avgAngle) << endl;
    float grad = tan(avgAngle);
    float len = 10000;

    //cout << avgX << ",   " << avgY << "      count: " << count    <<"     grad : "<<grad <<endl;
    //cout << "BEST FIT LINE: " << Vec4f(avgX - len, avgY - (len*grad), avgX + len, avgY + (len*grad)) << endl;
    return Vec4f(avgX - len, avgY - (len*grad), avgX + len, avgY + (len*grad));
}

/** Get the index of the largest contour */
int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        } // End if
    } // End for
    return maxAreaContourId;
} // End function

/** Return vector of all detected Hough lines in given image */
vector<Vec4f> getLines()
{
    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    cvtColor(src, HSV, COLOR_BGR2HSV);
    // Detect the field based on HSV Range Values
    //inRange(HSV, Scalar(20, 10, 50), Scalar(45, 255, 255), thresh);
    //inRange(HSV, Scalar(32, 124, 51), Scalar(46, 255, 191), thresh); // Stadium Test Pics
    //inRange(HSV, Scalar(27, 86, 2), Scalar(85, 255, 145), thresh); // broadcast
    //inRange(HSV, Scalar(31, 55, 70), Scalar(66, 255, 197), thresh); // renders
    inRange(HSV, Scalar(35, 100, 0), Scalar(55, 215, 255), thresh); // artificial
    //inRange(HSV, Scalar(31, 55, 45), Scalar(68, 255, 206), thresh);
    
    if(debugDisplay) imshow("thresh af", thresh);
    
    // opening and closing
    if(useMask){
        Mat opened;
        Mat closed;
        Mat kernel = Mat(3, 3, CV_8U, Scalar(1));
        morphologyEx(thresh, opened, MORPH_OPEN, kernel);
        morphologyEx(opened, closed, MORPH_ERODE, kernel);
        
        if(debugDisplay) imshow("MORPH OPS", closed);
        
        // Add one pixel white columns to both sides of the image to close contours
        if(selectedLine == 0){
            int count = 0;
            for(int i = 0; i < closed.rows; i++){
               //closed.at<uchar>(i, closed.cols-1) = 255;
                if( closed.at<uchar>(i, closed.cols-1) != 255 ){
                    count++; // Count black pixels
                } else {
                    if(count <= 5){ // If previous run of black pixels was 3 or less, leave them alone
                        count = 0;
                    } else if(count != 0){
                        for(int j = 0; j <= count; j++) closed.at<uchar>(i-j, closed.cols-1) = 255;
                        
                        count = 0;
                    }
                }
            }
        } else if(selectedLine == 2){
            for(int i = 0; i < closed.rows; i++){
               //closed.at<uchar>(i, 0) = 255;
            }
        }
        vector<vector<cv::Point> > contours;
        
        findContours(closed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        boundary = Mat(1080, 1920, CV_8UC1, Scalar(0));
        
        Mat threshClosed;
        kernel = Mat(12, 3, CV_8U, Scalar(1));
        morphologyEx(thresh, threshClosed, MORPH_DILATE, kernel);
        vector<vector<cv::Point> > findBound;
        findContours(threshClosed, findBound, RETR_TREE, CHAIN_APPROX_SIMPLE);
        drawContours(boundary, findBound, getMaxAreaContourId(findBound), Scalar(255,0,255), -1);
        //Mat cont;
        //cvtColor(closed, cont, COLOR_GRAY2BGR);
        morphologyEx(boundary, boundary, MORPH_DILATE, kernel);
        //imshow("BOUNDARY", boundary);
        
        Mat mask = Mat::ones( thresh.rows, thresh.cols, CV_8U);
        //drawContours(mask, contours, getMaxAreaContourId(contours), Scalar(255,255,255), -1);
        int j = 0;
        for( int i = 0; i < contours.size(); i++){
            if(contourArea(contours[i]) > 4000){
                drawContours(mask, contours, i, Scalar(255,255,255), -1);
                j++;
            }
        }
        
        /* Remove 1px columns from the sides
        for(int i = 0; i < mask.rows; i++){
            if(mask.at<uchar>(i, 1) > 0) mask.at<uchar>(i, 0) = 0;
            if(mask.at<uchar>(i, closed.cols-2) > 0) mask.at<uchar>(i, closed.cols-1) = 0;
        }*/
        
        if(debugDisplay) imshow("cont", mask);
        thresh = mask;
    }

    
    Mat dst, invdst, cdst;
    GaussianBlur( thresh, invdst, Size( 5, 5 ), 0, 0 );
    //imshow("gauss", invdst);
    Canny(invdst, dst, 50, 200, 5);
    //imshow("canny", dst);
    //Remove 1px columns from the sides if mask used
    if(useMask){
        for(int i = 0; i < dst.rows; i++){
            dst.at<uchar>(i, 1) = 0;
            dst.at<uchar>(i, dst.cols-1) = 0;
            dst.at<uchar>(i, dst.cols-2) = 0;
        }
    }
    
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    
    if(rotateInput){
        Mat M = getRotationMatrix2D(Point2f(1920/2,1080/2), -7, 1);
        warpAffine(dst, dst, M, Size(1920,1080));
        warpAffine(cdst, cdst, M, Size(1920,1080));
    }
    vector<Vec4f> lines;
    HoughLinesP(dst, lines, 2, CV_PI/360, 150, 175, 45 );
    //HoughLinesP(dst, lines, 2, CV_PI/360, 100, 75, 45 );
    
    //cout << lines.size() << " lines\n";
    for(int i = 0; i < lines.size(); i++ ){
        Scalar colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
         line( cdst, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), colour, 5, 0);
    }
    if(debugDisplay) imshow("Lines", cdst);
    
    //cout << "Line count: " << lines.size() << endl;
    
    return lines;
}

/**Split horizontal lines with matching labels into two groups, for top and bottom of the pitch */
vector<int> splitHorizontals( vector<Vec4f> lines ){
    vector<int> labels;
    
    float y1 = 0.0, y2 = 0.0;
    for(int i = 0; i < lines.size(); i++){
        y1 += lines[i][1];
        y2 += lines[i][3];
    }
    
    y1 /= lines.size();
    y2 /= lines.size();
    float avgY = (y1+y2)/2 ;
    Mat checkH = src.clone();
    line(checkH, Point(0, avgY), Point(4000, avgY), Scalar(255,255,255), 2, 0);
    if(debugDisplay) imshow("horiz split", checkH);
    
    //cout << "Y threshold: " << avgY<< endl;
    for(int i = 0; i < lines.size(); i++){
        if( getCenter(lines[i])[1] < avgY ){
            labels.push_back(0);
        } else {
            labels.push_back(1);
        }
    }
    
    return labels;
}

//  HT, HB, V, V... ordering
vector<Vec4f> trimLines(vector<Vec4f> inputLines){
    vector<Vec4f> outputLines;
    
    // Convert lines to homogenous form
    vector<Vec3f> hmgLines;
    for(int i = 0; i < inputLines.size(); i++){
        hmgLines.push_back( Vec3f(inputLines[i][0], inputLines[i][1], 1).cross( Vec3f(inputLines[i][2], inputLines[i][3], 1 ) ) );
    }
    
    int n = (int) inputLines.size();
    
    // Top horizontal
    Vec3f inter1 = intersect( hmgLines[0], hmgLines[2]);
    Vec3f inter2 = intersect( hmgLines[0], hmgLines[n-1]);
    outputLines.push_back( Vec4f( inter1[0], inter1[1], inter2[0], inter2[1]) );
    
    //cout << "TOP INTERSECTIONS : " << inter1 << endl << inter2 << endl << endl;
    //cout << inputLines[0] << endl << "intersects\n" << inputLines[2] << endl;
    
    // Bottom horizontal
    inter1 = intersect( hmgLines[1], hmgLines[2]);
    inter2 = intersect( hmgLines[1], hmgLines[n-1]);
    outputLines.push_back( Vec4f( inter1[0], inter1[1], inter2[0], inter2[1]) );
    
    // Vertical Lines.
    for(int i = 2; i < n; i++){
        inter1 = intersect( hmgLines[i], hmgLines[0]);
        inter2 = intersect( hmgLines[i], hmgLines[1]);
        outputLines.push_back( Vec4f( inter1[0], inter1[1], inter2[0], inter2[1]) );
    }
    
    return outputLines;
}

Mat clusterLines(vector<Vec4f> sortedLines){
    Mat labels;
    vector<int> sortedAngles;
    
    int threshold = 7; // Minimum angle difference between clusters
    
    
    int label = 0;
    float startAngle = getAngle(sortedLines[0]);
    cout << "Min Angle = " << startAngle << endl;
    for(int i = 0; i < sortedLines.size(); i++ ){
        float angle = getAngle(sortedLines[i]);

        if( abs(angle - startAngle) < threshold){
            sortedAngles.push_back(label);
        } else {
            label++;
            sortedAngles.push_back(label);
            startAngle = angle;
        }
    }
    
    /* Split horizontal lines into two clusters, for the bottom and top of the pitch */
    vector<Vec4f> horizontals;
    int k = 1;
    int lastOne = 0;
    int setLabel = 0;
    for(int i = 0; i < sortedAngles.size(); i++ ){
        if( i == 0){
            labels.push_back(setLabel);
            horizontals.push_back(sortedLines[i]);
        } else if( sortedAngles[i] == sortedAngles[i-1]){
            if(setLabel == 0) horizontals.push_back(sortedLines[i]);
            labels.push_back(setLabel);
        } else {
            //cout << "Angle :" << sortedAngles[i] << ", last: " << lastOne << " new: " << setLabel+1 << " new K: " << k+1 << endl;
            setLabel += 1;
            labels.push_back(setLabel);
            lastOne = setLabel;
            k += 1;
        }

    }
    
    Mat splitLabels;
    vector<int> splitVec = splitHorizontals( horizontals );
    splitLabels = Mat( splitVec );
    
    for(int i = 0; i < labels.rows; i++){
        if(labels.at<int>(i) == 0){
            labels.at<int>(i) = splitLabels.at<int>(i);
        } else {
            labels.at<int>(i) += 1;
        }
    }
    
    
    Mat clustered = src.clone();
    srand(83);
    for(int i = 0; i < k+1; i++){
        Scalar colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) )); // Random colour for each cluster
        for(int j = 0; j < labels.rows; j++){
            if(labels.at<int>(j) == i){
                line( clustered, Point(sortedLines[j][0], sortedLines[j][1]), Point(sortedLines[j][2], sortedLines[j][3]), colour, 2, 0);
            }
        }
    }
    if(debugDisplay) imshow("Clustering", clustered);
    return labels;
}

/** Find the best fitting line for each line cluster and return the line set of best fitting lines, horizontals first */
vector<Vec4f> cleanLines(vector<Vec4f> sortedLines, Mat labels){
    int clusterCount = labels.at<int>(labels.rows-1) + 1;
    vector<Vec4f> cleanedLines;
    
    int centroidX = 0;
    int centroidY = 0;
    
    for(int i = 0; i < sortedLines.size(); i++ ){
        Vec2f mid = getCenter(sortedLines[i]);
        centroidX += mid[0];
        centroidY += mid[1];
    }
    
    //cout << "cluster count:   " << clusterCount << endl;
    for(int i = 0; i < clusterCount; i++){
        vector<Vec4f> lines;
        
        for(int j = 0; j < labels.rows; j++){
            if(labels.at<int>(j) == i){
                lines.push_back( Vec4f(sortedLines[j][0], sortedLines[j][1], sortedLines[j][2], sortedLines[j][3]));
            }
        }
        
        bool isHori = false;
        if(i == 0 || i == 1){
            isHori = true;
        }
        Vec2f centroid = Vec2f( centroidX / sortedLines.size(), centroidY / sortedLines.size());
        //line( src, Point(centroid[0], 0), Point(centroid[0], 4000), Scalar(255,255,255), 5, 0);
        //line( src, Point(0, centroid[1]), Point(4000, centroid[1]), Scalar(255,255,255), 5, 0);
        Vec4f pushLine = fitBestLine(lines, centroid, isHori);
        cleanedLines.push_back( pushLine );
        
        line( src, Point(pushLine[0], pushLine[1]), Point(pushLine[2], pushLine[3]), Scalar(0,128,255), 4, 0);
    }

    //cout << cleanedLines.size() << endl;
    //waitKey();
    
    cleanedLines = trimLines(cleanedLines);
    for(int i = 0; i < cleanedLines.size(); i++){
        line( src, Point(cleanedLines[i][0], cleanedLines[i][1]), Point(cleanedLines[i][2], cleanedLines[i][3]), Scalar(0,0,255), 3, 0);
    }
    //cout << "cleaned size: " << cleanedLines.size() << endl;;
    return cleanedLines;
}

/** input of form, horizontal, horizontal, verticals after, in order of right to left */
vector<Vec4f> rectifyLines(vector<Vec4f> inputLines){
    finuks = Mat(1080, 1920, CV_8UC3, Scalar(0,0,0));
    
    // Convert lines to homogenous form
    vector<Vec3f> hmgLines;
    for(int i = 0; i < inputLines.size(); i++){
        hmgLines.push_back( Vec3f(inputLines[i][0], inputLines[i][1], 1).cross( Vec3f(inputLines[i][2], inputLines[i][3], 1 ) ) );
    }
    
    
    // Find intersections of parallel line pairs to finding vanishing points
    Vec3f intersection1 = constrainVec( hmgLines[0].cross(hmgLines[1]) ); // intersection of horizontals
    Vec3f intersection2 = constrainVec( hmgLines[2].cross(hmgLines[3]) ); // intersection of verticals
    
    // Find the line at infinity between the vanishing points.
    Vec3f infLine = constrainVec(intersection1.cross(intersection2));
    
    
    // Generate affine rectification matrix
    Mat affineTransform = Mat(3, 3, CV_32F, 0.0);
    affineTransform.at<float>(0,0) = 1.0;
    affineTransform.at<float>(1,1) = 1.0;
    affineTransform.at<float>(2,0) = infLine[0];
    affineTransform.at<float>(2,1) = infLine[1];
    affineTransform.at<float>(2,2) = infLine[2];
    
    Mat mirror = Mat(3, 3, CV_32F, Scalar(0));
    mirror.at<float>(0,0) = -1;
    mirror.at<float>(1,1) = 1;
    mirror.at<float>(2,2) = 1;
    
    vector<Point2f> in;
    vector<Point2f> affinePoints;
    for( int i = 0; i < inputLines.size(); i++){
        in.push_back( Point2f( inputLines[i][0], inputLines[i][1]) );
        in.push_back( Point2f( inputLines[i][2], inputLines[i][3]) );
    }
    
    for(int i = 0; i < in.size(); i += 2){
        line( finuks, Point(in[i].x, in[i].y), Point(in[i+1].x, in[i+1].y), Scalar(0,0,255), 5);
    }
    
    perspectiveTransform( in , affinePoints, affineTransform);
    
    int xaAdj = 0 - affinePoints[0].x;
    int yaAdj = 0 - affinePoints[0].y;
    
    for(int i = 0; i < in.size(); i += 2){
        line( finuks, Point(affinePoints[i].x + xaAdj, affinePoints[i].y + yaAdj), Point(affinePoints[i+1].x +xaAdj, affinePoints[i+1].y + yaAdj), Scalar(255,0,0), 5);
    }
    
    vector<Vec3f> hmgLinesAff;
    vector<Vec4f> affineLines;
    for( int i = 0; i < affinePoints.size(); i += 2){
        hmgLinesAff.push_back( constrainVec(Vec3f(affinePoints[i].x, affinePoints[i].y, 1).cross( Vec3f(affinePoints[i+1].x, affinePoints[i+1].y, 1) ) ));
        affineLines.push_back( Vec4f{affinePoints[i].x, affinePoints[i].y, affinePoints[i+1].x, affinePoints[i+1].y });
    }
    
    // Generate a constraint circle from a known angle between lines
    float a = (-hmgLinesAff[4][1]) / hmgLinesAff[4][0]; // vert
    float b = (-hmgLinesAff[0][1]) / hmgLinesAff[0][0]; // hori
    float theta = CV_PI / 2;
    
    float ca = (a+b)/2 ;
    float cb = ((a-b)/2) * cotan(theta);
    float r = abs( (a-b) / (2 * sin(theta)) );
    
    
    Vec3f intersectionLeft = constrainVec(hmgLinesAff[1].cross(hmgLinesAff[ hmgLinesAff.size()-1 ]));
    Vec3f intersectionRight = constrainVec(hmgLinesAff[1].cross(hmgLinesAff[ hmgLinesAff.size()-2 ]));
    Vec4f horizontalSegment = { intersectionLeft[0], intersectionLeft[1], intersectionRight[0], intersectionRight[1]};
    
    //Generate a constraint circle from known length ratio between two non parallel lines
    float dx1 = affineLines[3][0] - affineLines[3][2];
    float dy1 = affineLines[3][1] - affineLines[3][3];
    
    float dx2 = horizontalSegment[0] - horizontalSegment[2];
    float dy2 = horizontalSegment[1] - horizontalSegment[3];
    
    float ratio = 6.36;
    
    float ca2 = ((dx1*dy1) - ((ratio*ratio)*dx2*dy2)) / ((dy1*dy1)-((ratio*ratio)*(dy2*dy2)));
    float cb2 = 0;
    float r2 = abs( (ratio*(dx2*dy1-dx1*dy2)) / ((dy1*dy1)-(ratio*ratio)*(dy2*dy2)) );
    
    
    //Find where constraint circles intersect
    Point2f inter = intersectTwoCircles(ca, cb, r, ca2, cb2, r2);
    
    // Generate metric rectification matrix
    Mat metricTransform = Mat(3, 3, CV_32F, 0.0);
    metricTransform.at<float>(0,0) = 1 / inter.y;
    metricTransform.at<float>(0,1) = -(inter.x/inter.y);
    metricTransform.at<float>(1,1) = 1.0;
    metricTransform.at<float>(2,2) = 1.0;
    
    vector<Point2f> metricPoints;
    perspectiveTransform( affinePoints, metricPoints, metricTransform);
    
    float avgX = 0;
    float avgY = 0;
    for( int i = 0; i < metricPoints.size(); i++){
        avgX += metricPoints[i].x;
        avgY += metricPoints[i].y;
    }
    avgX /= metricPoints.size();
    avgY /= metricPoints.size();
    
    
    Vec4f horizLine = Vec4f(metricPoints[0].x, metricPoints[0].y, metricPoints[1].x, metricPoints[1].y);
    double angle = - getAngle(horizLine);
    //angle = 0;
    Mat rotationMat = getRotationMatrix2D(Point(avgX,avgY), angle, 1);
    for( int i = 0; i < metricPoints.size(); i++){
        Mat point;
        Mat(metricPoints[i], CV_64F).convertTo(point, CV_64F);
        point = point.t() * rotationMat;
        metricPoints[i].x = point.at<double>(0);
        metricPoints[i].y = point.at<double>(1);
    }
    
    
    
    
    
    
    int end = (int) metricPoints.size() - 1;
    if( (metricPoints[end].x+metricPoints[end-1].x)/2 > (metricPoints[4].x+metricPoints[5].x)/2 ){ // Final horizontal to to the right of first horizontal, order reversed.
        if((metricPoints[2].y+metricPoints[3].y)/2 > (metricPoints[0].y+metricPoints[1].y)/2){ // line 1 below line 0
            cout << "MIRRORED on X" << endl;
            perspectiveTransform( metricPoints, metricPoints, mirror); // Mirror rectification
        } else {
            cout << "FLIPMODE" << endl;
            
            
            float avgX = 0;
            float avgY = 0;
            for( int i = 0; i < metricPoints.size(); i++){
                avgX += metricPoints[i].x;
                avgY += metricPoints[i].y;
            }
            avgX /= metricPoints.size();
            avgY /= metricPoints.size();
            
            Mat rotationMat = getRotationMatrix2D(Point(avgX,avgY), 180, 1);
            for( int i = 0; i < metricPoints.size(); i++){
                Mat point;
                Mat(metricPoints[i], CV_64F).convertTo(point, CV_64F);
                point = point.t() * rotationMat;
                metricPoints[i].x = point.at<double>(0);
                metricPoints[i].y = point.at<double>(1);
            }
        }
    }
    
    if( (metricPoints[end].x+metricPoints[end-1].x)/2 < (metricPoints[4].x+metricPoints[5].x)/2 ){
        if((metricPoints[2].y+metricPoints[3].y)/2 < (metricPoints[0].y+metricPoints[1].y)/2){
            cout << "MIRRORED on Y" << endl;
            mirror.at<float>(0,0) = 1;
            mirror.at<float>(1,1) = -1;
            perspectiveTransform( metricPoints, metricPoints, mirror);
        }
    }
    
    
    
    
    
    vector<Vec4f> outputLines;
    for(int i = 0; i < metricPoints.size(); i += 2){
        outputLines.push_back( Vec4f(metricPoints[i].x, metricPoints[i].y, metricPoints[i+1].x, metricPoints[i+1].y) );
    }
    
    
    // Place rectified lines within frame for debug output;
    int xAdj = -metricPoints[1].x + 100;
    int yAdj = -metricPoints[1].y + 100;
    
    
    for(int i = 0; i < metricPoints.size(); i += 2){
        line( finuks, Point(metricPoints[i].x + xAdj, metricPoints[i].y + yAdj), Point(metricPoints[i+1].x + xAdj, metricPoints[i+1].y + yAdj), Scalar(0,255,0), 5);
    }
    
    if(debugDisplay) imshow("finuks", finuks);
    
    //Mat constraints = Mat(500, 500, CV_8UC1, Scalar(255,255,255));
    
    // Draw axes
    //line( constraints, Point(0, 250), Point(500, 250), Scalar(0,0,128));
    //line( constraints, Point(250, 0), Point(250, 500), Scalar(0,0,128));
    
    //circle(constraints, Point(ca+250 ,cb + 250), r, Scalar(0,255,0));
    //circle(constraints, Point(ca2+250 ,cb2 + 250), r2, Scalar(0,255,0));
    //imshow("constraints", constraints);
    
    return outputLines;
}

/** Update clickCoords. Left, middle, and right mouse buttons correspond to the chosen pitch line. */
void onMouse( int event, int x, int y, int, void* )
{
    clickCoords = Point2f(x, y);
    
    if  ( event == EVENT_LBUTTONDOWN )
    {
        selectedLine = 0;
    }
    else if  ( event == EVENT_MBUTTONDOWN )
    {
         selectedLine = 1;
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
         selectedLine = 2;
    }
}

Mat newClustering(Mat in, vector<Vec4f>& lines){
    Mat labels;
    vector<Vec4f> horizontals;
    vector<Vec4f> verticals;
    int horizontalCount = 0;
    
    vector<Point2f> vanishingPoints;
    int cells [192][108]{};
    for(int i = 0; i < lines.size(); i++){
        for(int j = 0; j < lines.size(); j++){
            if( i != j ){
                Point3f intr = intersect(lines[i], lines[j]);
                Point2f intr2D = Point2f(intr.x, intr.y);

                int x = intr.x/10;
                int y = intr.y/10;
                
                if(x < 192 && y < 54 && x >= 0 && y >= 0){
                    //cout << "incrementing (" << x << ","<< y<< ") \t\t " << intr << endl;;
                    if( boundary.at<uint>( intr.y, intr.x) == 0 ){

                        line(in, Point(intr.x, intr.y-1), Point(intr.x, intr.y+1), Scalar(0,255,255), 4);
                    
                        cells[x][y]++;
                        vanishingPoints.push_back( intr2D );
                    }
                }
            }
        }
    }
    
    int max = 0;
    Point2f maxCell;
    for(int i = 0; i < 192; i++){
        for(int j = 0; j < 108; j++){
            if( cells[i][j] > max ){
                max = cells[i][j];
                maxCell = Point2f(i,j);
            }
        }
    }
    
    //cout << "Max cell: " << maxCell << ", with size: "<< max << endl;
    int cellRangeH = 6; // Expand search beyond identified cell to account for any slight errors
    int cellRangeV = 1;
    for(int i = 0; i < lines.size(); i++){
        bool flag = false;
        for(int j = 0; j < lines.size(); j++){
            if( i != j ){
                Point3f intr = intersect(lines[i], lines[j]);
                int x = intr.x/10;
                int y = intr.y/10;
                if(x >= maxCell.x-cellRangeH && x <= maxCell.x+cellRangeH
                   && y >= maxCell.y-cellRangeV && y <= maxCell.y+cellRangeV){
                    line(in, Point(intr.x, intr.y-1), Point(intr.x, intr.y+1), Scalar(255,0,255), 4);
                    verticals.push_back(lines[i]);
                    flag = true;
                    break;
                }
            }
        }
        if(!flag){
            horizontals.push_back(lines[i]);
            horizontalCount++;
        }
    }
    //cout << endl;
    
    for(int i = 0; i < horizontals.size(); i++){
        line(in, Point(horizontals[i][0], horizontals[i][1]), Point(horizontals[i][2], horizontals[i][3]), Scalar(0,0,255), 4);
    }
    for(int i = 0; i < verticals.size(); i++){
        line(in, Point(verticals[i][0], verticals[i][1]), Point(verticals[i][2], verticals[i][3]), Scalar(0,255,0), 4);
    }
    
    if(debugDisplay) imshow("cluster", in);
    //waitKey();
    
    int topH = -1;
    int bottomH = -1;
    
    //float baseline = -1;
    int lowestMid = -1;
    int lowestLine = -1;
    
    int horizThresh = 70; // Max horizontal distance between lines before they're split into separate clusters
    
    for(int i = 0; i < lines.size(); i++){
        if( getCenter(lines[i])[1] > lowestMid && getCenter(lines[i])[1] < src.rows - 25 ){ // && an extra hack to avoid the bottom edge of the image getting detected
            lowestMid = getCenter(lines[i])[1];
            lowestLine = i;
        }
    }
    baseline = getAngle(lines[lowestLine]);
    //line(in, Point(lines[lowestLine][0], lines[lowestLine][1]), Point(lines[lowestLine][2], lines[lowestLine][3]), Scalar(0,255,0), 3);
    
    //cout << "baseline = " << baseline << endl;
    float grad = 0.0;
    //int blThresh = 15;
    
    
    for(int i = 0; i < lines.size(); i++){
        float angle = getAngle(lines[i]);
        if(angle < 0) angle += 180;
        //if( (angle > baseline - blThresh && angle < baseline + blThresh) || (angle > (180-baseline) - blThresh && angle < (180-baseline) + blThresh)){
        //if( (angle > baseline - blThresh && angle < baseline + blThresh) || (angle > (180 + (baseline - blThresh))) ){
        if( checkThreshold(angle, baseline, blThresh)){
            //cout << angle << " is within " << blThresh << " of " << baseline << endl;
            //line(in, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255,255,255), 5);
            //horizontals.push_back(lines[i]);
            if( topH == -1 || getCenter(lines[i])[1] < topH) topH = getCenter(lines[i])[1] ;
            if( bottomH == -1 || getCenter(lines[i])[1] > bottomH) bottomH = getCenter(lines[i])[1] ;
            grad += getGradient(lines[i]);
            //horizontalCount++;
        } else {
            //line(in, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255,0,0));
            //verticals.push_back(lines[i]);
        }
    }
    
    
    /* Added step to merge similar horizontal lines
    int dT = 50; //Distance threshold between endpoints for merging
    vector<Vec4f> mergedHorizontals;
    for(int i = 0; i < horizontalCount; i++){
        bool merged = false;
        for(int j = 0; j < horizontalCount; j++){
            if( lineLength( Vec4f(horizontals[i][0], horizontals[i][1], horizontals[j][0], horizontals[j][1] )) < dT ){
                if(i != j ){
                    mergedHorizontals.push_back ( Vec4f(horizontals[i][2], horizontals[i][3], horizontals[j][2], horizontals[j][3]  ));
                    merged = true;
                }
            }
            else if( lineLength( Vec4f(horizontals[i][0], horizontals[i][1], horizontals[j][2], horizontals[j][3] )) < dT ){
                if(i != j ){
                    mergedHorizontals.push_back ( Vec4f(horizontals[i][2], horizontals[i][3], horizontals[j][0], horizontals[j][1]  ));
                    merged = true;
                }
            }
            else if( lineLength( Vec4f(horizontals[i][2], horizontals[i][3], horizontals[j][0], horizontals[j][1] )) < dT ){
                if(i != j ){
                    mergedHorizontals.push_back ( Vec4f(horizontals[i][0], horizontals[i][1], horizontals[j][2], horizontals[j][3]  ));
                    merged = true;
                }
            }
            else if( lineLength( Vec4f(horizontals[i][2], horizontals[i][3], horizontals[j][2], horizontals[j][3] )) < dT ){
                if(i != j ){
                    mergedHorizontals.push_back ( Vec4f(horizontals[i][0], horizontals[i][1], horizontals[j][0], horizontals[j][1]  ));
                    merged = true;
                }
            }
        }
        if( !merged ) mergedHorizontals.push_back( horizontals[i] );
    }
    
    horizontals = mergedHorizontals;
    */
    //cout << "top: " << topH << endl;
    //cout << "bottom: " << bottomH << endl;
    int mid = topH + (abs(topH-bottomH) / 2);
    mid -= 100;
    //cout << mid << endl;
    //mid = 1080/2;
    //mid += 100;
    
    grad /= horizontalCount;
    //line(in, Point(0, mid-(grad*540)), Point(1920, mid+(grad*540)), Scalar(255,255,255), 3);
    divider = Vec4f(0, mid-(grad*540), 1920, mid+(grad*540));
    
    for(int i = 0; i < horizontals.size(); i++){
        //line(in, Point(horizontals[i][0], horizontals[i][1]), Point(horizontals[i][2], horizontals[i][3]), Scalar(255,255,255), 5);
        Vec2f vPoint = getCenter(horizontals[i]);
        Vec4f vLine( vPoint[0], 0, vPoint[0], 1080 );
        Vec3f inter = intersect(divider, vLine);
        
        if( inter[1] > vPoint[1]){
            line(in, Point(horizontals[i][0], horizontals[i][1]), Point(horizontals[i][2], horizontals[i][3]), Scalar(0,0,255), 4);
            labels.push_back(0);
        } else {
            line(in, Point(horizontals[i][0], horizontals[i][1]), Point(horizontals[i][2], horizontals[i][3]), Scalar(0,255,0), 4);
            labels.push_back(1);
        }
    }
    
    int lastX = -1;
    int cluster = 2;
    Scalar colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
    
    for(int i = 0; i < verticals.size(); i++){
        int x = intersect( verticals[i], divider)[0];
        if( lastX != -1 && abs(x - lastX) > horizThresh){
            colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
            cluster++;
        }
        line(in, Point(verticals[i][0], verticals[i][1]), Point(verticals[i][2], verticals[i][3]), colour, 4);
        labels.push_back(cluster);
        lastX = x;
    }
    
    if(debugDisplay) imshow("clustered", in);
    //waitKey();
    
    horizontals.insert( horizontals.end(), verticals.begin(), verticals.end() ); //
    lines = horizontals; // Need to put the input lines in same order as our labels.
    
    return labels;
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

/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//////////////                MAIN METHOD              /////////////
/////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv){
    
    clock_t start, end;
    double elapsed;
    start = clock();
    
    Mat gtPose = Mat();
    Mat rVecGT;
    
    if(argc >= 2){
        
        // Take image number as argument and load corresponding image frame.
        String strIn = argv[1];
        int parsed = stoi(strIn);
        //cout << "Image " << parsed << endl;
        if(parsed > 0 && parsed < 10){
            filename = "000" + strIn + ".png";
        } else if (parsed >= 10 && parsed < 100){
            filename = "00" + strIn + ".png";
        } else if (parsed < 1000){
            filename = "0" + strIn + ".png";
        } else {
            filename = strIn + ".png";
        }
        
        if(argc == 3){
            String strIn2 = argv[2];
            parsed = stoi(strIn2);
            
            if(parsed >= 0){
                imageSet = "Set"+strIn2+".txt";
            }
        }
        
        if(getPose && false){
            // Read in ground truth transformation matrix from file
            ifstream groundTruth(imageSet);
            String nextLine;
            bool flag = false;
            int lCount = 0;
            if(groundTruth.is_open()){
                while ( getline (groundTruth,nextLine) )
                {
                    if(flag && lCount < 19){
                        if(nextLine != ""){
                            gtPose.push_back( stof(nextLine) );
                            //cout << nextLine << endl;
                        }
                        lCount++;
                    }
                    
                    if( nextLine == "Image " + strIn){
                        flag = true;
                    }
                }
                groundTruth.close();
            }
            
            gtPose = gtPose.reshape(4,4);
            //cout << "Ground Truth Mat:\n" << gtPose << endl;
            //cout << "\n";
            
            
            
            Mat rMatGT = (Mat_<float>(3,3) <<  gtPose.at<float>(0,0), gtPose.at<float>(0,1), gtPose.at<float>(0,2),
                                                gtPose.at<float>(1,0), gtPose.at<float>(1,1), gtPose.at<float>(1,2),
                                                gtPose.at<float>(2,0), gtPose.at<float>(2,1), gtPose.at<float>(2,2));
            
            
            Rodrigues(rMatGT, rVecGT);
            /*
            for(int i = 0; i < 3; i++){
                cout << gtPose.at<float>(i, 3);
                if(i != 2) cout << ", " ;
            }
            cout << endl << endl;
            for(int i = 0; i < 3; i++){
                cout << rVecGT.at<float>(i)*(180/CV_PI);
                if(i != 2) cout << ", " ;
            }
            */
            
            //cout << endl << endl;
        }
    }
    
    
    
    src = imread(filename);
    if(mirrorInput) flip(src, src, +1);
    Mat finale = src.clone();
    finuks = src.clone();
    
    while(!src.data){
        cout << "." << endl;
    }
    
    if(guiMode){
        imshow("Select a Line.", src);
        setMouseCallback( "Select a Line.", onMouse, 0 );
        waitKey();
    } else {
        string imageSet = "realTest.txt";
        ifstream groundTruth(imageSet);
        String strIn = argv[1];
        int parsed = stoi(strIn);
        int count = 0;
        
        String nextLine;
        if(groundTruth.is_open()){
            while ( getline (groundTruth,nextLine) && count < parsed )
            {
                count++;
            }
            
            
            vector<string> result;
            stringstream s_stream(nextLine); //create string stream from the string
            while(s_stream.good()) {
               string substr;
               getline(s_stream, substr, ','); //get first string delimited by comma
               result.push_back(substr);
            }
            clickCoords = Point2f(stoi(result[0]),stoi(result[1]) );
            selectedLine = stoi(result[2]);
            groundTruth.close();
            
           //cout << "Click Coords: " << clickCoords << endl;
        }
    }
    start = clock();
    
    vector<Vec4f> rawLines = getLines();
    if(rotateInput){
        Mat M = getRotationMatrix2D(Point2f(1920/2,1080/2), -7, 1);
        warpAffine(src, src, M, Size(1920,1080));
    }

    vector<Vec4f> templateLines;
    
    if(selectedLine == 0){
        templateLines = {Vec4f(0,0,0,2800), Vec4f(440,0,440,2800), Vec4f(1420,0,1420,2800), Vec4f(0,0,1420,0), Vec4f(0,2800,1420,2800)};
    } else if(selectedLine == 1){
       templateLines = {Vec4f(0,0,0,2350), Vec4f(350,0,350,2350), Vec4f(700,0,700,2350), Vec4f(0,0,700,0), Vec4f(0,2350,700,2350)};
    } else if(selectedLine == 2){
        templateLines = {Vec4f(0,0,0,2800), Vec4f(960,0,960,2800), Vec4f(1400,0,1400,2800), Vec4f(0,0,1400,0), Vec4f(0,2800,1400,2800)};
    }
    
    vector<Vec4f> sortedLines = rawLines;
    sort(sortedLines.begin(), sortedLines.end(), compareVec); // Sort lines by gradient
    
    Mat labels = newClustering(src.clone(), sortedLines);
    //Mat labels = clusterLines(sortedLines);
    vector<Vec4f> lines = cleanLines(sortedLines, labels);
    
    vector<Vec4f> rectifiedLines;
    rectifiedLines = lines;

    
    for( size_t i = 0; i < templateLines.size(); i++ )
    {
        templateLines[i][0] /= 4;
        templateLines[i][2] /= 4;
        templateLines[i][1] /= 4;
        templateLines[i][3] /= 4;
     
        
        templateLines[i][0] += 150;
        templateLines[i][2] += 150;
        templateLines[i][1] += 150;
        templateLines[i][3] += 150;
    }
    
    float templateHeight = lineLength( templateLines[0] );
    float rectifiedLinesHeight = lineLength(rectifiedLines[rectifiedLines.size() - 1]);
    float diff = templateHeight/rectifiedLinesHeight;
    //cout << "diff: " << diff << endl;
    
    //diff = 3;
    diff = 1080/rectifiedLinesHeight;
    Mat scaling = Mat(3, 3, CV_32F, Scalar(0));
    scaling.at<float>(0,0) = diff;
    scaling.at<float>(1,1) = diff;
    scaling.at<float>(2,2) = 1;
    
    vector<Point2f> in;
    vector<Point2f> out;
    for( int i = 0; i < rectifiedLines.size(); i++){
        in.push_back( Point2f( rectifiedLines[i][0], rectifiedLines[i][1]) );
        in.push_back( Point2f( rectifiedLines[i][2], rectifiedLines[i][3]) );
    }
    
    perspectiveTransform(in, out, scaling);
    
    vector<Vec4f> scaledLines;
    for(int i = 0; i < out.size(); i += 2){
        scaledLines.push_back(Vec4f( out[i].x, out[i].y, out[i+1].x, out[i+1].y));
    }
    
    rectifiedLines = scaledLines;
    
    vector<Match> bestMatches;
    bestMatches.push_back( Match( templateLines[templateLines.size()-2], lines[0], 666) ); // top
    bestMatches.push_back( Match( templateLines[templateLines.size()-1], lines[1], 666) ); // bottom
    
    
    
    Vec4f horiz = Vec4f(0, 540+260, 1920, 540+260 );
    //line(src, Point(0,540) , Point(1920,540), Scalar(255,255,255), 3);
    
    if(selectedLine == 0 ){ // left
        // First vertical line matched is the line clicked by the user
        int closest = -1;
        int closestDist = -1;
        for(int i = 0; i < lines.size(); i++){
            if(minimum_distance(lines[i], clickCoords) < closestDist || closestDist == -1){
                closest = i;
                closestDist = minimum_distance(lines[i], clickCoords);
            }
        }
        bestMatches.push_back( Match( templateLines[0], lines[closest], 666) );
        
        for(int i = 1; i < templateLines.size() - 2; i++){
            int index = -1;
            int minDist = -1;
            for(int j = 0; j < lines.size(); j++){
                //if line is to the right of last matched line
                if( intersect(lines[j], horiz)[0] > intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0] ){
                    //if distance is smaller than last checked distance
                    if( abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]) < minDist || minDist == -1 ){
                        // Check if not already matched
                        if( !isMatched(j, lines, bestMatches) ){
                            /*
                            cout << lines[j] << "   hasn't been matched yet" << endl;
                            cout << "and is to the right of " << bestMatches[bestMatches.size()-1].l2 << endl;
                            cout << intersect(lines[j], horiz) << " > " << intersect(bestMatches[bestMatches.size()-1].l2, horiz)<< endl << endl;
                            
                            cout << "and the distance to the last line is " << abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]) << " which is less than " << minDist<< endl
                                << intersect(lines[j], horiz)[0] << " to " << intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]<< endl;
                            */
                            minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                            index = j;
                        }
                    }
                }
            }
            bestMatches.push_back( Match( templateLines[i], lines[index], 666) );
        }
    } else if (selectedLine == 1){ // center
        // First vertical line matched is the line clicked by the user
        int closest = -1;
        int closestDist = -1;
        for(int i = 0; i < lines.size(); i++){
            if(minimum_distance(lines[i], clickCoords) < closestDist || closestDist == -1){
                closest = i;
                closestDist = minimum_distance(lines[i], clickCoords);
            }
        }
        bestMatches.push_back( Match( templateLines[1], lines[closest], 666) );
        
        // Find line right of center
        int index = -1;
        int minDist = -1;
        for(int j = 0; j < lines.size(); j++){
            //if line is to the right of last matched line
            if( intersect(lines[j], horiz)[0] > intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0] ){
                //if distance is smaller than last checked distance
                if( abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]) < minDist || minDist == -1 ){
                    // Check if not already matched
                    if( !isMatched(j, lines, bestMatches) ){
                        /*
                        cout << lines[j] << "   hasn't been matched yet" << endl;
                        cout << "and is to the right of " << bestMatches[bestMatches.size()-1].l2 << endl;
                        cout << "and the distance to the last line is " << abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]) << " which is less than " << minDist<< endl
                            << intersect(lines[j], horiz)[0] << " to " << intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]<< endl;
                        */
                        minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                        index = j;
                    }
                }
            }
        }
        bestMatches.push_back( Match( templateLines[2], lines[index], 666) );
        
        
        
        // Find line left of center
        index = -1;
        minDist = -1;
        for(int j = 0; j < lines.size(); j++){
            //if line is to the left of center line
            if( intersect(lines[j], horiz)[0] < intersect(bestMatches[2].l2, horiz)[0] ){
                //if distance is smaller than last checked distance
                if( abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]) < minDist || minDist == -1 ){
                    // Check if not already matched
                    if( !isMatched(j, lines, bestMatches) ){
                        minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                        index = j;
                    }
                }
            }
        }
        bestMatches.push_back( Match( templateLines[0], lines[index], 666) );
        
    } else if (selectedLine == 2){ // right
        // Rightmost vertical line matched is the line clicked by the user
        int closest = -1;
        int closestDist = -1;
        for(int i = 0; i < lines.size(); i++){
            if(minimum_distance(lines[i], clickCoords) < closestDist || closestDist == -1){
                closest = i;
                closestDist = minimum_distance(lines[i], clickCoords);
            }
        }
        bestMatches.push_back( Match( templateLines[2], lines[closest], 666) );
        
        for(int i = 1; i >= 0; i--){
            int index = -1;
            int minDist = -1;
            for(int j = 0; j < lines.size(); j++){
                //if line is to the left of last matched line
                if( intersect(lines[j], horiz)[0] < intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0] ){
                    //cout << intersect(lines[j], horiz)[0] << " is < than " << intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0] << endl;
                    //if distance is smaller than last checked distance
                    if( abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]) < minDist || minDist == -1 ){
                        // Check if not already matched
                        if( !isMatched(j, lines, bestMatches) ){
                            minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                            index = j;
                        }
                    }
                }
            }
            bestMatches.push_back( Match( templateLines[i], lines[index], 666) );
        }
        
    }
    
    /*
    Vec4f oldLine = bestMatches[1].l2;
    Vec4f swapLine = Vec4f( oldLine[2], oldLine[3], oldLine[0], oldLine[1]);
    bestMatches[1].l2 = swapLine;
    */
    
    
    vector<Vec3f> templateH = vector<Vec3f>(templateLines.size());;
       vector<Vec3f> matchedH = vector<Vec3f>(templateLines.size());;
       
       // Take detected lines and template line sets and convert them to homogenous coordinates
       for(int i = 0; i < bestMatches.size(); i++){
           templateH[i] = Vec3f(bestMatches[i].l1[0], bestMatches[i].l1[1], 1).cross( Vec3f(bestMatches[i].l1[2], bestMatches[i].l1[3], 1 ) );
           if(templateH[i][2] != 0){
               templateH[i][0] /= templateH[i][2];
               templateH[i][1] /= templateH[i][2];
               templateH[i][2] /= templateH[i][2];
           }
           
           matchedH[i] = Vec3f(bestMatches[i].l2[0], bestMatches[i].l2[1], 1).cross( Vec3f(bestMatches[i].l2[2], bestMatches[i].l2[3], 1 ) );
           if(matchedH[i][2] != 0){
               matchedH[i][0] /= matchedH[i][2];
               matchedH[i][1] /= matchedH[i][2];
               matchedH[i][2] /= matchedH[i][2];
           }
       }
    
    
    Mat warp = Mat(1080, 1920, CV_8UC3, Scalar(0,0,0));
    
    line(warp, Point(horiz[0], horiz[1]), Point(horiz[2], horiz[3]), Scalar(255,255,255), 5);
    for(int i = 0; i < bestMatches.size(); i++)
    {
        line( warp, Point(bestMatches[i].l1[0], bestMatches[i].l1[1]), Point(bestMatches[i].l1[2], bestMatches[i].l1[3]), Scalar(0,255,255), 2, 0);
        line( warp, Point(bestMatches[i].l2[0], bestMatches[i].l2[1]), Point(bestMatches[i].l2[2], bestMatches[i].l2[3]), Scalar(255,0,255), 2, 0);
        Vec2f tempMid = getCenter(bestMatches[i].l1);
        Vec2f matchMid = getCenter(bestMatches[i].l2);
        line( warp, Point(tempMid[0], tempMid[1]), Point(matchMid[0], matchMid[1]), Scalar(0,255,100), 2, 0);
    }
    
    if(debugDisplay) imshow("warp", warp);
    //waitKey();
    
    
    
    
    // Homography computation using SVD
    Mat aMat = Mat(0,9,CV_32F);
    
    for(int i = 0; i < templateH.size(); i++){
        float x, y, u, v;
        
        x = templateH[i][0];
        y = templateH[i][1];
        u = matchedH[i][0];
        v = matchedH[i][1];
        
        Mat a1 = Mat(2, 9, CV_32F);
        
        a1.at<float>(0,0) = -u;
        a1.at<float>(0,1) = 0;
        a1.at<float>(0,2) = u*x;
        a1.at<float>(0,3) = -v;
        a1.at<float>(0,4) = 0;
        a1.at<float>(0,5) = v*x;
        a1.at<float>(0,6) = -1;
        a1.at<float>(0,7) = 0;
        a1.at<float>(0,8) = x;
        
        a1.at<float>(1,0) = 0;
        a1.at<float>(1,1) = -u;
        a1.at<float>(1,2) = u*y;
        a1.at<float>(1,3) = 0;
        a1.at<float>(1,4) = -v;
        a1.at<float>(1,5) = v*y;
        a1.at<float>(1,6) = 0;
        a1.at<float>(1,7) = -1;
        a1.at<float>(1,8) = y;
        
        vconcat(aMat, a1, aMat);
    }
    
    SVD aSVD = SVD(aMat);
    Mat rightSingular;
    transpose(aSVD.vt, rightSingular);
    Mat h = rightSingular.col( rightSingular.cols-1);
    
    
    Mat homography = Mat(3, 3, CV_32F);
    for (int i = 0 ; i < 3 ; i++){
        for (int j = 0 ; j < 3 ; j++){
            homography.at<float>(i,j) = h.at<float>(3*i+j, 0);
        }
    }
    
    //cout << homography << endl << endl;
    
    //////////////////////////////////////////////////////////
    ////////////// Display output for debugging //////////////
    //////////////////////////////////////////////////////////
    
    
    //Mat warp = src.clone();
    
    
    //cout << "TEMPLATE H SIZE: " << templateH.size() << endl;
    //cout << "BestMatchesSIZE: " << bestMatches.size() << endl;
    
    /*
    for(int i = 0; i < bestMatches.size(); i++)
    {
        line( warp, Point(bestMatches[i].l1[0], bestMatches[i].l1[1]), Point(bestMatches[i].l1[2], bestMatches[i].l1[3]), Scalar(0,255,255), 2, 0);
        line( warp, Point(bestMatches[i].l2[0], bestMatches[i].l2[1]), Point(bestMatches[i].l2[2], bestMatches[i].l2[3]), Scalar(255,0,255), 2, 0);
        Vec2f tempMid = getCenter(bestMatches[i].l1);
        Vec2f matchMid = getCenter(bestMatches[i].l2);
        line( warp, Point(tempMid[0], tempMid[1]), Point(matchMid[0], matchMid[1]), Scalar(0,255,100), 2, 0);
    }
    */
    
    
    //imshow("input", warp);
    
    Mat templateOriginal = Mat(warp.rows, warp.cols, CV_8UC3);
    Mat templateWarped = Mat(warp.rows, warp.cols, CV_8UC3);
    for(int i = 0; i < templateLines.size(); i++)
    {
        line( templateOriginal, Point(bestMatches[i].l1[0], bestMatches[i].l1[1]), Point(bestMatches[i].l1[2], bestMatches[i].l1[3]), Scalar(255,0,255), 2, 0);
    }
    warpPerspective(templateOriginal, templateWarped, homography, Size(templateOriginal.cols, templateOriginal.rows));
    for(int i = 0; i < templateH.size(); i++)
    {
        line( templateWarped, Point(bestMatches[i].l2[0], bestMatches[i].l2[1]), Point(bestMatches[i].l2[2], bestMatches[i].l2[3]), Scalar(255,255,255), 2, 0);
    }
    
    //imshow("Unwarpeeddd", templateWarped);
    
    if(!homography.empty()){

        std::vector<Point2f> in;
        std::vector<Point2f> out;
        
        //cout << "\n\nHomography : \n" << homography << "\n\n";
        
        //vector<Vec4f> warpLines {Vec4f(0,0,0,2800), Vec4f(440,0,440,2800), Vec4f(1400,0,1400,2800), Vec4f(0,0,4400,0), Vec4f(0,2800,4400,2800)};
        //templateLines = {Vec4f(150, 150, 150, 850), Vec4f(260, 150, 260, 850), Vec4f(500, 150, 500, 850), Vec4f(810, 150, 820, 850), Vec4f(150, 150, 1500, 150), Vec4f(150, 850, 700, 850)};
        
        
        for( int i = 0; i < templateLines.size(); i++){
            //cout << templateLines[i] << "\t\t!!!!!\n";
            in.push_back( Point2f( templateLines[i][0], templateLines[i][1]) );
            in.push_back( Point2f( templateLines[i][2], templateLines[i][3]) );
            //cout << Point(in[i].x, in[i].y) << "\t" << Point(in[i+1].x, in[i+1].y) << endl;
        }
        
        //cout << in << endl;
        
        //cout << "\n\n\n\n";
        
        perspectiveTransform( in , out, homography);
        
        if(getReprojection){
        // Print reprojected template corner positions
            vector<Point2f> reproj;
            for( int i = 0; i < out.size()-4; i++){
                reproj.push_back(out[i]);
            }
            // I didn't know how to change the object ordering in Blender, so I did this BS. Sorry.
            /*
            cout << reproj[1].x << ", " << reproj[1].y << endl;
            cout << reproj[5].x << ", " << reproj[5].y << endl;
            cout << reproj[3].x << ", " << reproj[3].y << endl;
            cout << reproj[0].x << ", " << reproj[0].y << endl;
            cout << reproj[4].x << ", " << reproj[4].y << endl;
            cout << reproj[2].x << ", " << reproj[2].y << endl;
             */
            //cout << "\n\n\n";
            
            cout << reproj[0].x << ", " << reproj[0].y << endl;
            cout << reproj[1].x << ", " << reproj[1].y << endl;
            cout << reproj[2].x << ", " << reproj[2].y << endl;
            cout << reproj[3].x << ", " << reproj[3].y << endl;
            cout << reproj[4].x << ", " << reproj[4].y << endl;
            cout << reproj[5].x << ", " << reproj[5].y << endl;
            //cout << "\n";
        }
            
        for( int i = 0; i < out.size(); i += 2){
            //line( finale, Point(in[i].x, in[i].y), Point(in[i+1].x, in[i+1].y), Scalar(0,255,255), 2, 0);
            line( finale, Point(out[i].x, out[i].y), Point(out[i+1].x, out[i+1].y), Scalar(255,0,255), 2, 0);
            //cout << Point(in[i].x, in[i].y) << "\t" << Point(in[i+1].x, in[i+1].y) << endl;
            //cout << Point(out[i].x, out[i].y) << "\t" << Point(out[i+1].x, out[i+1].y) << endl << endl << endl;
        }
        
        if(getPose){
            Mat imgPoints = (Mat_<float>(4,2) << out[0].x, out[0].y, out[2].x, out[2].y, out[1].x, out[1].y, out[3].x, out[3].y);
            
            Mat objPoints;
            if(selectedLine == 0){
                objPoints = (Mat_<float>(4,3) << 8.23, -63.24, 0, -1.34, -63.24, 0, 8.52, -4.86, 0, -1, -4.86, 0);
            } else if(selectedLine == 1) {
                objPoints = (Mat_<float>(4,3) << -40, -62.98, 0, -49.96, -62.96, 0, -39.91, -4.69, 0, -49.68, -4.58, 0);
            } else if(selectedLine == 2){
                objPoints = (Mat_<float>(4,3) << -77.33, -62.84, 0, -98.57, -62.71, 0, -77.07, -4.42, 0, -98.26, -4.33, 0);
            }
            
            //cout << "\n\n\n" << imgPoints << endl << objPoints << "\n\n\n";
            
            Mat rVec, tVec, distCoeffs;
            /*Mat cameraMatrix = (Mat_<float>(3, 3) << 1920, 0, 1920/2,
                                                        0, 1920, 1080/2,
                                                        0, 0, 1);
            */
            // Camera mat from Blender for synthetic testing
            Mat cameraMatrix = (Mat_<float>(3, 3) << -1600, 0, 960,
                                                    0, -1600, 540,
                                                    0, 0, 1);
            
            solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rVec, tVec);
            
            Mat rMat;
            Rodrigues(rVec, rMat);
            
            //cout << rMat;
            
            Mat flip = (Mat_<float>(4,4) << -1, 0, 0, 0,
                                          0, 1, 0, 0,
                                          0, 0, -1, 0,
                                          0, 0, 0, 1);
            
            Mat pose = (Mat_<float>(4,4) << rMat.at<float>(0,0), rMat.at<float>(0,1), rMat.at<float>(0,2), tVec.at<float>(0),
                                            rMat.at<float>(1,0), rMat.at<float>(1,1), rMat.at<float>(1,2), tVec.at<float>(1),
                                            rMat.at<float>(2,0), rMat.at<float>(2,1), rMat.at<float>(2,2), tVec.at<float>(2),
                                            0, 0, 0, 1);
            Mat invPose;
            invPose = pose.inv()*flip;
            
            Mat rMatInv = (Mat_<float>(3,3) <<  invPose.at<float>(0,0), invPose.at<float>(0,1), invPose.at<float>(0,2),
                                                invPose.at<float>(1,0), invPose.at<float>(1,1), invPose.at<float>(1,2),
                                                invPose.at<float>(2,0), invPose.at<float>(2,1), invPose.at<float>(2,2));
            
            Mat rVecInv;
            Rodrigues(rMatInv, rVecInv);
            //cout << "invpose:\n" << invPose << endl << endl;
            //cout << rMatInv << endl<<endl;
            //cout << rVecInv << endl<<endl;
            for(int i = 0; i < 3; i++){
                //cout << gtPose.at<float>(i, 3) << ", ";
                //if(i != 2) cout << ", " ;
            }
            for(int i = 0; i < 3; i++){
                //cout << invPose.at<float>(i, 3);
                //if(i != 2) cout << ", " ;
            }
            
            //cout << endl << endl;
            for(int i = 0; i < 3; i++){
                //cout << rVecGT.at<float>(i)*(180/CV_PI) << ", ";
                //if(i != 2) cout << ", " ;
            }
            for(int i = 0; i < 3; i++){
                //cout << rVecInv.at<float>(i)*(180/CV_PI);
                //if(i != 2) cout << ", " ;
            }
            
            //cout << endl;
        }
    }
    
    
    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    //cout << elapsed << endl;
    
    if(debugDisplay) imshow("Cleaned Lines", src);
    if(outputDisplay || debugDisplay){
        imshow("Finale", finale);
        waitKey();
    }
    //imshow("TEMPLATE", templateWarped);
    
}
