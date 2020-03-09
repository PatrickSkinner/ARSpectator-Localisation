//
//  main.cpp
//  LineDetectorOSXBundle
//
//  Created by Patrick Skinner on 28/08/19.
//  Copyright © 2019 Patrick Skinner. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

Mat src;
Mat thresh;
Mat lineOut;
Mat finale;
Mat clustered;

Mat finuks;

Mat homography = Mat(3, 3, CV_32F);

vector<Vec4f> templateLines {Vec4f(0,0,0,920), Vec4f(140,0,140,920), Vec4f(440,0,440,920), Vec4f(0,920,440,920), Vec4f(0,0,440,0)};
vector<Vec4f> templateLinesDefault {Vec4f(0,0,0,920), Vec4f(140,0,140,920), Vec4f(440,0,440,920), Vec4f(0,920,440,920), Vec4f(0,0,440,0)};
std::vector<Point2f> out;

class Match{
public:
    Vec4f l1; // Template Line x,y,x,y
    Vec4f l2; // Detected Line x,y,x,y
    
    double dist;
    
    Match(){
        l1 = NULL;
        l2 = NULL;
        dist = 99999;
    }
    
    Match(Vec4f line1, Vec4f line2, double distance){
        l1 = line1;
        l2 = line2;
        dist = distance;
    }
};

/** compared matches by distance for sorting purposes */
extern "C++" bool compareMatches(Match m1, Match m2){
    return (m1.dist < m2.dist);
}

/** Get center point of given line */
extern "C++" Vec2f getCenter(Vec4f line){
    return Vec2f( ((line[0] + line[2] )/2) , ((line[1] + line[3] )/2) );
}

/** Cotangent function */
float cotan(float i) {
    if( i < CV_PI/2 + 0.001 && i > CV_PI/2 - 0.001){
        return 0;
    }
    return(1 / tan(i));
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

/** Convert homogenous vector to form (x, y, 1) */
Vec3f constrainVec(Vec3f in){
    if(in[2] != 0){
        return Vec3f( in[0]/in[2], in[1]/in[2], in[2]/in[2]);
    } else {
        return in;
    }
}

/** Return intersection of two lines in homogenous coordinates */
Vec3f intersect(Vec3f a, Vec3f b){
    return constrainVec( a.cross(b) );
}

/** Take a set of intersecting lines and change the endpoints of each line to the intersection points */
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

/** for each L1, find the best matching L2 */
vector<Match> getBestMatches(vector<Match> matches, vector<Vec4f> templateLines){
    vector<Match> bestMatches;
    
    Match candidate;
    for(int i = 0; i < matches.size(); i++){
        if(matches[i].dist < candidate.dist){
            if(matches[i].l1 == templateLines[0]) candidate = matches[i]; // find best match for leftmost template line
        }
    }
    bestMatches.push_back(candidate);
    
    for(int i = 1; i < templateLines.size()-2; i++){
        candidate = Match();
        for(int j = 0; j < matches.size(); j++){
            if(matches[j].dist < candidate.dist){
                if(matches[j].l1 == templateLines[i]){
                    if( getCenter(matches[j].l2)[0] > getCenter(bestMatches[i-1].l2)[0] ){ // Candidate match midpoint is to the right of previous matched line
                        candidate = matches[j];
                    }
                }
            }
        }
        bestMatches.push_back(candidate);
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
    
    return bestMatches;
}


/** Get angle of given line */
extern "C++" double getAngle(Vec4f line1){
    double angle1 = atan2( ( line1[3] - line1[1] ), ( line1[2] - line1[0] ) );
    angle1 *= (180/ CV_PI);
    angle1 = ((int) angle1 + 360) % 360;
    return abs(angle1);
}

/** Return difference between angle of two given lines */
extern "C++" double getAngle(Vec4f line1, Vec4f line2){
    double angle1 = atan2( ( line1[3] - line1[1] ), ( line1[2] - line1[0] ) );
    double angle2 = atan2( ( line2[3] - line2[1] ), ( line2[2] - line2[0] ) );
    angle1 *= (180/ CV_PI);
    angle2 *= (180/ CV_PI);
    if(angle1 < 0) angle1 = 180 + angle1;
    if(angle2 < 0) angle2 = 180 + angle2;
    return abs(angle1-angle2);
}

/** Compare lines by angle */
extern "C++" bool compareVec(Vec4f v1, Vec4f v2)
{
    return (getAngle(v1) < getAngle(v2));
}

/** Calculate the length of a given line */
extern "C++" float lineLength(Vec4f line){
    return sqrt( pow((line[2] - line[0]), 2) + pow((line[1] - line[3]), 2) ) ;
}

/** Return the distance between the midpoints of two lines */
extern "C++" float midpointDistance(Vec4f line1, Vec4f line2){
    Vec2f mid1 = getCenter(line1);
    Vec2f mid2 = getCenter(line2);
    return abs( lineLength( Vec4f(mid1[0], mid1[1], mid2[0], mid2[1] )));
}

/** Given a cluster of lines return one line that best represents the cluster */
extern "C++" Vec4f fitBestLine( vector<Vec4f> inputLines, Vec2f center){
    float avgX = 0.0;
    float avgY = 0.0;
    float avgAngle = 0.0;
    
    float x = 0;
    float y = 0;
    float ang = 0;
    float closestDist = 99999;
    
    float searchRange = 10;
    
    for(int i = 0; i < inputLines.size(); i++){
        avgX += getCenter(inputLines[i])[0];
        avgY += getCenter(inputLines[i])[1];
        
        double angle = 0;
        angle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
        avgAngle += angle;
        
        //float dist = abs( lineLength( Vec4f(getCenter(inputLines[i])[0], getCenter(inputLines[i])[1], center[0], center[1] )));
        
        float dist = abs(getCenter(inputLines[i])[1] - center[1]);
        if( dist < closestDist ){
            closestDist = dist;
            x = getCenter(inputLines[i])[0];
            y = getCenter(inputLines[i])[1];
            cout << "angle: " << ang << ",   X: "<< inputLines[i][3] - inputLines[i][1] << ",   Y: " << inputLines[i][3] - inputLines[i][1] << endl;
            ang = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
            ang *= (180/CV_PI);
            if(ang < 0) ang += 180;
            //ang = getAngle(inputLines[i]);
        }
        //cout << getAngle(inputLines[i]) << endl;
        //avgAngle += getAngle(inputLines[i]);
    }
    
    
    avgX /= inputLines.size();
    avgY /= inputLines.size();
    avgAngle /= inputLines.size();
    
    /*
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
        if(ang >= 45 && ang <= 160){ // fix y axis
            int distAtX = 0;
            float grad = getGradient(inputLines[i]);
            
            float yDiff = y - inputLines[i][1];
            float steps = yDiff / grad;
            
            /*
             cout << "Grad: " << grad << endl;
             cout << "yDiff: " << yDiff << endl;
             cout << "Steps: " << steps << endl;
             */
            
            Point2f p = Point2f(inputLines[i][0] + steps , y);
            line( src, Point(inputLines[0][0], inputLines[0][1]), Point(inputLines[0][2], inputLines[0][3]), Scalar(200,0,255), 10, 0);
            line( src, Point( p.x - 7, p.y), Point( p.x + 7, p.y), Scalar(255,255,255), 5, 0);
            
            //cout << "POINT P: " << p << endl << endl;
            
            distAtX = abs( lineLength( Vec4f(compare.x, compare.y, p.x, p.y )));
            if(distAtX < searchRange){
                
                float thisAngle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
                //thisAngle *= (180/CV_PI);
                //if(thisAngle < 0) thisAngle += 180;
                cout << " thisAngle :    " << thisAngle << endl;
                
                
                
                avgX += p.x;
                avgY += p.y;
                avgAngle += thisAngle;
                count++;
            }
        } else { // fix x axis
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
            line( src, Point(inputLines[0][0], inputLines[0][1]), Point(inputLines[0][2], inputLines[0][3]), Scalar(200,0,255), 10, 0);
            line( src, Point( p.x, p.y-9), Point( p.x, p.y+9), Scalar(255,255,255), 5, 0);
            
            distAtX = abs( lineLength( Vec4f(compare.x, compare.y, p.x, p.y )));
            
            if(distAtX < searchRange){
                float thisAngle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
                //thisAngle *= (180/CV_PI);
                //if(thisAngle < 0) thisAngle += 180;

                avgX += p.x;
                avgY += p.y;
                avgAngle += thisAngle;
                count++;
                
            }
        }
    }
    
    
    avgX /= count;
    avgY /= count;
    avgAngle /= count;
    
    
    float grad = tan(avgAngle);
    float len = 1000;
    
    return Vec4f(avgX - len, avgY - (len*grad), avgX + len, avgY + (len*grad));
}

/** Calculate the total Hausdorff distance between two line sets */
extern "C++" float getSetDistance(vector<Vec4f> templateLines, vector<Vec4f> detectedLines){
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
            
            totalDistance += min(    max( lineLength(ac),lineLength(bd)) ,     max( lineLength(ad),lineLength(bc))       );
        }
    }
    
    return totalDistance;
}

/**Split horizontal lines with matching labels into two groups, for top and bottom of the pitch */
extern "C++" vector<int> splitHorizontals( vector<Vec4f> lines ){
    vector<int> labels;
    
    float y1 = 0.0, y2 = 0.0;
    for(int i = 0; i < lines.size(); i++){
        y1 += lines[i][1];
        y2 += lines[i][3];
        
    }
    
    y1 /= lines.size();
    y2 /= lines.size();
    float avgY = (y1+y2)/2;
    
    for(int i = 0; i < lines.size(); i++){
        if( getCenter(lines[i])[1] < avgY ){
            labels.push_back(0);
        } else {
            labels.push_back(1);
        }
    }
    
    return labels;
}

/** Find the best fitting line for each line cluster and return the line set of best fitting lines */
extern "C++" vector<Vec4f> cleanLines(vector<Vec4f> lines){
    int angleThreshold = 4;
    
    vector<Vec4f> sortedLines = lines;
    vector<int> sortedAngles;
    
    float centroidX = 0.0;
    float centroidY = 0.0;
    
    
    sort(sortedLines.begin(), sortedLines.end(), compareVec); // Sort lines by gradient to make removing duplicates easier
    int label = 0;
    float startAngle = getAngle(sortedLines[0]);
    for(int i = 0; i < sortedLines.size(); i++ ){
        
        float angle = getAngle(sortedLines[i]);
        if( angle >= 350 && angle <= 360 ){
            sortedAngles.push_back(0);
        }else if( (angle - startAngle) < angleThreshold){           // adjust cluster ranges here
            sortedAngles.push_back(label);
            startAngle = angle;
        } else {
            // Move to next cluster
            label++;
            sortedAngles.push_back(label);
            startAngle = angle;
        }
        
        Vec2f mid = getCenter(sortedLines[i]);
        centroidX += mid[0];
        centroidY += mid[1];
    }
    
    /* Split horizontal lines into two clusters, for the bottom and top of the pitch */
    vector<Vec4f> horizontals;
    Mat labels;
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
    
    clustered = src.clone();
    srand(83);
    for(int i = 0; i < k+1; i++){
        Scalar colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) )); // Random colour for each cluster
        for(int j = 0; j < labels.rows; j++){
            if(labels.at<int>(j) == i){
                line( clustered, Point(sortedLines[j][0], sortedLines[j][1]), Point(sortedLines[j][2], sortedLines[j][3]), colour, 5, 0);
            }
        }
    }
    
    vector<Vec4f> cleanedLines;
    
    for(int i = 0; i < k+1; i++){
        vector<Vec2f> points;
        vector<Vec4f> lines;
        
        for(int j = 0; j < labels.rows; j++){
            if(labels.at<int>(j) == i){
                points.push_back( Vec2f(sortedLines[j][0], sortedLines[j][1]));
                points.push_back( Vec2f(sortedLines[j][2], sortedLines[j][3]));
                lines.push_back( Vec4f(sortedLines[j][0], sortedLines[j][1], sortedLines[j][2], sortedLines[j][3]));
            }
        }
        if(points.size() != 0 ){
            
            Vec2f centroid = Vec2f( centroidX / sortedLines.size(), centroidY / sortedLines.size());
            Vec4f pushLine = fitBestLine(lines, centroid);
            cleanedLines.push_back( pushLine );
            
            line( clustered, Point(pushLine[0], pushLine[1]), Point(pushLine[2], pushLine[3]), Scalar(0,0,255), 2, 0);
        }
    }
    
    cleanedLines = trimLines(cleanedLines);
    return cleanedLines;
}

/** Use OpenCVs Hough line detector to find all lines in the input image, src */
extern "C++" vector<Vec4f> getLines()
{
    Mat HSV;
    
    if(! src.data ) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    
    cvtColor(src, HSV, COLOR_BGR2HSV);
    
    // Detect the field based on HSV Range Values
    inRange(HSV, Scalar(35, 30, 0), Scalar(80, 255, 255), thresh);
    Mat dst, invdst, cdst;
    GaussianBlur( thresh, invdst, Size( 5, 5 ), 0, 0 );
    Canny(invdst, dst, 50, 200, 3);
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    
    vector<Vec4f> lines;
    HoughLinesP(dst, lines, 2, CV_PI/360, 100, 250, 45 );
    
    for(int i = 0; i < lines.size(); i++ ) line( cdst, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255,0,0), 2, 0);
    lineOut = cdst;
    
    return lines;
}

/** Compute a rotation and translation to create an alignment between 2d image points and the corresponding 3D points on the virtual pitch */
extern "C" void ComputePNP(Vec3f *&op, Vec2f *&ip, float ** rv, float ** tv, int& width, int& height)
{
    Mat imagePoints = Mat(4, 2, CV_32F);
    Mat objectPoints;
    Mat cameraMatrix;
    Mat distCoeffs;
    
    Mat rotation;
    Mat translation;
    
    //imagePoints = Mat(4, 2, CV_32F, ip);
    objectPoints = Mat(4, 3, CV_32F, op);
    
    imagePoints.at<float>(2, 0) = out[6].x;
    imagePoints.at<float>(2, 1) = out[6].y;
    imagePoints.at<float>(3, 0) = out[7].x;
    imagePoints.at<float>(3, 1) = out[7].y;
    imagePoints.at<float>(0, 0) = out[8].x;
    imagePoints.at<float>(0, 1) = out[8].y;
    imagePoints.at<float>(1, 0) = out[9].x;
    imagePoints.at<float>(1, 1) = out[9].y;
    
    cout << "imagePoints = " << endl << " " << imagePoints << endl << endl;
    cout << "objectPoints = " << endl << " " << objectPoints << endl << endl;
    
    cameraMatrix = (Mat_<float>(3, 3) << 1400, 0, 1920/2, 0, 1400, 1080/2, 0, 0, 1);
    //cameraMatrix = (Mat_<float>(3, 3) << max(width, height), 0, width / 2, 0, max(width, height), height / 2, 0, 0, 1);
    
    solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rotation, translation);
    
    Mat rotationMat;
    Rodrigues(rotation, rotationMat);
    cout << "rotationMat = " << endl << " " << rotationMat << endl << endl;
    
    *rv = static_cast<float*>(malloc(rotationMat.total() * rotationMat.elemSize()));
    memcpy(*rv, rotationMat.data, rotationMat.total() * rotationMat.elemSize());
    
    *tv = static_cast<float*>(malloc(translation.total() * translation.elemSize()));
    memcpy(*tv, translation.data, translation.total() * translation.elemSize());
}

/** input of form, horizontal, horizontal, verticals after, in order of right to left */
vector<Vec4f> rectifyLines(vector<Vec4f> inputLines){
    finuks = Mat(1080, 1920, CV_8UC1, Scalar(128,0,0));
    
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
    
    perspectiveTransform( in , affinePoints, affineTransform);
    
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
            perspectiveTransform( metricPoints, metricPoints, mirror); // Mirrored rectification
        } else {
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
            mirror.at<float>(0,0) = 1;
            mirror.at<float>(1,1) = -1;
            perspectiveTransform( metricPoints, metricPoints, mirror); // Mirrored on Y axis
        }
    }
    
    vector<Vec4f> outputLines;
    for(int i = 0; i < metricPoints.size(); i += 2){
        outputLines.push_back( Vec4f(metricPoints[i].x, metricPoints[i].y, metricPoints[i+1].x, metricPoints[i+1].y) );
    }
    
    int xAdj = -metricPoints[1].x + 100;
    int yAdj = -metricPoints[1].y + 100;
    
    for(int i = 0; i < metricPoints.size(); i += 2){
        //line( finuks, Point(metricPoints[i].x + xAdj, metricPoints[i].y + yAdj), Point(metricPoints[i+1].x + xAdj, metricPoints[i+1].y + yAdj), Scalar(0,0,0), 5);
    }
    
    return outputLines;
}

extern "C" int sendImage(uint8_t *&image, int& width, int& height, int& crop)
{
    clock_t start, end;
    double elapsed;
    start = clock();
    
    src = Mat(height, width, CV_8UC3, *&image);
    flip(src, src, +1);
    finale = src.clone();
    
    if (!src.empty()) {
        
        Mat converted;
        cvtColor(src, converted, COLOR_RGB2BGRA);
        src = converted;
        
        vector<Vec4f> rawLines = getLines();
        vector<Vec4f> lines = cleanLines(rawLines);
        vector<Vec4f> rectifiedLines = rectifyLines(lines);
        
        
        float templateHeight = lineLength( templateLines[0] );
        float rectifiedLinesHeight = lineLength(rectifiedLines[rectifiedLines.size() - 1]);
        float diff = templateHeight/rectifiedLinesHeight;
        cout << "diff: " << diff << endl;
        
        diff = 3;
        Mat scaling = Mat(3, 3, CV_32F, Scalar(0));
        scaling.at<float>(0,0) = diff;
        scaling.at<float>(1,1) = diff;
        scaling.at<float>(2,2) = 1;
        
        vector<Point2f> in2;
        vector<Point2f> out2;
        for( int i = 0; i < rectifiedLines.size(); i++){
            in2.push_back( Point2f( rectifiedLines[i][0], rectifiedLines[i][1]) );
            in2.push_back( Point2f( rectifiedLines[i][2], rectifiedLines[i][3]) );
        }
        
        perspectiveTransform(in2, out2, scaling);
        
        vector<Vec4f> scaledLines;
        for(int i = 0; i < out2.size(); i += 2){
            scaledLines.push_back(Vec4f( out2[i].x, out2[i].y, out2[i+1].x, out2[i+1].y));
        }
        
        rectifiedLines = scaledLines;
        
        Vec2f templateCenter = getCenter(templateLines[0]);
        Vec2f rectifiedCenter = getCenter( rectifiedLines[ rectifiedLines.size() - 1]);
        float xAdjust = rectifiedCenter[0] - templateCenter[0];
        float yAdjust = rectifiedCenter[1] - templateCenter[1];
        
        for(int i = 0; i < rectifiedLines.size(); i++){
            rectifiedLines[i][0] -= xAdjust;
            rectifiedLines[i][1] -= yAdjust;
            rectifiedLines[i][2] -= xAdjust;
            rectifiedLines[i][3] -= yAdjust;
        }
        
        
        
        
        for( size_t i = 0; i < rectifiedLines.size(); i++ )
        {
            Vec4f l = rectifiedLines[i];
            line( lineOut, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 0);
        }
        
        // Record each possible match
        vector<Match> matches;
        for(int i = 0; i < templateLines.size(); i++)
        {
            for(int j = 0; j < lines.size(); j++)
            {
                float dist = midpointDistance(templateLines[i], lines[j]);
                if( (getAngle(templateLines[i], lines[j]) < 55) || (getAngle(templateLines[i], lines[j]) > 170)){
                    matches.push_back( Match(templateLines[i], lines[j], dist ));
                }
            }
        }
        
        sort(matches.begin(), matches.end(), compareMatches);
        vector<Match> bestMatches = getBestMatches(matches, templateLines);
        vector<Vec3f> templateH = vector<Vec3f>(templateLines.size());
        vector<Vec3f> matchedH = vector<Vec3f>(templateLines.size());
        
        // Take detected lines and template line sets and convert them to homogenous coordinates
        for(int i = 0; i < bestMatches.size(); i++){
            
            // Find the corresponding non-warped template line to use for Homography estimation
            int index = 0;
            for(int j = 0; j < templateLines.size(); j++){
                if( templateLines[j] == bestMatches[i].l1){
                    index = j;
                }
            }
            //templateH[i] = Vec3f(bestMatches[i].l1[0], bestMatches[i].l1[1], 1).cross( Vec3f(bestMatches[i].l1[2], bestMatches[i].l1[3], 1 ) );
            templateH[i] = Vec3f(templateLinesDefault[index][0], templateLinesDefault[index][1], 1).cross( Vec3f(templateLinesDefault[index][2], templateLinesDefault[index][3], 1 ) );
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
            
            if(matchedH[i][2] == 0){
                cout << "LINE MATCH MISSING, REJECTED";
                templateLines = {Vec4f(0,0,0,920), Vec4f(140,0,140,920), Vec4f(440,0,440,920), Vec4f(0,920,440,920), Vec4f(0,0,440,0)};
                return 0;
            }
        }
        
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
        
        for (int i = 0 ; i < 3 ; i++){
            for (int j = 0 ; j < 3 ; j++){
                homography.at<float>(i,j) = h.at<float>(3*i+j, 0);
            }
        }
        
        for(int i = 0; i < templateH.size(); i++)
        {
            line( clustered, Point(bestMatches[i].l1[0], bestMatches[i].l1[1]), Point(bestMatches[i].l1[2], bestMatches[i].l1[3]), Scalar(0,255,255), 2, 0);
            line( clustered, Point(bestMatches[i].l2[0], bestMatches[i].l2[1]), Point(bestMatches[i].l2[2], bestMatches[i].l2[3]), Scalar(255,0,255), 2, 0);
            Vec2f tempMid = getCenter(bestMatches[i].l1);
            Vec2f matchMid = getCenter(bestMatches[i].l2);
            line( clustered, Point(tempMid[0], tempMid[1]), Point(matchMid[0], matchMid[1]), Scalar(0,255,100), 2, 0);
        }
        
        cout << "\n\n\n Top Bottom Difference: " << getCenter(bestMatches[bestMatches.size()-2].l2)[1] - getCenter(bestMatches[bestMatches.size()-1].l2)[1] << "\n\n\n";
        
        // Check that top match is above bottom match
        if( getCenter(bestMatches[bestMatches.size()-1].l2)[1] > getCenter(bestMatches[bestMatches.size()-2].l2)[1]){
            cout << "\n\n\n REVERSED MATCHES, SHOULD BE REJECTED \n\n\n";
            templateLines = {Vec4f(0,0,0,920), Vec4f(140,0,140,920), Vec4f(440,0,440,920), Vec4f(0,920,440,920), Vec4f(0,0,440,0)};
            return 0;
        }
        
        if( getCenter(bestMatches[bestMatches.size()-2].l2)[1] - getCenter(bestMatches[bestMatches.size()-1].l2)[1] < 320){
            cout << "\n\n\n LINES TOO CLOSE, SHOULD BE REJECTED \n\n\n";
            templateLines = {Vec4f(0,0,0,920), Vec4f(140,0,140,920), Vec4f(440,0,440,920), Vec4f(0,920,440,920), Vec4f(0,0,440,0)};
            return 0;
        }
        
        if(!homography.empty()){
            std::vector<Point2f> in;
            
            for( int i = 0; i < templateLines.size(); i++){
                in.push_back( Point2f( templateLinesDefault[i][0], templateLinesDefault[i][1]) );
                in.push_back( Point2f( templateLinesDefault[i][2], templateLinesDefault[i][3]) );
            }
            
            for( int i = 0; i < in.size(); i += 2){
                cout << "OG Points: " << Point(in[i].x, in[i].y) << "\t" << Point(in[i+1].x, in[i+1].y) << endl;
            }
            
            perspectiveTransform( in , out, homography);
            
            for( int i = 0; i < out.size(); i += 2){
                line( clustered, Point(out[i].x, out[i].y), Point(out[i+1].x, out[i+1].y), Scalar(255,0,255), 5, 0);
                cout << "Transformed by Homo: " << Point(out[i].x, out[i].y) << "\t" << Point(out[i+1].x, out[i+1].y) << endl;
            }
            
            //Reprojection
            templateLines = {
                Vec4f(out[0].x,out[0].y, out[1].x,out[1].y),
                Vec4f(out[2].x,out[2].y, out[3].x,out[3].y),
                Vec4f(out[4].x,out[4].y, out[5].x,out[5].y),
                Vec4f(out[6].x,out[6].y, out[7].x,out[7].y),
                Vec4f(out[8].x,out[8].y, out[9].x,out[9].y)
            };
            
            end = clock();
            elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
            cout << "Time Taken: " << elapsed << endl;
            
            return 1;
        }
    }
    
    return 0;
}

// Pass an image mat back to Unity to be displayed as a texture
extern "C" void GetRawImageBytes(unsigned char*& data, int& width, int& height)
{
    //Resize Mat to match the array passed to it from C#
    if(!src.empty()){
        cv::Mat resizedMat(height, width, clustered.type());
        cv::resize(clustered, resizedMat, resizedMat.size(), cv::INTER_CUBIC);
        
        //Convert from RGB to ARGB
        cv::Mat argb_img;
        cv::cvtColor(resizedMat, argb_img, COLOR_BGR2RGBA);
        std::memcpy(data, argb_img.data, argb_img.total() * argb_img.elemSize());
    }
}



