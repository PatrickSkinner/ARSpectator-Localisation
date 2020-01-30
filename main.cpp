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
    
    Vec2f getCenter(Vec4f line){
        return Vec2f( ((line[0] + line[2] )/2) , ((line[1] + line[3] )/2) );
    }
    
    float xDistance(){
        if(dist == 99999) return dist;
        /*
        float grad1 = (l1[3]-l1[1]) / (l1[2]-l1[0]);
        float grad2 = (l2[3]-l2[1]) / (l2[2]-l2[0]);
        cout << "da gradientsz: " << grad1 << "    -    " << grad2 << endl;
        
        
        int y = 500;
        int x1 = l1[0]+(y / grad1);
        int x2 = l2[0]+(y / grad2);
        
        if((l1[2]-l1[0]) == 0) x1 = l1[0];
        if((l2[2]-l2[0]) == 0) x2 = l2[0];
        
        cout << "da goordinates: " << x1 << "    -    " << x2 << endl;
        */
        //return abs(x1-x2);
        return abs(getCenter(l1)[0] - getCenter(l2)[0]);
    }
    
    float yDistance(){
        if(dist == 99999) return dist;
        return abs(getCenter(l1)[1] - getCenter(l2)[1]);
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

/** for each template line (L1), find the best matching line in the image (L2) */
extern "C++" vector<Match> getBestMatches(vector<Match> matches, vector<Vec4f> templateLines){
    vector<Match> bestMatches;
    
    Match candidate;
    for(int i = 0; i < matches.size(); i++){
        if(matches[i].xDistance() < candidate.xDistance()){
            if(matches[i].l1 == templateLines[0]){
                candidate = matches[i]; // find best match for leftmost template line
            }
        }
    }
    bestMatches.push_back(candidate);
    
    for(int i = 1; i < templateLines.size()-2; i++){
        candidate = Match();
        for(int j = 0; j < matches.size(); j++){
            if(matches[j].xDistance() < candidate.xDistance()){
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
        if(matches[i].yDistance() < candidate.yDistance()){
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
        if(matches[i].yDistance() < candidate.yDistance()){
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

/*
extern "C++" Vec4f fitBestLine( vector<Vec4f> inputLines, int bias){
    float avgX = 0.0;
    float avgY = 0.0;
    float avgAngle = 0.0;
    
    for(int i = 0; i < inputLines.size(); i++){
        avgX += getCenter(inputLines[i])[0];
        avgY += getCenter(inputLines[i])[1];
        avgAngle += getAngle(inputLines[i]);
    }
    
    avgX /= inputLines.size();
    avgY /= inputLines.size();
    avgAngle /= inputLines.size();
    
    float grad = tan(avgAngle);
    float len = 150;
    
    return Vec4f(avgX - len, avgY - (len*grad), avgX + len, avgY + (len*grad));
}*/

extern "C++" Vec4f fitBestLine( vector<Vec4f> inputLines, Vec2f center){
    float avgX = 0.0;
    float avgY = 0.0;
    float avgAngle = 0.0;
    
    float x = 0;
    float y = 0;
    float closestDist = 99999;
    
    for(int i = 0; i < inputLines.size(); i++){
        avgX += getCenter(inputLines[i])[0];
        avgY += getCenter(inputLines[i])[1];
        
        double angle = 0;
        angle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
        avgAngle += angle;
        
        //float dist = abs( lineLength( Vec4f(getCenter(inputLines[i])[0], getCenter(inputLines[i])[1], center[0], center[1] )));+
        

        
        float dist = abs(getCenter(inputLines[i])[1] - center[1]);
        if( dist < closestDist ){
            closestDist = dist;
            x = getCenter(inputLines[i])[0];
            y = getCenter(inputLines[i])[1];
        }
    }
    
    avgX /= inputLines.size();
    avgY /= inputLines.size();
    
    avgX = x;
    avgY = y;
    avgAngle /= inputLines.size();
    
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
            
            /*
            Vec4f outputLine;
            fitLine(points, outputLine, DIST_L12, 0, 0.01, 0.01);
            
            // Convert from direction/point format to a line defined by its endpoints
            Vec4f pushLine = Vec4f(outputLine[2] + outputLine[0]*150, // 150 is arbitrary, line length isn't considered when converted to homogenous coordinates later.
                                   outputLine[3] + outputLine[1]*150,
                                   outputLine[2] - outputLine[0]*150,
                                   outputLine[3] - outputLine[1]*150
                                   );
            
             */
             cleanedLines.push_back( pushLine );
            
            line( clustered, Point(pushLine[0], pushLine[1]), Point(pushLine[2], pushLine[3]), Scalar(0,0,255), 2, 0);
        }
    }
    
    return cleanedLines;
}

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

extern "C" void ComputePNP(Vec3f *&op, Vec2f *&ip, float ** rv, float ** tv, int& width, int& height)
{
    Mat imagePoints;
    Mat objectPoints;
    Mat cameraMatrix;
    Mat distCoeffs;
    
    Mat rotation;
    Mat translation;
    
    imagePoints = Mat(4, 2, CV_32F, ip);
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

        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4f l = lines[i];
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
