//
//  lineClustering.cpp
//  lineDetector
//
//  Created by Patrick Skinner on 12/10/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "support.hpp"

using namespace cv;
using namespace std;

extern bool debugDisplay;
Vec4f divider;
extern Mat clustered;


/** Trim length of lines down to the intersection points
 *  HT, HB, V, V..., V ordering
 */
vector<Vec4f> trimLines(vector<Vec4f> inputLines){
    vector<Vec4f> outputLines;
    
    // Convert lines to homogenous form
    vector<Vec3f> hmgLines;
    for(int i = 0; i < inputLines.size(); i++){
        hmgLines.push_back( Vec3f(inputLines[i][0], inputLines[i][1], 1).cross( Vec3f(inputLines[i][2], inputLines[i][3], 1 ) ) );
    }
    
    int n = (int) inputLines.size();
    
    // Trim top horizontal
    Vec3f inter1 = intersect( hmgLines[0], hmgLines[2]);
    Vec3f inter2 = intersect( hmgLines[0], hmgLines[n-1]);
    outputLines.push_back( Vec4f( inter1[0], inter1[1], inter2[0], inter2[1]) );
    
    // Trim bottom horizontal
    inter1 = intersect( hmgLines[1], hmgLines[2]);
    inter2 = intersect( hmgLines[1], hmgLines[n-1]);
    outputLines.push_back( Vec4f( inter1[0], inter1[1], inter2[0], inter2[1]) );
    
    // Trim all vertical Lines.
    for(int i = 2; i < n; i++){
        inter1 = intersect( hmgLines[i], hmgLines[0]);
        inter2 = intersect( hmgLines[i], hmgLines[1]);
        outputLines.push_back( Vec4f( inter1[0], inter1[1], inter2[0], inter2[1]) );
    }
    
    return outputLines;
}


Mat clusterLines(Mat in, vector<Vec4f>& lines){
    cout << "Lines: " << lines.size() << endl;
    Mat labels;
    vector<Vec4f> horizontals;
    vector<Vec4f> verticals;
    int horizontalCount = 0;

    Mat colorBound;
    extern Mat boundary;
    cvtColor(boundary, colorBound, COLOR_GRAY2BGR);
    
    cout << "IMG SIZE: " << clustered.cols << " x " << clustered.rows << endl;
    
    // Using voting scheme to determine which cell corresponds to the dominant vanishing point.
    vector<Point2f> vanishingPoints;
    const int cellsX = 192;
    const int cellsY = 108;
    int cells [cellsX*3][cellsY*3]{};
   
    for(int i = 0; i < lines.size(); i++){
        for(int j = 0; j < lines.size(); j++){
            if(i != j){
                Point2f intr = Point2f( intersect(lines[i], lines[j])[0], intersect(lines[i], lines[j])[1] );
                intr.x += 1920;
                intr.y += 1080;
                
                if( intr.x > 0 && intr.x < 1920*3 && intr.y > 0 && intr.y < 1080*3){
                    int cX = intr.x/10;
                    int cY = intr.y/10;
                    
                    if( intr.x < 1920 || intr.x >= 1920*2 || intr.y < 1080 || intr.y >= 1080*2 || colorBound.at<Vec3b>( intr.y-1080, intr.x-1920) == Vec3b(0,0,0) ){
                        cout << "Intersecton: " << intr << "\t Cell: " << cX << ", " << cY << endl;
                        cells[cX][cY]++;
                        line(clustered, Point2f(intr.x-1, intr.y-1), Point2f(intr.x+1, intr.y+1), Scalar(0,255,0), 15);
                        vanishingPoints.push_back(intr);
                    }
                }
            }
        }
    }
    
    int max = 0;
    Point2f maxCell;
    for(int i = 0; i < cellsX*3; i++){
        for(int j = 0; j < cellsY*3; j++){
            if( cells[i][j] > max  ){
               // if(i > 192 && j > 108){
                    max = cells[i][j];
                    maxCell = Point2f(i,j);
               // }
            }
        }
    }
    
    cout << "Max Cell: " << maxCell << endl;
    cout << "MC Coords: " <<  (maxCell.x*10)-1920 << ", " <<  (maxCell.y*10)-1080 << endl;
    rectangle(clustered, Point2f((maxCell.x*10-1920), (maxCell.y*10-1080)), Point2f((maxCell.x*10-1920)+192, (maxCell.y*10-1080)+108), Scalar(0,0,0));
    
    
    int cellRangeH = 6; // Expand search beyond identified cell to account for any slight errors
    int cellRangeV = 1;
    for(int i = 0; i < lines.size(); i++){
        bool flag = false;
        for(int j = 0; j < lines.size(); j++){
            if( i != j ){
                Point3f intr = intersect(lines[i], lines[j]);
                int x = (intr.x/10)+cellsX;
                int y = (intr.y/10)+cellsY;
                if(x >= maxCell.x-cellRangeH && x <= maxCell.x+cellRangeH
                   && y >= maxCell.y-cellRangeV && y <= maxCell.y+cellRangeV){
                    verticals.push_back(lines[i]);
                    flag = true;
                    break;
                }
            }
        }
        if(!flag){ // Add line to horizontal group
            horizontals.push_back(lines[i]);
            horizontalCount++;
            line(clustered, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,0), 3);
        }
    }
    
    // Find the most common angle for horizontal lines
    int numFreq [180] = {};
    for(int i = 0; i < horizontals.size(); i++){
        int c = ceil( getAngle(horizontals[i]) );
        if( c > 170 ) c = abs(c- 180);
        numFreq[c]++;
    }
    
    int maxNum = 0;
    int maxIndex = -1;
    for(int i = 0; i < 180; i++){
        if( numFreq[i] > maxNum ){
            maxIndex = i;
            maxNum = numFreq[i];
        }
    }
    
    int median = maxIndex;
    vector<Vec4f> cleanedHorizontals;
    
    // Discard any lines that aren't within 15 degrees of the median angle
    for(int i = 0; i < horizontals.size(); i++){
        if( checkThreshold( getAngle(horizontals[i]), median, 15) ){
                cleanedHorizontals.push_back(horizontals[i]);
            }
    }
    horizontals = cleanedHorizontals;
    
    //if(debugDisplay) imshow("cluster", in);
    
    int topH = -1;
    int bottomH = -1;
    int lowestMid = -1;
    int lowestLine = -1;
    
    int horizThresh = 70; // Max horizontal distance between lines before they're split into separate clusters
    extern Mat src;
    
    for(int i = 0; i < lines.size(); i++){
        
        if( getCenter(lines[i])[1] > lowestMid
            && getCenter(lines[i])[1] < src.rows - 25 ){ //extra hack to avoid the bottom edge of the image getting detected as a line
            lowestMid = getCenter(lines[i])[1];
            lowestLine = i;
        }
    }
    int baseline = getAngle(lines[lowestLine]);
    float grad = 0.0;
    int blThresh = 14;
    
    // Get average gradient of horizontal lines ( Redundant? )
    for(int i = 0; i < lines.size(); i++){
        float angle = getAngle(lines[i]);
        if(angle < 0) angle += 180;
        if( checkThreshold(angle, baseline, blThresh)){
            if( topH == -1 || getCenter(lines[i])[1] < topH) topH = getCenter(lines[i])[1] ;
            if( bottomH == -1 || getCenter(lines[i])[1] > bottomH) bottomH = getCenter(lines[i])[1] ;
            grad += getGradient(lines[i]);
        }
    }


    int mid = topH + (abs(topH-bottomH) / 2);
    mid -= 100;
    grad /= horizontalCount;
    divider = Vec4f(1920/2, 0, 1920/2, 1080);
    
    
    int vertThresh = 20;
    int lastY = -1;
    int cluster = 0;
    Scalar colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
    sort(horizontals.begin(), horizontals.end(), compareLinesByY);
    int smallestX = -1;
    int largestX = -1;
    vector<int> ranges;
    vector<int> horiLabels;
    
    int avgMidX = 0;
    for(int i = 0; i < horizontals.size(); i++){
        avgMidX += getCenter(horizontals[i])[0];
    }
    avgMidX /= horizontals.size();
    divider = Vec4f(avgMidX, 0, avgMidX, 1080);
    
    // Replace this later
    for(int i = 0; i < horizontals.size(); i++){
        int y = intersect( horizontals[i], divider)[1];
        
        if( lastY != -1 && abs(y - lastY) > vertThresh){
            colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
            cluster++;
            ranges.push_back( largestX - smallestX);
            smallestX = -1;
            largestX = -1;
        }
        if(horizontals[i][0] < smallestX || smallestX == -1) smallestX = horizontals[i][0];
        if(horizontals[i][2] < smallestX || smallestX == -1) smallestX = horizontals[i][2];
        if(horizontals[i][0] > largestX || largestX == -1) largestX = horizontals[i][0];
        if(horizontals[i][2] > largestX || largestX == -1) largestX = horizontals[i][2];

        //line(clustered, Point(horizontals[i][0], horizontals[i][1]), Point(horizontals[i][2], horizontals[i][3]), colour, 4);
        horiLabels.push_back(cluster);
        
        lastY = y;
    }

    ranges.push_back( largestX - smallestX);
    
    vector<Vec4f> recleanedHorizontals;
    bool topFound = false;
    bool botFound = false;
    int minRange = 1800;
    
    // Find the largest horizontal line spans above and below the divider
    while( !topFound || !botFound ){
        bool matched = false;
        for( int cluster = 0; cluster <= horiLabels[horiLabels.size()-1 ]; cluster++){
            if( ranges[cluster] > minRange/* && ranges[cluster] < 1550*/){
                bool tempTopFound = false;
                bool tempBotFound = false;
                for(int i = 0; i < horizontals.size(); i++ ){
                    if( cluster == horiLabels[i] ){
                        if( getCenter(horizontals[i])[1]  < mid && !topFound){
                            recleanedHorizontals.push_back( horizontals[i] );
                            matched = true;
                            tempTopFound = true;
                        } else if ( getCenter(horizontals[i])[1]  > mid && !botFound){
                            recleanedHorizontals.push_back( horizontals[i] );
                            matched = true;
                            tempBotFound = true;
                        }
                    }
                }
                if(!botFound) botFound = tempBotFound;
                if(!topFound) topFound = tempTopFound;
            }
        }
        minRange -= 200;
        if(minRange < -200) break;
    }
    
    horizontals = recleanedHorizontals;
    divider = Vec4f(0, mid-(grad*540), 1920, mid+(grad*540));
    line(in, Point(divider[0], divider[1]), Point(divider[2], divider[3]), Scalar(255,255,255), 4);
    
    for(int i = 0; i < horizontals.size(); i++){
        Vec2f vPoint = getCenter(horizontals[i]);
        Vec4f vLine( vPoint[0], 0, vPoint[0], 1080 );
        Vec3f inter = intersect(divider, vLine);
        
        if( inter[1] > vPoint[1]){
            labels.push_back(0);
        } else {
            labels.push_back(1);
        }
    }
    
    cluster = 2;
    colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
    divider = Vec4f(0, mid-(grad*540), 1920, mid+(grad*540));
    
    int lastX = -1;

    for(int i = 0; i < verticals.size(); i++){
        int x = intersect( verticals[i], divider)[0];
        if( lastX != -1 && abs(x - lastX) > horizThresh){
            colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
            cluster++;
        }
        line(clustered, Point(verticals[i][0], verticals[i][1]), Point(verticals[i][2], verticals[i][3]), Scalar(0,255,255), 14); // vertical lines
        labels.push_back(cluster);
        lastX = x;
    }
    
    if(debugDisplay) imshow("clustered", in);
    //waitKey();
    
    horizontals.insert( horizontals.end(), verticals.begin(), verticals.end() ); //
    lines = horizontals; // Need to put the input lines in same order as our labels. HHVVVV(...)V
    
    return labels;
}

// Given a cluster of lines find one single line of best fit, biased twowards the center of the pitch to avoid outliers.
Vec4f fitBestLine( vector<Vec4f> inputLines, Vec2f center, bool isHori){
    if(inputLines.size() == 1 ) return inputLines[0];
    
    float avgX = 0.0;
    float avgY = 0.0;
    float avgAngle = 0.0;
    
    float x = 0;
    float y = 0;
    float ang = 0;
    float closestDist = 99999;
    
    float searchRange = 15;
    
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

        if( !isHori){ // Line is vertical
            Vec4f horiz = Vec4f( center[0]-100, center[1], center[0]+100, center[1]);
            dist = abs(intersect(inputLines[i], horiz)[0] - center[0]);
        } else { //Line is horizontal
            if( (lineLength(inputLines[i]) > (1080/3 + 1080/2)/3) || // Awful hack to avoid those short horizontal lines being chosen as best fit.
                  (intersect(divider, Vec4f( getCenter(inputLines[i])[0], 0, getCenter(inputLines[i])[0], 1080 ))[1] > getCenter(inputLines[i])[1]) ) {
                Vec4f vert = Vec4f( center[0], center[1]-1000, center[0], center[1]+1000);
                dist = abs(intersect(inputLines[i], vert)[1] - center[1]);
            }
        }

        if( dist < closestDist && dist != -1){
            closestDist = dist;
            x = getCenter(inputLines[i])[0];
            y = getCenter(inputLines[i])[1];
            ang = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
            ang *= (180/CV_PI);
            if(ang < 0) ang += 180;
        }
        
    }

    avgX = 0;
    avgY = 0;
    avgAngle = 0;
    
    
    Point2f compare = Point2f(x, y);
    int count = 0;
    
    for( int i = 0; i < inputLines.size(); i++){

        if( !isHori){ //  vertical

            int distAtX = 0;
            float grad = getGradient(inputLines[i]);
            
            float yDiff = y - inputLines[i][1];
            float steps = yDiff / grad;
            if(grad == 0) steps = 0;
            
            Point2f p = Point2f(inputLines[i][0] + steps , y);

            distAtX = abs( lineLength( Vec4f(compare.x, compare.y, p.x, p.y )));
            
            if(distAtX < searchRange+50){ //increased search range for vertical lines, hacky af
                float thisAngle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
                if(thisAngle < 0) thisAngle += CV_PI;
                avgX += p.x;
                avgY += p.y;
                avgAngle += thisAngle;
                count++;
            }
            
        } else { // fix x axis, horizontal line
            
            int distAtX = 0;
            float grad = getGradient(inputLines[i]);
            
            float xDiff = x - inputLines[i][0];
            float steps = xDiff * grad;
            
            Point2f p = Point2f(x, inputLines[i][1] + steps);

            distAtX = abs( lineLength( Vec4f(compare.x, compare.y, p.x, p.y )));
            
            if(distAtX < searchRange){
                float thisAngle = atan2( ( inputLines[i][3] - inputLines[i][1] ), ( inputLines[i][2] - inputLines[i][0] ) );
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
    
    float grad = tan(avgAngle);
    float len = 10000;
    return Vec4f(avgX - len, avgY - (len*grad), avgX + len, avgY + (len*grad));
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
    
    extern Mat src;
    
    cout << "Lines: " << sortedLines.size() << "\nClusters: " << clusterCount << endl;
    for(int i = 0; i < clusterCount; i++){
        vector<Vec4f> lines;
        
        for(int j = 0; j < labels.rows; j++){
            //if( i == 0 ) cout << labels.at<int>(j);
            if(labels.at<int>(j) == i){
                lines.push_back( Vec4f(sortedLines[j][0], sortedLines[j][1], sortedLines[j][2], sortedLines[j][3]));
            }
        }
        
        bool isHori = false;
        if(i == 0 || i == 1){
            isHori = true;
        }
        Vec2f centroid = Vec2f( centroidX / sortedLines.size(), centroidY / sortedLines.size());

        Vec4f pushLine = fitBestLine(lines, centroid, isHori);
        cleanedLines.push_back( pushLine );
        
        line( clustered, Point(pushLine[0], pushLine[1]), Point(pushLine[2], pushLine[3]), Scalar(0,0,255), 14, 0);
    }

    cleanedLines = trimLines(cleanedLines);
    for(int i = 0; i < cleanedLines.size(); i++){
        //line( clustered, Point(cleanedLines[i][0], cleanedLines[i][1]), Point(cleanedLines[i][2], cleanedLines[i][3]), Scalar(0,128,255), 4, 0);
    }
    return cleanedLines;
}


/** Create a set of matches between a set of detected lines and a set of template lines */
vector<Match> findMatches( vector<Vec4f> detectedLines, vector<Vec4f> templateLines, int selectedLine){
    vector<Vec4f> lines = detectedLines;
    
    float templateHeight = lineLength( templateLines[0] );
    float detectedLinesHeight = lineLength(detectedLines[detectedLines.size() - 1]);
    float diff = templateHeight/detectedLinesHeight;

    diff = 1080/detectedLinesHeight;
    Mat scaling = Mat(3, 3, CV_32F, Scalar(0));
    scaling.at<float>(0,0) = diff;
    scaling.at<float>(1,1) = diff;
    scaling.at<float>(2,2) = 1;
    
    vector<Point2f> in;
    vector<Point2f> out;
    for( int i = 0; i < detectedLines.size(); i++){
        in.push_back( Point2f( detectedLines[i][0], detectedLines[i][1]) );
        in.push_back( Point2f( detectedLines[i][2], detectedLines[i][3]) );
    }
    
    perspectiveTransform(in, out, scaling);
    
    vector<Vec4f> scaledLines;
    for(int i = 0; i < out.size(); i += 2){
        scaledLines.push_back(Vec4f( out[i].x, out[i].y, out[i+1].x, out[i+1].y));
    }
    
    detectedLines = scaledLines;
    
    vector<Match> bestMatches;
    bestMatches.push_back( Match( templateLines[templateLines.size()-2], lines[0], 666) ); // top
    bestMatches.push_back( Match( templateLines[templateLines.size()-1], lines[1], 666) ); // bottom
    
    
    
    Vec4f horiz = Vec4f(0, 540+260, 1920, 540+260 );
    extern Vec2f clickCoords;
    
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
    
    return bestMatches;
}

