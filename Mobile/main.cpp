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
#include "support.hpp"
#include "lineDetection.hpp"
#include "lineClustering.hpp"

using namespace cv;
using namespace std;

bool useRectification = true;
bool manualSelection = false;
bool useMask = true;
bool mirrorInput = false;
bool rotateInput = false;

bool debugDisplay = false;
bool outputDisplay = false;
bool getPose = true;
bool getReprojection = true;
bool getTiming = false;

bool showThresh = false;

bool autoThresh = false;
int hRange = 10;
int threshMid = 0;

//Mat src;
Mat HSV;
Mat thresh;

String filename = "0002.png";
String imageSet = "Set1.txt";
int selectedLine = 0; // 0 = leftmost, 1 = center, 2 = rightmost
bool guiMode = true;

float baseline = -1;
int blThresh = 14;

std::vector<Point2f> out;

Scalar lower = Scalar(35, 30, 0);
Scalar upper = Scalar(80, 219, 255);
//HoughLinesP(dst, lines, 2, CV_PI/360, 150, 275, 45 );
int rho = 2;
int vote = 150;
int minLen = 275;
int focalLength = 1410;

Mat OG;
Mat clustered;
Mat mask;
Mat showCluster;



clock_t start, end;
double elapsed;


Mat boundary = Mat(1080, 1920, CV_8UC1, Scalar(0));
Mat src;
Vec2f clickCoords;

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

extern "C" void sendClick(int x, int y){
    clickCoords = Point2f(x, y);
    //cout << "clickedy clickkkkkk" << endl;
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
    
    //cout << "imagePoints = " << endl << " " << imagePoints << endl << endl;
    //cout << "objectPoints = " << endl << " " << objectPoints << endl << endl;
    
    cameraMatrix = (Mat_<float>(3, 3) << focalLength, 0, 1920/2, 0, 1410, 1080/2, 0, 0, 1);
    //cameraMatrix = (Mat_<float>(3, 3) << max(width, height), 0, width / 2, 0, max(width, height), height / 2, 0, 0, 1);
    
    solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rotation, translation);
    
    Mat rotationMat;
    Rodrigues(rotation, rotationMat);
    //cout << "rotationMat = " << endl << " " << rotationMat << endl << endl;
    
    *rv = static_cast<float*>(malloc(rotationMat.total() * rotationMat.elemSize()));
    memcpy(*rv, rotationMat.data, rotationMat.total() * rotationMat.elemSize());
    
    *tv = static_cast<float*>(malloc(translation.total() * translation.elemSize()));
    memcpy(*tv, translation.data, translation.total() * translation.elemSize());
}

extern "C" int sendImage(uint8_t *&image, int& width, int& height, int& lineSelect)
{
    start = clock();
    
    src = Mat(height, width, CV_8UC3, *&image);
    if(!src.empty()){
        //cout << clickCoords << " = click\n";
        flip(src, src, +1);
        clustered = src.clone();
        Mat finale = src.clone();
        selectedLine = lineSelect;
        
        vector<Vec4f> templateLines;
         
         if(selectedLine == 0){
             templateLines = {Vec4f(0,0,0,2800), Vec4f(440,0,440,2800), Vec4f(1400,0,1400,2800), Vec4f(0,0,1400,0), Vec4f(0,2800,1400,2800)};
         } else if(selectedLine == 1){
            templateLines = {Vec4f(0,0,0,2800), Vec4f(440,0,440,2800), Vec4f(1400,0,1400,2800), Vec4f(0,0,1400,0), Vec4f(0,2800,1400,2800)};
         } else if(selectedLine == 2){
             templateLines = {Vec4f(0,0,0,2800), Vec4f(960,0,960,2800), Vec4f(1400,0,1400,2800), Vec4f(0,0,1400,0), Vec4f(0,2800,1400,2800)};
         }
        
        vector<Vec4f> rawLines = getLines( src, true, true );
        for(int i = 0; i < rawLines.size(); i++){
            line( clustered, Point(rawLines[i][0], rawLines[i][1]), Point(rawLines[i][2], rawLines[i][3]), Scalar(0,255,0), 4, 0);
        }
        
         vector<Vec4f> sortedLines = rawLines;
         sort(sortedLines.begin(), sortedLines.end(), compareVec); // Sort lines by gradient
         
         Mat labels = clusterLines(src, sortedLines);
         vector<Vec4f> lines = cleanLines(sortedLines, labels);
         
        for(int i = 0; i < lines.size(); i++){
            line( clustered, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,128,255), 4, 0);
        }
        
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
         //vector<Point2f> out;
         for( int i = 0; i < rectifiedLines.size(); i++){
             in.push_back( Point2f( rectifiedLines[i][0], rectifiedLines[i][1]) );
             in.push_back( Point2f( rectifiedLines[i][2], rectifiedLines[i][3]) );
         }
         
         perspectiveTransform(in, out, scaling);
         
         //cout << "in: " << in.size() << ",    out: " << out.size() << endl;
         vector<Vec4f> scaledLines;
         for(int i = 0; i < out.size(); i += 2){
             scaledLines.push_back(Vec4f( out[i].x, out[i].y, out[i+1].x, out[i+1].y));
         }
         
         rectifiedLines = scaledLines;
         
         vector<Match> bestMatches;
         bestMatches.push_back( Match( templateLines[templateLines.size()-2], lines[0], 666) ); // top
         bestMatches.push_back( Match( templateLines[templateLines.size()-1], lines[1], 666) ); // bottom
         
         
         
        Vec4f horiz = Vec4f(0, 540+260, 1920, 540+260 );
        extern Vec4f divider;
        //line(src, Point(0,540+260) , Point(1920,540+260), Scalar(255,255,255), 8);

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
            
            Vec3f firstMP = intersect( lines[closest], divider );
            Vec3f secondMP;
            float firstDist; // Distance between first matched line and the second
            
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
                                if(i != 2){
                                    minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                                    index = j;
                                } else {
                                    Vec3f thirdMP = intersect(lines[j], divider);
                                    float secondDist = lineLength( Vec4f( secondMP[0], secondMP[1], thirdMP[0], thirdMP[1]));
                                    if( secondDist > firstDist*1.5 ){
                                        minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                                        index = j;
                                    }
                                }
                            }
                        }
                    }
                }
                bestMatches.push_back( Match( templateLines[i], lines[index], 666) );
                if(i == 1){
                    secondMP = intersect(lines[index], divider);
                    firstDist = lineLength( Vec4f( firstMP[0], firstMP[1], secondMP[0], secondMP[1]));
                }
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
            
            Vec3f firstMP = intersect( lines[closest], divider );
            Vec3f secondMP;
            float firstDist; // Distance between first matched line and the second
            
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
                                if(i != 2){
                                    minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                                    index = j;
                                } else {
                                    Vec3f thirdMP = intersect(lines[j], divider);
                                    float secondDist = lineLength( Vec4f( secondMP[0], secondMP[1], thirdMP[0], thirdMP[1]));
                                    if( secondDist > firstDist*1.5 ){
                                        minDist = abs(intersect(lines[j], horiz)[0] - intersect(bestMatches[bestMatches.size()-1].l2, horiz)[0]);
                                        index = j;
                                    }
                                }
                            }
                        }
                    }
                }
                bestMatches.push_back( Match( templateLines[i], lines[index], 666) );
                if(i == 1){
                    secondMP = intersect(lines[index], divider);
                    firstDist = lineLength( Vec4f( firstMP[0], firstMP[1], secondMP[0], secondMP[1]));
                }
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
             line( clustered, Point(bestMatches[i].l2[0], bestMatches[i].l2[1]), Point(bestMatches[i].l2[2], bestMatches[i].l2[3]), Scalar(0,0,255), 3, 0);
             Vec2f tempMid = getCenter(bestMatches[i].l1);
             Vec2f matchMid = getCenter(bestMatches[i].l2);
             line( warp, Point(tempMid[0], tempMid[1]), Point(matchMid[0], matchMid[1]), Scalar(0,255,100), 2, 0);
         }
         
         //imshow("warp", warp);
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
         
         
        // Mat homography = (Mat_<float>(3,3) << 0.00069664459, -0.00089357677, 0.86346799, -9.946608e-05, -0.00023343519, 0.50440145, -2.3223721e-07, -5.5850325e-07, 0.00082679675);
         
         //cout << homography << endl << endl;
         
         //////////////////////////////////////////////////////////
         ////////////// Display output for debugging //////////////
         //////////////////////////////////////////////////////////
         
         
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
             //std::vector<Point2f> out;
             
             //cout << "\n\nHomography : \n" << homography << "\n\n";
             
             //vector<Vec4f> warpLines {Vec4f(0,0,0,2800), Vec4f(440,0,440,2800), Vec4f(1400,0,1400,2800), Vec4f(0,0,4400,0), Vec4f(0,2800,4400,2800)};
             
             for( int i = 0; i < templateLines.size(); i++){
                 in.push_back( Point2f( templateLines[i][0], templateLines[i][1]) );
                 in.push_back( Point2f( templateLines[i][2], templateLines[i][3]) );
                 //cout << Point(in[i].x, in[i].y) << "\t" << Point(in[i+1].x, in[i+1].y) << endl;
             }
             
         //cout << "\n\n\n\n";
             
             perspectiveTransform( in , out, homography);
             
             
             for( int i = 0; i < out.size(); i += 2){
                 line( finale, Point(in[i].x, in[i].y), Point(in[i+1].x, in[i+1].y), Scalar(0,255,255), 2, 0);
                 line( clustered, Point(out[i].x, out[i].y), Point(out[i+1].x, out[i+1].y), Scalar(255,0,255), 5, 0);
                 //cout << Point(in[i].x, in[i].y) << "\t" << Point(in[i+1].x, in[i+1].y) << endl;
                 //cout << Point(out[i].x, out[i].y) << "\t" << Point(out[i+1].x, out[i+1].y) << endl << endl << endl;
             }
                
             //end = clock();
             //elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
             //cout << "time taken:  " <<  elapsed << endl;
             
                return 1;
            }
    }
    
    return 0;
}






// Pass an image mat back to Unity to be displayed as a texture
extern "C" void GetRawImageBytes(unsigned char*& data, int& width, int& height)
{
    
    if(showThresh){
        //Resize Mat to match the array passed to it from C#
        if(!thresh.empty()){
            //cout << "Thresh not empty\n";
            cv::Mat resizedMat(height, width, thresh.type());
            cv::resize(thresh, resizedMat, resizedMat.size(), cv::INTER_CUBIC);
            
            //Convert from RGB to ARGB
            cv::Mat argb_img;
            cv::cvtColor(resizedMat, argb_img, COLOR_GRAY2RGBA);
            std::memcpy(data, argb_img.data, argb_img.total() * argb_img.elemSize());
        } else {
            cout << "Thresh Empty\n";
        }
        
    }else {
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
}

//Update threshold range based on input from debug menu in unity
extern "C" void updateThreshold(int hL, int hU, int sL, int sU, int vL, int vU){
    lower = Scalar(hL, sL, vL);
    upper = Scalar(hU, sU, vU);
}

//Update threshold range based on input from debug menu in unity
extern "C" void updateHough(int inRho, int inVote, int inMinLen, int focalLen, bool toggle){
    rho = inRho;
    vote = inVote;
    minLen = inMinLen;
    focalLength = focalLen;
    showThresh = toggle;
}
