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

bool debugDisplay = true;
bool outputDisplay = true;
bool getPose = true;
bool getReprojection = true;

bool autoThresh = true;
int hRange = 10;
int threshMid = 0;

//Mat src;
Mat HSV;
Mat thresh;
Mat finuks;

String filename = "0002.png";
String imageSet = "Set1.txt";
int selectedLine = 0; // 0 = leftmost, 1 = center, 2 = rightmost
bool guiMode = true;

float baseline = -1;
int blThresh = 14;

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

int main( int argc, char** argv){
    clock_t start, end;
    double elapsed;
    start = clock();
    auto t1 = std::chrono::steady_clock::now();
    
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
               //imageSet = "realTest.txt";
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
        int count = 1;
        
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
            
            //cout << "Click Coords: " << clickCoords << "\tline: " << selectedLine << endl;
        }
    }
    //start = clock();
    
    //auto t1 = std::chrono::steady_clock::now();
    
    vector<Vec4f> rawLines = getLines( src, true, true );
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
    
    Mat labels = clusterLines(src.clone(), sortedLines);
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
            //line( finale, Point(out[i].x, out[i].y), Point(out[i+1].x, out[i+1].y), Scalar(0,0,255), 4, 0);
            //cout << Point(in[i].x, in[i].y) << "\t" << Point(in[i+1].x, in[i+1].y) << endl;
            //cout << Point(out[i].x, out[i].y) << "\t" << Point(out[i+1].x, out[i+1].y) << endl << endl << endl;
        }
        
        if(getPose){
            Mat imgPoints = (Mat_<float>(4,2) << out[0].x, out[0].y, out[2].x, out[2].y, out[1].x, out[1].y, out[3].x, out[3].y);
            
            Mat objPoints;
            if(selectedLine == 0){
                //objPoints = (Mat_<float>(4,3) << 8.23, -63.24, 0, -1.34, -63.24, 0, 8.52, -4.86, 0, -1, -4.86, 0);
                objPoints = (Mat_<float>(4,3) <<   0,1200,0,    110,1200,0,      0,0,0,    110,0,0   );
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
            drawFrameAxes(finale, cameraMatrix, distCoeffs, rVec, tVec, 50);
            
            vector<Mat> rot, tran, norm;
            //decomposeHomographyMat(homography, cameraMatrix, rot, tran, norm);
            //for(int i = 0; i < rot.size(); i++ ) cout << rot[i] << endl;
            //cout << tran << endl;
            
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
    
    
    //end = clock();
    //elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    //cout << "Clock time: " << elapsed << endl;
    
    auto t2 = std::chrono::steady_clock::now();
    //std::cout << "Chrono time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds\n";
    
    //cout << chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();// << endl;
    
    if(debugDisplay) imshow("Cleaned Lines", src);
    if(outputDisplay || debugDisplay){
        imshow("Finale", finale);
        waitKey();
    }
    //imshow("TEMPLATE", templateWarped);
    
}
