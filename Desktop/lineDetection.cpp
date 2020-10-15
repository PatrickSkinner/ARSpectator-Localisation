//
//  lineDetection.cpp
//  lineDetector
//
//  Created by Patrick Skinner on 12/10/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#include "lineDetection.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "lineDetection.hpp"
#include "support.hpp"

using namespace std;
using namespace cv;

/** Return vector of all detected Hough lines in given image */
vector<Vec4f> getLines(Mat src, bool autoThresh = true, bool useMask = true)
{
    Mat HSV, thresh;
    
    
    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    cvtColor(src, HSV, COLOR_BGR2HSV);
    
    // Detect the field based on HSV Range Values
    if(autoThresh){
        /*
         Mat hOnly;
        inRange(HSV, Scalar(threshMid-hRange, 100, 0), Scalar(threshMid+hRange, 215, 215), hOnly);
        imshow("thresh af", hOnly);
        waitKey();
        */
        
        Mat hOnly;
        histThreshold(HSV, hOnly, 0);
        
        Mat masked;
        src.copyTo(masked, hOnly);
        imshow("mask", masked);

        Mat HSVmask;
        cvtColor(masked, HSVmask, COLOR_BGR2HSV);
        
        //histThreshold(HSVmask, thresh, 1);
        inRange(HSVmask, Scalar(0, 100, 0), Scalar(255, 215, 215), thresh);
        
    } else {
        //inRange(HSV, Scalar(20, 10, 50), Scalar(45, 255, 255), thresh);
        //inRange(HSV, Scalar(32, 124, 51), Scalar(46, 255, 191), thresh); // Stadium Test Pics
        //inRange(HSV, Scalar(27, 86, 2), Scalar(85, 255, 145), thresh); // broadcast
        //inRange(HSV, Scalar(31, 55, 70), Scalar(66, 255, 197), thresh); // renders
        inRange(HSV, Scalar(35, 100, 0), Scalar(55, 215, 215), thresh); // artificially lit stadium
        //inRange(HSV, Scalar(31, 55, 45), Scalar(68, 255, 206), thresh);
    }
    
    // opening and closing
    if(useMask){
        Mat opened;
        Mat closed;
        Mat kernel = Mat(3, 3, CV_8UC1, Scalar(1));
        morphologyEx(thresh, opened, MORPH_OPEN, kernel);
        morphologyEx(opened, closed, MORPH_ERODE, kernel);
        

        // Add one pixel white columns to both sides of the image to close contours
        //if(selectedLine == 0){
        int minRun = 5;
            int count = 0;
            for(int i = 0; i < closed.rows-1; i++){
                 if( closed.at<uchar>(i, closed.cols-1) != 255 ){
                    count++; // Count black pixels
                } else {
                    if(count <= minRun){ // If previous run of black pixels was 3 or less, leave them alone
                        count = 0;
                    } else if(count != 0){
                        //for(int j = 0; j <= count+1; j++) closed.at<uchar>((i+1)-j, closed.cols-1) = 255;
                        count = 0;
                    }
                }
            }
        
        count = 0;
        for(int i = 0; i < closed.rows; i++){
           //closed.at<uchar>(i, closed.cols-1) = 255;
            if( closed.at<Vec3b>(i, 0) == Vec3b(0,0,0) ){
                count++; // Count black pixels
            } else {
                if(count <= minRun){ // If previous run of black pixels was 3 or less, leave them alone
                    count = 0;
                } else if(count != 0){
                    //for(int j = 0; j <= count; j++) closed.at<uchar>(i-j, 1) = 255;
                    count = 0;
                }
            }
        }
        
        vector<vector<cv::Point> > contours;
        
        findContours(closed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        extern Mat boundary;
        boundary = Mat(1080, 1920, CV_8UC1, Scalar(0));
        
        Mat threshClosed;
        kernel = Mat(12, 12, CV_8U, Scalar(1));
        morphologyEx(thresh, threshClosed, MORPH_DILATE, kernel);
        vector<vector<cv::Point> > findBound;
        findContours(threshClosed, findBound, RETR_TREE, CHAIN_APPROX_SIMPLE);
        drawContours(boundary, findBound, getMaxAreaContourId(findBound), Scalar(255,0,255), -1);

        morphologyEx(boundary, boundary, MORPH_DILATE, kernel);
        
        Mat mask = Mat::ones( thresh.rows, thresh.cols, CV_8U);

        int j = 0;
        for( int i = 0; i < contours.size(); i++){
            if(contourArea(contours[i]) > 16000){
                drawContours(mask, contours, i, Scalar(255,255,255), -1);
                j++;
            }
        }
        
        /* Remove 1px columns from the sides
        for(int i = 0; i < mask.rows; i++){
            if(mask.at<uchar>(i, 1) > 0) mask.at<uchar>(i, 0) = 0;
            if(mask.at<uchar>(i, closed.cols-2) > 0) mask.at<uchar>(i, closed.cols-1) = 0;
        }*/

        thresh = mask;
    }

    
    Mat dst, invdst, cdst;
    GaussianBlur( thresh, invdst, Size( 5, 5 ), 0, 0 );
    Canny(invdst, dst, 50, 200, 5);
    
    //Remove 1px columns from the sides if mask used
    if(useMask){
        for(int i = 0; i < dst.rows; i++){
            dst.at<uchar>(i, 1) = 0;
            dst.at<uchar>(i, dst.cols-1) = 0;
            dst.at<uchar>(i, dst.cols-2) = 0;
        }
    }
    
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    vector<Vec4f> lines;
    HoughLinesP(dst, lines, 2, CV_PI/360, 100, 145, 45 );
    
    for(int i = 0; i < lines.size(); i++ ){
        Scalar colour = Scalar( ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ), ( rand() % (int) ( 255 + 1 ) ));
         line( cdst, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), colour, 5, 0);
    }
    
    return lines;
}


/** Automatic thresholding of input image based on Hue/Saturation values*/
int histThreshold(Mat inputImg, Mat& outputImg, int channel){
    Mat HSVmat;
    int hRange = 10;
    
    // Split image into H, S and V components
    HSVmat = inputImg.clone();
    vector<Mat> hsv_planes;
    split( HSVmat, hsv_planes );

    /// Establish the number of bins
    int histSize = 180;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 180 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &hsv_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &hsv_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &hsv_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 180; int hist_h = 255;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                         Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                         Scalar( 0, 255, 0), 2, 8, 0  );
        /*line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                         Scalar( 0, 0, 255), 2, 8, 0  );
         */
    }
    
    float max = 0.0;
    int maxIndex = -1;
    int threshMid = 1;
    
    if(channel == 0){
        for(int i = 0; i < 90; i++){
            if(maxIndex == -1 || b_hist.at<float>(i) > max){
                max = b_hist.at<float>(i);
                maxIndex = i;
            }
        }
        cout << "max index: " << maxIndex << " \t - \t " << max << endl;
        threshMid = maxIndex;
        
        inRange(inputImg, Scalar(threshMid-hRange, 0, 0), Scalar(threshMid+hRange, 255, 215), outputImg);
        //inRange(HSV, Scalar(threshMid-hRange, 100, 0), Scalar(threshMid+hRange, 215, 215), outputImg);
        
    } else if(channel == 1){
        for(int i = 1; i < 180; i++){
            if(maxIndex == -1 || g_hist.at<float>(i) > max){
                max = g_hist.at<float>(i);
                maxIndex = i;
            }
        }
        cout << "max index: " << maxIndex*(255.0/180.0) << " \t - \t " << max << endl;
        inRange(inputImg, Scalar(0, maxIndex-55, 0), Scalar(255, maxIndex+50, 215), outputImg);
    }
    
    /// Display
    line(histImage, Point(threshMid-hRange,0), Point(threshMid-hRange,255), Scalar(255,255,255), 1);
    line(histImage, Point(threshMid+hRange,0), Point(threshMid+hRange,255), Scalar(255,255,255), 1);
    //namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
    
    //imshow("Demo", outputImg );
    //imshow("calcHist Demo", histImage );
    return 0;
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
