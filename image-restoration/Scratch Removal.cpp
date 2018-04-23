/*-------------------------------------------|
|   Removes scratches from a damaged image.  |
|	Move your mouse on the histograms to	 |
|	dynamically update the threshold		 |
|	value and display the fixed image.		 |
|	Exit the program by pressing any key.	 |
--------------------------------------------*/
#include <stdlib.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//globals
Mat imgOG, histImgR, histImgB, histImgG;

//create histogram
Mat imHist(Mat hist, float scaleX = 1, float scaleY = 1) {
	double maxVal = 0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int rows = 64; //default height size
	int cols = hist.rows; //get the width size from the histogram
	Mat histImg = Mat::zeros(rows*scaleX, cols*scaleY, CV_8UC3);
	//for each bin
	for (int i = 0; i<cols - 1; i++) {
		float histValue = hist.at<float>(i, 0);
		float nextValue = hist.at<float>(i + 1, 0);
		Point pt1 = Point(i*scaleX, rows*scaleY);
		Point pt2 = Point(i*scaleX + scaleX, rows*scaleY);
		Point pt3 = Point(i*scaleX + scaleX, (rows - nextValue * rows / maxVal)*scaleY);
		Point pt4 = Point(i*scaleX, (rows - nextValue * rows / maxVal)*scaleY);

		int numPts = 5;
		Point pts[] = { pt1, pt2, pt3, pt4, pt1 };

		fillConvexPoly(histImg, pts, numPts, Scalar(255, 255, 255));
	}
	return histImg;
}

//adds 2 images pixel by pixel
Mat addImg(Mat img1, Mat img2, Mat dst) {
	//get rows & columns
	int y = img1.rows;
	int x = img1.cols;

	for (int i = 0; i< y; i++)
		for (int j = 0; j < x; j++) {
			dst.at<Vec3b>(i, j)[0] = img1.at<Vec3b>(i, j)[0] + img2.at<Vec3b>(i, j)[0];
			dst.at<Vec3b>(i, j)[1] = img1.at<Vec3b>(i, j)[1] + img2.at<Vec3b>(i, j)[1];
			dst.at<Vec3b>(i, j)[2] = img1.at<Vec3b>(i, j)[2] + img2.at<Vec3b>(i, j)[2];
		}
	return dst;
}

//applies threshold to image
void detailThreshold(Mat details, Mat blur, Mat filtered_details, uchar thresh) {
	//get rows & columns
	int y = details.rows;
	int x = details.cols;

	for (int i = 0; i< y; i++)
		for (int j = 0; j < x; j++) {
			//get intensity value for each channel
			Vec3b intensity = details.at<Vec3b>(i, j);
			uchar B = intensity.val[0];
			uchar G = intensity.val[1];
			uchar R = intensity.val[2];

			//if channel exceeds threshold, filter out that pixel 
			//and remove from 'filtered details'
			if ((B > thresh) || (G > thresh) || (R > thresh)) {
				details.at<Vec3b>(i, j)[0] = blur.at<Vec3b>(i, j)[0];
				details.at<Vec3b>(i, j)[1] = blur.at<Vec3b>(i, j)[1];
				details.at<Vec3b>(i, j)[2] = blur.at<Vec3b>(i, j)[2];
				filtered_details.at<Vec3b>(i, j)[0] = 0;
				filtered_details.at<Vec3b>(i, j)[1] = 0;
				filtered_details.at<Vec3b>(i, j)[2] = 0;
			}
			else {//set 'details' to black if below threshold
				details.at<Vec3b>(i, j)[0] = 0;
				details.at<Vec3b>(i, j)[1] = 0;
				details.at<Vec3b>(i, j)[2] = 0;
			}
		}
	return;
}

//fix the scratches
void fixImg(uchar thresh) {
	Mat blur = imgOG.clone(),
		details = imgOG.clone(),
		filtered_details = imgOG.clone(),
		result = imgOG.clone();

	//apply median filter with size 5
	medianBlur(imgOG, blur, 5);

	//subtract median blur from original to get 'details'
	//absdiff function gave best results compared to subtract and get_diff
	//had issues with get_diff from example code - could not fully remove scratches
	absdiff(imgOG, blur, details);

	//apply threshold to 'details' to get 'filtered details'
	detailThreshold(details, blur, filtered_details, thresh);
	addImg(filtered_details, details, result);

	//show fixed image
	imshow("Fixed Image", result);

	return;
}

void mouseHandler(int event, int x, int y, int flags, void *param) {
	//hold each histogram for redisplay
	Mat HR = histImgR.clone(),
		HG = histImgG.clone(),
		HB = histImgB.clone();

	//threshold is value of x coord divided by width of histogram bin 
	uchar thresh = (x / 3);

	switch (event) {
		//mouse move
	case CV_EVENT_MOUSEMOVE:

		//string to display threshold value
		string text = "Threshold: " + to_string(x / 3);

		//font properties
		int fontFace = CV_FONT_HERSHEY_COMPLEX;
		double fontScale = .5;
		int thickness = 1;
		int baseline = 0;
		Size textSize = getTextSize(text, fontFace,
			fontScale, thickness, &baseline);
		baseline += thickness;

		//text location - upper left corner
		Point textOrg(5, textSize.height);

		//display text
		putText(HR, text, textOrg, fontFace, fontScale,
			Scalar(0, 0, 128), thickness, 8);
		putText(HG, text, textOrg, fontFace, fontScale,
			Scalar(0, 0, 128), thickness, 8);
		putText(HB, text, textOrg, fontFace, fontScale,
			Scalar(0, 0, 128), thickness, 8);

		//redisplay histograms
		imshow("Red", HR);
		imshow("Green", HG);
		imshow("Blue", HB);

		//fix the image
		fixImg(thresh);

		break;
	}
}

int main(int argc, char** argv) {
	// check for supplied argument
	if (argc < 2) {
		cout << "Usage: loading <filename>\n" << endl;
		return 1;
	}

	// load the image
	imgOG = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// always check
	if (imgOG.data == NULL) {
		cout << "Cannot load file " << argv[1] << endl;
		return 1;
	}

	MatND hist; //Hold the histograms

	int nbins = 256;
	int hsize[] = { nbins };
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	int chnls[] = { 0 };

	//split into BGR color channls
	vector<Mat> bgr;
	split(imgOG, bgr);

	//compute histogams for each color
	calcHist(&bgr[0], 1, chnls, Mat(), hist, 1, hsize, ranges);
	histImgB = imHist(hist, 3, 3);
	imshow("Blue", histImgB);

	calcHist(&bgr[1], 1, chnls, Mat(), hist, 1, hsize, ranges);
	histImgG = imHist(hist, 3, 3);
	imshow("Green", histImgG);

	calcHist(&bgr[2], 1, chnls, Mat(), hist, 1, hsize, ranges);
	histImgR = imHist(hist, 3, 3);
	imshow("Red", histImgR);

	//set mouse interaction for all histograms
	setMouseCallback("Blue", mouseHandler, NULL);
	setMouseCallback("Red", mouseHandler, NULL);
	setMouseCallback("Green", mouseHandler, NULL);

	//press any key to quit
	waitKey(0);
	return 0;
}
