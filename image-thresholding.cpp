/*-------------------------------------------|
|	This program loads an image in			 |
|	grayscale and creates a histogram		 |
|	of its pixel intensities (0-255).		 |
|	Moving the mouse on the histogram		 |
|	dynamically displays the threshold		 |
|	value and updates the image	accordingly. | 
|	Exit the program by pressing any key.	 |
--------------------------------------------*/

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>

using namespace cv;
using namespace std;

//hold histogram and main image
MatND hist;
Mat histImg;
Mat nImg;

void mouseHandler(int event, int x, int y, int flags, void *param)
{
	Mat img1;	//To hold image
	Mat dst, src;	//Destination and source for threshold

	//Threshold and maxValue
	double thresh;
	double maxValue = 255;

	//for font and text attributes
	string text;
	int fontFace = CV_FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.5;
	int thickness = 1;
	Size textSize = getTextSize(text, fontFace,
		fontScale, thickness, 0);

	//put text at top left corner
	Point textOrg((img1.cols - textSize.width) / 24,
		(img1.rows + textSize.height));
	Point textOrg2((img1.cols - textSize.width) / 24,
		(img1.rows + textSize.height) *3);

	switch (event) {
		// mouse move 
	case CV_EVENT_MOUSEMOVE:
		img1 = histImg.clone();	//copy histogram image
		src = nImg.clone();	//copy image

		//draw circle around mouse center
		circle(img1,
			cvPoint(x, y),
			15,
			cvScalar(255, 255, 255, 0), 2, 8, 0);
		
		//to dynamically hold x,y coordinates
		Point p(x, y);	

		//output text for threshold
		putText(img1, format("Threshold: %d", (p.x)/4),
			textOrg, fontFace, fontScale, cvScalar(255, 255, 255, 0), thickness, 0);

		//threshold using Inverted Threshold To Zero
		threshold(src, dst, (p.x)/4, maxValue, THRESH_BINARY);

		int totalPixels = dst.rows * dst.cols;	//total pixels
		int blackPixels = totalPixels - countNonZero(dst);	//get number of black pixels

		//output text for area
		putText(img1, format("Area of bulbs: %d", blackPixels),
			textOrg2, fontFace, fontScale, cvScalar(255, 255, 255, 0), thickness, 0);

		//show updated histogram
		imshow("Grayscale Histogram", img1);

		//display updated image
		imshow("Thresholding", dst);

		
		break;
	}
}

/*-----------------------------------------*/

Mat imHist(Mat hist, float scaleX = 1, float scaleY = 1) {
	double maxVal = 0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int rows = 100; //default height size
	int cols = hist.rows; //get the width size from the histogram
	Mat histImg = Mat::zeros(rows*scaleX, cols*scaleY, CV_8UC3);
	
	//for each bin
	for (int i = 0; i<cols - 1; i++) {
		float histValue = hist.at<float>(i, 0);
		float nextValue = hist.at<float>(i + 1, 0);
		Point pt1 = Point(i*scaleX, rows*scaleY);
		Point pt2 = Point(i*scaleX + scaleX, rows*scaleY);
		Point pt3 = Point(i*scaleX + scaleX, (rows - nextValue*rows / maxVal)*scaleY);
		Point pt4 = Point(i*scaleX, (rows - nextValue*rows / maxVal)*scaleY);

		int numPts = 5;
		Point pts[] = { pt1, pt2, pt3, pt4, pt1 };

		fillConvexPoly(histImg, pts, numPts, Scalar(255, 255, 255));
	}
	return histImg;
}

int main(int argc, char** argv) {
	cout << endl << 
		"Move your mouse across the histogram to dynamically select a threshold value." <<
		endl << endl;
	cout << "Press any key to exit." << endl << endl;

	// check for supplied argument
	if (argc < 2) {
		cout << "Usage: load img <filename>\n" << endl;
		return 1;
	}

	//setup window and mouse callback
	namedWindow("Grayscale Histogram", CV_WINDOW_AUTOSIZE);
	setMouseCallback("Grayscale Histogram", mouseHandler, NULL);

	// load the image in grayscale
	nImg = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// always check
	if (nImg.data == NULL) {
		cout << "Cannot load file " << argv[1] << endl;
		return 1;
	}

	//Hold the histogram
	int nbins = 256; // lets hold 256 levels
	int hsize[] = { nbins }; // just one dimension
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	int chnls[] = { 0 };

	// create channel
	vector<Mat> colors;
	split(nImg, colors);

	// compute for grey channel
	calcHist(&colors[0], 1, chnls, Mat(), hist, 1, hsize, ranges);
	histImg = imHist(hist, 4, 4);
	imshow("Grayscale Histogram", histImg);	//display histogram image

	// show image
	imshow("Thresholding", nImg);

	// wait until user press a key
	waitKey(0);

	// no need to release the memory, Mat do it for you
	return 0;
}
