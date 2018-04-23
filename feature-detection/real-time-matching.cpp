#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <string>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

void onTrackbar(int value, void* data) {}

int main(int argc, char* argv[]) {
	
	//check for supplied argument
	if (argc < 2) {
		cout << "Please provide the filename of a testing image" << endl;
		return 0;
	}

	// open the default camera, use something different from 0 otherwise;
	VideoCapture cap;
	if (!cap.open(0))
		return 0;

	Mat displayImage;	//to show final result

	//read in the image
	Mat image1;
	image1 = imread(argv[1]);

	//convert to grayscale
	Mat grayImage1;
	cvtColor(image1, grayImage1, CV_BGR2GRAY);

	//SIFT Algorithm detector
	SiftFeatureDetector detector;

	// keypoints
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;

	// descriptors
	Mat descriptors1;
	Mat descriptors2;

	//Brute force descriptor matcher
	BFMatcher matchmaker;
	vector<DMatch> matches;

	//values for trackbar
	int maxVal = 1000;
	int threshold = 10;
	int prevThreshold = 0;

	//create window & trackbar
	namedWindow("Image Window", 1);
	createTrackbar("Match Thresh", "Image Window", &threshold, maxVal);

	//enter infinite loop
	while (1 == 1)
	{
		Mat frame;	//video frame
		cap >> frame;	//capture current frame
		if (frame.empty()) break; // end of video stream

		//detect keypoints and compute descriptors
		detector.detect(grayImage1, keypoints1);
		detector.detect(frame, keypoints2);
		detector.compute(image1, keypoints1, descriptors1);
		detector.compute(frame, keypoints2, descriptors2);

		// match descriptors
		matchmaker.match(descriptors1, descriptors2, matches);

		//update matches according to trackbar threshold	
		if (threshold != prevThreshold)
		{
			vector<DMatch> finalMatches;
			for (int i = 0; i<matches.size(); i++)
			{
				if (matches[i].distance < threshold)
				{
					finalMatches.push_back(matches[i]);
				}
			}

			//Draw the found matches of keypoints
			drawMatches(image1, keypoints1, frame, keypoints2, finalMatches, displayImage);

			//update threshold
			prevThreshold = threshold;
		}
		
		//display final result
		imshow("Image Window", displayImage);

		char key = waitKey(33);
		if (key == 'q') 
		{
			cap.release();	//release memory
			break;
		}
		if (key == ' ') 
		{
			imwrite("outputRT.png", displayImage);	//write image
		}
	}
}
