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

int main(int argc, char* argv[])
{
    //check for two supplied arguments
	if (argc < 3)
	{
		//example arguments: image1.jpg image2.jpg
		// first argument: example image
		// second argument: test image
		cout << "Please provide a filename of an example image and a testing image" << endl;
		return 0;
	}

	Mat displayImage;

	//read in the image
	Mat image1, image2;
	image1 = imread(argv[1]);
	image2 = imread(argv[2]);
    
    //convert to grayscale
	Mat grayImage1, grayImage2;
	cvtColor(image1, grayImage1, CV_BGR2GRAY);
	cvtColor(image2, grayImage2, CV_BGR2GRAY);

    //detect keypoints
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	detector.detect(grayImage1, keypoints1);
	detector.detect(grayImage2, keypoints2);

    //compute descriptors
	// http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_extractors.html
	Mat descriptors1;
	Mat descriptors2;
	detector.compute(image1, keypoints1, descriptors1);
	detector.compute(image2, keypoints2, descriptors2);

    //find matches
	//http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html#bfmatcher
	BFMatcher matchmaker;
	vector<DMatch> matches;
	matchmaker.match(descriptors1, descriptors2, matches);

    //threshold values for trackbar
	int maxVal = 1000;
	int threshold = 150;
	int prevThreshold = 0;

    //create window and trackbar
	namedWindow("Image Window", 1);
	createTrackbar("Match Thresh", "Image Window", &threshold, maxVal);

    //enter infinite loop
	while (1 == 1)
	{
        //calculate new matches based on trackbar threshold value
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
            //draw final matches
			//http://docs.opencv.org/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html?
			drawMatches(image1, keypoints1, image2, keypoints2, finalMatches, displayImage);
			prevThreshold = threshold;
		}
        
        //display match results
		imshow("Image Window", displayImage);
        
        //press q to quit, or space bar to write the image file
		char key = waitKey(33);
		if (key == 'q')
		{
			break;
		}
		if (key == ' ')
		{
			imwrite("output.png", displayImage);
		}
	}
}
