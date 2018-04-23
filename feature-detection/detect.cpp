#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"  //for SIFT
#include <string>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

Mat originalImage;
Mat displayImage;

int main(int argc, char* argv[])
{
    //check for supplied argument
	if (argc <= 1)
	{
		cout << "Please provide a filename of an image" << endl;
		return 0;
	}

	//read in the image
	originalImage = imread(argv[1]);
    //convert to grayscale
	Mat grayImage;
	cvtColor(originalImage, grayImage, CV_BGR2GRAY);
    
    //SIFT algorithm
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(grayImage, keypoints);

	// Add results to image and save
	drawKeypoints(originalImage, keypoints, displayImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //enter loop
	while (1 == 1)
	{
        //display image with keypoints
		imshow("Image Window", displayImage);
        
        //press q to quit
		char key = waitKey(33);
		if (key == 'q')
		{
			break;
		}
		if (key == ' ')
		{
		}
	}
}
