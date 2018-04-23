#include <stdio.h>  
#include "opencv2/opencv.hpp" 
#include "opencv2/stitching/stitcher.hpp"   //for stitcher
#include <stdio.h>
#include <iostream> 

using namespace cv;
using namespace std;


int main()
{
        //read images to stitch together
		Mat image1 = imread("S1.jpg", IMREAD_COLOR);
		Mat image2 = imread("S2.jpg", IMREAD_COLOR);
		Mat image3 = imread("S3.jpg", IMREAD_COLOR);

        //check for load error
		if (!image1.data || !image2.data || !image3.data)
		{
			std::cout << " --(!) Error reading images " << std::endl; return -1;
		}

        // store images in vector
		vector< Mat > vImg;
        // for final result
		Mat rImg;

        //push images to vector
		vImg.push_back(image1);
		vImg.push_back(image2);
		vImg.push_back(image3);

        //create stitcher
		Stitcher stitcher = Stitcher::createDefault();
		 

		unsigned long AAtime = 0, BBtime = 0; //check processing time
		AAtime = getTickCount(); //check processing time

        //stitch images together
		Stitcher::Status status = stitcher.stitch(vImg, rImg);

		BBtime = getTickCount(); //check processing time 
		printf("%.2lf sec \n", (BBtime - AAtime) / getTickFrequency()); //check processing time

        //if successful, display result and save image in current directory
		if (Stitcher::OK == status)
		{
			imshow("Result", rImg);
			cv::imwrite("./Result/stitching_result.jpg", rImg);
		}
		else
			printf("Stitching fail.");  // if stitching unsucessful

		 waitKey(0);    //press any key to exit

}
