/*--------------------------------------|
|	Program Description:				|
|	This program dynamically displays	|
|	the pixel coordinates and RGB value	|
|	at the current location of the		|
|	mouse pointer on an image, which	|
|	is surrounded by a red circle.		|
--------------------------------------*/


#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//to hold image
Mat nImg;

void mouseHandler(int event, int x, int y, int flags, void *param)
{
	Mat img1;	//to hold image
	Mat3b img_bgr;	//to hold vector of image color (blue, green, red)

	//for font and text attributes
	string text, text2;
	int fontFace = CV_FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.5;
	int thickness = 1;
	Size textSize = getTextSize(text, fontFace,
		fontScale, thickness, 0);
	Size textSize2 = getTextSize(text2, fontFace,
		fontScale, thickness, 0);

	//put text at top left corner
	Point textOrg((img1.cols - textSize.width) / 24,
		(img1.rows + textSize.height));
	Point textOrg2((img1.cols - textSize2.width) / 24,
		(img1.rows + textSize2.height) * 3);

	switch (event) {
		// mouse move 
	case CV_EVENT_MOUSEMOVE:
		// draw a red circle around mouse center
		img1 = nImg.clone();	//copy image
		circle(img1,
			cvPoint(x, y),
			15,
			cvScalar(0, 0, 255, 0), 2, 8, 0);
		img_bgr = nImg.clone();	//copy image
		Vec3b px = img_bgr(y, x);	//get BGR value at pixel position
		Point p(x, y);	//to dynamically hold x,y coordinates

		//output text for coordinates and rgb values
		putText(img1, format("Coordinates: (%d, %d)", p.x, p.y), 
			textOrg, fontFace, fontScale, cvScalar(0, 0, 255, 0), thickness, 0);
		putText(img1, format("RGB Value: (%d, %d, %d)", px[2], px[1], px[0]),
			textOrg2, fontFace, fontScale, cvScalar(0, 0, 255, 0), thickness, 0);
		imshow("Coordinates and RGB", img1);
		break;
	}
}

/*-----------------------------------------*/

int main(int argc, char** argv)
{
	//create window, mouse callback, and open image
	namedWindow("CCoordinates and RGB", CV_WINDOW_AUTOSIZE);
	setMouseCallback("Coordinates and RGB", mouseHandler, NULL);
	nImg = imread("ColorWheel.jpg", CV_LOAD_IMAGE_COLOR); 
	imshow("Coordinates and RGB", nImg);

	//create a copy
	Mat img1 = nImg.clone();

	//display original image
	namedWindow("Coordinates and RGB", CV_WINDOW_AUTOSIZE);

	int key;
	while (1) {
		//wait for keyboard input
		key = waitKey(0);
		// 'q' pressed, quit the program
		if (key == 'q') break;
	}

	//free memory
	destroyWindow("Coordinates and RGB");
	nImg.release();
	img1.release();

	return 0;
}
