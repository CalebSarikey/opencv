// Caleb Sarikey & Tim Densmore
// CSIT 463 Final Project
// Training-Data.cpp - Computes and writes training data to text files

//#include "stdafx.h"
#include <iostream>	//cin, cout
#include "opencv2/core/core.hpp"	//opencv libraries
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>	// for filenames
#include <vector>	// for pixel vectors
#include<iomanip>	// for decimal places
#include<fstream>	// write & read files

using namespace cv;
using namespace std;

//this will be used to store the slopes for every iteration (image)
//results will be appended to a text file, and then arraySize will be set back to zero
//repeat for every image
const int arrayMaxSize = 1000;
int arraySize = 0;	//to hold size of array
float variance[arrayMaxSize];	//keeps variance
float slopes[arrayMaxSize];//keep all the slopes
float slopeVariance[arrayMaxSize]; //keeps all the slope variances

/*------------------------------------------------------------------------------------------*/

//extracts just the red pixels from the image
//returns total number of red pixels
int extractRed(Mat img) {
	//get rows & columns
	int r = img.rows;
	int c = img.cols;

	int countRed = 0;	//count total # red pixels

	for (int i = 0; i< r; i++)
		for (int j = 0; j < c; j++) {
			//get intensity value for each channel
			Vec3b intensity = img.at<Vec3b>(i, j);
			uchar B = intensity.val[0];
			uchar G = intensity.val[1];
			uchar R = intensity.val[2];

			//filter pixels
			if (R > 200 && G < 150) {
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 255;
				countRed++;
			}
			else {//set to black if below threshold
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 0;
			}
		}
	return countRed;
}

/*------------------------------------------------------------------------------------------*/

//to hold coordinate values
struct coordinate
{
	int x;
	int y;
};

struct distances //edited to hold the slope and variance
{
	int x1, x2, y1, y2;
	int distance;
	float variance;
	float slope;
};

/*------------------------------ Algorithm for best fit line ---------------------------- */
void newGetLine(Mat L, int windowWidth, int windowHeight, int xStart)
{
	//window width & height
	int r = windowHeight;
	int c = windowWidth;
	int meanShiftHeight = 13;	// meanshift window of height 13
	int meanWidth = windowWidth;	//meanshift window width
	int meanCenter = windowHeight - 7;	// meanshift window center
	int meanBottom = meanCenter - 6;	// meanshift window bottom
	int meanTop = meanCenter + 6;	// meanshift window top

	//if window exceeds the bounds, we are done
	if (xStart + c > L.cols)
		return;

	//draw the window borders
	Point w1(xStart, 0);
	Point w2(xStart, 274);
	//line(L, w1, w2, cvScalar(0, 255, 0, 0), 1);

	// ----------------------- START MEAN SHIFT ----------------------- //
	// mean shift until:
	//	 1.  it has iterated 30 times
	//	 2.  all windows have been exhausted (check bounds)
	//	 3.  same mean is calculated repeatedly

	int meanCount = 0;	//to count iterations
	bool done = false;	//make true when finished
	int previousMean = 0;	//to hold mean from previous window
	vector<coordinate> redPixelsMean;	//holds red pixels within meanshift window
	coordinate tempMean;

	//cycle through from the bottom, get all red pixels in meanshift window
	while (!done) {
		int mean = 0;//initialize mean to 0

		// Calculate mean within window, then shift window based on mean 
		for (int i = xStart; i < xStart + meanWidth; i++) {
			for (int j = meanTop; j > meanBottom; j--) {
				Vec3b intensity = L.at<Vec3b>(j, i);
				uchar R = intensity.val[2];
				if (R > 0)
				{
					mean += j;
					tempMean.x = i;
					tempMean.y = j;
					redPixelsMean.push_back(tempMean);
				}
			}
		}

		//if there are red pixels in the mean window, 
		//shift the window center to the mean location
		if (redPixelsMean.size() > 100)
		{
			mean /= redPixelsMean.size();	//calculate mean
			if (mean == previousMean) {	//if mean does not change, we are done
				done = true;
				break;
			}
			previousMean = mean;
			meanCenter = mean;	//shift to new mean
			if ((meanCenter - 6) < 0 || (meanCenter + 6) > 979) { //check bounds first
				done = true;
				break;
			}
			meanBottom = meanCenter - 6;	//move window bottom
			meanTop = meanCenter + 6;	//move window top
			meanCount++;	//count an iteration
		}
		else
		{
			if ((meanBottom - 5) < 0) {	//check bounds first
				done = true;
				break;
			}
			else {
				//there are little to no red pixels, shift window up 
				meanTop -= 5;
				meanBottom -= 5;
				meanCenter -= 5;
			}
		}

		//if iterated 30 times or if window exceeds bounds, then we are done
		if (meanCount > 30 || meanBottom < 0 || meanTop > 979)
			done = true;
	}
	// ----------------------- END MEAN SHIFT ----------------------- //

	//if there are little to no red pixels, we don't need to map a line
	//in this case, the variance will be set to zero
	if (redPixelsMean.size() < 100)
	{
		slopeVariance[arraySize] = 0; //no variance in slope, since there is no line mapped
		variance[arraySize] = 0;
		arraySize++;	//increment arraySize and move to next element
		return newGetLine(L, 49, 275, xStart + 24);	//move to next window
	}

	//stores line info
	float deltaY = 0;
	float deltaX = 0;
	float slope;
	int yIntercept = 0;
	int xPlot1;
	int xPlot2;
	int yPlot1;
	int yPlot2;

	//to hold distances from line
	vector<distances> lineDistances;
	distances lowestDistance;
	float var = 0;

	//take a sample of 40% of the red pixel points to map lines
	for (int i = 0; i < redPixelsMean.size() / 2; i += (redPixelsMean.size() / 40))
	{
		for (int j = redPixelsMean.size() - 1; j > redPixelsMean.size() / 2; j -= (redPixelsMean.size() / 40))
		{
			deltaY = (redPixelsMean[j].y - redPixelsMean[i].y);
			deltaX = (redPixelsMean[j].x - redPixelsMean[i].x);

			if (deltaX == 0) //if vertical line, set slope to a really high number
				slope = 10000;
			else
				slope = deltaY / deltaX;

			//get y intercept
			yIntercept = redPixelsMean[j].y - slope * redPixelsMean[j].x;

			if (slope == 0)//if the slope is zero, horizontal line
			{
				xPlot1 = xStart;
				yPlot1 = yIntercept;

				xPlot2 = xStart + c - 1;
				yPlot2 = yIntercept;
			}
			else if (slope >= 10000)//vertical line
			{
				xPlot1 = redPixelsMean[j].x; //plot at x
				yPlot1 = r - 1; //plot at bound of box

				xPlot2 = redPixelsMean[j].x;//plot at x
				yPlot2 = 0;//plot at bound of box
			}
			else //if its not a horizontal or vertical line
			{	// simply use the points to plot

				//we can further update this to extend lines to 
				//window borders for a better visual representation,
				//but it doesn't matter as far as calculating distances
				xPlot1 = redPixelsMean[j].x;
				yPlot1 = redPixelsMean[j].y;

				xPlot2 = redPixelsMean[i].x;
				yPlot2 = redPixelsMean[i].y;
			}
			//points for test line
			Point p1(xPlot1, yPlot1);
			Point p2(xPlot2, yPlot2);
			//line(L, p1, p2, cvScalar(0, 255, 0, 0), 1);
			//^^un-commenting the above line of code will show all lines tested

			//get distance from red pixel to test line
			int totalDistance = 0;	//init to 0
			var = 0; //reset variance to 0
			for (int i = 0; i < redPixelsMean.size(); i++)
			{
				int lineY = slope * redPixelsMean[i].x + yIntercept; //get the y coordinate based on y = mx+b
				int distance = redPixelsMean[i].y - lineY;
				if (distance < 0) //get absolute distance value, not negative
					distance *= -1;

				//get variance
				//summation of (distance of point from mean^2 / number of total points - 1) for all points
				var += (distance * distance);
				totalDistance += distance;
			}
			var /= (redPixelsMean.size() - 1); //divide by total number of points - 1
			//store line coordinates
			distances d;
			d.x1 = xPlot1;
			d.y1 = yPlot1;
			d.x2 = xPlot2;
			d.y2 = yPlot2;
			d.slope = slope; //store the slope
			d.variance = var; //store the variance
			d.distance = totalDistance;//store distances

			lineDistances.push_back(d);//put total distances in a vector
		}
	}

	//find the lowest distance, then map a blue line onto it
	distances d = lineDistances.back();
	lowestDistance = d;
	lineDistances.pop_back();
	while (lineDistances.size()>0)
	{
		d = lineDistances.back();
		if (d.distance < lowestDistance.distance)
			lowestDistance = d;
		lineDistances.pop_back();
	}

	//set current element = the best line variance
	slopeVariance[arraySize] = lowestDistance.slope;
	variance[arraySize] = lowestDistance.variance;
	arraySize++;	//increment array size and move to next element

	//points for best fit line
	Point p3(lowestDistance.x1, lowestDistance.y1);
	Point p4(lowestDistance.x2, lowestDistance.y2);

	//line(L, p3, p4, cvScalar(255, 0, 0, 0), 1);	//draws best fit line
	//imshow("getLine", L); //display image with line drawn
	return newGetLine(L, 49, 275, xStart + 24);	//move to next window
}

/*------------------------------------------------------------------------------------------*/

int main() {
	cout << endl << "Press any key to exit." << endl << endl;
	string filename[2];
	string outfiles[2];

	//training data 135 normal images: {N2,N3,N4, N6,N7,N8, N10,N11,N12, N14,N15,N16, ..., N178, N179, N180}
	// and 77 AMD images: {A2,A3,A4, A6,A7,A8, A10,A11,A12, A14,A15,A16, ..., A102,A103}
	int N;
	int third;
	for (int k = 0; k <= 1; k++) {
		third = 0;
		if (k == 0)
			N = 180;
		else
			N = 103;
		for (int i = 2; i <= N; i++)
		{
			if (third == 3) {
				//reset
				third = 0;
			} else {
				string ext = ".png";	//file extension
				filename[0] = "C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\NORMAL\\N"; //filename prefix
				filename[1] = "C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\AMD\\A";	//filename prefix
				string num = to_string(i);	//convert i to a string
				filename[k].append(num);	//append num to file name
				filename[k].append(ext);	//append extension to complete full file path

				// load the image in color
				Mat OCT = imread(filename[k], CV_LOAD_IMAGE_COLOR);

				//always check
				if (OCT.data == NULL) {
					cout << "Cannot load file " << endl;
					return 1;
					}

				// show original image
				//imshow("OCT Scan", OCT);

				// get red pixels
				Mat R = OCT.clone();
				int countRed = extractRed(R);
				//imshow("Red Pixels", R);

				//if less than 3,500 total red pixels, do not apply median blur
				if (countRed < 3500) {
					// get best fit lines from red pixel image
					Mat L = R.clone();
					newGetLine(L, 49, 275, 0);
					//waitKey(0);	//wait for user to press a key
					//cvDestroyAllWindows(); //resets all imshow() windows for next image
				}
				else {
					// apply median blur if substantial red pixels
					Mat blurred;
					medianBlur(R, blurred, 3);
					//imshow("Blurred Red Pixels", blurred);

					// get best fit lines from blurred image
					Mat L = blurred.clone();
					newGetLine(L, 49, 275, 0);
					//waitKey(0);	//wait for user to press a key
					//cvDestroyAllWindows();	//resets all imshow() windows for next image
				}

				/*------------------------------------------------------------------------------------------*/

				//keeps slope variances
				float slV[arrayMaxSize];
				float slV2[arrayMaxSize];
				//index 0 is always zero; no change because it's the first window
				slV[0] = 0;
				slV2[0] = 0;

				// create output file in append mode
				// each image will generate 78 numbers (2 slope values for each of the 38 windows, plus 2 overall scores)
				// the data is written to the file with one number per line (78 lines) and a blank line at the very end
				outfiles[0] = "C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\Normal-Training.txt";
				outfiles[1] = "C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\AMD-Training.txt";
				ofstream outFile;
				outFile.open(outfiles[k], ios::app);

				if (k == 0)
					cout << "Writing N" << i << ".png data to file..." << endl << endl;
				else
					cout << "Writing A" << i << ".png data to file..." << endl << endl;

				//calculate slope variances
				for (int i = 1; i < arraySize; i++)
				{
					slV[i] = slopeVariance[i] - slopeVariance[i - 1];
					slV[i] *= 100;
					if (slV[i] < 0)
						slV[i] *= -1;//make sure it is positive
				}

				//calculate slope variances 2
				for (int i = 1; i < arraySize; i++)
				{
					slV2[i] = slV[i] - slV[i - 1];
					if (slV2[i] < 0)
						slV2[i] *= -1;	//make sure it is positive
					cout << showpoint << setprecision(8) << fixed << "Window " << i << "\t SLV: " << slV[i] << "\t SLV2: " << slV2[i] << endl;
					outFile << showpoint << setprecision(8) << fixed << slV[i] << endl << slV2[i] << endl;
				}

				//total score (sum) of slope variances
				float score = 0;
				float score2 = 0;
				for (int i = 0; i < arraySize; i++)
				{
					score += slV[i];
					score2 += slV2[i];
				}
				cout << showpoint << setprecision(8) << fixed << "Total Score: " << "\t SLV: " << score << "\t SLV2: " << score2 << endl;
				outFile << showpoint << setprecision(8) << fixed << score << endl << score2 << endl << endl;

				//close file
				outFile.close();
				if (k == 0)
					cout << "Done writing N" << i << ".png data to file..." << endl << endl;
				else
					cout << "Done writing A" << i << ".png data to file..." << endl << endl;

				//set arraySize and index position back to zero
				arraySize = 0;
				third++;
			}		
		}
	}
	/* -------------- done with current image at this point, now repeat for next image -------------- */

	// terminate when user presses a key
	waitKey(0);
	system("pause");
	return 0;
}