// Caleb Sarikey & Tim Densmore
// CSIT 463 Final Project
// SupportVectorMachine.cpp - Trains and tests SVM, outputs SVM results to text file

//#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include<iostream>
#include<fstream>
#include<iomanip>

using namespace std;
using namespace cv;

int main()
{
	// Set up training data:
	float labels[212];

	//first 77 are AMD (-1.0)
	for (int i = 0; i < 77; i++)
	{
		labels[i] = -1.0;
	}
	//last 135 are Normal (1.0)
	for (int i = 77; i < 212; i++) {
		labels[i] = 1.0;
	}

	Mat labelsMat(212, 1, CV_32FC1, labels);

	//training data:  212 images total with 78 values each
	float TrainingData[212][78];
	//testing data:  71 images total with 78 values each
	float TestingData[71][78];

	ifstream inFile;	//for input file stream

	/* ----------------------- READ AMD TRAINING DATA ----------------------- */
	//open file in input mode
	inFile.open("C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\AMD-Training.txt", ios::in);
	if (inFile.is_open()) {
		cout << "Now reading AMD-Training.txt..." << endl;
		for (int i = 0; i < 77; i++) {
			for (int j = 0; j < 78; j++) {
				inFile >> showpoint >> fixed >> setprecision(8) >> TrainingData[i][j];	//insert into array
				//cout << showpoint << fixed << setprecision(8) << TrainingData[i][j] << endl;
			}
		}
		inFile.close();	//close file
		cout << "Done reading AMD-Training.txt..." << endl;
		system("pause");
	}
	else {
		//check file open errors
		cout << "Unable to open file..." << endl;
	}


	/* ----------------------- READ NORMAL TRAINING DATA ----------------------- */
	//open file in input mode
	inFile.open("C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\Normal-Training.txt", ios::in);
	if (inFile.is_open()) {
		cout << endl << "Now reading Normal-Training.txt..." << endl;
		for (int i = 77; i < 212; i++) {
			for (int j = 0; j < 78; j++) {
				inFile >> showpoint >> fixed >> setprecision(8) >> TrainingData[i][j];	//insert into array
				//cout << showpoint << fixed << setprecision(8) << TrainingData[i][j] << endl;
			}
		}
		inFile.close();	//close file
		cout << "Done reading Normal-Training.txt..." << endl;
		system("pause");
	}
	else {
		//check file open errors
		cout << "Unable to open file..." << endl;
	}

	/* ----------------------- READ AMD TESTING DATA ----------------------- */
	//open file in input mode
	inFile.open("C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\AMD-Testing.txt", ios::in);
	if (inFile.is_open()) {
		cout << endl << "Now reading AMD-Testing.txt..." << endl;
		for (int i = 0; i < 26; i++) {
			for (int j = 0; j < 78; j++) {
				inFile >> showpoint >> fixed >> setprecision(8) >> TestingData[i][j];	//insert into array
				//cout << showpoint << fixed << setprecision(8) << TestingData[i][j] << endl;
			}
		}
		inFile.close();	//close file
		cout << "Done reading AMD-Testing.txt..." << endl;
		system("pause");
	}
	else {
		//check file open errors
		cout << "Unable to open file..." << endl;
	}


	/* ----------------------- READ NORMAL TESTING DATA ----------------------- */
	//open file in input mode
	inFile.open("C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\Normal-Testing.txt", ios::in);
	if (inFile.is_open()) {
		cout << endl << "Now reading Normal-Testing.txt..." << endl;
		for (int i = 26; i < 71; i++) {
			for (int j = 0; j < 78; j++) {
				inFile >> showpoint >> fixed >> setprecision(8) >> TestingData[i][j];	//insert into array
				//cout << showpoint << fixed << setprecision(8) << TestingData[i][j] << endl;
			}
		}
		inFile.close();	//close file
		cout << "Done reading Normal-Testing.txt..." << endl;
		system("pause");
	}
	else {
		//check file open errors
		cout << "Unable to open file..." << endl;
	}


	// ---------------------------------------------------------------------------

	Mat trainingDataMat(212, 78, CV_32FC1, TrainingData);

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-8);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	cout << endl << "SVM has been trained..." << endl;

	cout << "Press any key to test SVM" << endl;
	system("pause");

	float temp[1][78];	//to hold data for single test image
	float classifications[71];	//to hold predictions
	float distances[71];
	bool returnDFVal = true;
	//put data for one image at a time into temp
	//test temp and store the result into responses array
	for (int i = 0; i < 71; i++) {
		for (int j = 0; j < 78; j++) {
			temp[0][j] = TestingData[i][j];
		}
		Mat testMat(1, 78, CV_32FC1, temp);
		classifications[i] = SVM.predict(testMat);
		distances[i] = SVM.predict(testMat, returnDFVal);
	}

	cout << endl << "Testing complete.\nPress any key to write SVM test results" << endl;
	system("pause");

	ofstream outFile;
	outFile.open("C:\\Users\\csari\\Desktop\\A4\\A4\\x64\\Debug\\SVM-Results.txt", ios::out);

	// print out the prediction data
	// AMD predictions should be -1.0
	cout << "AMD Predictions:" << endl;
	outFile << "AMD Predictions:" << endl;
	outFile << "Image #" << " \t " << "Classifications" << " \t " << "Distances" << endl;
	int imgNum = 1;
	for (int i = 0; i < 26; i++) {
		outFile << "A" << imgNum << " \t\t " << classifications[i] << " \t\t\t " << distances[i] << endl;
		cout << "A" << imgNum << ".png ----> " << "Classification : " << classifications[i] << " \t Distance: " << distances[i] << endl;
		imgNum += 4;
	}
	imgNum = 1;
	// Normal predictions should be 1.0
	outFile << endl;
	outFile << "Normal Predictions:" << endl;
	outFile << "Image #" << " \t " << "Classifications" << " \t " << "Distances" << endl;
	cout << endl << "Normal Predictions:" << endl;
	for (int i = 26; i < 71; i++) {
		outFile << "N" << imgNum << " \t\t " << classifications[i] << " \t\t\t " << distances[i] << endl;
		cout << "N" << imgNum << ".png ----> " << "Classification : " << classifications[i] << " \t Distance: " << distances[i] << endl;
		imgNum += 4;
	}

	system("pause");
	return 0;
}