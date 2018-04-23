#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
    VideoCapture cap;
	if (!cap.open(0))
		return 0;
    
    //enter loop
	for (;;)
	{
        //capture frame
		Mat frame;
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("this is you!", frame);   //display webcame output
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	cap.release();  //release memory
}
