#ifndef ACTCONTOUR_H_
#define ACTCONTOUR_H_

#include <iostream>
#include <deque>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;  
using namespace cv;

class ActiveContour
{
private:
	int rows, cols;
	int BandWidth, lenGrid;
	Size s;
	float lambda;						//length penalty coefficient
	float step;
	float epsilon;
	int show;

	//Mats
	Mat imgMat, inBandMat, traveMat, parentMat;
	Mat extMat, levelSetMat, wtMat;		//extension, levelset, weight of extension
	Mat circleMask, locInSqrDiff, locOutSqrDiff;
	Mat curveMat;

	//Container of all curve points
	deque<int> x_crvs, y_crvs;
	deque<int> x_layer, y_layer;
	deque<int> x_bands, y_bands;
	deque<int> x_pmine, y_pmine, x_nmine, y_nmine;
	int num_crvPt, num_layerPt, num_bandPt;

	Mat fdiff_x, bdiff_x, fdiff_y, bdiff_y, cdiff_x, cdiff_y;
	Mat cvt;


public:
	ActiveContour(Mat& src);
	//~ActiveContour();

	void ResetTraveMat() { traveMat = Mat::zeros(rows, cols, CV_8UC1);}		//should be 8U, not 8S
	void ResetInBandMat() { inBandMat = Mat::zeros(rows, cols, CV_8UC1); }
	void ResetParentMat() { parentMat = Mat::zeros(rows, cols, CV_8UC1); }

	void ShowLevelSetMat();
	void ShowBand();
	void ShowCurve();

	//A simple initialization method for level set: a single large contour
	//src is gray image
	void SimpleInitialization();
	void InteractiveInitialization(int *);

	//Rebuild narrow-band and compute extension w.r.t current zero-level curve
	void Reinitialization();

	//Recover zero-level curve points; initFlag=1 when reinitialization is needed
	void FindCurvePoints1(bool initFlag);
	void FindCurvePoints2(bool initFlag);

	//Touch mine region or not. If yes, reinitialize.
	int TouchMine();

	//Inside-curve (u) and outside-curve (v) mean 
	float InCurveMean();
	float OutCurveMean();

	//Compute forward and backward difference for x and y
	void DiffXYFwdBcwd();

	//Entropy upwind scheme for computing || delt(level_set) ||
	float EntropyUpwindDiff(float coeff, int x, int y);
	float CentralDiff(int x, int y);

	//Curvature
	void Curvature();

	//Update level-set
	void UpdateLevelSet();

	//Chan-Vese extension "seed" on zero-level curve
	void ChanVeseSeed();

	void ExtendExternalField();
	
	float ComputeEnergy();

	void SetShow() {show = 1;}
};


#endif;
