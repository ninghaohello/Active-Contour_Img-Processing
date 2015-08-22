#include "ActContour.h"

#include <iostream>
#include <cmath>
#include <deque>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

ActiveContour::ActiveContour(Mat& src)
{
	//Initialize basic parameter
	s = src.size();							//extremely careful!! size = (cols, rows)
	lambda = 0.8; step = 0.009;			//0.7, 0.005 (appropriate step is important not to cause sandy effect, even 0.0002)
	//An improved performance with large lambda at early stage
	rows = src.rows; cols = src.cols;
	imgMat = src.clone();
	num_crvPt = 0; num_layerPt = 0; num_bandPt = 0;
	inBandMat = Mat::zeros(rows, cols, CV_8UC1);
	traveMat = Mat::zeros(rows, cols, CV_8UC1);
	parentMat = Mat::zeros(rows, cols, CV_8UC1);
	extMat = Mat::zeros(rows, cols, CV_32FC1); wtMat = Mat::zeros(rows, cols, CV_32FC1);
	levelSetMat = Mat::zeros(rows, cols, CV_32FC1);
	lenGrid = 5; epsilon = 30.;
	show = 0;
}


void ActiveContour::ShowLevelSetMat()
{
	//Show
	namedWindow("LevelSet", CV_WINDOW_AUTOSIZE );
	imshow("LevelSet", levelSetMat < 0);
	waitKey(0);
}

void ActiveContour::ShowCurve()
{
	//Show
	Mat dst;
	addWeighted(imgMat, 0.7, curveMat>0, 0.3, 0.0, dst);
	namedWindow("Curve", CV_WINDOW_AUTOSIZE );
	imshow("Curve", dst);
	waitKey(0);
}


void ActiveContour::ShowBand()
{
	//Show
	namedWindow("Band", CV_WINDOW_AUTOSIZE );
	imshow("Band", inBandMat > 0);
	waitKey(0);
}

//Initialization corresponding the simple fixed initialization step(not recommended in real images)
void ActiveContour::SimpleInitialization()
{
	//Level set initialization (store curve points coordinates)
	float* p; uchar* p_ib;
	int gap = 0.3 * std::min(rows, cols);							//distance from picture edge (0.15)
	for(int i = 0; i < rows; i++)
	{
		p = levelSetMat.ptr<float>(i);
		p_ib = inBandMat.ptr<uchar>(i);
		for(int j = 0; j < cols; j++)
		{
			if(i < gap || i > rows-gap || j < gap || j > cols - gap)
				p[j] = 50;
			else if(i > gap && i < rows-gap && j > gap && j < cols - gap)
				p[j] = -50;
			else													//Push curve points into a queue
				{
					num_crvPt ++; num_layerPt ++; num_bandPt ++;
					x_crvs.push_back(i); y_crvs.push_back(j);		//add to curve points
					x_layer.push_back(i); y_layer.push_back(j);		//add to layer points
					x_bands.push_back(i); y_bands.push_back(j);		//add to band points
					p_ib[j] = 1;			//in-band indicator
					traveMat.at<uchar>(i,j) = 1;					//traversed
				}
		}
	}
}

//Initialization corresponding the interactive initialization step(recommended)
void ActiveContour::InteractiveInitialization(int* xy12)
{
	int x1, y1, x2, y2;
	x1 = xy12[0]; y1 = xy12[1]; x2 = xy12[2]; y2 = xy12[3];

	//Level set initialization (store curve points coordinates)
	float* p; uchar* p_ib;
	for(int i = 0; i < rows; i++)
	{
		p = levelSetMat.ptr<float>(i);
		p_ib = inBandMat.ptr<uchar>(i);
		for(int j = 0; j < cols; j++)
		{
			if(i < x1 || i > x2 || j < y1 || j > y2)
				p[j] = 50;
			else if(i > x1 && i < x2 && j > y1 && j < y2)
				p[j] = -50;
			else													//Push curve points into a queue
				{
					num_crvPt ++; num_layerPt ++; num_bandPt ++;
					x_crvs.push_back(i); y_crvs.push_back(j);		//add to curve points
					x_layer.push_back(i); y_layer.push_back(j);		//add to layer points
					x_bands.push_back(i); y_bands.push_back(j);		//add to band points
					p_ib[j] = 1;			//in-band indicator
					traveMat.at<uchar>(i,j) = 1;					//traversed
				}
		}
	}
	curveMat = traveMat.clone();
}


//Reinitialization, which recompute the level set value with narrow band, is an important step in narrow-band method
//Before using this function, check "traveMat" and "inBandMat", make sure you have curve points and seeds on them 
void ActiveContour::Reinitialization()
{
	//erase old mine region
	x_pmine.erase(x_pmine.begin(), x_pmine.end()); y_pmine.erase(y_pmine.begin(), y_pmine.end());
	x_nmine.erase(x_nmine.begin(), x_nmine.end()); y_nmine.erase(y_nmine.begin(), y_nmine.end());

	BandWidth = 8;
	int n, xp, yp;
	double lsv, weight;
	int l = 1;
	cout << "Reinitialization!!!!!!!!" << endl;
	while(! x_layer.empty())
	{
		num_layerPt = x_layer.size();
		n = num_layerPt;
		//num_layerPt = 0;			//clear
		for(;n > 0; n--)			//expand from this layer
		{
			//get each layer point(curve points are first-layer points)
			xp = x_layer.front(); yp = y_layer.front();
			x_layer.pop_front(); y_layer.pop_front();

			//8 neighbours
			for(int i = xp-1; i <= xp+1; i++)
			for(int j = yp-1; j <= yp+1; j++)
			{
				if( !(i>=0 && j>=0 && i<rows && j<cols) || (xp == i && yp == j) )
					continue;

				if( traveMat.at<uchar>(i,j) == 0 )
				{
					//traversed
					traveMat.at<uchar>(i,j) = 1;

					//add to next layer
					x_layer.push_back(i); y_layer.push_back(j);

					//signed distance as level set value
					lsv = levelSetMat.at<float>(i,j);
					levelSetMat.at<float>(i,j) = lenGrid * l * ((lsv>0) - (lsv<0));

					//mine region (positive and negative)
					if(l == BandWidth && lsv > 0){ x_pmine.push_back(i); y_pmine.push_back(j); }
					else if(l == BandWidth && lsv < 0) { x_nmine.push_back(i); y_nmine.push_back(j); }	
				}

				if( inBandMat.at<uchar>(i,j) == 0 && l <= BandWidth)
				{
					//compute extension for next layer neighbours
					weight = 1./pow(abs(xp-i) + abs(yp-j), 4);
					extMat.at<float>(i,j) += extMat.at<float>(xp,yp) * weight;
					wtMat.at<float>(i,j) += weight;
				}
			}
		}

		if(l <= BandWidth)
		{
			//new layer points add to band
			for(int p = 0; p < x_layer.size(); p++)
			{
				inBandMat.at<uchar>(x_layer[p],y_layer[p]) = 1;
				x_bands.push_back(x_layer[p]);  y_bands.push_back(y_layer[p]);
				num_bandPt ++;

				extMat.at<float>(x_layer[p],y_layer[p]) /= wtMat.at<float>(x_layer[p],y_layer[p]);
			}
		}
		l ++;						//next layer
	}
}

//Before using this function, make sure you have assign seeds to the curve points
//Extend field to doubled-width band
void ActiveContour::ExtendExternalField()
{
	int n, xp, yp;
	double weight;
	for(int l = 1;l < 2*BandWidth; l++)
	{
		num_layerPt = x_layer.size();
		n = num_layerPt;
		for(;n > 0; n--)
		{
			xp = x_layer.front(); yp = y_layer.front();
			x_layer.pop_front(); y_layer.pop_front();

			//8 neighbours
			for(int i = xp-1; i <= xp+1; i++)
			for(int j = yp-1; j <= yp+1; j++)
			{
				if( i>=0 && j>=0 && i<rows && j<cols && !(xp == i && yp == j))
				{
					if(traveMat.at<uchar>(i,j) == 0)
					{
						traveMat.at<uchar>(i,j) = 1;
						//add to next layer
						x_layer.push_back(i); y_layer.push_back(j);
					}

					if( parentMat.at<uchar>(i,j) == 0 && inBandMat.at<uchar>(i,j) == 1)
					{
						//spread
						if(xp != i || yp != j){
						weight = 1./pow(abs(xp-i) + abs(yp-j), 4);
						extMat.at<float>(i,j) += extMat.at<float>(xp,yp) * weight;
						wtMat.at<float>(i,j) += weight;
						}
					}
				}
			}
		}
		//new layer points add to band
		for(int p = 0; p < x_layer.size(); p++)
		{
			parentMat.at<uchar>(x_layer[p], y_layer[p]) = 1;
			extMat.at<float>(x_layer[p],y_layer[p]) /= wtMat.at<float>(x_layer[p],y_layer[p]);
		}
	}
}

//Before using this function, "traveMat" and "inBandMat" should be reset, in case of reinitializaiton
void ActiveContour::FindCurvePoints1(bool initFlag)
{
	int xp, yp;
	//erase old curve points and layer points
	x_crvs.erase(x_crvs.begin(), x_crvs.end()); y_crvs.erase(y_crvs.begin(), y_crvs.end());
	x_layer.erase(x_layer.begin(), x_layer.end()); y_layer.erase(y_layer.begin(), y_layer.end());
	parentMat = Mat::zeros(rows, cols, CV_8UC1);

	//find new curve points from narrow-band points
	for(int b = 0; b < x_bands.size(); b++)
	{
		xp = x_bands[b]; yp = y_bands[b];
		//different for ChanVese and Local-ChanVese !!!!!!!!!!!!!!!!!!!(2)
		if( (levelSetMat.at<float>(xp,yp) + numeric_limits<float>::min() <= 0 ) && ( (xp-1>=0 && levelSetMat.at<float>(xp-1,yp)>0) 
			|| (yp-1>=0 && levelSetMat.at<float>(xp,yp-1)>0) || (xp+1<rows && levelSetMat.at<float>(xp+1,yp)>0) 
			|| (yp+1<cols && levelSetMat.at<float>(xp,yp+1)>0) 
			|| (xp-1>=0 && yp-1>=0 && levelSetMat.at<float>(xp-1,yp-1)>0) 
			|| (xp+1<rows && yp-1>=0 && levelSetMat.at<float>(xp+1,yp-1)>0)
			|| (xp+1<rows && yp+1<=cols && levelSetMat.at<float>(xp+1,yp+1)>0)
			|| (xp-1>=0 && yp+1<=cols && levelSetMat.at<float>(xp-1,yp+1)>0)) ) //(levelSetMat.at<float>(xp,yp) == 0)
		{
			//zero-level, curve point number ++, save curve point 
			levelSetMat.at<float>(xp,yp) = 0;
			num_crvPt ++;
			x_crvs.push_back(xp); y_crvs.push_back(yp);

			//extension spread
			parentMat.at<uchar>(xp,yp) = 1;
			traveMat.at<uchar>(xp,yp) = 1;

			//initial layer, number of layer points ++
			num_layerPt ++;
			x_layer.push_back(xp); y_layer.push_back(yp);
		}
	}
	if(initFlag)
	{
		x_bands.erase(x_bands.begin(), x_bands.end()); y_bands.erase(y_bands.begin(), y_bands.end());
		for(int p = 0; p < x_crvs.size(); p++)
		{
			x_bands.push_back(x_crvs[p]);
			y_bands.push_back(y_crvs[p]);
			inBandMat.at<uchar>(x_crvs[p],y_crvs[p]) = 1;
		}
	}
	curveMat = parentMat.clone();
}

void ActiveContour::FindCurvePoints2(bool initFlag)
{
	int xp, yp;
	//erase old curve points and layer points
	x_crvs.erase(x_crvs.begin(), x_crvs.end()); y_crvs.erase(y_crvs.begin(), y_crvs.end());
	x_layer.erase(x_layer.begin(), x_layer.end()); y_layer.erase(y_layer.begin(), y_layer.end());
	parentMat = Mat::zeros(rows, cols, CV_8SC1);

	Mat temp_abs = abs(levelSetMat);

}

//The contour should be kept within the narrow band, which region is defined by a positive boundary and a negative boundary.
//This function is used to check if contour will move beyond narrow band.  
int ActiveContour::TouchMine()
{
	int threshold = 1;
	int count = 0;
	//positive boundary become negative
	for(int i = 0; i < x_pmine.size(); i++)
		if(levelSetMat.at<float>(x_pmine[i], y_pmine[i]) < 0)
			if(++ count >= threshold)
				return 1;

	//negative boundary become positive
	for(int i = 0; i < x_nmine.size(); i++)
		if(levelSetMat.at<float>(x_nmine[i], y_nmine[i]) > 0)
			if(++ count >= threshold)
				return 1;

	return 0;
}

//Mean of pixels density inside curve
float ActiveContour::InCurveMean()
{
	float ci;
	Mat inMask = (levelSetMat < 0);
	ci = mean(imgMat, inMask)[0];

	return ci;
}

//Mean of pixels density outside curve
float ActiveContour::OutCurveMean()
{
	float co;
	Mat outMask = (levelSetMat > 0);
	co = mean(imgMat, outMask)[0];
	
	return co;
}

//4 difference matrix get ready
void ActiveContour::DiffXYFwdBcwd()
{
	Mat fdiff_x_mask = (Mat_<float>(1,3)<<0,-1,1);
	Mat bdiff_x_mask = (Mat_<float>(1,3)<<-1,1,0);
	Mat fdiff_y_mask = (Mat_<float>(3,1)<<0,-1,1);
	Mat bdiff_y_mask = (Mat_<float>(3,1)<<-1,1,0);

	filter2D(levelSetMat, fdiff_x, -1, fdiff_x_mask);
	filter2D(levelSetMat, bdiff_x, -1, bdiff_x_mask);
	filter2D(levelSetMat, fdiff_y, -1, fdiff_y_mask);
	filter2D(levelSetMat, bdiff_y, -1, bdiff_y_mask);
}

//Upwind entropy scheme
float ActiveContour::EntropyUpwindDiff(float coeff, int x, int y)
{
	int diffNorm = 0;
	if(coeff > 0)
		diffNorm = sqrt(pow(std::max(fdiff_x.at<float>(x,y),float(0)),2) + 
		pow(std::min(bdiff_x.at<float>(x,y),float(0)),2) + 
		pow(std::max(fdiff_y.at<float>(x,y),float(0)),2) + 
		pow(std::min(bdiff_y.at<float>(x,y),float(0)),2));
	else
		if(coeff < 0)
		diffNorm = sqrt(pow(std::min(fdiff_x.at<float>(x,y),float(0)),2) + 
		pow(std::max(bdiff_x.at<float>(x,y),float(0)),2) + 
		pow(std::min(fdiff_y.at<float>(x,y),float(0)),2) + 
		pow(std::max(bdiff_y.at<float>(x,y),float(0)),2));

	return diffNorm;
}

float ActiveContour::CentralDiff(int x, int y)
{
	return sqrt(pow(cdiff_x.at<float>(x,y),2) + pow(cdiff_y.at<float>(x,y),2));
}

//Compute curvature
void ActiveContour::Curvature()
{
	Mat cdiff_xx, cdiff_yy, cdiff_xy;
	
	Mat cdiff_x_mask = (Mat_<float>(1,3)<<-1,0,1)/2;
	Mat cdiff_y_mask = (Mat_<float>(3,1)<<-1,0,1)/2;
	Mat cdiff_xx_mask = (Mat_<float>(1,3)<<1,-2,1);
	Mat cdiff_yy_mask = (Mat_<float>(1,3)<<1,-2,1);
	Mat cdiff_xy_mask = (Mat_<float>(3,3)<<1,0,-1,0,0,0,-1,0,1)/4;

	filter2D(levelSetMat, cdiff_x, -1, cdiff_x_mask);
	filter2D(levelSetMat, cdiff_y, -1, cdiff_y_mask);
	filter2D(levelSetMat, cdiff_xx, -1, cdiff_xx_mask);
	filter2D(levelSetMat, cdiff_yy, -1, cdiff_yy_mask);
	filter2D(levelSetMat, cdiff_xy, -1, cdiff_xy_mask);

	Mat denom;
	double regularizer = 1;
	cv::pow(cdiff_x.mul(cdiff_x) + cdiff_y.mul(cdiff_y) + regularizer, 1.5, denom);
	cvt = -( (cdiff_y.mul(cdiff_y)).mul(cdiff_xx) - 2*(cdiff_x.mul(cdiff_y)).mul(cdiff_xy) +  (cdiff_x.mul(cdiff_x)).mul(cdiff_yy) 
		/ denom );
}

//Before using this function, build extension for all band points first.
//In the function, update base matrix first.
void ActiveContour::UpdateLevelSet()
{
	cout << "Update level set......" << endl;
	int x,y;
	double coeff1,coeff2;
	Curvature();					//curvature
	DiffXYFwdBcwd();				//4 difference matrix ready
	for(int n = 0; n < x_bands.size(); n++)
	{
		x = x_bands[n]; y = y_bands[n];
		coeff1 = -lambda*extMat.at<float>(x,y);		//(u-v)*(I-u+I-V), I is on curve
		coeff2 = (1-lambda) * cvt.at<float>(x,y);
		levelSetMat.at<float>(x,y) += step*( coeff1 * EntropyUpwindDiff(coeff1, x, y) + coeff2 * CentralDiff(x, y) );
	}
}

//Before using this funtion, make sure you have got all curve points
void ActiveContour::ChanVeseSeed()
{
	int show = 0;
	
	int x, y, I;
	float u, v;
	u = InCurveMean(); v = OutCurveMean(); if(show == 1) cout << "u: " << u << "v: " << v << endl;

	extMat = Mat::zeros(rows, cols, CV_32FC1);
	wtMat = Mat::zeros(rows, cols, CV_32FC1);

	for(int n = 0; n < x_crvs.size(); n++)
	{
		x = x_crvs[n]; y = y_crvs[n];
		I = imgMat.at<uchar>(x,y);
		extMat.at<float>(x,y) = (u - v)*(I - u + I - v);
		if(show == 1)
			cout << step*(u - v)*(I - u + I - v) << ' ';
	}

	double maxVal;
	minMaxLoc(abs(extMat), NULL, &maxVal, NULL, NULL, inBandMat);
	step = lenGrid/(1.1*maxVal);				//no sqrt(2) or 2 is needed, the reason may be that step is not Euclidian
												//larger step may be better, lambda in the denominator??(maybe) (1.15 good)
}

//Calculate the value of energy function
float ActiveContour::ComputeEnergy()
{
	double energy, sum1, sum2;
	sum1 = sum((imgMat - InCurveMean()).mul(imgMat - InCurveMean()).mul((levelSetMat <= 0)/255))[0];
	sum2 = sum((imgMat - OutCurveMean()).mul(imgMat - OutCurveMean()).mul((levelSetMat > 0)/255))[0];
	energy = (1 - lambda)*x_crvs.size() + lambda*(sum1 + sum2);

	return energy;
}
