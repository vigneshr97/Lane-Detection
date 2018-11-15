#include <iostream>
#include "tools.h"
#include <math.h>
#include <cmath>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth) 
{
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if(estimations.size() != ground_truth.size() || estimations.size() == 0)
	{
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i)
	{

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
	MatrixXd Hj(3,4);
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
	if(px*px+py*py < 0.000001)
	{
		cout<<"division by 0 error !!!!"<<endl;
		px = 0.001;
		py = 0.001;
	}
    
    Hj << 0,0,0,0,0,0,0,0,0,0,0,0;
	Hj(0,0) = px/sqrt(px*px+py*py);
    Hj(0,1) = py/sqrt(px*px+py*py);
    Hj(1,0) = -py/(px*px+py*py);
    Hj(1,1) = px/(px*px+py*py);
    Hj(2,2) = Hj(0,0);
    Hj(2,3) = Hj(0,1);
    Hj(2,0) = (vx*py-vy*px)*py/pow(px*px+py*py, 1.5);
    Hj(2,1) = (vy*px-vx*py)*px/pow(px*px+py*py, 1.5);
    
	//check division by zero
	
	//compute the Jacobian matrix

	return Hj;
}
