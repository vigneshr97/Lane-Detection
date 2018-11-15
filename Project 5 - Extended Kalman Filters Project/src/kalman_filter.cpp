#include "kalman_filter.h"
#include <iostream>
#include <math.h>
#include <cmath> 
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() 
{
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) 
{
  VectorXd y = z - H_*x_;
  common_update(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
  VectorXd h(3);
  h << 0, 0 ,0;
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  h(0) = sqrt(px*px+py*py);
  h(1) = atan2(py, px);
  h(2) = (px*vx + py*vy)/sqrt(px*px + py*py); 
  
  VectorXd y = z - h;
  int npi = 0;
  if(y(1) > M_PI)
  {
    npi = (int) (y(1)/M_PI);
  }
  if(y(1) < - M_PI)
  {
    npi = (int) (y(1)/M_PI);
  }
  y(1) -= (float) (npi*M_PI) ;
  
  cout <<"y(1):"<<y(1)<<endl;

  common_update(y);
}


void KalmanFilter::common_update(VectorXd &y)
{
  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();
  x_ = x_ + K*y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
