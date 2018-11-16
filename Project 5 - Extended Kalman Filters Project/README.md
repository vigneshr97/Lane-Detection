# Extended Kalman Filter Project

In this project a kalman filter is utilized to estimate the state of a moving object of interest with noisy lidar and radar measurements. The install file viz., install-mac.sh or install-ubuntu.sh is run to install the required dependencies.
This project involves the Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)
Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory. A build directory with a compiled file 'ExtendedKF' is available in this repository. A brief writeup explaining the various functions is at the bottom of the README

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

The following image shows the final rmse values of x, y, vx, vy
![image1](images/image1.jpg)

---

## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Move to build directory `cd build`
3. To edit and make
`make && cmake ..`
4. Run the compiled file: `./ExtendedKF `

## Explaining the code

The code in src directory has three C++ files FusionEKF.cpp, tools.cpp, kalman_filter.cpp that perform various functions and a main function contained in main.cpp. Eigen/Dense library of C++ was used to perform various matrix operations.

## tools.cpp
The file has two functions. The first function, CalculateRMSE finds the Root mean square error of the obtained values from the Sensor fusion against the ground truth values. The second function, CalculateJacobian finds the Jacobian with the help of the previous position and velocity values to obtain the measurement function for the RADAR.

## kalman_filter.cpp
The file has four functions after constructor, destructor and initializer. The first function is the motion prediction function, Predict(). The position and covariance are calculated from the State Transition Matrix, State covariance and noise covariance matrix using the Prediction formulae.
The function, Update() does the measurement update using the Kalman Filter Equations for the data obtained from the LIDAR. The LIDAR does not disturb the velocity values. The function common_update() performs the update operations common to both LIDAR and RADAR measurement update.
The function, UpdateEKF() performs the measurement update using Extended Kalman Filter equations. The initial position values and covariance matrix are passed on to this function. The measurment, y = z - Hx becomes y = z - h in the extended kalman filter equations. z is the measurement passed on to the RADAR sensor and they are in polar coordinates containing the values of radial position, angle and the radial velocity respecitively. h is the value calculated from the current position vector. h contains the position vector in polar coordinates. The measurement function is calculated using the function, CalculateJacobian(). The rest of the steps are the same as performed on a LIDAR.

## FusionEKF.cpp
The values are initialized either as given or arbitrarily in the class constructor. The function, ProcessMeasurement() has two parts, namely the initialization part which is called when the first measurement is recorded. The velocity values are initialized to 0 incase of a LIDAR as it cannot measure velocity. A LIDAR returns the x and y position values whereas the RADAR returns the radial position, angle and the radial velocity. So the polar coordinate values are converted to Cartesian coordinates. The acceleration noise in x and y directions are set to 9. The state covariance matrix is fed with 1 for variance in x and y positions and 1000 for variance in x and y velocity values. The measurement variance of LASER in both x and y directions are 0.0225. The measurement variance of RADAR in determining rho, phi and rho_dot values are 0.09, 0.0009 and 0.09 respectively. The transition matrix is varied depending on the difference in successive timestamp values. The Predict() function is run first followed by either Update() or UpdateEKF() depending on whether it is LASER or RADAR after setting the respective measurement function and measurement uncertainty.  

## main.cpp
The function runs and obtains the x and P values from ProcessMeasurement() function and compares them with ground truth values thus obtaining the Root Mean Squared error values of x, y, vx, vy respectively as 0.0973, 0.0855, 0.4513 and 0.4399 for the first dataset. All the values satisfy the threshold condition specified in the project requirement.
