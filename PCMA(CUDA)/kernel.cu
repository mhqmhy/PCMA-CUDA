
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include<time.h>

#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


#define TX 32 
#define TY 32
using namespace std;
using namespace cv;
double3 **I0;
double3 **I45;
double3 **I90;
double3 **I135;

__global__
void computeS0Kernel(double3 **out, double3 ** I0, double3 ** I45, double3 ** I90, double3 ** I135,int w,int h) {
	
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	const int i = row*w + col;
	out[i]->x = (I0[row][col].x + I45[row][col].x +I90[row][col].x + I135[row][col].x)/2.0;
	out[i]->y = (I0[row][col].y + I45[row][col].y + I90[row][col].y + I135[row][col].y) / 2.0;
	out[i]->z = (I0[row][col].z + I45[row][col].z + I90[row][col].z + I135[row][col].z) / 2.0;

}
void PCMA(Mat image0,Mat image45,Mat image90,Mat image135,double w,double pt_x
,double pt_y,double wSize_w,double wSize_h);
int main()
{	
	//声明变量
	clock_t startTime, endTime, tempS, tempE;
	Mat image0, image45, image90, image135;
	//读取图片
	tempS = clock();
	image0 = imread("1024/right_up.bmp");
	image0.convertTo(image0, CV_32FC3, 1.0 / 255.0);
	image45 = imread("1024/left_up.bmp");
	image45.convertTo(image45, CV_32FC3, 1.0 / 255.0);
	image90 = imread("1024/left_down.bmp");
	image90.convertTo(image90, CV_32FC3, 1.0 / 255.0);
	image135 = imread("1024/right_down.bmp");
	image135.convertTo(image135, CV_32FC3, 1.0 / 255.0);
	tempE = clock();
	cout << "读取图片：" << double(tempE - tempS)/ CLOCKS_PER_SEC << "s" << endl;
    
	
	I0 = (double3 **)calloc(441 *925, sizeof(double3));
		I45 = (double3 **)calloc(441 * 925, sizeof(double3));
	I90 = (double3 **)calloc(441 * 925, sizeof(double3));
	I135 = (double3 **)calloc(441 * 925, sizeof(double3));

	
	//裁剪图片
	int H = image0.rows, W = image0.cols, D = image0.channels();
	double pt_x = 100, pt_y = 100;
	Rect area(pt_x - 1, pt_y - 1, W - pt_x + 1, H - pt_y + 1);
	image0 = image0(area);
	image45 = image45(area);
	image90 = image90(area);
	image135 = image135(area);
	
	//cout << ptr[0] << " " << ptr[1] << endl;

	pt_x = 1, pt_y = 1;
	double wSize_w = 10, wSize_h = 10;
	//执行去雾算法
	PCMA(image0, image45, image90, image135, 1.7,pt_x, pt_y, wSize_w, wSize_h);
	free(I0);
free(I45);
	free(I90);
	free(I135);
	system("pause");
    return 0;
}


void Mat2Double3(Mat a, double3 **mat) {
	int h = a.rows, w = a.cols;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			double *ptr = a.ptr<double>(row, col);
			mat[row][col].x = ptr[0];
			mat[row][col].y = ptr[1];
			mat[row][col].z = ptr[2];
		}
	}
}

void PCMA(Mat image0, Mat image45, Mat image90, Mat image135, double w, double pt_x, double pt_y, double wSize_w, double wSize_h)
{
	int W = 441, H = 925;
	clock_t  tempS, tempE;
	cout << image0.rows <<" "<<image0.cols<< endl;
	

	 Mat2Double3(image0.clone(),(double3 **)I0);
	 Mat2Double3(image45.clone(), (double3 **)I45);
	  Mat2Double3(image90.clone(), (double3 **)I90);
	 Mat2Double3(image135.clone(), (double3 **)I135);

	tempS = clock();
	//计算S0=(I0+I45+I90+I135)/2.0
	//int W = image0.cols, H = image0.rows;
	double3 **out = (double3 **)calloc(W*H, sizeof(double3));
	double3 **d_out;
	cudaMalloc(&d_out, W*H * sizeof(double3));
	const dim3 blockSize(TX, TY);
	const int bx = (W + TX - 1) / TX;
	const int by = (H + TY - 1) / TY;
	const dim3 gridSize = dim3(bx, by);
	computeS0Kernel << <gridSize, blockSize>>> (d_out, I0, I45, I90, I135, W, H);
	cudaMemcpy(out, d_out, W*H * sizeof(double3), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	free(out);
	tempE = clock();
	cout << "计算S0:" << double(tempE - tempS) / CLOCKS_PER_SEC << "s" << endl;
}
