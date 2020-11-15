#pragma once
#ifndef UTILS_H
#define UTILS_H

//设置线程块数大小
#define TX 32
#define TY 32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<device_functions.h>
#include <iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <time.h>
using namespace std;
using namespace cv;
//申请W*H大小的double3数组
void mallocArray(float3*  &a,int W, int H);
//释放W*H大小的double3数组
void freeArray(float3* &a, int W,int H);
//将W*H大小的3通道Mat转为double3数组,注意Mat类型通道顺序为BGR,转换后通道顺序RGB
void Mat2Array(float3* &dst, const Mat &src, const int W, const int H);


//PCMA算法
void PCMA(float3 *I0, float3 *I45, float3 *I90, float3 *I135,int W,int H);

__device__ float3 sqrt(float3 a);
__device__ float3 abs(float3 a);
__device__ float3 multiply(float3 a, float3 b);
__device__ float3 multiply(float3 a, float b);
__device__ float3 add(float3 a, float3 b);
__device__ float3 add(float3 a, float3 b, float3 c, float3 d, float3 e );
__device__ float3 subtract(float3 a, float3 b);
__device__ float3 subtract(float3 a, float3 b,float3 c,float3 d);
__global__ void computeKernel(float3* d_out, float3 *I0, float3 * I45, float3 * I90, float3 * I135, int W, int H);



#endif // !UTILS_H




