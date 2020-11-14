#pragma once
#ifndef UTILS_H
#define UTILS_H

//�����߳̿�����С
#define TX 32
#define TY 32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;
//����W*H��С��double3����
void mallocArray(float3**  &a,int W, int H);
//�ͷ�W*H��С��double3����
void freeArray(float3** &a, int W,int H);
//��W*H��С��3ͨ��MatתΪdouble3����,ע��Mat����ͨ��˳��ΪBGR,ת����ͨ��˳��RGB
void Mat2Array(float3** &dst, const Mat &src, const int W, const int H);


//PCMA�㷨
void PCMA(float3 **&I0, float3 **&I45, float3 **&I90, float3 **I135,int W,int H);

__device__ float3 multiply(float3 a, float3 b);
__device__ float3 multiply(float3 a, float b);
__device__ float3 add(float3 a, float3 b);
__device__ float3 subtract(float3 a, float3 b);
__global__ void computeKernel(float3* &out, float3 **& I0, float3 **& I45, float3 **& I90, float3 ** I135, int W, int H);

#endif // !UTILS_H




