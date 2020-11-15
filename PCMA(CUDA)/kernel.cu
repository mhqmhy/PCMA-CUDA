
#include "utils.h"
#define TX 32 
#define TY 32
using namespace std;
using namespace cv;


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
    
	
	
	//裁剪图片，裁剪黑边
	int H = image0.rows, W = image0.cols, D = image0.channels();
	double pt_x = 100, pt_y = 100;
	Rect area(pt_x - 1, pt_y - 1, W - pt_x + 1, H - pt_y + 1);
	image0 = image0(area);
	image45 = image45(area);
	image90 = image90(area);
	image135 = image135(area);
	
	H = image0.rows, W = image0.cols, D = image0.channels();

	tempS = clock();
	//数据结构转换 Mat to Double3数组
	float3 *I0,*I45,*I90,*I135;
	//申请空间
	mallocArray(I0, W, H);
	mallocArray(I45, W, H);
	mallocArray(I90, W, H);
	mallocArray(I135, W, H);
	//转换数据结构
	Mat2Array(I0, image0, W, H);
	Mat2Array(I45, image45, W, H);
	Mat2Array(I90, image90, W, H);
	Mat2Array(I135, image135, W, H);

	tempE = clock();
	//cout << "转换类型：" << double(tempE - tempS) / CLOCKS_PER_SEC << "s" << endl;
	//cout << "裁剪后:" << W << "*" << H << endl;
	double sum = 0,tempTime;
	pt_x = 1, pt_y = 1;
	double wSize_w = 10, wSize_h = 10;
	//执行去雾算法
	for (int i = 0; i < 10; i++) {
		cout << "**************************" << endl;
		tempS = clock();
		PCMA(I0, I45, I90, I135, W, H);
		tempE = clock();
		tempTime = double(tempE - tempS) / CLOCKS_PER_SEC;
		cout <<"第"<<i <<"次去雾算法：" << tempTime << "s" << endl;
	}
	
	//一定要释放空间
	freeArray(I0, W, H);
	freeArray(I45, W, H);
	freeArray(I90, W, H);
	freeArray(I135, W, H);
	system("pause");
    return 0;
}



