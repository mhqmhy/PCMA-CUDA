#include "utils.h"
#include <cstdlib>
#include <time.h>
void mallocArray(float3 *&a, int W, int H)//按引用传递，否则是临时变量a
{
	a = new float3[W*H];
}

void freeArray(float3 * &a,int W,int H)
{
	delete[] a;
}

void Mat2Array(float3 * &dst, const Mat & src, const int W, const int H)
{
	srand((unsigned)time(NULL));
	for (int row = 0; row < H; row++) {
	
		for (int col = 0; col < W; col++) {
			//blue
			int index = row*W + col;
			float b = *(src.data + src.step[0] * row + src.step[1] * col);
			//green  elemSize1指的是一个通道所占字节数
			float g = *(src.data + src.step[0] * row + src.step[1] * col+src.elemSize1());
			//red
			float r = *(src.data + src.step[0] * row + src.step[1] * col+ src.elemSize1()*2);

			dst[index].x =r;
			dst[index].y =g;
			dst[index].z =b;
		}
	}
}



__global__ void computeKernel(float3* d_out, float3* I0, float3* I45, float3* I90, float3* I135, int W, int H)
{
	//利用线程索引和块索引计算数组下标
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	const int index = r*W+c;
	//计算S0
	if (r >= H || c >= W)
		return;
	float3 s0,i0,i45,i90,i135;
	i0 = I0[index],i45 = I45[index], i90 = I90[index], i135 = I135[index];
	s0.x = (i0.x + i45.x + i90.x + i135.x);
	s0.y = (i0.y + i45.y + i90.y + i135.y);
	s0.z = (i0.z + i45.z + i90.z + i135.z);
	d_out[index].x = index;
	//计算w0_45,w0_90,w0_135,w45_90,w45_135,w90_135
	float3 w0_45, w0_90, w0_135, w45_90, w45_135, w90_135;
	w0_45 = multiply(i0, i45);
	w0_90 = multiply(i0 ,i90);
	w0_135 = multiply(i0, i135);
	w45_90 = multiply(i45, i90);
	w45_135 = multiply(i45, i135);
	w90_135 = multiply(i90, i135);
	
	//-ε45,-ε'45,γ45,-γ45,-γ'45
	float3 epsilon45_1,epsilon45_2,gamma45_1,gamma45_2,gamma45_3;
	epsilon45_1 =subtract( i45 , i90);
	epsilon45_2 = subtract(i0, multiply(i45, 2));
	gamma45_1 = add(i0, i90);
	gamma45_2 = subtract(i45, i90);
	gamma45_3 = subtract(i0, multiply(i45, 2));


	
	float3 DolP_A=sqrt(abs(add(multiply(epsilon45_1,epsilon45_1), multiply(epsilon45_2, epsilon45_2), multiply(multiply(i0,i0), (float)2),
		multiply(i135, i135),multiply(subtract(w45_135,w0_45,w0_135,w45_90),2))));
	float3 DolP_Sky;
}


void PCMA(float3 * I0, float3 * I45, float3 * I90, float3 * I135, int W, int H)
{

	cudaEvent_t startKernel, stopKernel,startUpload,stopUpload;
	float computeTime = 0.0,uploadTime=0.0;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);
	cudaEventCreate(&startUpload);
	cudaEventCreate(&stopUpload);
	clock_t  tempS, tempE;
	//计算网格大小
	dim3 blockSize(TX, TY);
	int bx = (W + blockSize.x - 1) / blockSize.x;
	int by = (H + blockSize.y - 1) / blockSize.y;
	dim3 gridSize(bx, by);
	//申请变量空间
	float3 *out, *d_out, *i0,*i45,*i90,*i135;
	cudaEventRecord(startUpload);
	out = (float3*)calloc(W*H ,sizeof(float3));
	cudaMalloc(&d_out, W*H*sizeof(float3));
	cudaMalloc(&i0, W*H * sizeof(float3));
	cudaMalloc(&i45, W*H * sizeof(float3));
	cudaMalloc(&i90, W*H * sizeof(float3));
	cudaMalloc(&i135, W*H * sizeof(float3));

	//CUP->GPU
	cudaMemcpy(i0, I0, W*H * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(i45, I45, W*H * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(i90, I90, W*H * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(i135, I135, W*H * sizeof(float3), cudaMemcpyHostToDevice);

	cudaEventRecord(stopUpload);
	cudaEventSynchronize(stopUpload);
	cudaEventElapsedTime(&uploadTime, startUpload, stopUpload);
	cout << "数据上传:" << uploadTime << "ms" << endl;

	//GPU运行计时
	
	cudaEventRecord(startKernel);
	
	//执行核函数
	computeKernel << <gridSize, blockSize >> > (d_out, i0,i45, i90, i135, W, H);
	
	cudaEventRecord(stopKernel);
	cudaEventSynchronize(stopKernel);
	cudaEventElapsedTime(&computeTime, startKernel, stopKernel);
	cout << "GPU运算:" << computeTime << "ms" << endl;
	tempS = clock();
	cudaMemcpy(out, d_out, W*H *sizeof(float), cudaMemcpyDeviceToHost);
	
	tempE = clock();
	cout << "数据下载:" << double(tempE - tempS) / CLOCKS_PER_SEC *1000 << "ms" << endl;
	
	
	cudaEventDestroy(startKernel);
	cudaEventDestroy(stopKernel);
	

	//释放空间
	cudaFree(d_out);
	cudaFree(i0);
	cudaFree(i45);
	cudaFree(i90);
	cudaFree(i135);
	free(out);

	
}

__device__ float3 sqrt(float3 a)
{
	float3 t;
	t.x = sqrt(a.x);
	t.y = sqrt(a.y);
	t.y = sqrt(a.y);
	return t;
}

__device__ float3 abs(float3 a)
{
	float3 t;
	t.x = a.x >= 0 ? a.x : -a.x;
	t.y = a.y >= 0 ? a.y : -a.y;
	t.z = a.z >= 0 ? a.z : -a.z;
	return t;
}

__device__ float3 multiply(float3 a, float3 b)
{

	float3 t;
	t.x = a.x*b.x;
	t.y = a.y*b.y;
	t.z = a.z*b.z;
	return t;

}

__device__ float3 multiply(float3 a, float b)
{
	float3 t;
	t.x = a.x*b;
	t.y = a.y*b;
	t.z = a.z*b;
	return t;
}

__device__ float3 add(float3 a, float3 b)
{
	float3 t;
	t.x = a.x + b.x;
	t.y = a.y + b.y;
	t.z = a.z + b.z;
	return t;
}

__device__ float3 add(float3 a, float3 b, float3 c, float3 d, float3 e)
{
	float3 t;
	t = add(a, b);
	t = add(t, c);
	t = add(t, d);
	t = add(t, e);
	return t;
}

__device__ float3 subtract(float3 a, float3 b)
{
	float3 t;
	t.x = a.x - b.x;
	t.y = a.y - b.y;
	t.z = a.z - b.z;
	return t;
}

__device__ float3 subtract(float3 a, float3 b, float3 c, float3 d)
{
	float3 t;
	t = subtract(a ,b);
	t = subtract(t, c);
	t = subtract(t, d);
	return t;
}

