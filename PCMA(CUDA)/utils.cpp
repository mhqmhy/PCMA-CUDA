#include "utils.h"

void mallocArray(float3 **&a, int W, int H)//按引用传递，否则是临时变量a
{
	a = new float3*[H];
	for (int i = 0; i < H; i++) {
		a[i] = new float3[W];
	}
}

void freeArray(float3 **& a,int W,int H)
{
	for (int i = 0; i < H; i++) {
		delete[] a[i];
	}
	delete[] a;
}

void Mat2Array(float3 **& dst, const Mat & src, const int W, const int H)
{
	for (int row = 0; row < H; row++) {
	
		for (int col = 0; col < W; col++) {
			//blue
			float b = *(src.data + src.step[0] * row + src.step[1] * col);
			//green  elemSize1指的是一个通道所占字节数
			float g = *(src.data + src.step[0] * row + src.step[1] * col+src.elemSize1());
			//red
			float r = *(src.data + src.step[0] * row + src.step[1] * col+ src.elemSize1()*2);

			dst[row][col].x = r;
			dst[row][col].y = g;
			dst[row][col].z = b;
		}
	}
}



__global__ void computeKernel(float3* &out, float3 **& I0, float3 **& I45, float3 **& I90, float3 ** I135, int W, int H)
{
	//利用线程索引和块索引计算数组下标
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	const int index = r*W + H;
	//计算S0
	float3 s0;
	s0.x = (I0[r][c].x + I45[r][c].x + I90[r][c].x + I135[r][c].x)/2;
	s0.y = (I0[r][c].y + I45[r][c].y + I90[r][c].y + I135[r][c].y)/2;
	s0.z = (I0[r][c].z + I45[r][c].z + I90[r][c].z + I135[r][c].z)/2;
	//计算w0_45,w0_90,w0_135,w45_90,w45_135,w90_135
	float3 w0_45, w0_90, w0_135, w45_90, w45_135, w90_135;
	w0_45 = multiply(I0[r][c], I45[r][c]);
	w0_90 = multiply(I0[r][c], I90[r][c]);
	w0_135 = multiply(I0[r][c], I135[r][c]);
	w45_90 = multiply(I45[r][c], I90[r][c]);
	w45_135 = multiply(I45[r][c], I135[r][c]);
	w90_135 = multiply(I90[r][c], I135[r][c]);
	
	//-ε45,-ε'45,γ45,-γ45,-γ'45
	float3 epsilon45_1,epsilon45_2,gamma45_1,gamma45_2,gamma45_3;
	
}


void PCMA(float3 **& I0, float3 **& I45, float3 **& I90, float3 ** I135, int W, int H)
{
	clock_t  tempS, tempE;
	//计算网格大小
	dim3 blockSize(TX, TY);
	int bx = (W + blockSize.x - 1) / blockSize.x;
	int by = (H + blockSize.y - 1) / blockSize.y;
	dim3 gridSize(bx, by);
	//申请变量空间
	float3 **out, *d_out;
	mallocArray(out, W, H);
	cudaMalloc(&d_out, W*H * sizeof(float3));
	tempS = clock();
	//执行核函数
	computeKernel << <gridSize, blockSize >> > (d_out, I0, I45, I90, I135, W, H,test);
	cudaMemcpy(out, d_out, W*H * sizeof(float3), cudaMemcpyDeviceToHost);
	tempE = clock();
	cout << "纯算法(不考虑除上传下载外其他开销):" << double(tempE - tempS) / CLOCKS_PER_SEC *1000 << "ms" << endl;
	
	//释放空间
	cudaFree(d_out);
	freeArray(out, W, H);
	
	
}

__device__ float3 multiply(float3 a, float3 b)
{

	float3 t;
	t.x = a.x*b.x;
	t.y = a.y*b.y;
	t.z = a.z*b.z;
	return t;

	//return __device__ double3();
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

__device__ float3 subtract(float3 a, float3 b)
{
	float3 t;
	t.x = a.x - b.x;
	t.y = a.y - b.y;
	t.z = a.z - b.z;
	return t;
}
