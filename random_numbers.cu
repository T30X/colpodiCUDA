#include <iostream>
#include "randomgen.h"

using namespace std;

__global__ void Kernel(unsigned int* S, double* W){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	Randomgen obj(S[4*i],S[4*i+1],S[4*i+2],S[4*i+3]);
	W[i]=obj.Rand();
}

int main(){
	srand(17);
	unsigned int *_S;
	double *_W;
	int dim_vec=1024;

	unsigned int *S = new unsigned int[4*dim_vec];
	double *W = new double[dim_vec];

	size_t sizeS = 4*dim_vec * sizeof(unsigned int);
	size_t sizeW = dim_vec * sizeof(double);

	cudaMalloc((void**)& _S,sizeS);
	cudaMalloc((void**)& _W,sizeW);

	for(int i=0; i<4*dim_vec; i++){
		S[i]=rand()+128;
	}

	cudaMemcpy(_S, S, sizeS, cudaMemcpyHostToDevice);

	int blockSize=512;
	int gridSize = (dim_vec + blockSize - 1) / blockSize;

	Kernel<<<gridSize, blockSize>>>(_S, _W);

	cudaMemcpy(W, _W, sizeW, cudaMemcpyDeviceToHost);

	for(int i=0; i<10; i++){
		cout<<W[i]<<endl;
	}

	cudaFree(_S);
	cudaFree(_W);

	delete[] S;
	delete[] W;

    return 0;
}
