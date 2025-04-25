#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define ITMAX 20
#define MAXEPS 0.5f
#define SIZEX 8
#define SIZEY 8
#define SIZEZ 8

typedef double real_t;

template<typename T>
__device__ static T MyatomicMax(T* address, T val) {
    if (sizeof(T) == sizeof(float)) {
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    } else if (sizeof(T) == sizeof(double)) {
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
    return 0;
}

__global__ void computeEpsAndCopy(real_t* __restrict A, const real_t* __restrict B, real_t* eps, int L) {
    __shared__ real_t shared_max[SIZEX*SIZEY*SIZEZ];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    __syncthreads();
    
    real_t my_max = 0.0;

    if (i > 0 && i < L-1 && j > 0 && j < L-1 && k > 0 && k < L-1) {
	int idx = (k * L + j) * L + i;
        real_t b_val = B[idx];
	real_t tmp = fabs(b_val - A[idx]);
        my_max = tmp;
	A[idx] = b_val;
    }
    
    shared_max[tid] = my_max;
    __syncthreads();

    for (int i = 256; i > 0; i = i/2) {
        if (tid < i) {
	    shared_max[tid] = fmax(shared_max[tid], shared_max[tid+i]);
        }
        __syncthreads();
    }

    __syncthreads();
    
    if (tid == 0){
	MyatomicMax(eps, shared_max[0]);
    }
}

__global__ void computeNewValues(const real_t* __restrict A, __restrict real_t* B, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i > 0 && i < L-1 && j > 0 && j < L-1 && k > 0 && k < L-1) {
	int idx = (k * L + j) * L + i;
        B[idx] = (
            A[idx+1] + 
            A[idx-1] + 
            A[idx + L] + 
            A[idx - L] + 
            A[idx + L*L] + 
            A[idx - L*L]
        ) / 6.0f;
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    int L;
    size_t element_size = sizeof(real_t);

    if (argc == 1){
    	size_t available_mem = free_mem * 0.9;
    	size_t array_size_per_matrix = available_mem / (2 * element_size);
    
    	L = (int)cbrt((double)array_size_per_matrix);
    	L = (L / 32) * 32;
    	if (L < 32) L = 32; 
    }
    else L = 900;    

    real_t *d_A, *d_B, *d_eps;
    size_t matrix_size = L * L * L * sizeof(real_t);
    
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_B, matrix_size);
    cudaMalloc(&d_eps, sizeof(real_t));
    
    real_t *h_A = (real_t*)malloc(matrix_size);
    real_t *h_B = (real_t*)malloc(matrix_size);
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                h_A[(i * L + j) * L + k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) {
                    h_B[(i * L + j) * L + k] = 0;
                } else {
                    h_B[(i * L + j) * L + k] = 4 + i + j + k;
                }
            }
        }
    }
    
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(SIZEX, SIZEY, SIZEZ);
    dim3 gridSize(
        (L + blockSize.x - 1) / blockSize.x,
        (L + blockSize.y - 1) / blockSize.y,
        (L + blockSize.z - 1) / blockSize.z
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int it = 1; it <= ITMAX; it++) {
        real_t h_eps = 0;
        cudaMemcpy(d_eps, &h_eps, sizeof(real_t), cudaMemcpyHostToDevice);
        
	computeEpsAndCopy<<<gridSize, blockSize>>>(d_A, d_B, d_eps, L);
        
        computeNewValues<<<gridSize, blockSize>>>(d_A, d_B, L);
        
        cudaMemcpy(&h_eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost);
        
        printf(" IT = %4i   EPS = %14.7E\n", it, h_eps);
        //if (h_eps < MAXEPS) break;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2f\n", milliseconds / 1000.0f);
    printf(" Operation type  =     %s\n", (sizeof(real_t) == sizeof(float)) ? "float" : "double");
    printf(" GPU Device: %s\n", prop.name);
    printf(" Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf(" GPU Memory used =     %.2f MB\n", (2.0 * L * L * L * sizeof(real_t)) / (1024.0 * 1024.0));
    printf(" END OF Jacobi3D Benchmark\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_eps);
    free(h_A);
    free(h_B);
    
    return 0;
}
