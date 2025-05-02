#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define ITMAX 20
#define MAXEPS 0.5f
#define SIZEX 16
#define SIZEY 4
#define SIZEZ 4

typedef double real_t;

template<typename T>
__device__ static T MyatomicMax(T* address, T val) {
    if (sizeof(T) == sizeof(float)) {
        int* address_as_int = (int*)address;
        int vl1 = *address_as_int, vl2;
        do {
            vl2 = vl1;
            vl1 = atomicCAS(address_as_int, vl2,
                __float_as_int(fmaxf(val, __int_as_float(vl2))));
        } while (vl2 != vl1);
        return __int_as_float(vl1);
    } else if (sizeof(T) == sizeof(double)) {
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int vl1 = *address_as_ull, vl2;
        do {
            vl2 = vl1;
            vl1 = atomicCAS(address_as_ull, vl2,
                __double_as_longlong(fmax(val, __longlong_as_double(vl2))));
        } while (vl2 != vl1);
        return __longlong_as_double(vl1);
    }
    return 0;
}

__global__ void computeEpsAndCopy(real_t* __restrict A, const real_t* __restrict B, real_t* eps, int L) {
    __shared__ real_t shared_max[SIZEX*SIZEY*SIZEZ];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    __syncthreads();

    real_t my_max = 0.0;    

    if (i < L-1 && j < L-1 && k < L-1) {
	int idx = (k * L + j) * L + i;
        real_t b_val = B[idx];
	my_max = fabs(b_val - A[idx]);
	A[idx] = b_val;
    }
    
    shared_max[tid] = my_max;
    __syncthreads();

    for (int i = SIZEX*SIZEY*SIZEZ / 2; i > 0; i = i/2) {
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
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (i < L-1 && j < L-1 && k < L-1) {
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

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void printB(const char* filename, real_t* B, int size){
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%lf\n", B[i]);
    }

    fclose(file);
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
    else{
	int ll;
	sscanf(argv[1], "%d", &ll);
	L = ll;
    }

    real_t *d_A, *d_B, *d_eps;
    size_t matrix_size = L * L * L * sizeof(real_t);
    
    checkCudaError(cudaMalloc(&d_A, matrix_size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, matrix_size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_eps, sizeof(real_t)), "cudaMalloc d_esp failed");
    
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
    
    checkCudaError(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice), "Memcpy d_A failed");
    checkCudaError(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice), "Memcpy d_B failed");
    
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
        checkCudaError(cudaMemcpy(d_eps, &h_eps, sizeof(real_t), cudaMemcpyHostToDevice), "Memcpy d_eps failed");
        
	computeEpsAndCopy<<<gridSize, blockSize>>>(d_A, d_B, d_eps, L);
	checkCudaError(cudaGetLastError(), "computeEpsAndCopy failed");        

        computeNewValues<<<gridSize, blockSize>>>(d_A, d_B, L);
	checkCudaError(cudaGetLastError(), "computeNewValues failed");        
        
        checkCudaError(cudaMemcpy(&h_eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost), "Memcpy h_eps failed");
        
        printf(" IT = %4i   EPS = %14.7E\n", it, h_eps);
        if (h_eps < MAXEPS) break;
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

    if (argc == 3){
    	checkCudaError(cudaMemcpy(h_B, d_B, matrix_size, cudaMemcpyDeviceToHost), "Memcpy h_B failed");
	printB("gpu_output.txt", h_B, L*L*L);
    }   
 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_eps);
    free(h_A);
    free(h_B);
    
    return 0;
}
