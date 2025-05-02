#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef double real_t;

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define SIZEX 32
#define SIZEY 32

int nx, ny, nz;

void init(real_t *a, int nx, int ny, int nz);
void checkCudaError(cudaError_t err, const char* msg);

template<typename T>
__device__ static T atomicMax(T* address, T val) {
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

__global__ void x_sweep(real_t* a, int nx, int ny, int nz) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        for (int i = 1; i < nx-1; i++) {
            int idx = i * ny * nz + j * nz + k;
            int idx_prev = (i-1) * ny * nz + j * nz + k;
            int idx_next = (i+1) * ny * nz + j * nz + k;

	    a[idx] = (a[idx_prev] + a[idx_next]) / 2;
        }
    }
}

__global__ void y_sweep(real_t* a, int nx, int ny, int nz) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= 1 && i < nx-1 && k >= 1 && k < nz-1) {
        for (int j = 1; j < ny-1; j++) {
            int idx = i * ny * nz + j * nz + k;
            int idx_prev = i * ny * nz + (j-1) * nz + k;
            int idx_next = i * ny * nz + (j+1) * nz + k;

            a[idx] = (a[idx_prev] + a[idx_next]) / 2;
        }
    }
}

__global__ void reorder_to_z_major(real_t* a, real_t* b, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            int idx_a = k * ny * nz + i * nz + j;         // [i][j][k]
            int idx_b = j * nx * ny + k * ny + i;         // [k][i][j]
            b[idx_b] = a[idx_a];
        }
    }
}

__global__ void z_sweep(real_t* b, int nx, int ny, int nz, real_t* d_eps) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    real_t local_eps = 0;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        for (int k = 1; k < nz - 1; k++) {
            int idx      = k * nx * ny + i * ny + j;     // b[k][i][j]
            int idx_prev = (k - 1) * nx * ny + i * ny + j;
            int idx_next = (k + 1) * nx * ny + i * ny + j;

            real_t tmp1 = (b[idx_prev] + b[idx_next]) / 2;
            real_t tmp2 = fabs(b[idx] - tmp1);
            local_eps = Max(local_eps, tmp2);
            b[idx] = tmp1;
        }
    }

    __shared__ real_t shared_eps[SIZEX * SIZEY];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    shared_eps[tid] = local_eps;
    __syncthreads();

    for (int s = SIZEX * SIZEY / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_eps[tid] = Max(shared_eps[tid], shared_eps[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(d_eps, shared_eps[0]);
    }
}

__global__ void reorder_from_z_major(real_t* b, real_t* a, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            int idx_b = i * nx * ny + k * ny + j;         // [k][i][j]
            int idx_a = k * ny * nz + j * nz + i;         // [i][j][k]
            a[idx_a] = b[idx_b];
        }
    }
}

void printA(const char* filename, real_t* a, int size){
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%lf\n", a[i]);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    real_t maxeps, eps;
    real_t *a, *d_a, *d_eps, *d_b;
    int it, itmax;
    cudaEvent_t start, stop;
    float elapsedTime;
    
    maxeps = (real_t)0.01;
    itmax = 10;
    
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t maxMatrixSize = freeMem / 2 * 0.9;
   
    if (argc == 1){
    	nx = (int)cbrt((double)maxMatrixSize / sizeof(real_t));
    	ny = nx;
    	nz = nx;
    }
    else {
	int ll;
        sscanf(argv[1], "%d", &ll);
	nx = ny = nz = ll;
    }
    
    size_t requiredMem = nx * ny * nz * sizeof(real_t) + sizeof(real_t); 
    while (requiredMem > freeMem && nx > 32) {
        nx = nx * 9 / 10;
        ny = nx;
        nz = nx;
        requiredMem = nx * ny * nz * sizeof(real_t) + sizeof(real_t);
    }
    
    a = (real_t*)malloc(nx * ny * nz * sizeof(real_t));
    if (!a) {
        printf("Host memory allocation failed\n");
        return 1;
    }
    
    init(a, nx, ny, nz);
    
    checkCudaError(cudaMalloc(&d_a, nx * ny * nz * sizeof(real_t)), "cudaMalloc d_a failed");
    checkCudaError(cudaMalloc(&d_b, nx * ny * nz * sizeof(real_t)), "cudaMalloc d_b failed");
    checkCudaError(cudaMalloc(&d_eps, sizeof(real_t)), "cudaMalloc d_eps failed");
    
    checkCudaError(cudaMemcpy(d_a, a, nx * ny * nz * sizeof(real_t), cudaMemcpyHostToDevice), "Initial memcpy failed");
    
    dim3 blockSize(SIZEX, SIZEY);
    dim3 gridSizeX((ny + blockSize.x - 1) / blockSize.x, (nz + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeY((nx + blockSize.x - 1) / blockSize.x, (nz + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeZ((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    for (it = 1; it <= itmax; it++) {
        eps = (real_t)0;
        checkCudaError(cudaMemcpy(d_eps, &eps, sizeof(real_t), cudaMemcpyHostToDevice), "cudaMemcpy d_eps failed");
        
        x_sweep<<<gridSizeX, blockSize>>>(d_a, nx, ny, nz);
        checkCudaError(cudaGetLastError(), "x_sweep failed");
        
        y_sweep<<<gridSizeY, blockSize>>>(d_a, nx, ny, nz);
        checkCudaError(cudaGetLastError(), "y_sweep failed");

	reorder_to_z_major<<<gridSizeZ, blockSize>>>(d_a, d_b, nx, ny, nz);
        checkCudaError(cudaGetLastError(), "reorder_to failed");
        
        z_sweep<<<gridSizeZ, blockSize>>>(d_b, nx, ny, nz, d_eps);
        checkCudaError(cudaGetLastError(), "z_sweep failed");

	
	reorder_from_z_major<<<gridSizeZ, blockSize>>>(d_b, d_a, nx, ny, nz);       
        checkCudaError(cudaGetLastError(), "reorder_from failed");

        checkCudaError(cudaMemcpy(&eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost), "cudaMemcpy eps failed");
        
	printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps) break;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    
    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2f\n", elapsedTime / 1000.0f);
    printf(" Operation type  =     %s\n", (sizeof(real_t) == sizeof(float)) ? "float" : "double");
    printf(" GPU Device: %s\n", prop.name);
    printf(" Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf(" GPU Memory used =     %.2f MB\n", (2 * nx * ny * nz * sizeof(real_t)) / (1024.0 * 1024.0));
    printf(" END OF ADI Benchmark\n");

    if (argc == 3){
        checkCudaError(cudaMemcpy(a, d_a, nx*ny*nz*sizeof(real_t), cudaMemcpyDeviceToHost), "Memcpy h_B failed");
        printA("gpu_output.txt", a, nx*ny*nz);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_eps);
    free(a);

    return 0;
}

void init(real_t *a, int nx, int ny, int nz) {
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                int idx = i * ny * nz + j * nz + k;
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[idx] = (real_t)10.0 * i / (nx - 1) + (real_t)10.0 * j / (ny - 1) + (real_t)10.0 * k / (nz - 1);
                else
                    a[idx] = (real_t)0;
            }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
