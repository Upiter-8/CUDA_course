#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

void printA(const char* filename, double* a, int size){
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

int main(int argc, char *argv[]){
    int nx, ny, nz;
    if (argc > 2){
	int ll;
        sscanf(argv[1], "%d", &ll);
        nx = ny = nz = ll;
    }
    else
	nx = ny = nz = 900;

    double maxeps;
    double *a;
    int it, itmax, i, j, k;
    double startt, endt;
    maxeps = 0.01;
    itmax = 10;
    a = (double*)malloc(nx * ny * nz * sizeof(double));
    
    #pragma omp parallel for
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[i*ny*nx + j*nx + k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[i*ny*nx + j*nx + k] = 0;

    startt = omp_get_wtime();

    for (it = 1; it <= itmax; it++){
        double eps = 0;
	
	#pragma omp parallel for collapse(2)
        for (j = 1; j < ny - 1; j++)
            for (k = 1; k < nz - 1; k++)
                for (i = 1; i < nx - 1; i++)
                    a[i*ny*nx + j*nx + k] = (a[(i-1)*ny*nx + j*nx + k] + a[(i+1)*ny*nx + j*nx + k]) / 2;

	#pragma omp parallel for collapse(2)
        for (i = 1; i < nx - 1; i++)
            for (k = 1; k < nz - 1; k++)
                for (j = 1; j < ny - 1; j++)
                    a[i*ny*nx + j*nx + k] = (a[i*ny*nx + (j-1)*nx + k] + a[i*ny*nx + (j+1)*nx + k]) / 2;

	#pragma omp parallel for collapse(2) reduction(max:eps)
        for (i = 1; i < nx - 1; i++)
            for (j = 1; j < ny - 1; j++)
                for (k = 1; k < nz - 1; k++){
                    double tmp1 = (a[i*ny*nx + j*nx + k-1] + a[i*ny*nx + j*nx + k+1]) / 2;
                    double tmp2 = fabs(a[i*ny*nx + j*nx + k] - tmp1);
                    eps = Max(eps, tmp2);
                    a[i*ny*nx + j*nx + k] = tmp1;
                }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }

    endt = omp_get_wtime();

    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2lf\n", endt - startt);
    printf(" Operation type  =   double precision\n");
    printf(" All threads     =       %12d\n", omp_get_max_threads());
    printf(" END OF ADI Benchmark\n");

    if (argc == 3){
        printA("cpu_output.txt", a, nx*ny*nz);
    }
    
    free(a);

    return 0;
}
