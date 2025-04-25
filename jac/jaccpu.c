#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define ITMAX 20
#define MAXEPS 0.5

int main(int argc, char** argv) {
    int L = 900;

    if (argc > 1){
	int ll;
	sscanf(argv[1], "%d", &ll);
	L = ll;
    }
    
    double *A = (double*)malloc(L * L * L * sizeof(double));
    double *B = (double*)malloc(L * L * L * sizeof(double));
    
    if (!A || !B) {
        printf("Failed in malloc\n");
        return 1;
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                A[i*L*L + j*L + k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L-1 || j == L-1 || k == L-1) {
                    B[i*L*L + j*L + k] = 0;
                } else {
                    B[i*L*L + j*L + k] = 4 + i + j + k;
                }
            }
        }
    }

    double start = omp_get_wtime();
    
    for (int it = 1; it <= ITMAX; it++) {
        double eps = 0;
        
        #pragma omp parallel for collapse(3) reduction(max:eps)
        for (int i = 1; i < L-1; i++) {
            for (int j = 1; j < L-1; j++) {
                for (int k = 1; k < L-1; k++) {
                    double tmp = fabs(B[i*L*L + j*L + k] - A[i*L*L + j*L + k]);
                    eps = Max(tmp, eps);
                    A[i*L*L + j*L + k] = B[i*L*L + j*L + k];
                }
            }
        }

        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L-1; i++) {
            for (int j = 1; j < L-1; j++) {
                for (int k = 1; k < L-1; k++) {
                    B[i*L*L + j*L + k] = (
                        A[(i-1)*L*L + j*L + k] + 
                        A[i*L*L + (j-1)*L + k] + 
                        A[i*L*L + j*L + (k-1)] + 
                        A[i*L*L + j*L + (k+1)] + 
                        A[i*L*L + (j+1)*L + k] + 
                        A[(i+1)*L*L + j*L + k]
                    ) / 6.0;
                }
            }
        }

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS) break;
    }

    double end = omp_get_wtime();

    printf(" Jacobi3D CPU Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2f\n", end - start);
    printf(" Operation type  =     double\n");
    printf(" All threads     =       %12d\n", omp_get_max_threads());
    printf(" END OF Jacobi3D Benchmark\n");

    free(A);
    free(B);
    return 0;
}
