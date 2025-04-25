#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE 512

double get_time(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    double time = -1.0;

    while (fgets(line, MAX_LINE, fp)) {
        if (strstr(line, "Time in seconds")) {
            sscanf(line, " Time in seconds = %lf", &time);
        }
    }

    fclose(fp);
    return time;
}

double get_eps(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    double last_eps = -1.0;

    while (fgets(line, MAX_LINE, fp)) {
        if (strstr(line, "EPS")) {
            sscanf(line, " IT = %*d   EPS = %lf", &last_eps);
        }
    }

    fclose(fp);
    return last_eps;
}

int main(int argc, char** argv) {
    int L = 900;
    if (argc > 1){
	int ll;
	sscanf(argv[1], "%d", &ll);
	L = ll;
    }

    char command1[256];
    char command2[256];

    snprintf(command1, sizeof(command1), "./jaccpu %d > cpu_output.txt", L);
    snprintf(command2, sizeof(command2), "./jacgpu %d > gpu_output.txt", L);

    printf("Running CPU version...\n");
    system(command1);

    printf("Running GPU version...\n");
    system(command2);

    double time_cpu = get_time("cpu_output.txt");
    double time_gpu = get_time("gpu_output.txt");
    double eps_cpu = get_eps("cpu_output.txt");
    double eps_gpu = get_eps("gpu_output.txt");

    printf("\nComparison results\n");
    printf("CPU time: %.2f s\n", time_cpu);
    printf("GPU time: %.2f s\n", time_gpu);
    printf("Speedup (CPU/GPU): %.2fx\n", time_cpu / time_gpu);

    if (fabs(eps_cpu - eps_gpu) < 1e-6) {
        printf("Eps matched\n");
    } else {
        printf("Eps differ\n");
	printf("Eps CPU: %.8lf\n", eps_cpu);
    	printf("Eps GPU: %.8lf\n", eps_gpu);
    }

    return 0;
}
