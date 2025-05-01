#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define MAX_LINE 512

bool compare(const char* file1, const char* file2) {
    FILE *f1 = fopen(file1, "r");
    FILE *f2 = fopen(file2, "r");

    if (f1 == NULL || f2 == NULL) {
	fprintf(stderr, "Could not open\n");
	if (f1) fclose(f1);
        if (f2) fclose(f2);
        return false;
    }

    double num1, num2;
    bool compare = true;

    while (1) {
        int res1 = fscanf(f1, "%lf", &num1);
        int res2 = fscanf(f2, "%lf", &num2);
	
	if (res1 == EOF && res2 == EOF) {
            break;
        }

	if (res1 == EOF || res2 == EOF || num1 != num2) {
            compare = false;
            break;
        }
    }

    fclose(f1);
    fclose(f2);

    return compare;

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

    snprintf(command1, sizeof(command1), "./jaccpu %d compare", L);
    snprintf(command2, sizeof(command2), "./jacgpu %d compare", L);

    printf("\nRun CPU version\n");
    system(command1);

    printf("\nRun GPU version\n");
    system(command2);

    bool flag = compare("cpu_output.txt", "gpu_output.txt");
    printf("--- Results compare ---\n");

    if (flag) {
        printf("Matrix matched\n");
    } else {
        printf("Matrix differ\n");
    }

    return 0;
}
