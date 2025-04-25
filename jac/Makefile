NVCC = nvcc
CC = gcc

CFLAGS = -fopenmp -std=c99 -O3
COMPARE_FLAGS = -std=c99 -O3

TARGETS = jaccpu jacgpu compare

all: $(TARGETS)

jaccpu: jaccpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

jacgpu: jacgpu.cu
	$(NVCC) -o $@ $<

compare: compare.c
	$(CC) $(COMPARE_FLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGETS) *.o *.txt

.PHONY: all clean

