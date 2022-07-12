CUDA= /usr/local/cuda
CC=$(CUDA)/bin/nvcc
OBJDUMP=$(CUDA)/bin/cuobjdump
CFLAGS = -O3

all: stream_cuda

%.o: %.cu
	$(CC) -c $< -o $@

stream_cuda: stream.o maxflops.o cache.o
	$(CC) $(CFLAGS) $^ -o ./bin/stream_cuda
	$(OBJDUMP) -sass ./maxflops.o > maxflops.sass
	grep -c "FFMA" maxflops.sass > maxflops.count || true
	grep -c -e "FADD" -e "FMUL" maxflops.sass >> maxflops.count || true

clean:
	rm -f ./bin/stream_cuda *.o
