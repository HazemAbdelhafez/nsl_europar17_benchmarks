CC = gcc
CFLAGS = -O2
CLIBS = -mfpu=neon -lopenblas -lpapi -fopenmp

all: mb roofline

mb: main.c mb_*.c benchmark.c common.c
	$(CC) $(CFLAGS) main.c mb_*.c benchmark.c common.c $(CLIBS) -o ./bin/mb

roofline: roofline.c mb_*.c benchmark.c common.c
	$(CC) $(CFLAGS) roofline.c mb_*.c benchmark.c common.c $(CLIBS) -o ./bin/roofline

clean:
	rm -f ./bin/mm ./bin/mb ./bin/roofline *.o
