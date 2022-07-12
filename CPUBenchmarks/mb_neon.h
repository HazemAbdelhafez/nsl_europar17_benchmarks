#ifndef MB_NEON_H
#define MB_NEON_H

#include "common.h"

typedef struct {
	int loops;
} data_neon;

data_neon NEON_DATA;

// Setup benchmark parameters
void prepare_neon_data(int loops);

// NEON SIMD (floating point)
void microbenchmark_neon(int thread_num);

#endif

