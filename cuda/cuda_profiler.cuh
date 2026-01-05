#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_profiler_init();
void cuda_profiler_record_start(const char* name, cudaStream_t stream);
void cuda_profiler_record_end(const char* name, cudaStream_t stream);
void cuda_profiler_print_results();

#ifdef __cplusplus
}
#endif
