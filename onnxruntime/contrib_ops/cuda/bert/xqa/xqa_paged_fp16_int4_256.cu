#if defined(USE_INT4_KV_CACHE)
#define HEAD_ELEMS 256
#define HEAD_DIM_NAMESPACE H256
#define XQA_PAGED_CACHE_ELEM 0
#define XQA_PAGED_INT4 1
#define XQA_PAGED_GROUP6_ONLY 1
#define XQA_PAGED_INPUT_FP16 1
#define XQA_PAGED_QUERY_T half
#define XQA_PAGED_FAMILY fp16_int4
#define XQA_PAGED_LAUNCH_FN LaunchXQAPagedInt4Kernel

#include "xqa_paged_loader_impl.cuh"
#endif