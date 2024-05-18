#define CPU_CAPABILITY_AVX512 TRUE
#define CPU_CAPABILITY CPU_CAPABILITY_AVX512
#include "contrib_ops/cpu/flash_attention/flash_attention.h"
#undef CPU_CAPABILITY
#undef CPU_CAPABILITY_AVX512
