#include <cuda.h>
#include <stdint.h>
#include <assert.h>
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_v1_api.h"

// Dispatcher files are generated.
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_dispatcher_fp16_sm75.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_dispatcher_fp16_sm80.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_dispatcher_bf16_sm80.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace sparse_attention_v1 {

int get_algo_id(SparseAttentionParams& params) {
  int block_n = params.kernel_block_size;
  int block_m = block_n;
  bool even_m = (params.sequence_length % block_m == 0);
  bool even_n = (params.total_sequence_length % block_n == 0);

  if (params.head_size == 128) {
    if (block_m == 16) {
      if (!even_m) {
        return even_n ? 1 : 0;
      } else {
        return even_n ? 3 : 2;
      }
    } else if (block_m == 64) {
      if (!even_m) {
        return even_n ? 5 : 4;
      } else {
        return even_n ? 7 : 6;
      }
    }
  }

  return -1;
}

bool is_supported_device(const cudaDeviceProp& dprops) {
  return dprops.major == 8 || (dprops.major == 7 && dprops.minor == 5);
}

bool is_supported_sparse_attention(int head_size, int sparse_block_size) {
  return head_size == 128 && sparse_block_size == 64;
}

static std::once_flag load_sparse_attention_sm75_fp16_flag;
static std::once_flag load_sparse_attention_sm80_fp16_flag;
static std::once_flag load_sparse_attention_sm80_bf16_flag;

void load_sparse_attention_fp16(int sm) {
  if (sm == 75) {
    std::call_once(load_sparse_attention_sm75_fp16_flag, load_sparse_attention_fp16_sm75);
  } else {
    assert(sm == 80 || sm == 86 || sm == 89);
    std::call_once(load_sparse_attention_sm80_fp16_flag, load_sparse_attention_fp16_sm80);
  }
}

void load_sparse_attention_bf16([[maybe_unused]] int sm) {
  if (sm == 80 || sm == 86 || sm == 89) {
    std::call_once(load_sparse_attention_sm80_bf16_flag, load_sparse_attention_bf16_sm80);
  } else {
    assert(false);
  }
}

Status run_sparse_attention_bf16(SparseAttentionParams& params) {
  int algo_id = get_algo_id(params);
  if (algo_id < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "no algo found for the parameters");
  }

  // bfloat16 requires sm >= 80
  assert(params.sm == 80 || params.sm == 86 || params.sm == 89);
  return sparse_attention_bf16_sm80(params, algo_id);
}

Status run_sparse_attention_fp16(SparseAttentionParams& params) {
  int algo_id = get_algo_id(params);
  if (algo_id < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "no algo found for the parameters");
  }

  if (params.sm == 75) {
    return sparse_attention_fp16_sm75(params, algo_id);
  } else {
    assert(params.sm == 80 || params.sm == 86 || params.sm == 89);
    return sparse_attention_fp16_sm80(params, algo_id);
  }
}

}  // namespace sparse_attention_v1
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
