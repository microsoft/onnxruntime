#include <cuda.h>
#include <stdint.h>
#include <assert.h>
#include "contrib_ops/cuda/sparse/sparse_attention_trition/sparse_attention_api.h"

// Dispatcher files are generated.
#include "contrib_ops/cuda/sparse/sparse_attention_trition/sparse_attention_dispatcher_fp16_sm80.h"
#include "contrib_ops/cuda/sparse/sparse_attention_trition/sparse_attention_dispatcher_bf16_sm80.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

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

bool is_supported_sparse_attention(const cudaDeviceProp& dprops) {
  return dprops.major == 8;
}

bool is_supported_sparse_attention(int head_size, int sparse_block_size) {
  return head_size == 128 && sparse_block_size == 64;
}

// -----------------------------------------------------------------------
// FP16

Status run_sparse_attention_fp16(SparseAttentionParams& params) {
  int algo_id = get_algo_id(params);
  if (algo_id < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "no algo found for the parameters");
  }

  // Right now we only support sm_8x.
  // If we want to support more architectures, we need to dispatch according to SM.
  return sparse_attention_fp16_sm80(params, algo_id);
}

static std::once_flag load_sparse_attention_fp16_flag;

void load_sparse_attention_fp16(void) {
  // Right now we only support sm_8x.
  // If we want to support more architectures, we need to dispatch according to SM.
  std::call_once(load_sparse_attention_fp16_flag, load_sparse_attention_fp16_sm80);
}

void unload_sparse_attention_fp16(void) {
  unload_sparse_attention_fp16_sm80();
}

// -----------------------------------------------------------------------
// BF16

Status run_sparse_attention_bf16(SparseAttentionParams& params) {
  int algo_id = get_algo_id(params);
  if (algo_id < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "no algo found for the parameters");
  }

  return sparse_attention_bf16_sm80(params, algo_id);
}

static std::once_flag load_sparse_attention_bf16_flag;

void load_sparse_attention_bf16(void) {
  std::call_once(load_sparse_attention_bf16_flag, load_sparse_attention_fp16_sm80);
}

void unload_sparse_attention_bf16(void) {
  unload_sparse_attention_bf16_sm80();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
