#include <cuda.h>
#include <stdint.h>
#include <assert.h>
#include "contrib_ops/cuda/sparse/sparse_attention_trition_cubin/sparse_attention_api.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// launcher for: fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2
extern Status fbsa_sm80_fp16_7528cbef(SparseAttentionParams& params);

Status fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_7528cbef(params);
}

// load for: fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2
extern Status load_fbsa_sm80_fp16_7528cbef();
Status load_fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2() {
  return load_fbsa_sm80_fp16_7528cbef();
}

// unload for: fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2
extern Status unload_fbsa_sm80_fp16_7528cbef();
Status unload_fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2() {
  return unload_fbsa_sm80_fp16_7528cbef();
}

// launcher for: fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2
extern Status fbsa_sm80_fp16_e5e3ee05(SparseAttentionParams& params);

Status fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_e5e3ee05(params);
}

// load for: fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2
extern Status load_fbsa_sm80_fp16_e5e3ee05();
Status load_fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2() {
  return load_fbsa_sm80_fp16_e5e3ee05();
}

// unload for: fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2
extern Status unload_fbsa_sm80_fp16_e5e3ee05();
Status unload_fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2() {
  return unload_fbsa_sm80_fp16_e5e3ee05();
}

// launcher for: fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2
extern Status fbsa_sm80_fp16_721800fe(SparseAttentionParams& params);

Status fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_721800fe(params);
}

// load for: fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2
extern Status load_fbsa_sm80_fp16_721800fe();
Status load_fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2() {
  return load_fbsa_sm80_fp16_721800fe();
}

// unload for: fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2
extern Status unload_fbsa_sm80_fp16_721800fe();
Status unload_fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2() {
  return unload_fbsa_sm80_fp16_721800fe();
}

// launcher for: fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2
extern Status fbsa_sm80_fp16_d6436af3(SparseAttentionParams& params);

Status fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_d6436af3(params);
}

// load for: fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2
extern Status load_fbsa_sm80_fp16_d6436af3();
Status load_fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2() {
  return load_fbsa_sm80_fp16_d6436af3();
}

// unload for: fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2
extern Status unload_fbsa_sm80_fp16_d6436af3();
Status unload_fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2() {
  return unload_fbsa_sm80_fp16_d6436af3();
}

// launcher for: fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2
extern Status fbsa_sm80_fp16_0e745d1c(SparseAttentionParams& params);

Status fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_0e745d1c(params);
}

// load for: fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2
extern Status load_fbsa_sm80_fp16_0e745d1c();
Status load_fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2() {
  return load_fbsa_sm80_fp16_0e745d1c();
}

// unload for: fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2
extern Status unload_fbsa_sm80_fp16_0e745d1c();
Status unload_fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2() {
  return unload_fbsa_sm80_fp16_0e745d1c();
}

// launcher for: fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2
extern Status fbsa_sm80_fp16_3c9faa92(SparseAttentionParams& params);

Status fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_3c9faa92(params);
}

// load for: fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2
extern Status load_fbsa_sm80_fp16_3c9faa92();
Status load_fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2() {
  return load_fbsa_sm80_fp16_3c9faa92();
}

// unload for: fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2
extern Status unload_fbsa_sm80_fp16_3c9faa92();
Status unload_fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2() {
  return unload_fbsa_sm80_fp16_3c9faa92();
}

// launcher for: fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2
extern Status fbsa_sm80_fp16_10917ade(SparseAttentionParams& params);

Status fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_10917ade(params);
}

// load for: fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2
extern Status load_fbsa_sm80_fp16_10917ade();
Status load_fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2() {
  return load_fbsa_sm80_fp16_10917ade();
}

// unload for: fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2
extern Status unload_fbsa_sm80_fp16_10917ade();
Status unload_fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2() {
  return unload_fbsa_sm80_fp16_10917ade();
}

// launcher for: fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2
extern Status fbsa_sm80_fp16_4b5cfdb7(SparseAttentionParams& params);

Status fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2(SparseAttentionParams& params) {
  return fbsa_sm80_fp16_4b5cfdb7(params);
}

// load for: fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2
extern Status load_fbsa_sm80_fp16_4b5cfdb7();
Status load_fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2() {
  return load_fbsa_sm80_fp16_4b5cfdb7();
}

// unload for: fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2
extern Status unload_fbsa_sm80_fp16_4b5cfdb7();
Status unload_fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2() {
  return unload_fbsa_sm80_fp16_4b5cfdb7();
}

typedef Status (*kernel_func_t)(SparseAttentionParams& params);
kernel_func_t fbsa_sm80_fp16_kernels[] = {
    fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2,
    fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2,
    fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2,
    fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2,
    fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2,
    fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2,
    fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2,
    fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2,
};

int fbsa_sm80_fp16_get_num_algos(void) {
  return (int)sizeof(fbsa_sm80_fp16_kernels);
}

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

Status run_sparse_attention_fp16(SparseAttentionParams& params) {
  int algo_id = get_algo_id(params);
  if (algo_id < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "no algo found for the parameters");
    ;
  }

  assert(algo_id < (int)sizeof(fbsa_sm80_fp16_kernels));
  return fbsa_sm80_fp16_kernels[algo_id](params);
}

void TryLoadSparseAttentionKernel() {
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2());
  ORT_THROW_IF_ERROR(load_fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2());
}

static std::once_flag load_sparse_attention_kernel_flag;

void load_sparse_attention_fp16(void) {
  std::call_once(load_sparse_attention_kernel_flag, TryLoadSparseAttentionKernel);
}

Status unload_sparse_attention_fp16(void) {
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_16x0x64x0x64x2_warps1xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_16x0x64x1x64x2_warps1xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_16x1x64x0x64x2_warps1xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_16x1x64x1x64x2_warps1xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_64x0x64x0x64x2_warps4xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_64x0x64x1x64x2_warps4xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_64x1x64x0x64x2_warps4xstages2());
  ORT_RETURN_IF_ERROR(unload_fbsa_sm80_fp16_64x1x64x1x64x2_warps4xstages2());
  return Status::OK();
}

// void run_sparse_attention(){
//       struct {
//         void* out;
//         const void* q;
//         const void* k;
//         const void* v;
//         const void* layout_crow_ptr;
//         const void* layout_col_ptr;
//         int layout_crow_stride_h;
//         int layout_col_stride_h;
//         int num_layout;
//         float softmax_scale;
//         int stride_qb;
//         int stride_qh;
//         int stride_qm;
//         int stride_kb;
//         int stride_kh;
//         int stride_kn;
//         int stride_vb;
//         int stride_vh;
//         int stride_vn;
//         int stride_ob;
//         int stride_oh;
//         int stride_om;
//         int num_heads;
//         int total_sequence_length;
//         int past_sequence_length;
//       } args = {
//           params.output,
//           params.q,
//           params.k,
//           params.v,
//           params.layout_crow,
//           params.layout_col,
//           params.layout_crow_stride_h,
//           params.layout_col_stride_h,
//           params.num_layout,
//           params.softmax_scale,
//           params.num_heads * params.sequence_length * params.head_size,
//           params.sequence_length * params.head_size,
//           params.head_size,
//           params.num_heads * params.total_sequence_length * params.head_size,
//           params.total_sequence_length * params.head_size,
//           params.head_size,
//           params.num_heads * params.total_sequence_length * params.head_size,
//           params.total_sequence_length * params.head_size,
//           params.head_size,
//           params.num_heads * params.sequence_length * params.head_size,
//           params.sequence_length * params.head_size,
//           params.head_size,
//           params.num_heads,
//           params.total_sequence_length,
//           params.past_sequence_length};

//       int grid_0 = (params.sequence_length + block_m - 1) / block_m;
//       int grid_1 = params.batch_size * params.num_heads;
//       return onnxruntime::cuda::LaunchTritonKernel(params.StreamHandle(), i, grid_0, grid_1, 1, &args, sizeof(args));
//     };

// }

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
