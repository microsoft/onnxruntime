#include <cuda.h>
#include "contrib_ops/cuda/sparse/sparse_attention_v2/sparse_attention_v2_common.h"

using onnxruntime::Status;

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace sparse_attention_v2 {

bool is_supported_device(int sm);
bool is_supported_sparse_attention(int head_size, int sparse_block_size);

void load_sparse_attention_fp16(int sm);
void load_sparse_attention_bf16(int sm);

Status run_sparse_attention_fp16(SparseAttentionParams& params);
Status run_sparse_attention_bf16(SparseAttentionParams& params);

}  // namespace sparse_attention_v2
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
