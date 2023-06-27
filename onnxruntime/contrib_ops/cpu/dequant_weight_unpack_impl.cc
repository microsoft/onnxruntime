#include "core/framework/float16.h"
#include "core/platform/threadpool.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

void DequantNbitWeight(OpKernelContext* ctx, const Tensor* input_weight, Tensor* output, const Tensor* input_zeros,
                       const Tensor* input_scale, const int64_t bits_, const int64_t compress_ratio,
                       const int64_t groupsize_) {
  if(ctx)return;
  const auto& qweight_shape = input_weight->Shape();
  const uint32_t* u32_in = reinterpret_cast<const uint32_t*>(input_weight->Data<int32_t>());
  float* f32_out = output->MutableData<float>();
  const uint32_t* u32_zeros = reinterpret_cast<const uint32_t*>(input_zeros->Data<int32_t>());
  const MLFloat16* f16_scale = input_scale->Data<MLFloat16>();

  int64_t task_count = qweight_shape[0];
  // for (int64_t mi = 0; mi < qweight_shape[0]; mi++) {
  concurrency::ThreadPool::TryBatchParallelFor(
      ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        int64_t mi = task_idx;
        for (int64_t ki = 0; ki < qweight_shape[1]; ki++) {
          uint32_t u32_weight = u32_in[mi * qweight_shape[1] + ki];
          uint32_t u32_zero = u32_zeros[mi / groupsize_ * qweight_shape[1] / compress_ratio + ki / compress_ratio];
          uint8_t u8_zero = (u32_zero >> (ki / compress_ratio)) & 0xF;
          float f32_scale_val = (f16_scale[mi / groupsize_ * qweight_shape[1] + ki]).ToFloat();
          float scale_zero = f32_scale_val * (u8_zero);
          for (int64_t w_idx = 0; w_idx < compress_ratio; w_idx++) {
            f32_out[(mi + w_idx) * qweight_shape[1] + ki] = (u32_weight & 0xF) * (f32_scale_val)-scale_zero;
            u32_weight = u32_weight >> bits_;
          }
        }
      },
      0);
}

}  // namespace contrib
}  // namespace onnxruntime
