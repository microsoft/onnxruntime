#include "bias_gelu_fusion.h"

#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    BiasGelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BiasGelu<float>);

template <typename T>
static void ComputeOneTask(int64_t task_idx,
                          int64_t bias_len,
                          const T* __restrict p_input,
                          const T* __restrict p_bias,
                          T* __restrict p_output,
                          T* __restrict p_output_tmp) {
  p_input = p_input + task_idx * bias_len;
  p_output = p_output + task_idx * bias_len;
  p_output_tmp = p_output_tmp + task_idx * bias_len;

  for (int64_t h = 0; h < bias_len; h++) {
    T value = p_input[h] + p_bias[h];
    p_output[h] = value * static_cast<T>(M_SQRT1_2);
    p_output_tmp[h] = value * 0.5f;
  }

  MlasComputeErf(p_output, p_output, bias_len);

  for (int64_t h = 0; h < bias_len; h++) {
    p_output[h] = p_output_tmp[h] * (p_output[h] + 1.0f);
  }
}

template <typename T>
Status BiasGelu<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const auto input_dims = X->Shape().GetDims();

  const Tensor* B = ctx->Input<Tensor>(1);
  const auto bias_dims = B->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 is expected to have 1 dimensions, got ", bias_dims.size());
  }

  int64_t bias_len = bias_dims[0];
  if (bias_len != input_dims[input_dims.size() - 1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as the last dimension of input 0");
  }

  Tensor* Y = ctx->Output(0, X->Shape());

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr temp_data_buf_ptr = BufferUniquePtr(alloc->Alloc(sizeof(T) * X->Shape().Size()), BufferDeleter(alloc));
  T* p_output_tmp = static_cast<T*>(temp_data_buf_ptr.get());

  const T* p_input = X->template Data<T>();
  const T* p_bias = B->template Data<T>();
  T* p_output = Y->template MutableData<T>();
  int64_t task_count = X->Shape().Size() / bias_len;

  if (concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool()) {
    int block_count = tp->NumThreads() + 1;
    tp->ParallelFor(block_count, [p_input,
                                  p_bias,
                                  p_output,
                                  p_output_tmp,
                                  bias_len,
                                  task_count,
                                  block_count](int32_t blk_idx) {
      int64_t task_start = blk_idx * task_count / block_count;
      int64_t task_end = (blk_idx + 1) * task_count / block_count;
      for (int64_t task_idx = task_start; task_idx < task_end; task_idx++) {
        ComputeOneTask(task_idx, bias_len, p_input, p_bias, p_output, p_output_tmp);
      }
    });
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t task_idx = 0; task_idx < task_count; task_idx++) {
      ComputeOneTask(task_idx, bias_len, p_input, p_bias, p_output, p_output_tmp);
    }
  }

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
