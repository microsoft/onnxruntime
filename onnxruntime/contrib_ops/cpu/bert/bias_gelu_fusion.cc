#include "bias_gelu_fusion.h"

#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include "contrib_ops/cpu/bert/fast_gelu_helper.h"

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
Status BiasGelu<T>::Compute(OpKernelContext* ctx) const {
  ORT_RETURN_IF_ERROR(fast_gelu::CheckInputs(ctx));

  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  Tensor* Y = ctx->Output(0, X->Shape());

  int64_t bias_len = B->Shape().Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr temp_data_buf_ptr = BufferUniquePtr(alloc->Alloc(SafeInt<size_t>(sizeof(T)) * X->Shape().Size()),
                                                      BufferDeleter(alloc));
  T* tmp_data = static_cast<T*>(temp_data_buf_ptr.get());

  const T* X_data = X->template Data<T>();
  const T* B_data = B->template Data<T>();
  T* Y_data = Y->template MutableData<T>();
  int64_t task_count = X->Shape().Size() / bias_len;

  concurrency::ThreadPool::TryBatchParallelFor(ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
                                               [&](ptrdiff_t task_idx) {
                                                 const T* p_input = X_data + task_idx * bias_len;
                                                 T* p_output = Y_data + task_idx * bias_len;
                                                 T* p_output_tmp = tmp_data + task_idx * bias_len;

                                                 for (int64_t h = 0; h < bias_len; h++) {
                                                   T value = p_input[h] + B_data[h];
                                                   p_output[h] = value * static_cast<T>(M_SQRT1_2);
                                                   p_output_tmp[h] = value * 0.5f;
                                                 }

                                                 MlasComputeErf(p_output, p_output, bias_len);

                                                 for (int64_t h = 0; h < bias_len; h++) {
                                                   p_output[h] = p_output_tmp[h] * (p_output[h] + 1.0f);
                                                 }
                                               }, 0);

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
