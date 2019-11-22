#include "add_gelu_fusion.h"

#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    AddGeluFusion,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AddGeluFusion<float>);

template <typename T>
static void ComputeWithParallelFor(int64_t batch_sequence,
                                   int64_t bias_length,
                                   const T* __restrict input,
                                   const T* __restrict bias,
                                   T* __restrict output,
                                   T* __restrict temp_output_data,
                                   concurrency::ThreadPool* tp) {
  if (tp != nullptr) {
    int task_count = tp->NumThreads() + 1;
    tp->ParallelFor(task_count, [input,
                                 bias,
                                 output,
                                 temp_output_data,
                                 bias_length,
                                 batch_sequence,
                                 task_count](int32_t i) {
      int64_t elem_inx_start = i * batch_sequence / task_count;
      int64_t elem_inx_end = (i + 1) * batch_sequence / task_count;
      for (int64_t elem_inx = elem_inx_start; elem_inx < elem_inx_end; elem_inx++) {
        const T* input_start = input + elem_inx * bias_length;
        T* output_start = output + elem_inx * bias_length;
        T* temp_output_start = temp_output_data + elem_inx * bias_length;
        const T* input_src = input_start;
        const T* bias_src = bias;
        T* ouput_src = output_start;
        T* temp_output_src = temp_output_start;
        for (int64_t h = 0; h < bias_length; h++) {
          T value = *input_src++ + *bias_src++;
          *ouput_src++ = value * static_cast<T>(M_SQRT1_2);
          *temp_output_src++ = value * 0.5f;
        }

        MlasComputeErf(output_start, output_start, bias_length);

        ouput_src = output_start;
        temp_output_src = temp_output_start;
        for (int64_t h = 0; h < bias_length; h++, ouput_src++) {
          *ouput_src = *temp_output_src++ * (*ouput_src + 1.0f);
        }
      }
    });
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < batch_sequence; i++) {
      const T* input_start = input + i * bias_length;
      T* output_start = output + i * bias_length;
      T* temp_output_start = temp_output_data + i * bias_length;
      const T* input_src = input_start;
      const T* bias_src = bias;
      T* ouput_src = output_start;
      T* temp_output_src = temp_output_start;
      for (int64_t h = 0; h < bias_length; h++) {
        T value = *input_src++ + *bias_src++;
        *ouput_src++ = value * static_cast<T>(M_SQRT1_2);
        *temp_output_src++ = value * 0.5f;
      }

      MlasComputeErf(output_start, output_start, bias_length);

      ouput_src = output_start;
      temp_output_src = temp_output_start;
      for (int64_t h = 0; h < bias_length; h++, ouput_src++) {
        *ouput_src = *temp_output_src++ * (*ouput_src + 1.0f);
      }
    }
  }
}

template <typename T>
Status AddGeluFusion<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const auto input_dims = X->Shape().GetDims();

  const Tensor* B = ctx->Input<Tensor>(1);
  const auto bias_dims = B->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 is expected to have 1 dimensions, got ", bias_dims.size());
  }

  int64_t bias_length = bias_dims[0];
  if (bias_length != input_dims[input_dims.size() - 1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as the last dimension of input 0");
  }

  Tensor* Y = ctx->Output(0, X->Shape());

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr temp_data_buf_ptr = BufferUniquePtr(alloc->Alloc(sizeof(T) * X->Shape().Size()), BufferDeleter(alloc));
  T* temp_output_data = static_cast<T*>(temp_data_buf_ptr.get());

  const T* input = X->template Data<T>();
  const T* bias = B->template Data<T>();
  T* output = Y->template MutableData<T>();
  int64_t BS = X->Shape().Size() / bias_length;

  ComputeWithParallelFor(BS,
                         bias_length,
                         input,
                         bias,
                         output,
                         temp_output_data,
                         ctx->GetOperatorThreadPool());
  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
