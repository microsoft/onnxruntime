// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

// Environment variable to disable fused attention kernel. Default is false.
constexpr const char* kDisableFusedAttention = "ORT_DISABLE_FUSED_ATTENTION";

static inline bool HasFusedFp16Kernel(int sm, int head_size, int sequence_length) {
  if (!(sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86)) {
    return false;
  }

  if (head_size != 64) {
    return false;
  }

  // For sequence length 512, SM86 could fall back to SM80.
  // In our test, T4 GPU has no enough shared memory to load fmha_v2_fp16_512_64_sm75_kernel so we removed it.
  if (!(sequence_length == 64 || sequence_length == 128 || sequence_length == 192 ||
        sequence_length == 256 || sequence_length == 384 || (sequence_length == 512 && sm >= kSM_80))) {
    return false;
  }

  return true;
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, false, false) {
  disable_fused_runner_ = sizeof(T) != 2 || ParseEnvironmentVariableWithDefault<bool>(kDisableFusedAttention, false);
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* extra_add_qk = context->Input<Tensor>(5);
  const Tensor* key = context->Input<Tensor>(6);
  const Tensor* value = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  nullptr == weights ? nullptr : &(weights->Shape()),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  extra_add_qk,
                                  key,
                                  value,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{2, parameters.batch_size, parameters.num_heads,
                                    parameters.total_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;
  bool use_fused_runner = (!disable_fused_runner_ &&
                           nullptr != mask_index && mask_index->Shape().NumDimensions() == 1 &&
                           nullptr == past &&
                           nullptr == present &&
                           nullptr == extra_add_qk &&
                           !is_unidirectional_ &&
                           parameters.hidden_size == parameters.v_hidden_size &&
                           parameters.sequence_length == parameters.kv_sequence_length &&
                           HasFusedFp16Kernel(sm, parameters.head_size, sequence_length));

  MHARunner* fused_runner = nullptr;
  if (use_fused_runner) {
    if (nullptr == fused_fp16_runner_.get()) {
      fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(num_heads_, parameters.head_size, sm));
    }
    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    if (fused_fp16_runner_->isValid(sequence_length)) {
      fused_runner = fused_fp16_runner_.get();
    }
  }

  cublasHandle_t cublas = CublasHandle();
  constexpr size_t element_size = sizeof(T);

  IAllocatorUniquePtr<T> gemm_buffer;
  if (weights != nullptr) {
    // Use GEMM for fully connection.
    int m = batch_size * sequence_length;
    int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
    int k = parameters.input_hidden_size;
    size_t gemm_buffer_size = static_cast<size_t>(batch_size) * sequence_length * n * element_size;
    gemm_buffer = GetScratchBuffer<T>(gemm_buffer_size);

    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT one = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);

    // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
        reinterpret_cast<const CudaT*>(input->Data<T>()), k,
        &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));
  }

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner);

  auto work_space = GetScratchBuffer<void>(workSpaceSize);

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = (nullptr == weights) ? nullptr : reinterpret_cast<const CudaT*>(gemm_buffer.get());
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = (nullptr != weights) ? nullptr : reinterpret_cast<const CudaT*>(input->Data<T>());
  data.key = (nullptr == key) ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (nullptr == value) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = (nullptr == mask_index) ? nullptr : mask_index->Data<int>();
  data.mask_index_dims = (nullptr == mask_index) ? gsl::span<const int64_t>() : mask_index->Shape().GetDims();
  data.past = (nullptr == past) ? nullptr : reinterpret_cast<const CudaT*>(past->Data<T>());
  data.extra_add_qk = (nullptr == extra_add_qk) ? nullptr : reinterpret_cast<const CudaT*>(extra_add_qk->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = (nullptr == present) ? nullptr : reinterpret_cast<CudaT*>(present->MutableData<T>());

  return QkvToContext<CudaT>(device_prop, cublas, Stream(), parameters, data, reinterpret_cast<void*>(fused_runner));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
