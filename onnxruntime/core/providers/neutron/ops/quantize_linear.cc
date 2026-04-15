// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include "core/providers/neutron/ops/quantize_linear.h"
#include "core/framework/element_type_lists.h"
#include "core/util/qmath.h"
#include "core/providers/neutron/ops/common.h"
#include "core/providers/neutron/neutron_fwd.h"
#include "core/providers/neutron/neutron_allocator.h"

namespace onnxruntime {
namespace neutron {

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

#define REGISTER_Q_KERNEL_TYPED(T)                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                            \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      13, 18,                                                         \
      T,                                                              \
      kNeutronExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

#define REGISTER_Q_KERNEL_TYPED_19(T)                                 \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                  \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      19,                                                             \
      T, float,                                                       \
      kNeutronExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

REGISTER_Q_KERNEL_TYPED(uint8_t)
REGISTER_Q_KERNEL_TYPED(int8_t)
REGISTER_Q_KERNEL_TYPED_19(int8_t)
REGISTER_Q_KERNEL_TYPED_19(uint8_t)

template <typename InputType, typename OutputType>
void ParQuantizeLinear(const InputType* Input,
                       OutputType* Output,
                       size_t N,
                       InputType Scale,
                       size_t bd,
                       const OutputType* ZeroPoint,
                       bool saturate,
                       concurrency::ThreadPool* thread_pool) {
  if constexpr (!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value) {
    ORT_UNUSED_PARAMETER(saturate);
    ParQuantizeLinearStd(Input, Output, N, Scale,
                         ZeroPoint != nullptr ? ZeroPoint[bd] : (OutputType)0,
                         thread_pool);
  } else {
    OutputType default_zp = OutputType(static_cast<InputType>(static_cast<float>(0)), true);
    ParQuantizeLinearSat(Input, Output, N, Scale,
                         ZeroPoint != nullptr ? ZeroPoint[bd] : default_zp,
                         saturate, thread_pool);
  }
}

template <typename T, typename InT>
void ComputeLoop(OpKernelContext* ctx,
                 const InT* input, const InT* scale, const T* zero_point, T* output,
                 int64_t N, int64_t broadcast_dim, int64_t block_size, bool saturate) {
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      ParQuantizeLinear(input, output,
                        static_cast<size_t>(block_size),
                        scale[bd], bd, zero_point, saturate,
                        ctx->GetOperatorThreadPool());
      input += block_size;
      output += block_size;
    }
  }
}

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, broadcast_dim, block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;

  // Allocate output tensor using ORT's allocator (CPU memory)
  Tensor* y = ctx->Output(0, x_shape);
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }

#if NEUTRON_AARCH64
  // Use Neutron allocator for temporary buffer, then copy to output
  // This matches the pattern used in MatMul ops
  size_t m_handle = neutronAlloc->getMemoryHandle();
  neutronAlloc->pushMemoryState(m_handle);

  const size_t out_size = x_shape.Size() * sizeof(T);
  T* y_temp = static_cast<T*>(neutronAlloc->AllocReserved(out_size, m_handle));

  // Compute quantization to temporary buffer
  if (x.IsDataType<float>()) {
    ComputeLoop<T, float>(ctx, x.Data<float>(), y_scale.Data<float>(), zero_point,
                          y_temp, N, broadcast_dim, block_size, saturate_);
  } else {
    neutronAlloc->popMemoryState(m_handle);
    ORT_THROW("Unsupported input type.");
  }

  // Copy result to output tensor
  memcpy(y->MutableDataRaw(), y_temp, out_size);

  neutronAlloc->popMemoryState(m_handle);
#else
  // Non-ARM: compute directly to output buffer
  T* out_ptr = y->MutableData<T>();
  if (x.IsDataType<float>()) {
    ComputeLoop<T, float>(ctx, x.Data<float>(), y_scale.Data<float>(), zero_point,
                          out_ptr, N, broadcast_dim, block_size, saturate_);
  } else {
    ORT_THROW("Unsupported input type.");
  }
#endif

  return Status::OK();
}

}  // namespace neutron
}  // namespace onnxruntime
