// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include "core/providers/neutron/ops/dequantize_linear.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/common/float16.h"
#include "core/providers/neutron/ops/common.h"
#include "core/providers/neutron/neutron_fwd.h"

namespace onnxruntime {
namespace neutron {

#define REGISTER_DQ_KERNEL_TYPED(T)                               \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      DequantizeLinear,                                           \
      kOnnxDomain,                                                \
      13, 18,                                                     \
      T,                                                          \
      kNeutronExecutionProvider,                                  \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

#define REGISTER_DQ_KERNEL_TYPED_19(T)                                 \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                   \
      DequantizeLinear,                                                \
      kOnnxDomain,                                                     \
      19,                                                              \
      T, float,                                                        \
      kNeutronExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()), \
      DequantizeLinear<T>);

REGISTER_DQ_KERNEL_TYPED(uint8_t)
REGISTER_DQ_KERNEL_TYPED(int8_t)
REGISTER_DQ_KERNEL_TYPED(int32_t)
REGISTER_DQ_KERNEL_TYPED_19(int8_t)
REGISTER_DQ_KERNEL_TYPED_19(uint8_t)
REGISTER_DQ_KERNEL_TYPED_19(int32_t)

template <typename T, typename OutT>
struct DequantizeLinearApply {
  void op(int64_t N, int64_t broadcast_dim, int64_t block_size,
          const T* input, const OutT* scale, OutT* output, const T* zero_point) {
    for (size_t n = 0; n < static_cast<size_t>(N); n++) {
      for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
        auto zp = zero_point ? static_cast<int32_t>(zero_point[bd]) : 0;
        auto sc = static_cast<float>(scale[bd]);
        for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {
          *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
        }
      }
    }
  }
};

// formula is Y = (X - ZeroPoint) * Scale
template <typename T>
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, N, broadcast_dim, block_size);

  const T* zero_point = x_zero_point ? x_zero_point->Data<T>() : nullptr;

  if constexpr (boost::mp11::mp_contains<boost::mp11::mp_append<element_type_lists::AllFloat8,
                                                                TypeList<int32_t>>,
                                         T>::value) {
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](T zp) { return zp == T{0}; }),
                "DequantizeLinear with type int32 or float8 should have no zero point or all zero points should be 0");
  }

  const auto to = x_scale.GetElementType();
  const T* input = x.Data<T>();

  if (to == ONNX_NAMESPACE::TensorProto::FLOAT) {
    const float* scale = x_scale.Data<float>();
    float* output = y.MutableData<float>();
    DequantizeLinearApply<T, float>().op(N, broadcast_dim, block_size, input, scale, output, zero_point);
  } else if (to == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    const MLFloat16* scale = x_scale.Data<MLFloat16>();
    MLFloat16* output = y.MutableData<MLFloat16>();
    DequantizeLinearApply<T, MLFloat16>().op(N, broadcast_dim, block_size, input, scale, output, zero_point);
  } else if (to == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    ORT_THROW("DequantizeLinear into BFLOAT16 is not implemented yet.");
  } else {
    ORT_THROW("DequantizeLinear only outputs FLOAT16, FLOAT or BFLOAT16.");
  }

  return Status::OK();
}

}  // namespace neutron
}  // namespace onnxruntime
