// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/bitshift_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class BitShift final : public BinaryElementwise<ShouldBroadcast> {
 public:
  explicit BitShift(const OpKernelInfo& info) : BinaryElementwise<ShouldBroadcast>(info) {
    std::string direction;
    ORT_THROW_IF_ERROR(info.GetAttr("direction", &direction));
    ORT_ENFORCE(direction == "LEFT" || direction == "RIGHT",
                "Invalid direction value of '", direction, "'. Valid values are 'LEFT' or 'RIGHT'.");
    shift_left_ = direction == "LEFT";
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    BinaryElementwisePreparation prepare;
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));
    Impl_BitShift<T>(
        Stream(context),
        prepare.output_rank_or_simple_broadcast,
        &prepare.lhs_padded_strides,
        prepare.lhs_tensor->Data<T>(),
        &prepare.rhs_padded_strides,
        prepare.rhs_tensor->Data<T>(),
        &prepare.fdm_output_strides,
        prepare.fdm_H,
        prepare.fdm_C,
        prepare.output_tensor->MutableData<T>(),
        prepare.output_tensor->Shape().Size(),
        shift_left_);
    return Status::OK();
  }

 private:
  bool shift_left_;
};

#define REGISTER_BITSHIFT_VERSIONED(T)                                                     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      BitShift, kOnnxDomain, 11, 27, T, kCudaExecutionProvider,                            \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BitShift<T>);

#define REGISTER_BITSHIFT(T)                                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      BitShift, kOnnxDomain, 28, T, kCudaExecutionProvider,                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BitShift<T>);

REGISTER_BITSHIFT_VERSIONED(uint8_t)
REGISTER_BITSHIFT_VERSIONED(uint16_t)
REGISTER_BITSHIFT_VERSIONED(uint32_t)
REGISTER_BITSHIFT_VERSIONED(uint64_t)

REGISTER_BITSHIFT(uint8_t)
REGISTER_BITSHIFT(uint16_t)
REGISTER_BITSHIFT(uint32_t)
REGISTER_BITSHIFT(uint64_t)
REGISTER_BITSHIFT(int8_t)
REGISTER_BITSHIFT(int16_t)
REGISTER_BITSHIFT(int32_t)
REGISTER_BITSHIFT(int64_t)

#undef REGISTER_BITSHIFT_VERSIONED
#undef REGISTER_BITSHIFT

}  // namespace cuda
}  // namespace onnxruntime
