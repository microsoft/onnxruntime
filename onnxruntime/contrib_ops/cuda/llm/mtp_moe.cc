// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/mtp_moe.h"

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class MtpMoe final : public CudaKernel {
 public:
  explicit MtpMoe(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("gate_up_global_scale", &gate_up_global_scale_).IsOK(),
                "MtpMoe requires gate_up_global_scale");
    ORT_ENFORCE(info.GetAttr<float>("down_global_scale", &down_global_scale_).IsOK(),
                "MtpMoe requires down_global_scale");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* input = context->Input<Tensor>(0);
    const Tensor* router = context->Input<Tensor>(1);
    const Tensor* gate_up = context->Input<Tensor>(2);
    const Tensor* down = context->Input<Tensor>(3);
    ORT_ENFORCE(input != nullptr && router != nullptr && gate_up != nullptr && down != nullptr);

    const TensorShape& input_shape = input->Shape();
    const TensorShape& router_shape = router->Shape();
    const TensorShape& gate_up_shape = gate_up->Shape();
    const TensorShape& down_shape = down->Shape();
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 3 && input_shape[0] == 1 &&
                          input_shape[2] == mtp_moe::kHiddenSize,
                      "MtpMoe input must have shape [1, R, 2560], got ", input_shape);
    const int64_t rows = input_shape[1];
    ORT_RETURN_IF_NOT(rows > 0 && rows <= std::numeric_limits<int>::max() &&
                          router_shape.NumDimensions() == 3 &&
                          router_shape[0] == 1 && router_shape[1] == rows &&
                          router_shape[2] == mtp_moe::kExpertCount,
                      "MtpMoe router logits must have shape [1, R, 512], got ", router_shape);
    ORT_RETURN_IF_NOT(gate_up_shape.NumDimensions() == 3 &&
                          gate_up_shape[0] == mtp_moe::kExpertCount &&
                          gate_up_shape[1] == 2 * mtp_moe::kFfnSize &&
                          gate_up_shape[2] == mtp_moe::kHiddenSize * 9 / 16,
                      "MtpMoe NVFP4 gate_up weights must have shape [512, 1280, 1440], got ", gate_up_shape);
    ORT_RETURN_IF_NOT(down_shape.NumDimensions() == 3 &&
                          down_shape[0] == mtp_moe::kExpertCount &&
                          down_shape[1] == mtp_moe::kHiddenSize &&
                          down_shape[2] == mtp_moe::kFfnSize * 9 / 16,
                      "MtpMoe NVFP4 down weights must have shape [512, 2560, 360], got ", down_shape);

    Tensor* output = context->Output(0, input_shape);
    Tensor* scratch = context->Output(
        1, TensorShape({rows, static_cast<int64_t>(mtp_moe::kScratchStride)}));
    ORT_ENFORCE(output != nullptr && scratch != nullptr);

    return mtp_moe::LaunchNvfp4MtpMoe(
        reinterpret_cast<const __nv_bfloat16*>(input->Data<BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(router->Data<BFloat16>()),
        gate_up->Data<uint8_t>(), down->Data<uint8_t>(),
        gate_up_global_scale_, down_global_scale_,
        scratch->MutableData<float>(),
        reinterpret_cast<__nv_bfloat16*>(output->MutableData<BFloat16>()),
        static_cast<int>(rows), Stream(context));
  }

 private:
  float gate_up_global_scale_;
  float down_global_scale_;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MtpMoe,
    kMSDomain,
    1,
    BFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    MtpMoe);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
