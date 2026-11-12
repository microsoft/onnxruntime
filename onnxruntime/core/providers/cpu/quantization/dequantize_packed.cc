// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/quantization/dequantize_packed.h"

namespace onnxruntime {

class DequantizePacked final : public OpKernel {
 public:
  explicit DequantizePacked(const OpKernelInfo& info) : OpKernel(info), attributes_(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    ORT_RETURN_IF_ERROR(attributes_.ValidateInput(input->Shape()));
    auto* output = context->Output(0, TensorShape(attributes_.shape));
    if (attributes_.elements == 0) return Status::OK();

    const auto* src = input->Data<uint8_t>();
    auto run = [&](auto* dst) {
      using T = std::remove_pointer_t<decltype(dst)>;
      dequantize_packed::Dispatch(attributes_.format.id, [&](auto kind, auto elements, auto bytes) {
        concurrency::ThreadPool::TryParallelFor(
            context->GetOperatorThreadPool(), static_cast<std::ptrdiff_t>(attributes_.blocks),
            TensorOpCost{static_cast<double>(bytes), static_cast<double>(elements * sizeof(T)),
                         static_cast<double>(elements * 12)},
            [&](std::ptrdiff_t first, std::ptrdiff_t last) {
              for (auto block = first; block < last; ++block) {
                const auto* packed = src + block * bytes;
                auto* decoded = dst + block * elements;
                dequantize_packed::DecodeBlockCpu<decltype(kind)::value, decltype(elements)::value>(packed, decoded);
              }
            });
      });
    };
    switch (attributes_.output_dtype) {
      case 1:
        run(output->MutableData<float>());
        break;
      case 10:
        run(output->MutableData<MLFloat16>());
        break;
      case 16:
        run(output->MutableData<BFloat16>());
        break;
      case 11:
        run(output->MutableData<double>());
        break;
    }
    return Status::OK();
  }

 private:
  dequantize_packed::Attributes attributes_;
};

ONNX_OPERATOR_KERNEL_EX(
    DequantizePacked, kOnnxDomain, 29, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<MLFloat16>(),
                                            DataTypeImpl::GetTensorType<BFloat16>(),
                                            DataTypeImpl::GetTensorType<double>()}),
    DequantizePacked);

}  // namespace onnxruntime
