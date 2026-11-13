// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/providers/common.h"
#include "core/providers/cuda/math/logcumsumexp_impl.h"

namespace onnxruntime {
namespace cuda {
namespace {

template <typename T>
struct LaunchLogCumSumExp {
  Status operator()(cudaStream_t stream, const Tensor& input, Tensor& output, size_t axis) const {
    using CudaT = typename ToCudaType<T>::MappedType;
    const int64_t width = input.Shape()[axis];
    if (width == 1) {
      return CUDA_CALL(cudaMemcpyAsync(output.MutableDataRaw(), input.DataRaw(), input.SizeInBytes(),
                                       cudaMemcpyDeviceToDevice, stream));
    }
    return LogCumSumExpImpl(
        stream, reinterpret_cast<const CudaT*>(input.Data<T>()),
        reinterpret_cast<CudaT*>(output.MutableData<T>()), width,
        input.Shape().SizeFromDimension(axis + 1), input.Shape().Size() / width);
  }
};

}  // namespace

class LogCumSumExp final : public CudaKernel {
 public:
  explicit LogCumSumExp(const OpKernelInfo& info)
      : CudaKernel(info), axis_(info.GetAttrOrDefault<int64_t>("axis", 0)) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    const int64_t rank = static_cast<int64_t>(input.Shape().NumDimensions());
    ORT_RETURN_IF_NOT(rank > 0, "LogCumSumExp requires rank >= 1; reshape scalar X to a one-element tensor.");
    ORT_RETURN_IF_NOT(axis_ >= -rank && axis_ < rank, "LogCumSumExp axis ", axis_,
                      " is outside [-", rank, ", ", rank - 1, "]; choose an axis within X's rank.");
    const auto axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
    auto& output = *ctx->Output(0, input.Shape());
    if (input.Shape().Size() == 0) {
      return Status::OK();
    }
    return utils::MLTypeCallDispatcher<float, double, MLFloat16, BFloat16>(input.GetElementType())
        .InvokeRet<Status, LaunchLogCumSumExp>(Stream(ctx), input, output, axis);
  }

 private:
  int64_t axis_;
};

ONNX_OPERATOR_KERNEL_EX(
    LogCumSumExp, kOnnxDomain, 29, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16, BFloat16>()),
    LogCumSumExp);

}  // namespace cuda
}  // namespace onnxruntime
