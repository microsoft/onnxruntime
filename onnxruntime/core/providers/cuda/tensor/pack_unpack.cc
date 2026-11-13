// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/tensor/pack_unpack_impl.h"
#include "core/providers/cpu/tensor/pack_unpack.h"

namespace onnxruntime {
namespace cuda {
namespace {

template <bool Unpack>
class PackUnpack final : public CudaKernel {
 public:
  explicit PackUnpack(const OpKernelInfo& info) : CudaKernel(info), attributes_(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    TensorShapeVector shape;
    packing::Layout layout{};
    ORT_RETURN_IF_ERROR(attributes_.Prepare(input, Unpack ? ctx->Input<Tensor>(1) : nullptr, Unpack, shape, layout));
    auto& output = *ctx->Output(0, TensorShape(shape));
    if (output.Shape().Size() == 0) return Status::OK();
    const auto& p = attributes_.parameters;
    const bool word32 = (Unpack ? input.GetElementType() : attributes_.output_type) == ONNX_NAMESPACE::TensorProto::UINT32;
    if (!word32 && p.group_size == 1 && p.radix == 256) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableDataRaw(), input.DataRaw(), input.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream(ctx)));
      return Status::OK();
    }
    const uint64_t units = Unpack ? layout.groups : layout.words;
    const uint64_t work = SafeInt<uint64_t>(output.Shape().Size() / (Unpack ? layout.count : layout.words)) * units;
    return LaunchPacking(Stream(ctx), Unpack, word32, input.DataRaw(), output.MutableDataRaw(), work, p, layout);
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PackUnpack);
  const packing::Attributes attributes_;
};

using Pack = PackUnpack<false>;
using Unpack = PackUnpack<true>;
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    Pack, kOnnxDomain, 29, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()).TypeConstraint("TStorage", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<uint32_t>()}),
    Pack);

ONNX_OPERATOR_KERNEL_EX(
    Unpack, kOnnxDomain, 29, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 1).TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()).TypeConstraint("TStorage", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<uint32_t>()}),
    Unpack);
}  // namespace cuda
}  // namespace onnxruntime
