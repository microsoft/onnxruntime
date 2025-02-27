// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "slice.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cpu/tensor/utils.h"
namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME slice_kernel_src
#include "opencl_generated/tensor/kernels/slice.cl.inc"
}  // namespace

template <bool dynamic>  // dynamic slice(version >= 10,dynamic = true)
class Slice : public OpenCLKernel, public SliceBase {
 public:
  explicit Slice(const OpKernelInfo& info)
      : OpenCLKernel(info), SliceBase(info, dynamic) {
    LoadProgram(slice_kernel_src, slice_kernel_src_len);
    LoadKernel("Slice");
  };

  Status Compute(OpKernelContext* ctx) const override;

 private:
  virtual const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const;
  virtual Status FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                  TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                  TensorShapeVector& input_steps) const;
};

void ComputeSliceStrides(const TensorShape& input_shape,
                         TensorShapeVector& input_strides,
                         TensorShapeVector& output_strides,
                         onnxruntime::SliceOp::PrepareForComputeMetadata& compute_metadata) {
  const auto input_dimensions = input_shape.GetDims();
  size_t rank = compute_metadata.p_flattened_input_dims_ ? compute_metadata.p_flattened_input_dims_->size()
                                                         : input_dimensions.size();

  input_strides.resize(gsl::narrow_cast<int32_t>(rank));
  const gsl::span<int64_t> input_strides_span = gsl::make_span(input_strides.data(), input_strides.size());

  if (compute_metadata.p_flattened_input_dims_) {
    TensorPitches::Calculate(input_strides_span, *compute_metadata.p_flattened_input_dims_);
  } else {
    TensorPitches::Calculate(input_strides_span, input_dimensions);
  }

  size_t start_len = compute_metadata.starts_.size();
  size_t step_len = compute_metadata.steps_.size();

  if (start_len < compute_metadata.output_dims_.size()) {
    compute_metadata.starts_.resize(compute_metadata.output_dims_.size(), 0);
  }
  if (step_len < compute_metadata.output_dims_.size()) {
    compute_metadata.steps_.resize(compute_metadata.output_dims_.size(), 1);
  }

  const auto& output_dims = compute_metadata.p_flattened_output_dims_
                                ? *compute_metadata.p_flattened_output_dims_
                                : compute_metadata.output_dims_;

  output_strides.resize(gsl::narrow_cast<int32_t>(output_dims.size()));
  const gsl::span<int64_t> output_strides_span = gsl::make_span(output_strides.data(), output_strides.size());

  TensorPitches::Calculate(output_strides_span, output_dims);
}

template <bool dynamic>
const Tensor* Slice<dynamic>::GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const {
  return ctx->Input<Tensor>(0);
}

template <bool dynamic>
Status Slice<dynamic>::FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                        TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                        TensorShapeVector& input_steps) const {
  return FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2), ctx->Input<Tensor>(3),
                              ctx->Input<Tensor>(4), input_starts, input_ends, input_axes, input_steps);
}
template <bool dynamic>
Status Slice<dynamic>::Compute(OpKernelContext* ctx) const {
  const Tensor* input_tensor = GetSlicedOrUnslicedTensor(ctx);
  ORT_ENFORCE(nullptr != input_tensor);
  const auto& input_shape = input_tensor->Shape();
  const auto input_dimensions = input_shape.GetDims();
  if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
  TensorShapeVector input_starts, input_ends, input_axes, input_steps;
  if (dynamic) {
    ORT_RETURN_IF_ERROR(FillInputVectors(ctx, input_starts, input_ends, input_axes, input_steps));
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  } else {
    // no support now
    // ORT_RETURN_IF_ERROR(PrepareForCompute(StartsAttribute(), EndsAttribute(), AxesAttribute(), compute_metadata));
  }

  TensorShape output_shape(compute_metadata.output_dims_);
  auto* output_tensor = ctx->Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }
  TensorShapeVector input_strides;
  TensorShapeVector output_strides;
  ComputeSliceStrides(input_shape, input_strides, output_strides, compute_metadata);

  auto Input_strides = exec_->GetScratchBufferTmp(input_strides.size() * sizeof(int64_t));
  auto Output_strides = exec_->GetScratchBufferTmp(output_strides.size() * sizeof(int64_t));
  exec_->WriteToCLBuffer(Input_strides, input_strides.data(), input_strides.size() * sizeof(int64_t));
  exec_->WriteToCLBuffer(Output_strides, output_strides.data(), output_strides.size() * sizeof(int64_t));

  auto Input_shape = exec_->GetScratchBufferTmp(input_shape.NumDimensions() * sizeof(int64_t));
  auto Output_shape = exec_->GetScratchBufferTmp(output_shape.NumDimensions() * sizeof(int64_t));

  exec_->WriteToCLBuffer(Input_shape, input_shape.GetDims().data(), input_shape.NumDimensions() * sizeof(int64_t));
  exec_->WriteToCLBuffer(Output_shape, output_shape.GetDims().data(), output_shape.NumDimensions() * sizeof(int64_t));
  cl_mem input_starts_mem;
  cl_mem input_steps_mem;
  input_starts_mem = exec_->GetScratchBufferTmp(compute_metadata.starts_.size() * sizeof(int64_t));
  exec_->WriteToCLBuffer(input_starts_mem, compute_metadata.starts_.data(), compute_metadata.starts_.size() * sizeof(int64_t));
  input_steps_mem = exec_->GetScratchBufferTmp(compute_metadata.steps_.size() * sizeof(int64_t));
  exec_->WriteToCLBuffer(input_steps_mem, compute_metadata.steps_.data(), compute_metadata.steps_.size() * sizeof(int64_t));

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Slice")}
          .SetBuffers(*input_tensor, *output_tensor)
          .SetArg<cl_long>((cl_long)input_dimensions.size())
          .SetArg<cl_long>((cl_long)input_tensor->DataType()->Size())
          .SetBuffers(input_starts_mem, input_steps_mem)
          .SetBuffers(Input_strides, Output_strides)
          .SetArg<cl_long>((cl_long)compute_metadata.output_dims_.size())
          .SetArg<cl_long>((cl_long)input_tensor->SizeInBytes())
          .Launch(*exec_, {output_tensor->SizeInBytes() / input_tensor->DataType()->Size(), 1, 1}));

  exec_->ReleaseCLBuffer(Input_shape);
  exec_->ReleaseCLBuffer(Output_shape);
  exec_->ReleaseCLBuffer(Input_strides);
  exec_->ReleaseCLBuffer(Output_strides);
  exec_->ReleaseCLBuffer(input_starts_mem);
  exec_->ReleaseCLBuffer(input_steps_mem);
  exec_->Sync();
  return Status::OK();
}

#define REGISTER_V13_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      13,                                                               \
      TIND,                                                             \
      kOpenCLExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V13_TYPED_SLICE(int32_t)
REGISTER_V13_TYPED_SLICE(int64_t)

}  // namespace opencl
}  // namespace onnxruntime
