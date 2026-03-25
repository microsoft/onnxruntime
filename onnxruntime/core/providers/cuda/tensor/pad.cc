// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pad.h"
#include "pad_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      2, 10,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      11, 12,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      13, 17,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      18, 18,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      19, 20,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      21, 22,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      23, 23,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      24, 24,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Pad,                                                        \
      kOnnxDomain,                                                \
      25,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);

using PadsVector = PadBase::PadsVector;

// In the plugin build, PadBase::ComputePads is not accessible because it
// depends on CPU provider internals. ComputePadsImpl is a minimal inline
// equivalent. Keep in sync with PadBase::ComputePads in pad.h.
template <typename KernelContextType>
static void ComputePadsLocal(KernelContextType& ctx,
                             size_t data_rank,
                             gsl::span<const int64_t> pads_data,
                             PadsVector& pads) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  PadBase::ComputePadsImpl(ctx, data_rank, pads_data, pads);
#else
  PadBase::ComputePads(ctx, data_rank, pads_data, pads);
#endif
}

// In the plugin build, PadBase::HandleDimValueZero lives in CPU provider code
// that cannot be linked into the plugin. Inline the same validation here.
// Keep in sync with PadBase::HandleDimValueZero in pad.h.
static Status HandleDimValueZeroLocal(const Mode& mode,
                                      const TensorShape& input_shape,
                                      const TensorShape& output_shape) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  switch (mode) {
    case Mode::Constant:
      break;
    case Mode::Edge:
    case Mode::Reflect: {
      for (size_t i = 0, end = input_shape.NumDimensions(); i < end; ++i) {
        if (input_shape[i] == 0 && output_shape[i] > 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Cannot use '", mode == Mode::Edge ? "edge" : "reflect",
                                 "' mode to pad dimension with a value of 0. Input shape:",
                                 input_shape);
        }
      }
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected mode of ", static_cast<int>(mode));
  }

  return Status::OK();
#else
  return PadBase::HandleDimValueZero(mode, input_shape, output_shape);
#endif
}

static bool IsNCHWInputWithPaddingAlongHAndW(size_t input_rank,
                                             const TArray<int64_t>& lower_pads,
                                             const TArray<int64_t>& upper_pads) {
  if (input_rank == 2) {  // N = 1 and C = 1
    return true;
  }

  // Is CHW input AND no padding along C dim
  if (input_rank == 3 &&
      lower_pads[0] == 0 &&  // start padding along C
      upper_pads[0] == 0) {  // end padding along C
    return true;
  }

  // Is NCHW input AND no padding along N and C dims
  if (input_rank == 4 &&
      lower_pads[0] == 0 && lower_pads[1] == 0 &&  // start padding along N and C
      upper_pads[0] == 0 && upper_pads[1] == 0) {  // end padding along N and C
    return true;
  }

  return false;
}

template <typename T>
typename ToCudaType<T>::MappedType ToCudaValue(const T& value) {
  return value;
}

template <>
typename ToCudaType<MLFloat16>::MappedType ToCudaValue<MLFloat16>(const MLFloat16& value) {
  return *reinterpret_cast<const typename ToCudaType<MLFloat16>::MappedType*>(&value.val);
}

template <typename T>
Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto& input_tensor = *ctx->Input<Tensor>(0);
  auto const& input_shape = input_tensor.Shape();
  const int32_t dimension_count = narrow<int32_t>(input_shape.NumDimensions());

  const PadsVector* p_pads = &pads_;
  const PadsVector* p_slices = &slices_;
  CudaT value = ToCudaType<T>::FromFloat(value_);

  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  PadsVector pads;
  PadsVector slices;
  if (is_dynamic_) {
    const Tensor& pads_tensor = *ctx->Input<Tensor>(1);
    const auto pads_tensor_dims = pads_tensor.Shape().GetDims();
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * num_axes] or a 2D tensor of shape [1, 2 * num_axes]");

    const auto pads_data = pads_tensor.DataAsSpan<int64_t>();

    PadBase::ComputePadsImpl(*ctx, input_shape.NumDimensions(), pads_data, pads);

    // Separate out any negative pads into the slices array
    PadBase::SeparateNegativeToSlices(pads, slices);

    T raw_value{};
    const Tensor* value_tensor = ctx->Input<Tensor>(2);
    if (nullptr != value_tensor) {
      ORT_ENFORCE(utils::IsPrimitiveDataType<T>(value_tensor->DataType()) &&
                      value_tensor->Shape().Size() == 1,
                  "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
      raw_value = value_tensor->Data<T>()[0];
      value = ToCudaValue<T>(raw_value);
    }
    p_pads = &pads;
    p_slices = &slices;
  }

  TensorPitches input_pitches(input_shape.GetDims());
  TArray<int64_t> input_dims(input_shape.GetDims());
  TArray<int64_t> input_strides(input_pitches);

  auto output_dims(input_shape.AsShapeVector());
  ORT_ENFORCE(dimension_count * 2 == narrow<int32_t>(p_pads->size()), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  TArray<int64_t> lower_pads(dimension_count);
  TArray<int64_t> upper_pads(dimension_count);
  for (int32_t i = 0; i < dimension_count; i++) {
    lower_pads[i] = SafeInt<int64_t>((*p_pads)[i]) + (*p_slices)[i];
    upper_pads[i] = SafeInt<int64_t>((*p_pads)[i + dimension_count]) + (*p_slices)[i + dimension_count];
    output_dims[i] += SafeInt<int64_t>(lower_pads[i]) + upper_pads[i];
  }

  TensorShapeVector effective_input_extents;
  effective_input_extents.reserve(dimension_count);
  for (int32_t i = 0; i < dimension_count; i++) {
    int64_t extent = std::max<int64_t>(SafeInt<int64_t>(input_dims[i]) +
                                           (*p_slices)[i] + (*p_slices)[i + dimension_count],
                                       0LL);
    effective_input_extents.push_back(extent);
  }

  TArray<int64_t> input_offsets(dimension_count);
  for (int32_t i = 0; i < dimension_count; ++i) {
    input_offsets[i] = -(*p_slices)[i];
  }

  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);

  // If the input size is zero, but output shape is not, need padding only
  // this is expected for constant mode only, otherwise the output is empty
  // no error
  if (input_shape.Size() == 0) {
    ORT_RETURN_IF_ERROR(HandleDimValueZeroLocal(mode_, input_shape, output_shape));
    if (mode_ == Mode::Constant) {
      const int64_t output_size = output_shape.Size();
      if (output_size > 0) {
        Fill<CudaT>(Stream(ctx), reinterpret_cast<CudaT*>(output_tensor.MutableData<T>()), value,
                    output_size);
      }
    }
    // No error for other modes (preserve CPU historical behavior),
    // but no output should be expected either
    return Status::OK();
  }

  // Early constant-fill: input is not empty as above
  // However, if any effective input extent is zero, no data to copy
  // only padding if any.
  const bool no_effective_data_to_copy = std::any_of(effective_input_extents.begin(), effective_input_extents.end(),
                                                     [](int64_t v) { return v == 0; });

  if (no_effective_data_to_copy) {
    if (mode_ == Mode::Constant) {
      // Attempt to pad constant mode in case output is not empty
      // all other modes are an error
      const int64_t output_size = output_shape.Size();
      if (output_size > 0) {
        Fill<CudaT>(Stream(ctx), reinterpret_cast<CudaT*>(output_tensor.MutableData<T>()), value,
                    output_size);
      }
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Pad: invalid mode: ", static_cast<int>(mode_), " with zero effective input extent");
  }

  // Special case for Reflect mode: ensure all extents >= 2 after slicing;
  // otherwise reflection is not possible. Also validate that pads do not
  // exceed extent - 1 on each side, as required by the ONNX spec, which
  // aligns with NumPy behavior where start and end positions must be distinct.
  if (mode_ == Mode::Reflect) {
    for (int32_t i = 0; i < dimension_count; ++i) {
      const int64_t extent = effective_input_extents[i];  // length after slicing
      const bool reflect_on_axis =
          (*p_pads)[i] > 0 || (*p_pads)[i + dimension_count] > 0;
      if (reflect_on_axis && extent < 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Pad reflect requires axis length >= 2 after slicing. Input shape:",
                               input_shape);
      }
      // ONNX spec: reflect pads must not exceed extent - 1 on each side
      if ((*p_pads)[i] > extent - 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Pad reflect: pre-pad (", (*p_pads)[i],
                               ") exceeds maximum allowed (", extent - 1,
                               ") for axis ", i, ". Input shape:", input_shape);
      }
      if ((*p_pads)[i + dimension_count] > extent - 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Pad reflect: post-pad (", (*p_pads)[i + dimension_count],
                               ") exceeds maximum allowed (", extent - 1,
                               ") for axis ", i, ". Input shape:", input_shape);
      }
    }
  }

  // Case of all pads and slices being zero: just copy input to output
  if (std::all_of(p_pads->begin(), p_pads->end(), [](const int64_t v) { return v == 0; }) &&
      std::all_of(p_slices->begin(), p_slices->end(), [](const int64_t v) { return v == 0; }) &&
      output_shape.Size() > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        output_tensor.MutableData<T>(), input_tensor.Data<T>(),
        sizeof(typename ToCudaType<T>::MappedType) * output_shape.Size(),
        cudaMemcpyDeviceToDevice, Stream(ctx)));
    return Status::OK();
  }

  if (mode_ != Mode::Wrap &&
      IsNCHWInputWithPaddingAlongHAndW(dimension_count, lower_pads, upper_pads)) {
    // If we have entered here, it means the input can only be 4-D (NCHW), 3-D (CHW), or 2-D (HW)

    // NCHW input
    int height_dim = 2;
    int width_dim = 3;

    if (dimension_count == 3) {  // CHW input
      height_dim = 1;
      width_dim = 2;
    } else if (dimension_count == 2) {  // HW input
      height_dim = 0;
      width_dim = 1;
    }

    PadNCHWInputWithPaddingAlongHAndWImpl(
        Stream(ctx),
        dimension_count == 4 ? input_dims[0] : 1,
        dimension_count == 4 ? input_dims[1] : (dimension_count == 3 ? input_dims[0] : 1),
        input_dims[height_dim],
        output_dims[height_dim],
        input_dims[width_dim],
        output_dims[width_dim],
        lower_pads[height_dim],
        lower_pads[width_dim],
        value,
        static_cast<int>(mode_),
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input_tensor.Data<T>()),
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor.MutableData<T>()),
        output_tensor.Shape().Size());

    return Status::OK();
  }

  TArray<fast_divmod> fdm_output_strides(dimension_count);
  TensorPitches output_strides(output_dims);
  for (int32_t i = 0; i < dimension_count; i++) {
    fdm_output_strides[i] = fast_divmod(static_cast<int>(output_strides[i]));
  }

  PadImpl(
      Stream(ctx),
      dimension_count,
      input_dims,
      input_strides,
      lower_pads,
      TArray<int64_t>(effective_input_extents),
      input_offsets,
      value,
      static_cast<int>(mode_),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input_tensor.Data<T>()),
      fdm_output_strides,
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor.MutableData<T>()),
      output_tensor.Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)
SPECIALIZED_COMPUTE(bool)

}  // namespace cuda
};  // namespace onnxruntime
