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
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Pad,                                                        \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);

using PadsVector = PadBase::PadsVector;

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
  int32_t dimension_count = static_cast<int32_t>(input_shape.NumDimensions());

  const PadsVector* p_pads = &pads_;
  const PadsVector* p_slices = &slices_;
  CudaT value = ToCudaType<T>::FromFloat(value_);

  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  PadsVector pads;
  PadsVector slices;
  if (is_dynamic_) {
    const Tensor& pads_tensor = *ctx->Input<Tensor>(1);
    const auto pads_tensor_dims = pads_tensor.Shape().GetDims();
    ORT_ENFORCE(utils::IsPrimitiveDataType<int64_t>(pads_tensor.DataType()),
                "Pads tensor should be an INT64 tensor");
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]");

    const int64_t* pads_tensor_raw_data = pads_tensor.Data<int64_t>();
    size_t pads_size = static_cast<size_t>(pads_tensor.Shape().Size());
    ORT_ENFORCE(pads_size == 2 * static_cast<size_t>(dimension_count),
                "Pads tensor size should be equal to twice the input dimension count ");

    pads.reserve(2LL * dimension_count);
    for (size_t i = 0; i < pads_size; ++i) {
      pads.push_back(pads_tensor_raw_data[i]);
    }
    // Separate out any negative pads into the slices array
    slices.resize(pads.size(), 0);
    for (size_t index = 0; index < pads.size(); index++) {
      if (pads[index] < 0) {
        slices[index] = pads[index];
        pads[index] = 0;
      }
    }

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
  ORT_ENFORCE(static_cast<size_t>(dimension_count * 2) == p_pads->size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  TArray<int64_t> lower_pads(dimension_count);
  TArray<int64_t> upper_pads(dimension_count);
  for (auto i = 0; i < dimension_count; i++) {
    lower_pads[i] = (*p_pads)[i] + (*p_slices)[i];
    upper_pads[i] = (*p_pads)[i + dimension_count] + (*p_slices)[i + dimension_count];
    output_dims[i] += lower_pads[i] + upper_pads[i];
  }

  TensorShape output_shape(output_dims);

  // special case when there is a dim value of 0 in the shape. behavior depends on mode
  if (input_shape.Size() == 0) {
    ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode_, input_shape, output_shape));
  }

  auto& output_tensor = *ctx->Output(0, output_shape);

  if (std::all_of(p_pads->begin(), p_pads->end(), [](const int64_t v) { return v == 0; }) &&
      std::all_of(p_slices->begin(), p_slices->end(), [](const int64_t v) { return v == 0; }) &&
      output_shape.Size() > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        output_tensor.MutableData<T>(), input_tensor.Data<T>(),
        sizeof(typename ToCudaType<T>::MappedType) * output_shape.Size(),
        cudaMemcpyDeviceToDevice, Stream()));
    return Status::OK();
  }

  if (IsNCHWInputWithPaddingAlongHAndW(static_cast<size_t>(dimension_count), lower_pads, upper_pads)) {
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
        Stream(),
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
  for (auto i = 0; i < dimension_count; i++) {
    fdm_output_strides[i] = fast_divmod(static_cast<int>(output_strides[i]));
  }

  PadImpl(
      Stream(),
      dimension_count,
      input_dims,
      input_strides,
      lower_pads,
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
