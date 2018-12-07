// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/eye_like.h"
#include "core/framework/tensorprotoutils.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    EyeLike,
    9,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{
                                          DataTypeImpl::GetTensorType<float>(),
                                          DataTypeImpl::GetTensorType<int64_t>(),
                                          DataTypeImpl::GetTensorType<uint64_t>(),
                                      })
                        .TypeConstraint("T2",
                                        std::vector<MLDataType>{
                                            DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<uint64_t>(),
                                            DataTypeImpl::GetTensorType<int64_t>(),
                                        }),
    EyeLike);

Status EyeLike::Compute(OpKernelContext* context) const {
  const Tensor* T1 = context->Input<Tensor>(0);
  ONNXRUNTIME_ENFORCE(T1 != nullptr);  
  
  auto output_tensor_dtype = has_dtype ? static_cast<onnx::TensorProto::DataType>(dtype_) : utils::GetTensorProtoType(*T1);
  switch (output_tensor_dtype) {
    case onnx::TensorProto_DataType_FLOAT:
      return ComputeImpl<float>(context);
    case onnx::TensorProto_DataType_INT64:
      return ComputeImpl<int64_t>(context);
    case onnx::TensorProto_DataType_UINT64:
      return ComputeImpl<uint64_t>(context);
    default:
      ONNXRUNTIME_THROW("Unsupported 'dtype' value: ", output_tensor_dtype);
  }
}

template <typename T>
Status EyeLike::ComputeImpl(OpKernelContext* context) const {
  const Tensor* T1 = context->Input<Tensor>(0);
  const std::vector<int64_t>& input_dims = T1->Shape().GetDims();
  if (input_dims.size() != 2) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "EyeLike : Input tensor dimension is not 2");
  }

  // set output tensor shape same as input tensor and set all values to zero
  auto* T2 = context->Output(0, input_dims);
  auto* data = T2->MutableData<T>();
  T zero_value = static_cast<T>(0);
  auto out = gsl::make_span(data, T2->Shape().Size());
  std::fill(out.begin(), out.end(), zero_value);

  int64_t diag_start = 0;
  int64_t diag_end = 0;
  if (k_ >= 0) {
    diag_start = k_;
    diag_end = (input_dims[1] - k_) * (input_dims[1]);
  } else {
    diag_start = (-k_) * input_dims[1];
    diag_end = diag_start + (input_dims[0] + k_) * input_dims[1];
  }
  
  T one_value = static_cast<T>(1); 
  for (auto i = diag_start; i < diag_end; i += input_dims[1] + 1) {
    data[i] = one_value;
  }
  
  return Status::OK();
}

}  // namespace onnxruntime
