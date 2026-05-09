// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/common/float8.h"

namespace onnxruntime {
namespace cuda {

template <typename SrcT>
class Cast final : public CudaKernel {
 public:
  Cast(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

    int64_t saturate = info.GetAttrOrDefault("saturate", int64_t{1});
    if (saturate == 0 &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2 &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E8M0) {
      ORT_THROW("Attribute saturate is only used for cast to float 8 types.");
    }
    saturate_ = saturate == 1;

#if !defined(DISABLE_FLOAT8_TYPES)
    std::string round_mode_str = info.GetAttrOrDefault("round_mode", std::string("up"));
    if (round_mode_str == "up") {
      round_mode_ = Float8E8M0::RoundMode::Up;
    } else if (round_mode_str == "down") {
      round_mode_ = Float8E8M0::RoundMode::Down;
    } else if (round_mode_str == "nearest") {
      round_mode_ = Float8E8M0::RoundMode::Nearest;
    } else {
      ORT_THROW("Attribute round_mode must be 'up', 'down', or 'nearest'.");
    }
    if (round_mode_ != Float8E8M0::RoundMode::Up &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E8M0) {
      ORT_THROW("Attribute round_mode is only used for cast to float8e8m0.");
    }
#endif
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  bool saturate_;
#if !defined(DISABLE_FLOAT8_TYPES)
  Float8E8M0::RoundMode round_mode_{Float8E8M0::RoundMode::Up};
#endif
};

namespace cast_helper_impl {
template <class OutT, class InT>
Status CudaCastPairwise(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_elements);

#if !defined(DISABLE_FLOAT8_TYPES)
template <class OutT, class InT>
Status CudaCastStd(cudaStream_t stream, const InT* input, OutT* output, size_t num_of_elements);

// Cast to Float8E8M0 with saturate and round_mode support.
template <class InT>
Status CudaCastToE8M0(cudaStream_t stream, const InT* input, Float8E8M0* output,
                      size_t num_of_elements, bool saturate, Float8E8M0::RoundMode round_mode);
#endif
}  // namespace cast_helper_impl

}  // namespace cuda
}  // namespace onnxruntime
