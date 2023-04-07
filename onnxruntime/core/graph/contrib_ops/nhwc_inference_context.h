// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnx/defs/shape_inference.h"

namespace onnxruntime {
namespace contrib {

/** Adapter class to enable ONNX shape inferencing to be used with the equivalent NHWC operators.
 *
 * Can be used with operators that match the ONNX spec but have NHWC layout for input 0 and output 0. This adapter
 * context will handle conversion between NHWC and NCHW for the relevant shapes. Once ONNX shape inferencing has
 * completed, call PropagateOutputShape to convert the inferred NCHW shape to NHWC, and propagate it to output 0.
 */
class NhwcInferenceContext : public ONNX_NAMESPACE::InferenceContext {
 public:
  NhwcInferenceContext(ONNX_NAMESPACE::InferenceContext& ctx) : ctx_(ctx) {
    // copy any existing type and shape info, and convert to NCHW for usage with the ONNX inferencing
    const auto* nhwc_type = ctx_.getInputType(0);
    if (nhwc_type != nullptr) {
      input_type_ = *nhwc_type;
      TransposeToNchw(*nhwc_type, input_type_);
    }

    nhwc_type = ctx_.getOutputType(0);
    if (nhwc_type != nullptr) {
      output_type_ = *nhwc_type;
      TransposeToNchw(*nhwc_type, output_type_);
    }
  }

  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override {
    return ctx_.getAttribute(name);
  }

  size_t getNumInputs() const noexcept override {
    return ctx_.getNumInputs();
  }

  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override {
    return (index == 0) ? &input_type_ : ctx_.getInputType(index);
  }

  const ONNX_NAMESPACE::TensorProto* getInputData(size_t index) const override {
    // we can't return the NHWC input data without transposing it, but wouldn't expect to be asked for it
    // during shape inferencing as getInputData is only used to retrieve things that may have small
    // constant initializers (e.g. something like the min and max values of a Clip operator).
    return index == 0 ? nullptr : ctx_.getInputData(index);
  }

  size_t getNumOutputs() const noexcept override {
    return ctx_.getNumOutputs();
  }

  ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) override {
    return (index == 0) ? &output_type_ : ctx_.getOutputType(index);
  }

  ONNX_NAMESPACE::GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
    return ctx_.getGraphAttributeInferencer(attribute_name);
  }

  const ONNX_NAMESPACE::TensorShapeProto* getSymbolicInput(size_t index) const override {
    return ctx_.getSymbolicInput(index);
  }

  const ONNX_NAMESPACE::SparseTensorProto* getInputSparseData(size_t index) const override {
    return ctx_.getInputSparseData(index);
  }

  // Propagate the inferred type/shape info to output 0, converting any inferred shape from NCHW to NHWC.
  void PropagateOutputShape() {
    auto& nhwc_tp = *ctx_.getOutputType(0);

    // copy latest type/shape info.
    nhwc_tp = output_type_;

    // convert shape to channels last
    if (output_type_.tensor_type().has_shape()) {
      const auto& nchw_shape = output_type_.tensor_type().shape();
      const int rank = nchw_shape.dim_size();
      // N and C dims are required. Some operators like AveragePool allow 1D input
      if (rank < 3) {
        fail_shape_inference("Output tensor must have at least 3 dimensions");
      }

      // Convert output shape from N, C, H {, W, ...} to N, H {, W, ...}, C.
      auto& nhwc_shape = *nhwc_tp.mutable_tensor_type()->mutable_shape();
      nhwc_shape.Clear();
      *nhwc_shape.add_dim() = nchw_shape.dim(0);
      for (int i = 2; i < rank; i++) {
        *nhwc_shape.add_dim() = nchw_shape.dim(i);
      }

      *nhwc_shape.add_dim() = nchw_shape.dim(1);
    }
  }

 private:
  void TransposeToNchw(const ONNX_NAMESPACE::TypeProto& nhwc_tp, ONNX_NAMESPACE::TypeProto& nchw_tp) {
    if (nhwc_tp.tensor_type().has_shape()) {
      const auto& nhwc_shape = nhwc_tp.tensor_type().shape();
      const int rank = nhwc_shape.dim_size();
      // N and C dims are required. Some operators like AveragePool allow 1D input.
      if (rank < 3) {
        fail_shape_inference(
            "Tensor must have at least 3 dimensions to convert between channels first and channels last.");
      }

      // Convert input shape from {N, H, W, ..., C} to {N, C, H, W, ...}.
      auto& nchw_shape = *nchw_tp.mutable_tensor_type()->mutable_shape();
      nchw_shape.Clear();
      *nchw_shape.add_dim() = nhwc_shape.dim(0);
      *nchw_shape.add_dim() = nhwc_shape.dim(rank - 1);
      for (int i = 1; i < rank - 1; i++) {
        *nchw_shape.add_dim() = nhwc_shape.dim(i);
      }
    }
  }

  InferenceContext& ctx_;
  ONNX_NAMESPACE::TypeProto input_type_;
  ONNX_NAMESPACE::TypeProto output_type_;
};

}  // namespace contrib
}  // namespace onnxruntime
