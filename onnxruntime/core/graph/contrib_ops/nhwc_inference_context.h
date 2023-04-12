// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

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

#ifdef USE_QNN
/** Adapter class to enable ONNX shape inferencing to be used with the NHWC Resize layout-sensitive operator.
 *
 * After layout transformation, the Resize operator's input0 and output0 have an NHWC layout. However, ONNX shape
 * inferencing expects a NCHW layout for inputs and outputs. This adapter ensures that ONNX sees a NCHW version
 * of the operator during shape inferencing.
 *
 * The NhwcInferenceContext class is not used for NHWC Resize because it only handles input0 and output0. We also
 * need to convert the 'scales' and 'sizes' constant inputs to their NCHW counterparts.
 * Ex: a NHWC 'scales' input [1.0f, 2.0f, 3.0f, 4.0f] needs to be converted to [1.0f, 4.0f, 2.0f, 3.0f] for NCHW.
 */
class NhwcResizeInferenceContext : public ONNX_NAMESPACE::InferenceContext {
 public:
  NhwcResizeInferenceContext(ONNX_NAMESPACE::InferenceContext& ctx) : ctx_(ctx) {
    // copy any existing type and shape info, and convert to NCHW for usage with the ONNX inferencing
    const auto* nhwc_type = ctx_.getInputType(0);
    if (nhwc_type != nullptr) {
      input_type_ = *nhwc_type;
      TransposeToNchw(*nhwc_type, input_type_);
    }

    const size_t num_inputs = ctx_.getNumInputs();

    // Skip ROI input.

    // Scales input
    if (num_inputs > 2) {
      TransposeScalesInput();
    }

    // Sizes input
    if (num_inputs > 3) {
      TransposeSizesInput();
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
    // Return NCHW data for the 'scales' and 'sizes' constant inputs.
    //
    // For other inputs, we can't return the NHWC input data without transposing it, but wouldn't expect to be
    // asked for it during shape inferencing as getInputData is only used to retrieve things that may have small
    // constant initializers.
    switch (index) {
      case 2:
        return &scales_input_data_;
      case 3:
        return &sizes_input_data_;
      default:
        return ctx_.getInputData(index);
    }
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

  template <typename ElemType>
  void TransposeRawConstantInput(const ONNX_NAMESPACE::TensorProto& nhwc_tp, ONNX_NAMESPACE::TensorProto& nchw_tp,
                                 const char* input_name) {
    const std::vector<ElemType> nhwc_vals = ONNX_NAMESPACE::ParseData<ElemType>(&nhwc_tp);
    const size_t num_vals = nhwc_vals.size();

    if (num_vals < 3) {
      fail_shape_inference(
          "Resize operator's '", input_name, "' input must have at least 3 elements "
          "to convert between channels first and channels last.");
      return;
    }

    std::vector<ElemType> nchw_vals;
    nchw_vals.reserve(nhwc_vals.size());

    // Convert 1D scales from [N, H, W, ..., C] to [N, C, H, W, ...].
    nchw_vals.push_back(nhwc_vals[0]);
    nchw_vals.push_back(nhwc_vals.back());
    for (size_t i = 1; i < num_vals - 1; ++i) {
      nchw_vals.push_back(nhwc_vals[i]);
    }

    // Copy nchw values into tensor proto.
    std::string& mut_vals = *nchw_tp.mutable_raw_data();
    mut_vals.resize(nchw_vals.size() * sizeof(ElemType));
    std::copy(nchw_vals.cbegin(), nchw_vals.cend(), reinterpret_cast<ElemType*>(mut_vals.data()));
  }

  void TransposeScalesInput() {
    auto* scales_data = ctx_.getInputData(2);
    if (scales_data == nullptr) {
      return;
    }

    const ONNX_NAMESPACE::TensorProto& nhwc_tp = *scales_data;

    // Copy nhwc_tp to scales_input_data_ to ensure both have the same type and dims.
    scales_input_data_ = nhwc_tp;

    const auto& dims = nhwc_tp.dims();

    if (dims.size() != 1) {
      fail_shape_inference("Resize operator's 'scales' input must have a rank of 1.");
    }

    // Exit if scales input is empty. In this case, the sizes input must be provided.
    if (dims[0] == 0) {
      return;
    }

    const auto data_type = nhwc_tp.data_type();

    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      fail_shape_inference("Resize operator's 'scales' input must be of type float32.");
      return;
    }

    if (nhwc_tp.has_raw_data()) {
      TransposeRawConstantInput<float>(nhwc_tp, scales_input_data_, "scales");
    } else {
      const auto& nhwc_vals = nhwc_tp.float_data();
      const int num_vals = nhwc_tp.float_data_size();

      if (num_vals < 3) {
        fail_shape_inference("Resize operator's 'scales' input must have at least 3 elements "
                             "to convert between channels first and channels last.");
      }

      auto& nchw_vals = *scales_input_data_.mutable_float_data();

      // Convert 1D scales from [N, H, W, ..., C] to [N, C, H, W, ...].
      int j = 0;
      nchw_vals[j++] = nhwc_vals[0];
      nchw_vals[j++] = nhwc_vals[num_vals - 1];
      for (int i = 1; i < num_vals - 1; ++i) {
        nchw_vals[j++] = nhwc_vals[i];
      }
    }
  }

  void TransposeSizesInput() {
    auto* sizes_data = ctx_.getInputData(3);
    if (sizes_data == nullptr) {
      return;
    }

    const ONNX_NAMESPACE::TensorProto& nhwc_tp = *sizes_data;

    // Copy nhwc_tp to sizes_input_data_ to ensure both have the same type and dims.
    sizes_input_data_ = nhwc_tp;

    const auto& dims = nhwc_tp.dims();

    if (dims.size() != 1) {
      fail_shape_inference("Resize operator's 'sizes' input must have a rank of 1.");
      return;
    }

    const auto data_type = nhwc_tp.data_type();

    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      fail_shape_inference("Resize operator's 'sizes' input must be of type int64.");
      return;
    }

    if (nhwc_tp.has_raw_data()) {
      TransposeRawConstantInput<int64_t>(nhwc_tp, sizes_input_data_, "sizes");
    } else {
      const auto& nhwc_vals = nhwc_tp.int64_data();
      const int num_vals = nhwc_tp.int64_data_size();

      if (num_vals < 3) {
        fail_shape_inference("Resize operator's 'sizes' input must have at least 3 elements "
                             "to convert between channels first and channels last.");
      }

      auto& nchw_vals = *sizes_input_data_.mutable_int64_data();

      // Convert 1D sizes from [N, H, W, ..., C] to [N, C, H, W, ...].
      int j = 0;
      nchw_vals[j++] = nhwc_vals[0];
      nchw_vals[j++] = nhwc_vals[num_vals - 1];
      for (int i = 1; i < num_vals - 1; ++i) {
        nchw_vals[j++] = nhwc_vals[i];
      }
    }
  }

  InferenceContext& ctx_;
  ONNX_NAMESPACE::TypeProto input_type_;
  ONNX_NAMESPACE::TensorProto scales_input_data_;
  ONNX_NAMESPACE::TensorProto sizes_input_data_;
  ONNX_NAMESPACE::TypeProto output_type_;
};
#endif  // defined(USE_QNN)

}  // namespace contrib
}  // namespace onnxruntime
