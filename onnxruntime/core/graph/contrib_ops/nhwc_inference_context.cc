// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/contrib_ops/nhwc_inference_context.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cassert>

namespace onnxruntime {
namespace contrib {

/**
 * Helper function that sets the shape of the destination TypeProto to the transpose (i.e., NCHW)
 * of the source TypeProto (NHWC).
 *
 * /param nhwc_src The source TypeProto in NHWC layout.
 * /param nchw_dst The destination TypeProto in NCHW layout.
 */
static void TransposeShapeToNchw(const ONNX_NAMESPACE::TypeProto& nhwc_src, ONNX_NAMESPACE::TypeProto& nchw_dst) {
  // Copy to ensure both have the same initial shape.
  nchw_dst = nhwc_src;

  if (nhwc_src.tensor_type().has_shape()) {
    const auto& nhwc_shape = nhwc_src.tensor_type().shape();
    const int rank = nhwc_shape.dim_size();
    // N and C dims are required. Some operators like AveragePool allow 1D input.
    if (rank < 3) {
      fail_shape_inference(
          "Tensor must have at least 3 dimensions to convert between channels first and channels last.");
    }

    // Convert input shape from {N, H, W, ..., C} to {N, C, H, W, ...}.
    auto& nchw_shape = *nchw_dst.mutable_tensor_type()->mutable_shape();
    nchw_shape.Clear();
    *nchw_shape.add_dim() = nhwc_shape.dim(0);
    *nchw_shape.add_dim() = nhwc_shape.dim(rank - 1);
    for (int i = 1; i < rank - 1; i++) {
      *nchw_shape.add_dim() = nhwc_shape.dim(i);
    }
  }
}

/**
 * Helper function that sets the shape of the destination TypeProto to the transpose (i.e., NHWC)
 * of the source TypeProto (NCHW).
 *
 * /param nhwc_src The source TypeProto in NHWC layout.
 * /param nchw_dst The destination TypeProto in NCHW layout.
 */
static void TransposeShapeToNhwc(const ONNX_NAMESPACE::TypeProto& nchw_src, ONNX_NAMESPACE::TypeProto& nhwc_dst) {
  // Copy to ensure both have the same initial shape.
  nhwc_dst = nchw_src;

  // convert shape to channels last
  if (nchw_src.tensor_type().has_shape()) {
    const auto& nchw_shape = nchw_src.tensor_type().shape();
    const int rank = nchw_shape.dim_size();
    // N and C dims are required. Some operators like AveragePool allow 1D input
    if (rank < 3) {
      fail_shape_inference("Output tensor must have at least 3 dimensions");
    }

    // Convert output shape from N, C, H {, W, ...} to N, H {, W, ...}, C.
    auto& nhwc_shape = *nhwc_dst.mutable_tensor_type()->mutable_shape();
    nhwc_shape.Clear();
    *nhwc_shape.add_dim() = nchw_shape.dim(0);
    for (int i = 2; i < rank; i++) {
      *nhwc_shape.add_dim() = nchw_shape.dim(i);
    }

    *nhwc_shape.add_dim() = nchw_shape.dim(1);
  }
}

/**
 * Helper function that sets the values of the destination 1D TensorProto to the transpose (i.e., NCHW)
 * of the source TensorProto (NHWC).
 *
 * The TensorProtos are assumed to have raw data.
 *
 * Ex: a NHWC input [1.0f, 2.0f, 3.0f, 4.0f] is converted to [1.0f, 4.0f, 2.0f, 3.0f] for NCHW.
 *
 * /param nhwc_src The source TypeProto in NHWC layout.
 * /param nchw_dst The destination TypeProto in NCHW layout.
 */
template <typename ElemType>
static void TransposeRawConstantInput(const ONNX_NAMESPACE::TensorProto& nhwc_src, ONNX_NAMESPACE::TensorProto& nchw_dst,
                                      const char* input_name) {
  // This function must only be called with tensor protos with raw data.
  assert(nhwc_src.has_raw_data());
  assert(nchw_dst.has_raw_data());

  const std::vector<ElemType> nhwc_vals = ONNX_NAMESPACE::ParseData<ElemType>(&nhwc_src);
  const size_t num_vals = nhwc_vals.size();

  if (num_vals < 3) {
    fail_shape_inference(
        "Resize operator's '", input_name,
        "' input must have at least 3 elements "
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
  std::string& mut_vals = *nchw_dst.mutable_raw_data();
  mut_vals.resize(nchw_vals.size() * sizeof(ElemType));
  std::copy(nchw_vals.cbegin(), nchw_vals.cend(), reinterpret_cast<ElemType*>(mut_vals.data()));
}

//
// NhwcInferenceContext
//

NhwcInferenceContext::NhwcInferenceContext(ONNX_NAMESPACE::InferenceContext& ctx) : ctx_(ctx) {
  // copy any existing type and shape info, and convert to NCHW for usage with the ONNX inferencing
  const auto* nhwc_type = ctx_.getInputType(0);
  if (nhwc_type != nullptr) {
    TransposeShapeToNchw(*nhwc_type, input_type_);
  }

  nhwc_type = ctx_.getOutputType(0);
  if (nhwc_type != nullptr) {
    TransposeShapeToNchw(*nhwc_type, output_type_);
  }
}

const ONNX_NAMESPACE::AttributeProto* NhwcInferenceContext::getAttribute(const std::string& name) const {
  return ctx_.getAttribute(name);
}

size_t NhwcInferenceContext::getNumInputs() const noexcept {
  return ctx_.getNumInputs();
}

const ONNX_NAMESPACE::TypeProto* NhwcInferenceContext::getInputType(size_t index) const {
  return (index == 0) ? &input_type_ : ctx_.getInputType(index);
}

const ONNX_NAMESPACE::TensorProto* NhwcInferenceContext::getInputData(size_t index) const {
  // we can't return the NHWC input data without transposing it, but wouldn't expect to be asked for it
  // during shape inferencing as getInputData is only used to retrieve things that may have small
  // constant initializers (e.g. something like the min and max values of a Clip operator).
  return index == 0 ? nullptr : ctx_.getInputData(index);
}

size_t NhwcInferenceContext::getNumOutputs() const noexcept {
  return ctx_.getNumOutputs();
}

ONNX_NAMESPACE::TypeProto* NhwcInferenceContext::getOutputType(size_t index) {
  return (index == 0) ? &output_type_ : ctx_.getOutputType(index);
}

ONNX_NAMESPACE::GraphInferencer* NhwcInferenceContext::getGraphAttributeInferencer(const std::string& attribute_name) {
  return ctx_.getGraphAttributeInferencer(attribute_name);
}

const ONNX_NAMESPACE::TensorShapeProto* NhwcInferenceContext::getSymbolicInput(size_t index) const {
  return ctx_.getSymbolicInput(index);
}

const ONNX_NAMESPACE::SparseTensorProto* NhwcInferenceContext::getInputSparseData(size_t index) const {
  return ctx_.getInputSparseData(index);
}

// Propagate the inferred type/shape info to output 0, converting any inferred shape from NCHW to NHWC.
void NhwcInferenceContext::PropagateOutputShape() {
  auto& nhwc_tp = *ctx_.getOutputType(0);
  TransposeShapeToNhwc(output_type_, nhwc_tp);
}

//
// NhwcResizeInferenceContext
//

NhwcResizeInferenceContext::NhwcResizeInferenceContext(ONNX_NAMESPACE::InferenceContext& ctx) : ctx_(ctx) {
  // copy any existing type and shape info, and convert to NCHW for usage with the ONNX inferencing
  const auto* nhwc_type = ctx_.getInputType(0);
  if (nhwc_type != nullptr) {
    TransposeShapeToNchw(*nhwc_type, input_type_);
  }

  const size_t num_inputs = ctx_.getNumInputs();

  // Skip ROI input processing.
  // Enable processing ROI when an EP which supports ROI starts using layout transformer.
  // NNAPI and QNN, which currently use the layout transformer, do not support it.

  // Scales input
  if (num_inputs > scales_input_index) {
    TransposeScalesInput();
  }

  // Sizes input
  if (num_inputs > sizes_input_index) {
    TransposeSizesInput();
  }

  nhwc_type = ctx_.getOutputType(0);
  if (nhwc_type != nullptr) {
    TransposeShapeToNchw(*nhwc_type, output_type_);
  }
}

  const ONNX_NAMESPACE::AttributeProto* NhwcResizeInferenceContext::getAttribute(const std::string& name) const {
  return ctx_.getAttribute(name);
}

size_t NhwcResizeInferenceContext::getNumInputs() const noexcept {
  return ctx_.getNumInputs();
}

const ONNX_NAMESPACE::TypeProto* NhwcResizeInferenceContext::getInputType(size_t index) const {
  return (index == 0) ? &input_type_ : ctx_.getInputType(index);
}

const ONNX_NAMESPACE::TensorProto* NhwcResizeInferenceContext::getInputData(size_t index) const {
  // Return NCHW data for the 'scales' and 'sizes' constant inputs.
  //
  // For other inputs, we can't return the NHWC input data without transposing it, but wouldn't expect to be
  // asked for it during shape inferencing as getInputData is only used to retrieve things that may have small
  // constant initializers.
  switch (index) {
    case scales_input_index:
      return &scales_input_data_;
    case sizes_input_index:
      return &sizes_input_data_;
    default:
      return ctx_.getInputData(index);
  }
}

size_t NhwcResizeInferenceContext::getNumOutputs() const noexcept {
  return ctx_.getNumOutputs();
}

ONNX_NAMESPACE::TypeProto* NhwcResizeInferenceContext::getOutputType(size_t index) {
  return (index == 0) ? &output_type_ : ctx_.getOutputType(index);
}

ONNX_NAMESPACE::GraphInferencer* NhwcResizeInferenceContext::getGraphAttributeInferencer(const std::string& attribute_name) {
  return ctx_.getGraphAttributeInferencer(attribute_name);
}

const ONNX_NAMESPACE::TensorShapeProto* NhwcResizeInferenceContext::getSymbolicInput(size_t index) const {
  return ctx_.getSymbolicInput(index);
}

const ONNX_NAMESPACE::SparseTensorProto* NhwcResizeInferenceContext::getInputSparseData(size_t index) const {
  return ctx_.getInputSparseData(index);
}

// Propagate the inferred type/shape info to output 0, converting any inferred shape from NCHW to NHWC.
void NhwcResizeInferenceContext::PropagateOutputShape() {
  auto& nhwc_tp = *ctx_.getOutputType(0);
  TransposeShapeToNhwc(output_type_, nhwc_tp);
}

// Initializes scales_input_data_ with the NCHW version of the NHWC scales input.
void NhwcResizeInferenceContext::TransposeScalesInput() {
  auto* scales_data = ctx_.getInputData(scales_input_index);
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

// Initializes sizes_input_data_ with the NCHW version of the NHWC sizes input.
void NhwcResizeInferenceContext::TransposeSizesInput() {
  auto* sizes_data = ctx_.getInputData(sizes_input_index);
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

}  // namespace contrib
}  // namespace onnxruntime
