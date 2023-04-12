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
  NhwcInferenceContext(ONNX_NAMESPACE::InferenceContext& ctx);

  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override;

  size_t getNumInputs() const noexcept override;

  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override;

  const ONNX_NAMESPACE::TensorProto* getInputData(size_t index) const override;

  size_t getNumOutputs() const noexcept override;

  ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) override;

  ONNX_NAMESPACE::GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override;

  const ONNX_NAMESPACE::TensorShapeProto* getSymbolicInput(size_t index) const override;

  const ONNX_NAMESPACE::SparseTensorProto* getInputSparseData(size_t index) const override;

  // Propagate the inferred type/shape info to output 0, converting any inferred shape from NCHW to NHWC.
  void PropagateOutputShape();

 private:
  InferenceContext& ctx_;

  // These are the NCHW versions of the actual input and output types.
  // Provided to ONNX type/shape inferencing via the overridden InferenceContext::getInputType function.
  ONNX_NAMESPACE::TypeProto input_type_;
  ONNX_NAMESPACE::TypeProto output_type_;
};

/** Adapter class to enable ONNX shape inferencing to be used with the NHWC Resize layout-sensitive operator.
 * Currently only used by the QNN EP.
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
  explicit NhwcResizeInferenceContext(ONNX_NAMESPACE::InferenceContext& ctx);

  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override;

  size_t getNumInputs() const noexcept override;

  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override;

  const ONNX_NAMESPACE::TensorProto* getInputData(size_t index) const override;

  size_t getNumOutputs() const noexcept override;

  ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) override;

  ONNX_NAMESPACE::GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override;

  const ONNX_NAMESPACE::TensorShapeProto* getSymbolicInput(size_t index) const override;

  const ONNX_NAMESPACE::SparseTensorProto* getInputSparseData(size_t index) const override;

  // Propagate the ONNX inferred type/shape info to output 0, converting any inferred shape from NCHW to NHWC.
  void PropagateOutputShape();

 private:

  // Initializes scales_input_data_ with the NCHW version of the NHWC scales input.
  void TransposeScalesInput();

  // Initializes sizes_input_data_ with the NCHW version of the NHWC sizes input.
  void TransposeSizesInput();

  InferenceContext& ctx_;

  // These are the NCHW versions of the actual inputs and outputs.
  // Provided to ONNX type/shape inferencing via overridden
  // InferenceContext functions (e.g., getInputType, getInputData).
  ONNX_NAMESPACE::TypeProto input_type_;
  ONNX_NAMESPACE::TensorProto scales_input_data_;
  ONNX_NAMESPACE::TensorProto sizes_input_data_;
  ONNX_NAMESPACE::TypeProto output_type_;

  static constexpr size_t scales_input_index = 2;
  static constexpr size_t sizes_input_index = 3;
};

}  // namespace contrib
}  // namespace onnxruntime
