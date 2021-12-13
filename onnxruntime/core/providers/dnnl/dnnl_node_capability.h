// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include <unordered_set>

namespace onnxruntime {

// redefinition of `enum TensorProto_DataType : int` to help code readability by using shorter names.
enum ORT_DataType : int {
  type_undefined = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
  type_float32 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
  type_uint8 = ONNX_NAMESPACE::TensorProto_DataType_UINT8,
  type_int8 = ONNX_NAMESPACE::TensorProto_DataType_INT8,
  type_uint16 = ONNX_NAMESPACE::TensorProto_DataType_UINT16,
  type_int16 = ONNX_NAMESPACE::TensorProto_DataType_INT16,
  type_int32 = ONNX_NAMESPACE::TensorProto_DataType_INT32,
  type_int64 = ONNX_NAMESPACE::TensorProto_DataType_INT64,
  type_string = ONNX_NAMESPACE::TensorProto_DataType_STRING,
  type_bool = ONNX_NAMESPACE::TensorProto_DataType_BOOL,
  type_float16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
  type_double = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
  type_uint32 = ONNX_NAMESPACE::TensorProto_DataType_UINT32,
  type_uint64 = ONNX_NAMESPACE::TensorProto_DataType_UINT64,
  type_complex64 = ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64,
  type_complex128 = ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128,
  type_bfloat16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
};
/**
 * Pure virtual base class
 *
 * Individual implementations of this class are expected
 * to implement the Supported() member function.
 *
 * The Supported() function is expected to use the contents
 * of the onnxruntime::Node to decided if that node is
 * supported in the DnnlExecutionProvider
 */
class DnnlNodeCapability {
 public:
  virtual ~DnnlNodeCapability(){};
  /**
   * virtual function expected to be implemented for different node
   * types.
   * @param node a onnxruntime::Node from the model
   *
   * @return true if the onnxRuntime::Node is supported in the
   * DnnlExecutionProvider return false otherwise.
   */
  virtual bool Supported(const Node* node, const GraphViewer& graph_viewer) const = 0;
};

/**
 * Default impelementation of the DnnlNodeCapability interface
 * This class can be used if the only thing needed to
 * decided if the we are capable of running the node
 * is the input data type.
 *
 * The default constructor assumes that type_float32 data
 * type is supported and no other data types.
 *
 * To add additional data types an array of ORT_DataTypes
 * can be passed into the custructor i.e.
 * `DnnlDefaultNodeCapability({type_float32, type_int8})`
 * Would indicate that "float" and "int8" are supported.
 *
 * This currently only checks the data type of input[0]. If
 * this does not work for the Node then this class will need
 * to be updated or another DnnlNodeCapability class will need to be
 * implemented for the operator in question.
 */
class DnnlDefaultNodeCapability : public DnnlNodeCapability {
 public:
  DnnlDefaultNodeCapability();
  DnnlDefaultNodeCapability(std::vector<ORT_DataType> inputTypes);

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 protected:
  bool IsTypeSupported(const Node* node) const;

 private:
  std::vector<ORT_DataType> inputTypes_;
};

/*
* Works similar to the `DnnlDefaultNodeCapability` class except that this
* will check the input of all input nodes.
*
* Example usage:
* std::unordered_set T1 = {type_float32};
* std::unordered_set T2 = {type_uint8, type_int32, type_float32};
* DnnlDefaultMultiInputNodeCapability({T1, T2});
*
* The number of inputs and the number of unordered_sets must match. For this reason
* this capability class is not sutable for nodes that may have a varible number of
* inputs.
*
* All types for all inputs must be specified.
*/
class DnnlDefaultMultiInputNodeCapability : public DnnlNodeCapability {
 public:
  DnnlDefaultMultiInputNodeCapability(std::vector<std::unordered_set<ORT_DataType>> inputTypes);

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 protected:
  bool IsTypeSupported(const Node* node) const;

 private:
  std::vector<std::unordered_set<ORT_DataType>> inputTypes_;
};

/**
 * Decide if a Pool op is supported by DnnlExecutionProvider
 *
 * Dnnl does not support all dimension types for Pooling operators
 * In addition the "dilations" attribute is not yet supported for
 * MaxPool operator.
 */
class DnnlPoolNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlPoolNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsAttributeSupported(const Node* node) const;
  bool IsDimensionSupported(const Node* node) const;
  bool IsMaxPoolIndicesSupported(const Node* node) const;
};

/**
 * Decide if a BatchNormalization op is supported by DnnlExecutionProvider
 */
class DnnlBatchNormalizationNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlBatchNormalizationNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a ReduceMean op is supported by DnnlExecutionProvider
 *
 * Dnnl does not support the "keepdims" attribute when it is `0`
 */
class DnnlReduceMeanNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlReduceMeanNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsAttributeSupported(const Node* node) const;
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a Softmax op is supported by DnnlExecutionProvider
 *
 * Dnnl Softmax doesnt support few attribute values for opset < 13 with axis values anything other than 2
 */
class DnnlSoftmaxNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlSoftmaxNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsAttributeSupported(const Node* node) const;
};

/**
 * Decide if a MatMul op is supported by DnnlExecutionProvider
 */
class DnnlMatMulNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlMatMulNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a MatMulInteger op is supported by DnnlExecutionProvider
 */
class DnnlMatMulIntegerNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlMatMulIntegerNodeCapability() : DnnlDefaultNodeCapability({type_int8, type_uint8}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node, const GraphViewer& graph_viewer) const;
  bool IsWeightZeroPointConstantZero(const NodeArg* node, const GraphViewer& graph_viewer) const;
};

/**
 * Decide if aSum op is supported by DnnlExecutionProvider
 * OneDNN does not support Numpy-style broadcasting for 'Sum'
 */
class DnnlSumNodeCapability : public DnnlDefaultNodeCapability {
 public:
  // OneDNN reports support for sum of type f32, f16, bf16, i8 and u8
  // Onnx reports support for float, float16, bfloat16, and double
  // Onnxruntime only has unittests for float and double.
  // To enable float16 and bfloat16 we will should add tests to verify those data types.
  DnnlSumNodeCapability() : DnnlDefaultNodeCapability({type_float32 /*, type_float16, type_bfloat16, type_int8, type_uint8*/}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a Binary op is supported by DnnlExecutionProvider
 */
class DnnlBinaryNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlBinaryNodeCapability() : DnnlDefaultNodeCapability({type_int8, type_uint8, type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
* Decide if an Elementwise op is supported by DnnlExecutionProvider
* Elementwise ops are:
* Abs, Elu, Exp, Log, Relu, Round, Sigmoid, Softplus, Sqrt, Tanh
*/
class DnnlElementwiseCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlElementwiseCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

class DnnlPowNodeCapability : public DnnlDefaultMultiInputNodeCapability {
 public:
  DnnlPowNodeCapability()
    : DnnlDefaultMultiInputNodeCapability({/*T */{type_float32},
                                           /*T1*/{type_uint8, type_int8, type_int32, type_float32}}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node, const GraphViewer& graph_viewe) const;
};

/**
 * Decide if a Gemm op is supported by DnnlExecutionProvider
 */
class DnnlGemmNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlGemmNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  DnnlMatMulNodeCapability _matmul;
  DnnlBinaryNodeCapability _binary;
};

class DnnlReshapeNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlReshapeNodeCapability() : DnnlDefaultNodeCapability({type_float32,
                                                           type_float16,
                                                           type_bfloat16,
                                                           type_int32,
                                                           type_int8,
                                                           type_uint8}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a DynamicQuantizeLinear op is supported by DnnlExecutionProvider
 */
class DnnlDynamicQuantizeLinearNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlDynamicQuantizeLinearNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
};

class DnnlSqueezeNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlSqueezeNodeCapability() : DnnlDefaultNodeCapability({type_float32,
                                                           type_float16,
                                                           type_bfloat16,
                                                           type_int32,
                                                           type_int8,
                                                           type_uint8}) {}

  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node, const GraphViewer& graph_viewer) const;
};

class DnnlErfNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlErfNodeCapability() : DnnlDefaultNodeCapability({type_float32}) {}
  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsErfPartOfGelu(const Node* node, const GraphViewer& graph_viewer) const;
  bool IsInitilizedWithExpectedValue(const GraphViewer& graph_viewer, const NodeArg* node_arg, float expected_value) const;
  const Node* FirstParentByType(const Node& node, const std::string& parent_type) const;
  bool IsNodeFusable(const Node* node, const GraphViewer& graph_viewer) const;
  DnnlBinaryNodeCapability _binary;
};


class DnnlQAttentionNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlQAttentionNodeCapability() : DnnlDefaultNodeCapability({type_float32,
                                                              type_int8,
                                                              type_uint8}) {}
  bool Supported(const Node* node, const GraphViewer& graph_viewer) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};
                                 
}  // namespace onnxruntime
