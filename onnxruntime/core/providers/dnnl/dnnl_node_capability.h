// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

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
  virtual bool Supported(const Node* node) const = 0;
};

/**
 * Default impelementation of the DnnlNodeCapability interface
 * This class can be used if the only thing needed to
 * decided if the we are capable of running the node using
 * the DnnlExecutionProvider is the input data type.
 *
 * The default constructor assumes that "float" data
 * type is supported and no other data types.
 *
 * To add additional data types an array of data types
 * can be passed as strings i.e.
 * `DnnlDefaultNodeCapability({"float", "int8"})`
 * Would indicate that "float" and "int8" are supported.
 *
 * At this time the possible data types strings are:
 * - "float"
 * - "float16"
 * - "bfloat16"
 * - "double"
 * - "int8"
 * - "int16"
 * - "int32"
 * - "int64"
 * - "uint8"
 * - "uint16"
 * - "uint32"
 * - "uint64"
 * - "complex64"
 * - "complex128"
 * - "string"
 * - "bool"
 *
 * The strings are from the data_type_utils.cc
 * TypesWrapper::TypesWrapper() member function. If a type
 * is expected but not found in the above list see if it is
 * assigned in the data_type_utils.cc file.
 *
 * This currently only checks the data type of input[0]. If
 * this does not work for the Node then this class will need
 * to be updated or another DnnlNodeCapability class will need to be
 * implemented for the operator in question.
 */
class DnnlDefaultNodeCapability : public DnnlNodeCapability {
 public:
  DnnlDefaultNodeCapability();
  DnnlDefaultNodeCapability(std::vector<std::string> inputTypes);

  bool Supported(const Node* node) const override;

 protected:
  bool IsTypeSupported(const Node* node) const;

 private:
  std::vector<std::string> inputTypes_;
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
  DnnlPoolNodeCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

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
  DnnlBatchNormalizationNodeCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

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
  DnnlReduceMeanNodeCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

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
  DnnlSoftmaxNodeCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

 private:
  bool IsAttributeSupported(const Node* node) const;
};

/**
 * Decide if a MatMul op is supported by DnnlExecutionProvider
 */
class DnnlMatMulNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlMatMulNodeCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a MatMulInteger op is supported by DnnlExecutionProvider
 */
class DnnlMatMulIntegerNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlMatMulIntegerNodeCapability() : DnnlDefaultNodeCapability({"int8", "uint8"}) {}

  bool Supported(const Node* node) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
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
  DnnlSumNodeCapability() : DnnlDefaultNodeCapability({"float" /*, "float16", "bfloat16", "int8", "uint8"*/}) {}

  bool Supported(const Node* node) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a Binary op is supported by DnnlExecutionProvider
 */
class DnnlBinaryNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlBinaryNodeCapability() : DnnlDefaultNodeCapability({"int8", "uint8", "float"}) {}

  bool Supported(const Node* node) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

class DnnlElementwiseCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlElementwiseCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

 private:
  bool IsDimensionSupported(const Node* node) const;
};

/**
 * Decide if a Gemm op is supported by DnnlExecutionProvider
 */
class DnnlGemmNodeCapability : public DnnlDefaultNodeCapability {
 public:
  DnnlGemmNodeCapability() : DnnlDefaultNodeCapability({"float"}) {}

  bool Supported(const Node* node) const override;

 private:
  DnnlMatMulNodeCapability _matmul;
  DnnlBinaryNodeCapability _binary;
};

}  // namespace onnxruntime
