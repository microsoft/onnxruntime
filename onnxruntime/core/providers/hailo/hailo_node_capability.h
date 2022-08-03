/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include <unordered_set>

namespace onnxruntime {

// redefinition of `enum TensorProto_DataType : int` to help code readability by using shorter names.
enum ORT_DataType : int {
    type_undefined = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
    type_float32 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    type_uint8 = ONNX_NAMESPACE::TensorProto_DataType_UINT8,
    type_uint16 = ONNX_NAMESPACE::TensorProto_DataType_UINT16,
};
/**
 * Pure virtual base class
 *
 * Individual implementations of this class are expected
 * to implement the Supported() member function.
 *
 * The Supported() function is expected to use the contents
 * of the onnxruntime::Node to decided if that node is
 * supported in the HailoExecutionProvider
 */
class HailoNodeCapability {
 public:
    virtual ~HailoNodeCapability(){};
    /**
     * virtual function expected to be implemented for different node
     * types.
     * @param node a onnxruntime::Node from the model
     *
     * @return true if the onnxRuntime::Node is supported in the
     * HailoExecutionProvider return false otherwise.
     */
    virtual bool Supported(const Node& node, const GraphViewer& graph_viewer) const = 0;
};

/**
 * Default implementation of the HailoNodeCapability interface
 * This class can be used if the only thing needed to decide if
 * the operation is supported is the input data type.
 *
 **
 * This currently only checks the data type of input[0].
 * TODO: consider adding more checks to determine if an operation
 * is supported. currently only checking for first input dtype.
 */
class HailoDefaultNodeCapability : public HailoNodeCapability {
 public:
    HailoDefaultNodeCapability();

    bool Supported(const Node& node, const GraphViewer& graph_viewer) const override;

 protected:
    virtual bool IsTypeSupported(const Node& node) const;
    std::vector<ORT_DataType> m_supported_input_types;
};

/*
* Works similar to the `HailoDefaultNodeCapability` class except that this
* will check all the inputs of the node.
*/
class HailoMultiInputNodeCapability: public HailoDefaultNodeCapability {
 public:
    HailoMultiInputNodeCapability();

 protected:
    bool IsTypeSupported(const Node& node) const override;
};

}  // namespace onnxruntime
