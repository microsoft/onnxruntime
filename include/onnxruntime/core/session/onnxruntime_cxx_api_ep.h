// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api_ep.h"

namespace Ort {
namespace PluginEP {

struct ValueInfoRef {
explicit ValueInfoRef(OrtValueInfoRef*);
~ValueInfoRef();
const std::vector<int64_t> GetShape();
const ONNXTensorElementDataType GetTensorElementType();
private:
OrtValueInfoRef* value_info_;
};

struct Graph {
explicit Graph(const OrtGraphViewer*);
const OrtGraphViewer* GetGraph() { return graph_; }
const char* GetName();
bool IsConstantInitializer(const char* name, bool check_outer_scope);
const std::vector<size_t> GetNodesIndexInTopologicalOrder(int execution_order);
bool IsSubgraph();
std::shared_ptr<Node> GetParenNode();
std::filesystem::path GetModelPath();
std::vector<std::string> GetRequiredInputs();
std::vector<std::string> GetAllInputs();
std::vector<std::string> GetAllInitializers();
Node GetOrtNode(size_t node_index);
std::vector<Node> GetNodesConsumingInput(const char* input_name);
Node GetNodeProducingOutput(const char* output_name);
int NumberOfNodes();
int MaxNodeIndex();
size_t GetOutputSize();
std::string GetIthOutputName(size_t i);
int32_t GetIthOutputElemType(size_t i);
// std::shared_ptr<TensorRef> GetInitializerTensor(const char* initializer_name);
std::shared_ptr<ValueInfoRef> GetValueInfo(const char* name);
// void SerializeToArray(void** data, size_t* data_size);
// void DumpOnnxModel(const std::filesystem::path& onnx_model_path);
// CreateOrUpdateEpCtxGraph();
std::shared_ptr<Graph> GetSubGraph(std::vector<size_t> node_indices);
// bool IsSameGraph(const Graph& other);

private:
const OrtGraphViewer* graph_;
};

struct Node {
explicit Node(const OrtNode*);
const char* GetName();
const std::string GetDescription();
const std::string GetDomain();
int SinceVersion();
const std::string GetExecutionProviderType();
const std::string GetOpType();
size_t GetImplicitInputSize();
const std::string GetIthImplicitInputName(size_t i);
size_t GetNumInputs();
const std::string GetIthInputName(size_t i);
size_t GetNumOutputs();
const std::string GetIthOutputName(size_t i);
size_t GetIndex();
std::vector<std::string> GetAttributeNames();
size_t GetAttributeSize();
int GetAttributeType(std::string attribute_name);
size_t GetAttributeKeyCount(std::string attribute_name);
int GetAttributeIntSize(std::string attribute_name);
int GetAttributeFloatSize(std::string attribute_name);
int GetAttributeStringSize(std::string attribute_name);
int64_t GetAttributeIthInt(std::string attribute_name, size_t i);
float GetAttributeIthFloat(std::string attribute_name, size_t i);
const std::string GetAttributeIthStr(std::string attribute_name, size_t i);
// GetAttributeIthStrWithSize
const std::string GetAttributeStr(std::string attribute_name);
// GetAttributeStrWithSize
int64_t GetAttributeInt(std::string attribute_name);
float GetAttributeFloat(std::string attribute_name);
// GetSubgraphs
private:
const OrtNode* node_;
};

}
}

#include "onnxruntime_cxx_inline_ep.h"
