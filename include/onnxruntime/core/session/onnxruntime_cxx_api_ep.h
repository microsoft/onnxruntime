// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api_ep.h"

namespace Ort {
namespace PluginEP {

using VoidPtr = std::unique_ptr<void, std::function<void(void*)>>;

struct TensorRef {
explicit TensorRef(OrtTensorRef*);
~TensorRef();
const std::vector<int64_t> GetShape();
const ONNXTensorElementDataType GetTensorElementType();
const char* GetData();
size_t GetDataLen();
private:
OrtTensorRef* tensor_;
};

struct ValueInfoRef {
explicit ValueInfoRef(OrtValueInfoRef*);
~ValueInfoRef();
const std::vector<int64_t> GetShape();
const ONNXTensorElementDataType GetTensorElementType();
private:
OrtValueInfoRef* value_info_;
};

struct Graph {
explicit Graph(const OrtGraph*);
const OrtGraph* GetGraph() { return graph_; }
void DumpOnnxModel(const std::filesystem::path& onnx_model_path);
private:
const OrtGraph* graph_;
};
using GraphPtr = std::unique_ptr<PluginEP::Graph, std::function<void(PluginEP::Graph*)>>;

struct GraphViewer {
explicit GraphViewer(const OrtGraphViewer*);
const OrtGraphViewer* GetGraphViewer() { return graph_; }
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
std::shared_ptr<TensorRef> GetInitializerTensor(const char* initializer_name);
std::shared_ptr<ValueInfoRef> GetValueInfo(const char* name);
std::pair<VoidPtr, size_t> SerializeToArray();
GraphPtr CreateOrUpdateEpCtxGraph(const char* node_name,
                                                const int64_t main_context,
                                                const int64_t embed_mode,
                                                const char* cache_path,
                                                char* cache_data,
                                                size_t size,
                                                const char* const* extra_attr_keys,
                                                const char* const* extra_attr_values,
                                                size_t extra_attr_num);
GraphViewerPtr GetSubGraph(std::vector<size_t> node_indices);
bool IsSameGraph(GraphViewer& other);

private:
const OrtGraphViewer* graph_;
};
using GraphViewerPtr = std::unique_ptr<PluginEP::GraphViewer, std::function<void(PluginEP::GraphViewer*)>>;

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
const std::string GetAttributeStr(std::string attribute_name);
int64_t GetAttributeInt(std::string attribute_name);
float GetAttributeFloat(std::string attribute_name);
// TODO: add GetSubgraphs wrapper here
private:
const OrtNode* node_;
};

}
}

#include "onnxruntime_cxx_inline_ep.h"
