#pragma once

namespace OrtGraphApis {
ORT_API(const OrtGraphApi*, GetGraphApi, uint32_t version);

ORT_API_STATUS_IMPL(OrtGraph_GetName, const OrtGraphViewer* graph, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtGraph_IsConstantInitializer, const OrtGraphViewer* graph, const char* name, bool check_outer_scope, _Out_ bool* out);

ORT_API_STATUS_IMPL(OrtGraph_GetNodesIndexInTopologicalOrder, const OrtGraphViewer* graph, int execution_order, _Out_ const size_t** nodes_index_in_topological_order, _Out_ size_t* num_nodes);

ORT_API_STATUS_IMPL(OrtGraph_IsSubgraph, const OrtGraphViewer* graph, _Out_ bool* out);

ORT_API_STATUS_IMPL(OrtGraph_GetParenNode, const OrtGraphViewer* graph, _Outptr_ const OrtNode** parent_node);

ORT_API_STATUS_IMPL(OrtGraph_GetModelPath, const OrtGraphViewer* graph, _Outptr_ const void** model_path);

ORT_API_STATUS_IMPL(OrtGraph_GetRequiredInputs, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len);

ORT_API_STATUS_IMPL(OrtGraph_GetAllInputs, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len);

ORT_API_STATUS_IMPL(OrtGraph_GetAllInitializers, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len);

ORT_API_STATUS_IMPL(ReleaseCharArray, const char** char_array);

ORT_API_STATUS_IMPL(OrtGraph_GetOrtNode, const OrtGraphViewer* graph, size_t node_index, _Outptr_ const OrtNode** node);

ORT_API_STATUS_IMPL(OrtGraph_GetNodesConsumingInput, const OrtGraphViewer* graph, const char* input_name, _Outptr_ const OrtNode*** consumers, _Out_ size_t* num_consumers);

ORT_API_STATUS_IMPL(ReleaseOrtNodeArray, const OrtNode** nodes);

ORT_API_STATUS_IMPL(OrtGraph_GetNodeProducingOutput, const OrtGraphViewer* graph, const char* output_name, _Outptr_ const OrtNode** node);

ORT_API_STATUS_IMPL(OrtGraph_NumberOfNodes, const OrtGraphViewer* graph, _Out_ int* num_nodes);

ORT_API_STATUS_IMPL(OrtGraph_MaxNodeIndex, const OrtGraphViewer* graph, _Out_ int* max_node_index);

ORT_API_STATUS_IMPL(OrtGraph_GetOutputSize, const OrtGraphViewer* graph, _Out_ size_t* output_len);

ORT_API_STATUS_IMPL(OrtGraph_GetIthOutputName, const OrtGraphViewer* graph, size_t i, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtGraph_GetIthOutputElemType, const OrtGraphViewer*, size_t i, _Out_ int32_t* out);

ORT_API_STATUS_IMPL(OrtGraph_GetInitializerTensor, const OrtGraphViewer* graph, const char* initializer_name, _Outptr_ OrtTensorRef** tensor);

ORT_API_STATUS_IMPL(OrtGraph_ReleaseInitializerTensor, OrtTensorRef* tensor);

ORT_API_STATUS_IMPL(OrtGraph_GetValueInfo, const OrtGraphViewer* graph, const char* name, _Outptr_ OrtValueInfoRef** out);

ORT_API_STATUS_IMPL(OrtGraph_ReleaseValueInfo, OrtValueInfoRef* value_info);

ORT_API_STATUS_IMPL(OrtGraph_SerializeToArray, const OrtGraphViewer* graph, _Out_ void** data, _Out_ size_t* data_size);

ORT_API_STATUS_IMPL(OrtGraph_DumpOnnxModel, const OrtGraph* graph, const char* onnx_model_path);

ORT_API_STATUS_IMPL(OrtGraph_CreateOrUpdateEpCtxGraph,
                    const OrtGraphViewer* graph,
                    const char* node_name,
                    const int64_t main_context,
                    const int64_t embed_mode,
                    const char* cache_path,
                    char* cache_data,
                    size_t size,
                    const char* const* extra_attr_keys,
                    const char* const* extra_attr_values,
                    size_t extra_attr_num,
                    _Outptr_ OrtGraph** ep_context_graph);

ORT_API_STATUS_IMPL(OrtGraph_GetSubGraph, const OrtGraphViewer* graph, const int node_num, const size_t* node_indices, _Outptr_ const OrtGraphViewer** subgraph);

ORT_API_STATUS_IMPL(OrtGraph_ReleaseGraph, const OrtGraph* graph);

ORT_API_STATUS_IMPL(OrtGraph_ReleaseGraphViewer, const OrtGraphViewer* graph);

ORT_API_STATUS_IMPL(OrtGraph_IsSameGraph, const OrtGraphViewer* graph1, const OrtGraphViewer* graph2, _Out_ bool* is_same);

ORT_API_STATUS_IMPL(OrtNode_GetName, const OrtNode* node, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetDescription, const OrtNode* node, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetDomain, const OrtNode* node, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_SinceVersion, const OrtNode* node, _Out_ int* out);

ORT_API_STATUS_IMPL(OrtNode_GetExecutionProviderType, const OrtNode* node, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetOpType, const OrtNode* node, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetImplicitInputSize, const OrtNode* node, _Out_ size_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetIthImplicitInputName, const OrtNode* node, size_t i, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetNumInputs, const OrtNode* node, _Out_ size_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetIthInputName, const OrtNode* node, size_t i, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetNumOutputs, const OrtNode* node, _Out_ size_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetIthOutputName, const OrtNode* node, size_t i, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetIndex, const OrtNode* node, _Out_ size_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeNames, const OrtNode* node, _Out_ const char*** names, _Out_ size_t* num);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeSize, const OrtNode* node, _Out_ size_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeType, const OrtNode* node, const char* attribute, _Out_ int* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeKeyCount, const OrtNode* node, const char* key, _Out_ size_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeIntSize, const OrtNode* node, const char* key, _Out_ int* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeFloatSize, const OrtNode* node, const char* key, _Out_ int* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeStringSize, const OrtNode* node, const char* key, _Out_ int* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeIthInt, const OrtNode* node, const char* key, int i, _Out_ int64_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeIthFloat, const OrtNode* node, const char* key, int i, _Out_ float* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeIthStr, const OrtNode* node, const char* key, int i, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeIthStrWithSize, const OrtNode* node, const char* key, int i, _Outptr_ const char** out, _Outptr_ size_t* size);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeStr, const OrtNode* node, const char* key, _Outptr_ const char** out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeStrWithSize, const OrtNode* node, const char* key, _Outptr_ const char** out, _Outptr_ size_t* size);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeInt, const OrtNode* node, const char* key, _Out_ int64_t* out);

ORT_API_STATUS_IMPL(OrtNode_GetAttributeFloat, const OrtNode* node, const char* key, _Out_ float* out);

ORT_API_STATUS_IMPL(OrtNode_GetSubgraphs, const OrtNode* node, _Outptr_ const OrtGraphViewer*** subgraphs, _Out_ size_t* num_subgraphs);

ORT_API_STATUS_IMPL(OrtFreeMem, void* p);

}
