// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(ExecutionProvider);
ORT_RUNTIME_CLASS(ExecutionProviderFactory);
ORT_RUNTIME_CLASS(Node);
ORT_RUNTIME_CLASS(Graph);
ORT_RUNTIME_CLASS(GraphViewer);

typedef struct OrtCreateStream {
  int device_type;
  void*(ORT_API_CALL* CreateStreamFunc)(const OrtDevice*);
} OrtCreateStream;

typedef struct OrtMetaDef {
  char* name;
  char* domain;
  int since_version;

  char** inputs;
  size_t input_len;
  char** outputs;
  size_t output_len;
  char** constant_initializers;
  size_t initializer_len;

  char* doc_string;
} OrtMetaDef;

typedef struct OrtIndexedSubGraph {
  OrtMetaDef* meta_def; // TODO(leca): how to define a nested structure pointer?
  size_t* node_index;
  size_t node_index_len;
} OrtIndexedSubGraph;

typedef struct OrtComputeContext {
  void*(ORT_API_CALL* AllocateFunc)(void*, size_t, size_t);
  void(ORT_API_CALL* DestroyFunc)(void*, void*);
  void* allocator_handle;
  const char* node_name;
} OrtComputeContext;

typedef struct OrtNodeComputeInfo {
  int(ORT_API_CALL* CreateFunctionStateFunc)(OrtComputeContext*, void*, void**);
  OrtStatusPtr(ORT_API_CALL* ComputeFunc)(void*, void*, const OrtApi*, OrtKernelContext*);
  void(ORT_API_CALL* DestroyFunctionStateFunc)(void*);
} OrtNodeComputeInfo;

typedef struct OrtTensorRef {   // TODO(leca): OrtValueInfoRef inside OrtTensorRef?
  int64_t* shape;
  size_t shape_len;
  ONNXTensorElementDataType data_type;
  const char* data;
  size_t data_len;
} OrtTensorRef;

typedef struct OrtValueInfoRef {
  int64_t* shape;
  size_t shape_len;
  ONNXTensorElementDataType data_type;
} OrtValueInfoRef;

typedef struct OrtExecutionProvider {
#ifdef __cplusplus
  OrtExecutionProvider() : GetCapability{nullptr}, Compile{nullptr}, RegisterKernels{nullptr}, CanCopy{nullptr}, CopyTensor{nullptr}, CreatePreferredAllocators{nullptr}, type{nullptr}, create_stream{nullptr}, default_device{nullptr},
                           extra_param_for_create_state_func{nullptr}, extra_param_for_compute_func{nullptr} {}
#endif
  void(ORT_API_CALL* GetCapability)(const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph***);
  OrtStatusPtr(ORT_API_CALL* Compile)(OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo* node_compute_info);
  void(ORT_API_CALL* RegisterKernels)(OrtKernelRegistry* kernel_registry);
  bool(ORT_API_CALL* CanCopy)(const OrtDevice* source, const OrtDevice* target);
  OrtStatusPtr(ORT_API_CALL* CopyTensor)(const void* src, OrtMemoryInfoDeviceType source_device_type, OrtMemoryType source_mem_type, void* dst, OrtMemoryInfoDeviceType target_device_type, size_t count, void* stream);
  int(ORT_API_CALL* CreatePreferredAllocators)(OrtExecutionProvider* this_, OrtAllocator*** ort_allocators);
  void(ORT_API_CALL* ReleaseIndexedSubGraphs)(OrtIndexedSubGraph** indexed_sub_graphs, size_t num_sub_graph);
  const char* type;
  OrtCreateStream* create_stream;
  const OrtDevice* default_device;
  void* extra_param_for_create_state_func;
  void* extra_param_for_compute_func;
} OrtExecutionProvider;

typedef struct OrtExecutionProviderFactory {
  OrtExecutionProvider*(ORT_API_CALL* CreateExecutionProvider)(OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size);
} OrtExecutionProviderFactory;

struct OrtGraphApi {
const char*(ORT_API_CALL* OrtGraph_GetName)(const OrtGraphViewer*)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

bool(ORT_API_CALL* OrtGraph_IsConstantInitializer)(const OrtGraphViewer* graph, const char* name, bool check_outer_scope)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

size_t(ORT_API_CALL* OrtGraph_GetNodesIndexInTopologicalOrder)(const OrtGraphViewer* graph, int execution_order, _Out_ const size_t** nodes_index_in_topological_order);

bool(ORT_API_CALL* OrtGraph_IsSubgraph)(const OrtGraph* graph);

const OrtGraph*(ORT_API_CALL* OrtGraph_GetParentGraph)(const OrtGraph* graph);

const OrtNode*(ORT_API_CALL* OrtGraph_GetParenNode)(const OrtGraphViewer* graph);

const void*(ORT_API_CALL* OrtGraph_GetModelPath)(const OrtGraphViewer* graph);

const OrtGraph*(ORT_API_CALL* OrtGraph_GetOrtGraph)(const OrtGraphViewer* graph_viewer);

size_t(ORT_API_CALL* OrtGraph_GetInputsIncludingInitializers)(const OrtGraphViewer* graph, _Outptr_ const char*** input_names);

const OrtNode*(ORT_API_CALL* OrtGraph_GetOrtNode)(const OrtGraphViewer* graph, size_t node_index);

size_t(ORT_API_CALL* OrtGraph_GetNodesConsumingInput)(const OrtGraphViewer* graph, const char* input_name, _Outptr_ const OrtNode*** consumers); // TODO(leca): ValueConsumers::comprehensive ?

const OrtNode*(ORT_API_CALL* OrtGraph_GetNodeProducingOutput)(const OrtGraphViewer* graph, const char* output_name);

int(ORT_API_CALL* OrtGraph_NumberOfNodes)(const OrtGraphViewer*)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

int(ORT_API_CALL* OrtGraph_MaxNodeIndex)(const OrtGraphViewer* graph);

size_t(ORT_API_CALL* OrtGraph_GetOutputSize)(const OrtGraphViewer*)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

const char*(ORT_API_CALL* OrtGraph_GetIthOutputName)(const OrtGraphViewer*, size_t i)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

int32_t(ORT_API_CALL* OrtGraph_GetIthOutputElemType)(const OrtGraphViewer*, size_t i)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

bool(ORT_API_CALL* OrtGraph_GetInitializerTensor)(const OrtGraphViewer* graph, const char* initializer_name, _Outptr_ OrtTensorRef**);

bool(ORT_API_CALL* OrtGraph_GetValueInfo)(const OrtGraphViewer* graph, const char* name, _Outptr_ OrtValueInfoRef**);

size_t(ORT_API_CALL* OrtGraph_SerializeToArray)(const OrtGraphViewer*, _Out_ void** data)NO_EXCEPTION;  // TODO(leca): review and discuss

ORT_API2_STATUS(OrtGraph_GetSubGraph, const OrtGraphViewer* graph, const int node_num, const size_t* node_indices, _Outptr_ const OrtGraphViewer** subgraph); // TODO(yang): review and discuss

ORT_API2_STATUS(OrtGraph_ReleaseGraph, const OrtGraphViewer* graph);

const char*(ORT_API_CALL* OrtNode_GetName)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetDescription)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetDomain)(const OrtNode* node);

int(ORT_API_CALL* OrtNode_SinceVersion)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetExecutionProviderType)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetOpType)(const OrtNode* node);

size_t(ORT_API_CALL* OrtNode_GetImplicitInputSize)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetIthImplicitInputName)(const OrtNode* node, size_t i);

size_t(ORT_API_CALL* OrtNode_GetInputSize)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetIthInputName)(const OrtNode* node, size_t i);

size_t(ORT_API_CALL* OrtNode_GetOutputSize)(const OrtNode* node);

const char*(ORT_API_CALL* OrtNode_GetIthOutputName)(const OrtNode* node, size_t i);

size_t(ORT_API_CALL* OrtNode_GetIndex)(const OrtNode* node);

size_t(ORT_API_CALL* OrtNode_GetAttributeNames)(const OrtNode*, _Out_ const char*** names);

size_t(ORT_API_CALL* OrtNode_GetAttributeSize)(const OrtNode* node);

int(ORT_API_CALL* OrtNode_GetAttributeType)(const OrtNode* node, const char* attribute)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;  // AttributeProto_AttributeType

size_t(ORT_API_CALL* OrtNode_GetAttributeKeyCount)(const OrtNode* node, const char* key);

int(ORT_API_CALL* OrtNode_GetAttributeIntSize)(const OrtNode* node, const char* key);

int(ORT_API_CALL* OrtNode_GetAttributeFloatSize)(const OrtNode* node, const char* key);

int(ORT_API_CALL* OrtNode_GetAttributeStringSize)(const OrtNode* node, const char* key);

int64_t(ORT_API_CALL* OrtNode_GetAttributeIthInt)(const OrtNode* node, const char* key, int i);

float(ORT_API_CALL* OrtNode_GetAttributeIthFloat)(const OrtNode* node, const char* key, int i);

const char*(ORT_API_CALL* OrtNode_GetAttributeIthStr)(const OrtNode* node, const char* key, int i);

const char*(ORT_API_CALL* OrtNode_GetAttributeStr)(const OrtNode*, const char* key)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

int64_t(ORT_API_CALL* OrtNode_GetAttributeInt)(const OrtNode*, const char* key)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

float(ORT_API_CALL* OrtNode_GetAttributeFloat)(const OrtNode*, const char* key)NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

size_t(ORT_API_CALL* OrtNode_GetSubgraphs)(const OrtNode* node, _Outptr_ const OrtGraphViewer*** subgraphs);

ORT_API2_STATUS(OrtFreeMem, void* p);
};
typedef struct OrtGraphApi OrtGraphApi;
