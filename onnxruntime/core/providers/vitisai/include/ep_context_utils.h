#pragma once

// Standard headers/libs.
#include <vector>
#include <string>
#include <memory>

// 1st-party headers/libs.
#include "core/providers/shared_library/provider_api.h"

namespace fs = std::filesystem;

namespace onnxruntime {

static constexpr const char* kEPContextOp = "EPContext";
static constexpr const char* kMainContextAttr = "main_context";
static constexpr const char* kEPCacheContextAttr = "ep_cache_context";
static constexpr const char* kEmbedModeAttr = "embed_mode";
static constexpr const char* kPartitionNameAttr = "partition_name";
static constexpr const char* kSourceAttr = "source";
static constexpr const char* kEPSDKVersionAttr = "ep_sdk_version";
static constexpr const char* kONNXModelFileNameAttr = "onnx_model_filename";
static constexpr const char* kNotesAttr = "notes";
static constexpr const char* kEPContextOpDomain = "com.microsoft";
static constexpr const char* kEPContextOpName = "VitisAIEPContextOp";

std::unique_ptr<ONNX_NAMESPACE::FunctionProto>
ConvertIndexedSubGraphToFunctionProto(const IndexedSubGraph&, const Graph&);

std::unique_ptr<IndexedSubGraph> ConvertFunctionProtoToIndexedSubGraph(
    const std::unique_ptr<ONNX_NAMESPACE::FunctionProto>&);

std::string SerializeCapabilities(
    const std::vector<std::unique_ptr<ComputeCapability>>&, const Graph&);

void DeserializeCapabilities(
    const std::string&, std::vector<std::unique_ptr<ComputeCapability>>&);

std::string SerializeOrigialGraph(const GraphViewer&);

// Ref.: `CreateEpContextModel()` in the file "graph_partitioner.cc".
ONNX_NAMESPACE::ModelProto* CreateEPContexModel(const GraphViewer&, const std::string&, const std::string&, const int64_t,
                                                const std::string&, const std::string&, bool, const logging::Logger*);

// Ref.: `static common::Status Save(Model& model, int fd)` in the file "model.h".
void DumpEPContextModel(const std::unique_ptr<ONNX_NAMESPACE::ModelProto>&, const std::string&);

const Node* GetEPContextNodePtr(const Graph&);

bool ValidateEPContextNode(const Graph&);

void CreateEPContexNodes(Graph*, const std::vector<IExecutionProvider::FusedNodeAndGraph>&, const std::string&, const std::string&,
                         const int64_t, const std::string&, const std::string&, bool, const logging::Logger*);

std::string RetrieveEPContextCache(const Graph&, const PathString&, bool binary_mode = true);

void RetrieveBackendCacheInfo(const Graph&, std::string&, std::string&);

std::unique_ptr<GraphViewer> RetrieveOriginalGraph(const Graph&);

bool GraphHasEPContextNode(const Graph&);

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>&);

const Path& GetTopLevelModelPath(const GraphViewer&);

bool GetEPContextModelFileLocation(
    const std::string&, const PathString&, bool, PathString&);

// The file for EP context cache is in the same folder as the EP context model file.
PathString GetEPContextCacheFileLocation(const PathString&, const PathString&);

std::string Slurp(const fs::path&);

std::string GetBackendCompileCache(const fs::path&);

void RestoreBackendCompileCache(const fs::path&, const std::string&);

std::vector<const NodeArg*> FilterOutputNodeArgs(const Node&);

std::vector<int64_t> GetNodeArgShape_I64(const NodeArg&);

std::string GetModelSignature(const GraphViewer&);

std::string HashFileContentWithMD5(const std::string&);

}  // namespace onnxruntime
