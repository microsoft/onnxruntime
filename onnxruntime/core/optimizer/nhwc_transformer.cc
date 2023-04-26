// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

/**
 * @brief For identifying layout sensive operators
 * as candidates for transforming to NHWC ops.
*/
struct OpIdInfo{
  const std::string optype_;
  const std::string domain_;
  const api::DataType data_type_;

  OpIdInfo(
    const std::basic_string_view<char>& op,
    const std::basic_string_view<char>& domain,
    api::DataType data_type
  ) : optype_(op), domain_(domain), data_type_(data_type)
  {}

  bool operator==(const OpIdInfo& other) const{
    return optype_ == other.optype_ && domain_ == other.domain_ && data_type_ == other.data_type_;
  }
};

/**
 * @brief Hash function for \ref OpIdInfo
*/
class OpIdHash{
  public:
  size_t operator()(const OpIdInfo& op) const
  {
    size_t h1 = std::hash<std::string>{}(op.optype_);
    size_t h2 = std::hash<std::string>{}(op.domain_);
    size_t h3 = size_t(op.data_type_);
    return h2 ^ (h1 << 4) ^ (h3 << 16);
  }
};

/**
 * @brief Information needed for operator layout transformation
*/
struct OpTransformInfo{
  const std::string optype_;
  const std::string domain_;
  const int version_;
  const bool has_channels_last_attrib_;
};

/**
 * Constructing a mapping table to identify operators that need
 * to be transformed, and map them to the new operators with
 * NHWC format
 */
std::unordered_map<OpIdInfo, OpTransformInfo, OpIdHash> conv_table;

static inline const OpTransformInfo* NhwcConvLookup(const api::GraphRef& graph, api::NodeRef& node){
  const auto& optype = node.OpType();
  const auto& domain = node.Domain();
  const auto info = graph.GetValueInfo(node.Inputs()[0]);
  const api::DataType dtype = info->DType();
  OpIdInfo key{optype, domain, dtype};

  const auto iter = conv_table.find(key);
  if (iter == conv_table.end()){
    return nullptr;
  }
  return &(iter->second);
}


NhwcTransformer::NhwcTransformer(AllocatorPtr cpu_allocator, std::shared_ptr<KernelRegistry> cpu_kernel_registry) noexcept
    : GraphTransformer("NhwcTransformer"), cpu_allocator_(std::move(cpu_allocator))
{
  if (!cpu_kernel_registry) {
      // This is a CPU op nodes optimizer, not useful if cpu EP is not available.
    return;
  }

  if (!conv_table.empty()){
    return;
  }

  // Make sure that the new nodes we are about to create during graph transformation,
  // their kernels are available in the cpu EP.

  {
    // int8 qconv
    OpKernelRegistryId qconv_int8{
        "QLinearConv", kMSDomain, 1, {{"T1", {DataTypeImpl::GetTensorType<int8_t>()}}}};
    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, qconv_int8.op_type_, qconv_int8.domain_,
        qconv_int8.version_, qconv_int8.type_constraints_, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table.emplace(
          OpIdInfo("QLinearConv", kOnnxDomain, api::DataType::INT8),
          OpTransformInfo{qconv_int8.op_type_, qconv_int8.domain_, qconv_int8.version_, true});
      conv_table.emplace(
          OpIdInfo("QLinearConv", kMSDomain, api::DataType::INT8),
          OpTransformInfo{qconv_int8.op_type_, qconv_int8.domain_, qconv_int8.version_, true});
    }
  }

  {
    // uint8 qconv
    OpKernelRegistryId qconv_uint8{
        "QLinearConv", kMSDomain, 1, {{"T1", {DataTypeImpl::GetTensorType<uint8_t>()}}}};
    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, qconv_uint8.op_type_, qconv_uint8.domain_,
        qconv_uint8.version_, qconv_uint8.type_constraints_, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table.emplace(
          OpIdInfo("QLinearConv", kOnnxDomain, api::DataType::UINT8),
          OpTransformInfo{qconv_uint8.op_type_, qconv_uint8.domain_, qconv_uint8.version_, true});
      conv_table.emplace(
          OpIdInfo("QLinearConv", kMSDomain, api::DataType::UINT8),
          OpTransformInfo{qconv_uint8.op_type_, qconv_uint8.domain_, qconv_uint8.version_, true});
    }
  }

  {
    // fp16 nhwc conv
    OpKernelRegistryId nhwc_conv_fp16{
        "NhwcFusedConv", kMSDomain, 1, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_,
        nhwc_conv_fp16.version_, nhwc_conv_fp16.type_constraints_, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table.emplace(
          OpIdInfo("Conv", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_, nhwc_conv_fp16.version_, false});
      conv_table.emplace(
          OpIdInfo("FusedConv", kMSDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_, nhwc_conv_fp16.version_, false});
    }
  }

  {
    // fp16 nhwc maxpool
    OpKernelRegistryId nhwc_maxpool_fp16{
        "MaxPool", kMSInternalNHWCDomain, 12, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_maxpool_fp16.op_type_, nhwc_maxpool_fp16.domain_,
        nhwc_maxpool_fp16.version_, nhwc_maxpool_fp16.type_constraints_, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table.emplace(
          OpIdInfo("MaxPool", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_maxpool_fp16.op_type_, nhwc_maxpool_fp16.domain_, nhwc_maxpool_fp16.version_, false});
    }
  }

  {
    // fp16 average maxpool
    OpKernelRegistryId nhwc_avgpool_fp16{
        "AveragePool", kMSInternalNHWCDomain, 11, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_avgpool_fp16.op_type_, nhwc_avgpool_fp16.domain_,
        nhwc_avgpool_fp16.version_, nhwc_avgpool_fp16.type_constraints_, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table.emplace(
          OpIdInfo("AveragePool", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_avgpool_fp16.op_type_, nhwc_avgpool_fp16.domain_, nhwc_avgpool_fp16.version_, false});
    }
  }

  {
    // fp16 global average maxpool
    OpKernelRegistryId nhwc_gavgpool_fp16{
        "GlobalAveragePool", kMSInternalNHWCDomain, 1, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_gavgpool_fp16.op_type_, nhwc_gavgpool_fp16.domain_,
        nhwc_gavgpool_fp16.version_, nhwc_gavgpool_fp16.type_constraints_, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table.emplace(
          OpIdInfo("GlobalAveragePool", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_gavgpool_fp16.op_type_, nhwc_gavgpool_fp16.domain_, nhwc_gavgpool_fp16.version_, false});
    }
  }

};

bool NhwcTransformer::IsActive() {
  return !conv_table.empty();
}

Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
#if defined(ORT_MINIMAL_BUILD)
  // update the producer/consumer info as previous optimizations may have invalidated it.
  // in a full build this will happen as part of Graph::Resolve.
  ORT_RETURN_IF_ERROR(graph.PopulateNodeArgToProducerConsumerLookupsFromNodes());
#endif

  GraphViewer graph_viewer(graph);
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  auto api_graph = MakeApiGraph(graph, cpu_allocator_, kCpuExecutionProvider);

  modified = false;
  for (std::unique_ptr<api::NodeRef>& node : api_graph->Nodes()) {
    // If the node is not supported in the CPU EP, skip it
    if (node->GetExecutionProviderType() != kCpuExecutionProvider) {
      continue;
    }

    // Only Conv and QLinearConv needs to be handled explicitly. The rest will be
    //  transformed if needed during transpose optimization.
    const auto* transform = NhwcConvLookup(*api_graph, *node);
    if (nullptr == transform) {
      continue;
    }

    // Skip if already transformed
    if (transform->has_channels_last_attrib_ &&
        node->GetAttributeIntDefault("channels_last", 0) == 1) {
      continue;
    }

    // Skip if unknown rank
    auto shape = NodeFromApiNode(*node).InputDefs()[0]->Shape();
    if (shape == nullptr) {
      continue;
    }

    // Convert to channels last
    if (transform->has_channels_last_attrib_) {
      node->SetAttributeInt("channels_last", 1);
    }
    size_t rank = shape->dim_size();
    std::vector<int64_t> input_perm = ChannelFirstToLastPerm(rank);
    std::vector<int64_t> output_perm = ChannelLastToFirstPerm(rank);
    WrapTransposesAroundNode(*api_graph, *node, {&input_perm}, {&output_perm});

    // Replace the operator if needed
    if (node->Domain() != transform->domain_ ||
        node->OpType() != transform->optype_ ||
        node->SinceVersion() != transform->version_) {
      SwapNodeOpTypeDomainAndSinceVersion(
        *api_graph, *node, transform->optype_,
        transform->domain_, transform->version_);
    }

    modified = true;
  }

  if (modified) {
    Optimize(*api_graph, /*allow_extended_ops*/ true, kCpuExecutionProvider, OptimizerMode::OPTIMIZE_TRANSPOSE,
             OrtEPCostCheck);
  }

  return Status::OK();
}

}  // namespace onnxruntime
