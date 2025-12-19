// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <cstdint>
#include <deque>
#include <vector>
#include "core/graph/constants.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/layout_transformation/layout_transformation.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_transpose_optimization;
using namespace nhwc_map_internal;

namespace onnxruntime {

using namespace layout_transformation;

#ifdef USE_KLEIDIAI
bool KleidiFp32NhwcFilter(const onnx_transpose_optimization::api::GraphRef& graph,
                          onnx_transpose_optimization::api::NodeRef& node) {
  auto& base_node = NodeFromApiNode(node);

  ORT_UNUSED_PARAMETER(graph);
  if (base_node.InputDefs().size() < 2) {
    return false;
  }

  const auto* input_shape = base_node.InputDefs()[0]->Shape();
  if (input_shape == nullptr || input_shape->dim_size() != 4) {
    return false;
  }

  const auto& batch_dim = input_shape->dim(0);
  if (!utils::HasDimValue(batch_dim) || batch_dim.dim_value() != 1) {
    return false;
  }

  const auto pads_attr = node.GetAttributeInts("pads");
  if (pads_attr.has_value()) {
    const auto& pads = pads_attr.value();
    if (pads.size() != 4 || pads[0] != pads[2] || pads[1] != pads[3]) {
      return false;
    }
  }

  const auto inputs = node.Inputs();
  if (inputs.size() > 3 && !inputs[3].empty()) {
    return false;
  }

  const auto* weight_shape = base_node.InputDefs()[1]->Shape();
  if (weight_shape == nullptr || weight_shape->dim_size() != 4) {
    return false;
  }

  const auto& filter_dim = weight_shape->dim(0);
  const auto& kernel_h_dim = weight_shape->dim(2);
  const auto& kernel_w_dim = weight_shape->dim(3);

  if (!utils::HasDimValue(filter_dim) || filter_dim.dim_value() <= 1 ||
      !utils::HasDimValue(kernel_h_dim) || kernel_h_dim.dim_value() < 3 ||
      !utils::HasDimValue(kernel_w_dim) || kernel_w_dim.dim_value() < 3) {
    return false;
  }

  const auto dilations_opt = node.GetAttributeInts("dilations");
  if (dilations_opt.has_value()) {
    const auto& dilations = dilations_opt.value();
    if ((dilations.size() >= 1 && dilations[0] != 1) ||
        (dilations.size() >= 2 && dilations[1] != 1)) {
      return false;
    }
  }

  const auto group_opt = node.GetAttributeInt("group");
  if (group_opt.has_value() && group_opt.value() != 1) {
    return false;
  }

  return true;
}
#endif

static inline const OpTransformInfo*
NhwcConvLookup(
    const OpTransformMap& conv_table,
    const api::GraphRef& graph,
    api::NodeRef& node) {
  const auto& optype = node.OpType();
  const auto& domain = node.Domain();
  const auto inputs = node.Inputs();
  if (inputs.empty()) {
    // node with no input, can't be our transformation candidate.
    return nullptr;
  }
  const auto info = graph.GetValueInfo(inputs[0]);
  const api::DataType dtype = info->DType();
  OpIdInfo key{optype, domain, dtype};

  const auto iter = conv_table.find(key);
  if (iter == conv_table.end()) {
    return nullptr;
  }

  if (iter->second.filter_ != nullptr) {
    if (!iter->second.filter_(graph, node)) {
      return nullptr;
    }
  }

  return &(iter->second);
}

NhwcTransformer::NhwcTransformer(AllocatorPtr cpu_allocator,
                                 std::shared_ptr<KernelRegistry> cpu_kernel_registry,
                                 const logging::Logger& logger) noexcept
    : GraphTransformer("NhwcTransformer"), cpu_allocator_(std::move(cpu_allocator)) {
  if (!cpu_kernel_registry) {
    // This is a CPU op nodes optimizer, not useful if cpu EP is not available.
    return;
  }

  //
  // Constructing a mapping table from operators to be transformed to their target.
  // Make sure that the new nodes we are about to create during graph transformation,
  // their kernels are available in the cpu EP.
  //

  {
    // int8 qconv -> int8 nhwc qconv
    OpKernelRegistryId qconv_int8{
        "QLinearConv", kMSDomain, 1, {{"T1", {DataTypeImpl::GetTensorType<int8_t>()}}}};
    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, qconv_int8.op_type_, qconv_int8.domain_,
        qconv_int8.version_, qconv_int8.type_constraints_, logger, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table_.emplace(
          OpIdInfo("QLinearConv", kOnnxDomain, api::DataType::INT8),
          OpTransformInfo{qconv_int8.op_type_, qconv_int8.domain_, qconv_int8.version_, true});
      conv_table_.emplace(
          OpIdInfo("QLinearConv", kMSDomain, api::DataType::INT8),
          OpTransformInfo{qconv_int8.op_type_, qconv_int8.domain_, qconv_int8.version_, true});
    }
  }

  {
    // uint8 qconv -> int8 nhwc qconv
    OpKernelRegistryId qconv_uint8{
        "QLinearConv", kMSDomain, 1, {{"T1", {DataTypeImpl::GetTensorType<uint8_t>()}}}};
    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, qconv_uint8.op_type_, qconv_uint8.domain_,
        qconv_uint8.version_, qconv_uint8.type_constraints_, logger, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table_.emplace(
          OpIdInfo("QLinearConv", kOnnxDomain, api::DataType::UINT8),
          OpTransformInfo{qconv_uint8.op_type_, qconv_uint8.domain_, qconv_uint8.version_, true});
      conv_table_.emplace(
          OpIdInfo("QLinearConv", kMSDomain, api::DataType::UINT8),
          OpTransformInfo{qconv_uint8.op_type_, qconv_uint8.domain_, qconv_uint8.version_, true});
    }
  }

  {
    // fp16 conv -> fp16 nhwc conv
    OpKernelRegistryId nhwc_conv_fp16{
        "NhwcFusedConv", kMSDomain, 1, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_,
        nhwc_conv_fp16.version_, nhwc_conv_fp16.type_constraints_, logger, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      const auto filter = [](const api::GraphRef&, api::NodeRef& node) {
        const auto dilations_opt = node.GetAttributeInts("dilations");
        if (dilations_opt.has_value()) {
          const auto& dilations = dilations_opt.value();
          if ((dilations.size() >= 1 && dilations[0] != 1) ||
              (dilations.size() >= 2 && dilations[1] != 1)) {
            return false;
          }
        }

        const auto group_opt = node.GetAttributeInt("group");
        if (group_opt.has_value() && group_opt.value() != 1) {
          return false;
        }

        return true;
      };

      conv_table_.emplace(
          OpIdInfo("Conv", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_, nhwc_conv_fp16.version_, false, filter});
      conv_table_.emplace(
          OpIdInfo("FusedConv", kMSDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_, nhwc_conv_fp16.version_, false, filter});
    }
  }

#ifdef USE_KLEIDIAI
  // Klediai specific block for NhwcFusedConvolutions
  {
    // F32 Conv -> F32 NHWC Conv
    OpKernelRegistryId nhwc_conv_fp32{
        "NhwcFusedConv", kMSDomain, 1, {{"T", {DataTypeImpl::GetTensorType<float>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_conv_fp32.op_type_, nhwc_conv_fp32.domain_,
        nhwc_conv_fp32.version_, nhwc_conv_fp32.type_constraints_, logger, &kernel_create_info);

    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;

      const auto filter = [](const api::GraphRef& graph, api::NodeRef& node) {
        return KleidiFp32NhwcFilter(graph, node);
      };

      conv_table_.emplace(
          OpIdInfo("Conv", kOnnxDomain, api::DataType::FLOAT),
          OpTransformInfo{nhwc_conv_fp32.op_type_, nhwc_conv_fp32.domain_, nhwc_conv_fp32.version_, false, filter});
      conv_table_.emplace(
          OpIdInfo("FusedConv", kMSDomain, api::DataType::FLOAT),
          OpTransformInfo{nhwc_conv_fp32.op_type_, nhwc_conv_fp32.domain_, nhwc_conv_fp32.version_, false, filter});
    }
  }
#endif

  {
    // fp16 MaxPool -> fp16 nhwc MaxPool
    OpKernelRegistryId nhwc_maxpool_fp16{
        "MaxPool", kMSInternalNHWCDomain, 12, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_maxpool_fp16.op_type_, nhwc_maxpool_fp16.domain_,
        nhwc_maxpool_fp16.version_, nhwc_maxpool_fp16.type_constraints_, logger, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table_.emplace(
          OpIdInfo("MaxPool", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_maxpool_fp16.op_type_, nhwc_maxpool_fp16.domain_, nhwc_maxpool_fp16.version_, false});
    }
  }

  {
    // fp16 AveragePool -> fp16 nhwc AveragePool
    OpKernelRegistryId nhwc_avgpool_fp16{
        "AveragePool", kMSInternalNHWCDomain, 11, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_avgpool_fp16.op_type_, nhwc_avgpool_fp16.domain_,
        nhwc_avgpool_fp16.version_, nhwc_avgpool_fp16.type_constraints_, logger, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table_.emplace(
          OpIdInfo("AveragePool", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_avgpool_fp16.op_type_, nhwc_avgpool_fp16.domain_, nhwc_avgpool_fp16.version_, false});
    }
  }

  {
    // fp16 GlobalAveragePool -> fp16 nhwc GlobalAveragePool
    OpKernelRegistryId nhwc_gavgpool_fp16{
        "GlobalAveragePool", kMSInternalNHWCDomain, 1, {{"T", {DataTypeImpl::GetTensorType<MLFloat16>()}}}};

    const KernelCreateInfo* kernel_create_info{};
    const auto status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, nhwc_gavgpool_fp16.op_type_, nhwc_gavgpool_fp16.domain_,
        nhwc_gavgpool_fp16.version_, nhwc_gavgpool_fp16.type_constraints_, logger, &kernel_create_info);
    if (status.IsOK() && kernel_create_info != nullptr) {
      kernel_create_info = nullptr;
      conv_table_.emplace(
          OpIdInfo("GlobalAveragePool", kOnnxDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_gavgpool_fp16.op_type_, nhwc_gavgpool_fp16.domain_, nhwc_gavgpool_fp16.version_, false});
    }
  }
};

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
    const auto ep = node->GetExecutionProviderType();
    if ((ep != kCpuExecutionProvider) && (ep != kAclExecutionProvider)) {
      continue;
    }

    // Only Conv and QLinearConv needs to be handled explicitly. The rest will be
    //  transformed if needed during transpose optimization.
    const auto* transform = NhwcConvLookup(conv_table_, *api_graph, *node);
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

    if (node->OpType() == "Conv" || node->OpType() == "FusedConv") {
      const auto group_opt = node->GetAttributeInt("group");
      if (group_opt.has_value() && group_opt.value() != 1) {
        continue;
      }

      const auto dilations_opt = node->GetAttributeInts("dilations");
      if (dilations_opt.has_value()) {
        const auto& dilations = dilations_opt.value();
        if ((dilations.size() >= 1 && dilations[0] != 1) ||
            (dilations.size() >= 2 && dilations[1] != 1)) {
          continue;
        }
      }
    }

    size_t rank = shape->dim_size();
    std::vector<int64_t> input_perm = ChannelFirstToLastPerm(rank);
    std::vector<int64_t> output_perm = ChannelLastToFirstPerm(rank);
    const auto inputs = node->Inputs();
    std::vector<const std::vector<int64_t>*> input_perms(inputs.size(), nullptr);
    if (!inputs.empty()) {
      input_perms[0] = &input_perm;
    }
    // Optional Sum (Z) input for FusedConv variants resides at index 3. When present,
    // it must be converted to NHWC alongside the activation tensor.
    const bool has_fused_sum_input = (node->Domain() == kMSDomain && node->OpType() == "FusedConv");
    if (has_fused_sum_input && inputs.size() > 3 && !inputs[3].empty()) {
      input_perms[3] = &input_perm;
    }

    WrapTransposesAroundNode(*api_graph, *node, input_perms, {&output_perm});

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
    Optimize(*api_graph, kCpuExecutionProvider, OrtEPCostCheck, OrtExtendedHandlers());
  }

  return Status::OK();
}

}  // namespace onnxruntime
