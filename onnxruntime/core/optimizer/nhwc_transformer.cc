// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <deque>
#include <vector>
#include "core/common/cpuid_info.h"
#include "core/graph/constants.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/layout_transformation/layout_transformation.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"
#include "core/providers/common.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_transpose_optimization;
using namespace nhwc_map_internal;

namespace onnxruntime {

using namespace layout_transformation;

#ifdef USE_KLEIDIAI
bool TryGetDimValueAsSizeT(const ONNX_NAMESPACE::TensorShapeProto& shape, int index, size_t& value) {
  if (shape.dim_size() <= index || !shape.dim(index).has_dim_value()) {
    return false;
  }

  const int64_t dim_value = shape.dim(index).dim_value();
  if (dim_value < 0) {
    return false;
  }

  value = narrow<size_t>(dim_value);
  return true;
}

bool TryReadPositiveOrZeroInts(const std::vector<int64_t>& values, std::array<size_t, 2>& out) {
  if (values.size() != out.size()) {
    return false;
  }

  for (size_t i = 0; i < out.size(); ++i) {
    if (values[i] < 0) {
      return false;
    }

    out[i] = narrow<size_t>(values[i]);
  }

  return true;
}

bool TryReadPositiveOrZeroInts(const std::vector<int64_t>& values, std::array<size_t, 4>& out) {
  if (values.size() != out.size()) {
    return false;
  }

  for (size_t i = 0; i < out.size(); ++i) {
    if (values[i] < 0) {
      return false;
    }

    out[i] = narrow<size_t>(values[i]);
  }

  return true;
}

bool TryParseAutoPadType(std::string_view value, AutoPadType& auto_pad_type) {
  if (value.empty() || value == "NOTSET") {
    auto_pad_type = AutoPadType::NOTSET;
    return true;
  }

  if (value == "VALID") {
    auto_pad_type = AutoPadType::VALID;
    return true;
  }

  if (value == "SAME_UPPER") {
    auto_pad_type = AutoPadType::SAME_UPPER;
    return true;
  }

  if (value == "SAME_LOWER") {
    auto_pad_type = AutoPadType::SAME_LOWER;
    return true;
  }

  return false;
}

bool TryComputeFloatNhwcPads(const api::NodeRef& node,
                             const std::array<size_t, 2>& input_shape,
                             const std::array<size_t, 2>& kernel_shape,
                             const std::array<size_t, 2>& strides,
                             const std::array<size_t, 2>& dilations,
                             std::array<size_t, 4>& pads) {
  const auto auto_pad_value = node.GetAttributeString("auto_pad");
  AutoPadType auto_pad = AutoPadType::NOTSET;
  if (!TryParseAutoPadType(auto_pad_value.value_or("NOTSET"), auto_pad)) {
    return false;
  }

  if (auto_pad == AutoPadType::NOTSET) {
    const auto pads_opt = node.GetAttributeInts("pads");
    if (!pads_opt.has_value()) {
      pads.fill(0);
      return true;
    }

    return TryReadPositiveOrZeroInts(*pads_opt, pads);
  }

  std::array<int64_t, 4> pads_int64{};
  for (size_t i = 0; i < 2; ++i) {
    int64_t pad_head = 0;
    int64_t pad_tail = 0;
    int64_t out_dim = 0;
    const auto status = ComputePadAndOutputShape(
        narrow<int64_t>(input_shape[i]),
        narrow<int64_t>(strides[i]),
        narrow<int64_t>(kernel_shape[i]),
        narrow<int64_t>(dilations[i]),
        auto_pad,
        pad_head,
        pad_tail,
        out_dim,
        /*force_symmetric_auto_padding*/ false);
    if (!status.IsOK() || pad_head < 0 || pad_tail < 0 || out_dim < 0) {
      return false;
    }

    pads_int64[i] = pad_head;
    pads_int64[i + 2] = pad_tail;
  }

  for (size_t i = 0; i < pads.size(); ++i) {
    pads[i] = narrow<size_t>(pads_int64[i]);
  }

  return true;
}

bool FloatNhwcWrapperFilter(const onnx_transpose_optimization::api::GraphRef& graph,
                            onnx_transpose_optimization::api::NodeRef& node) {
  auto& base_node = NodeFromApiNode(node);

  ORT_UNUSED_PARAMETER(graph);
#if !defined(__aarch64__)
  return false;
#else
  if (!CPUIDInfo::GetCPUIDInfo().HasArm_SME()) {
    return false;
  }

  if (base_node.InputDefs().size() < 2) {
    return false;
  }

  const auto* input_shape = base_node.InputDefs()[0]->Shape();
  if (input_shape == nullptr || input_shape->dim_size() != 4) {
    return false;
  }

  const auto* weight_shape = base_node.InputDefs()[1]->Shape();
  if (weight_shape == nullptr || weight_shape->dim_size() != 4) {
    return false;
  }

  const auto inputs = node.Inputs();
  if (base_node.OpType() == "FusedConv" && inputs.size() > 3 && !inputs[3].empty()) {
    return false;
  }

  const auto group = node.GetAttributeInt("group").value_or(1);
  if (group != 1) {
    return false;
  }

  std::array<size_t, 2> input_spatial_shape{};
  std::array<size_t, 2> kernel_spatial_shape{};
  std::array<size_t, 2> dilations{1, 1};
  std::array<size_t, 2> strides{1, 1};
  std::array<size_t, 4> pads{};
  size_t batch_count = 0;
  size_t filter_count = 0;

  if (!TryGetDimValueAsSizeT(*input_shape, 0, batch_count) ||
      !TryGetDimValueAsSizeT(*input_shape, 2, input_spatial_shape[0]) ||
      !TryGetDimValueAsSizeT(*input_shape, 3, input_spatial_shape[1]) ||
      !TryGetDimValueAsSizeT(*weight_shape, 0, filter_count) ||
      !TryGetDimValueAsSizeT(*weight_shape, 2, kernel_spatial_shape[0]) ||
      !TryGetDimValueAsSizeT(*weight_shape, 3, kernel_spatial_shape[1])) {
    return false;
  }

  const auto dilations_opt = node.GetAttributeInts("dilations");
  if (dilations_opt.has_value() && !TryReadPositiveOrZeroInts(*dilations_opt, dilations)) {
    return false;
  }

  const auto strides_opt = node.GetAttributeInts("strides");
  if (strides_opt.has_value() && !TryReadPositiveOrZeroInts(*strides_opt, strides)) {
    return false;
  }

  if (!TryComputeFloatNhwcPads(node, input_spatial_shape, kernel_spatial_shape, strides, dilations, pads)) {
    return false;
  }

  return MlasConvSupportsSymmetricChannelsLast2DFloatKernel(
      /*Dimensions*/ 2,
      batch_count,
      /*GroupCount*/ 1,
      input_spatial_shape.data(),
      kernel_spatial_shape.data(),
      dilations.data(),
      pads.data(),
      strides.data(),
      filter_count,
      /*Beta*/ 0.0f);
#endif
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
          OpTransformInfo{qconv_int8.op_type_, qconv_int8.domain_, qconv_int8.version_, true, false});
      conv_table_.emplace(
          OpIdInfo("QLinearConv", kMSDomain, api::DataType::INT8),
          OpTransformInfo{qconv_int8.op_type_, qconv_int8.domain_, qconv_int8.version_, true, false});
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
          OpTransformInfo{qconv_uint8.op_type_, qconv_uint8.domain_, qconv_uint8.version_, true, false});
      conv_table_.emplace(
          OpIdInfo("QLinearConv", kMSDomain, api::DataType::UINT8),
          OpTransformInfo{qconv_uint8.op_type_, qconv_uint8.domain_, qconv_uint8.version_, true, false});
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
          OpTransformInfo{nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_, nhwc_conv_fp16.version_, false, false, filter});
      conv_table_.emplace(
          OpIdInfo("FusedConv", kMSDomain, api::DataType::FLOAT16),
          OpTransformInfo{nhwc_conv_fp16.op_type_, nhwc_conv_fp16.domain_, nhwc_conv_fp16.version_, false, true, filter});
    }
  }

#ifdef USE_KLEIDIAI
  // KleidiAI specific block for NhwcFusedConvolutions
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
        return FloatNhwcWrapperFilter(graph, node);
      };

      conv_table_.emplace(
          OpIdInfo("Conv", kOnnxDomain, api::DataType::FLOAT),
          OpTransformInfo{nhwc_conv_fp32.op_type_, nhwc_conv_fp32.domain_, nhwc_conv_fp32.version_, false, true, filter});
      conv_table_.emplace(
          OpIdInfo("FusedConv", kMSDomain, api::DataType::FLOAT),
          OpTransformInfo{nhwc_conv_fp32.op_type_, nhwc_conv_fp32.domain_, nhwc_conv_fp32.version_, false, true, filter});
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
          OpTransformInfo{nhwc_maxpool_fp16.op_type_, nhwc_maxpool_fp16.domain_, nhwc_maxpool_fp16.version_, false, false});
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
          OpTransformInfo{nhwc_avgpool_fp16.op_type_, nhwc_avgpool_fp16.domain_, nhwc_avgpool_fp16.version_, false, false});
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
          OpTransformInfo{nhwc_gavgpool_fp16.op_type_, nhwc_gavgpool_fp16.domain_, nhwc_gavgpool_fp16.version_, false, false});
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

    size_t rank = shape->dim_size();
    std::vector<int64_t> input_perm = ChannelFirstToLastPerm(rank);
    std::vector<int64_t> output_perm = ChannelLastToFirstPerm(rank);
    const auto inputs = node->Inputs();
    std::vector<const std::vector<int64_t>*> input_perms(inputs.size(), nullptr);
    if (!inputs.empty()) {
      input_perms[0] = &input_perm;
    }
    // Some transformed operators require the optional fused Sum (Z) input at index 3
    // to be converted alongside the activation tensor.
    if (transform->transpose_fused_sum_input_ && inputs.size() > 3 && !inputs[3].empty()) {
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
