// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state_flatbuffers_utils.h"

namespace onnxruntime::fbs::utils {

std::string GetSubgraphId(const NodeIndex node_idx, const std::string& attr_name) {
  return std::to_string(node_idx) + "_" + attr_name;
}

FbsSessionStateViewer::FbsSessionStateViewer(const fbs::SessionState& fbs_session_state)
    : fbs_session_state_{fbs_session_state} {
}

Status FbsSessionStateViewer::Validate() const {
  if (fbs_session_state_.sub_graph_session_states() == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SessionState for subgraphs is null. Invalid ORT format model.");
  }

  const auto* const fbs_kcis = fbs_session_state_.kernels();
  if (fbs_kcis == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel create info is null. Invalid ORT format model.");
  }

  const auto* const fbs_node_indices = fbs_kcis->node_indices();
  if (fbs_node_indices == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel create info node indices are null. Invalid ORT format model.");
  }

  const auto* const fbs_kernel_def_hashes = fbs_kcis->kernel_def_hashes();
  if (fbs_kernel_def_hashes == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Kernel create info hashes are null. Invalid ORT format model.");
  }

  if (fbs_node_indices->size() != fbs_kernel_def_hashes->size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Size mismatch for kernel create info node indexes and hashes. Invalid ORT format model.",
                           fbs_node_indices->size(), " != ", fbs_kernel_def_hashes->size());
  }

  return Status::OK();
}

FbsSessionStateViewer::NodeKernelInfo FbsSessionStateViewer::GetNodeKernelInfo(Index idx) const {
  const auto* const fbs_kcis = fbs_session_state_.kernels();
  const auto* const fbs_node_indices = fbs_kcis->node_indices();
  const auto* const fbs_kernel_def_hashes = fbs_kcis->kernel_def_hashes();

  return {fbs_node_indices->Get(idx), fbs_kernel_def_hashes->Get(idx)};
}

FbsSessionStateViewer::Index FbsSessionStateViewer::GetNumNodeKernelInfos() const {
  return fbs_session_state_.kernels()->node_indices()->size();
}

Status FbsSessionStateViewer::GetSubgraphSessionState(NodeIndex node_idx, const std::string& attr_name,
                                                      const fbs::SessionState*& fbs_subgraph_session_state_out) const {
  const auto key = GetSubgraphId(node_idx, attr_name);
  const auto* const fbs_subgraph_session_state_entry =
      fbs_session_state_.sub_graph_session_states()->LookupByKey(key.c_str());
  if (fbs_subgraph_session_state_entry == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Subgraph SessionState entry for ", key, " is missing. Invalid ORT format model.");
  }

  const auto* const fbs_subgraph_session_state = fbs_subgraph_session_state_entry->session_state();
  if (fbs_subgraph_session_state == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Subgraph SessionState for ", key, " is null. Invalid ORT format model.");
  }

  fbs_subgraph_session_state_out = fbs_subgraph_session_state;
  return Status::OK();
}

}  // namespace onnxruntime::fbs::utils
