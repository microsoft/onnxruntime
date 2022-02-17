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

  HashValue hash = fbs_kernel_def_hashes->Get(idx);
  UpdateHashForBackwardsCompatibility(hash);

  return {fbs_node_indices->Get(idx), hash};
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

void UpdateHashForBackwardsCompatibility(HashValue& hash) {
  // map of old hash to new hash if we were forced to break backwards compatibility for a kernel registration
  //
  // If we need to update the hash for an existing registration, an entry needs to be added here to map the
  // old hash to the new. This should rarely be required as historically the only need for it was fixing
  // kernel registrations with invalid type constraints. Please carefully read through the information at the top of
  // onnxruntime/test/providers/kernel_def_hash_test.cc regarding how/when hashes might change and the best way to
  // address that.
  static const std::unordered_map<HashValue, HashValue> hashes{
      // old                   new                          domain, operator, opset[, type]
      {2832535737534577496ULL, 16708009824840936392ULL},    // kOnnxDomain, Dropout, 7
      {12198479371038564912ULL, 1718418059112844640ULL},    // kOnnxDomain, Scan, 9
      {2560955351529676608ULL, 3668627007850399040ULL},     // kOnnxDomain, Scan, 11
      {10232409728231027688ULL, 5212043150202938416ULL},    // kOnnxDomain, Not, 1
      {11912523891622051440ULL, 10225383741733918632ULL},   // kOnnxDomain, RoiAlign, 10, float
      {18084231515768318048ULL, 17022700455473327752ULL},   // kOnnxDomain, RoiAlign, 10, double
      {14033689580222898712ULL, 634727773751317256ULL},     // kOnnxDomain, GatherND, 11
      {646512416908411600ULL, 3064028185911332496ULL},      // kOnnxDomain, GatherND, 12
      {15019893097608892000ULL, 11311962292460032936ULL},   // kOnnxDomain, GatherND, 13
      {14259324427750852648ULL, 7767393334034626736ULL},    // kOnnxDomain, StringNormalizer, 10
                                                            // contrib ops
      {7642430665819070720ULL, 8620498355864235632ULL},     // kMSDomain, CropAndResize, 1
      {15019666093341768288ULL, 11924582339825775592ULL}};  // kMSDomain, GridSample, 1

  auto iter = hashes.find(hash);
  if (iter != hashes.cend()) {
    // hash was updated in newer version of ORT kernel registrations
    hash = iter->second;
  }
}

}  // namespace onnxruntime::fbs::utils
