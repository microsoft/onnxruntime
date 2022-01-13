// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/common.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/graph/basic_types.h"

namespace onnxruntime::fbs::utils {

/**
 * Gets the key that can be used to look up a fbs::SubGraphSessionState in a fbs::SessionState.
 *
 * @param node_idx The index of the node in the current graph.
 * @param attr_name The name of the node attribute that contains the subgraph.
 * @return The subgraph key.
 */
std::string GetSubgraphId(const NodeIndex node_idx, const std::string& attr_name);

/**
 * Provides read-only helper functions for a fbs::SessionState instance.
 */
class FbsSessionStateViewer {
 public:
  /**
   * Creates an instance.
   * Validation is not performed here, but in Validate().
   *
   * @param fbs_session_state The fbs::SessionState instance.
   */
  FbsSessionStateViewer(const fbs::SessionState& fbs_session_state);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FbsSessionStateViewer);

  /**
   * Validates the underlying fbs::SessionState instance.
   * WARNING: Other methods assume that the fbs::SessionState is valid!
   *
   * @return Whether the fbs::SessionState instance is valid.
   */
  Status Validate() const;

  using Index = flatbuffers::uoffset_t;

  struct NodeKernelInfo {
    NodeIndex node_index;
    HashValue kernel_def_hash;
  };

  /**
   * Retrieves the node kernel info element.
   *
   * @param index The index of the node kernel info element.
   * @return The node kernel info element.
   */
  NodeKernelInfo GetNodeKernelInfo(Index idx) const;

  /**
   * Gets the number of node kernel info elements.
   */
  Index GetNumNodeKernelInfos() const;

  /**
   * Retrieves the subgraph session state from the fbs::SessionState instance.
   *
   * @param node_idx The index of the node containing the subgraph.
   * @param attr_name The name of the attribute containing the subgraph.
   * @param[out] fbs_subgraph_session_state The subgraph session state. Non-null if successful.
   * @return Whether the retrieval was successful.
   */
  Status GetSubgraphSessionState(NodeIndex node_idx, const std::string& attr_name,
                                 const fbs::SessionState*& fbs_subgraph_session_state) const;

 private:
  const fbs::SessionState& fbs_session_state_;
};

}  // namespace onnxruntime::fbs::utils
