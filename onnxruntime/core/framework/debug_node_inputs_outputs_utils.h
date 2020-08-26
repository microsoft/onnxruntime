// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// to create a build with these enabled run the build script with:
//   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

#pragma once

#include "core/common/path.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace utils {

// environment variables that control debug node dumping behavior
namespace debug_node_inputs_outputs_env_vars {
// set to non-zero to dump node input data
constexpr const char* kDumpInputData = "ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA";
// set to non-zero to dump node output data
constexpr const char* kDumpOutputData = "ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA";
// specify a node name filter to limit the nodes that are dumped
constexpr const char* kNodeNameFilter = "ORT_DEBUG_NODE_IO_NODE_NAME_FILTER";
// set to non-zero to dump data to files instead of stdout
constexpr const char* kDumpDataToFiles = "ORT_DEBUG_NODE_IO_DUMP_DATA_TO_FILES";
// specify the output directory for any data files produced
constexpr const char* kOutputDir = "ORT_DEBUG_NODE_IO_OUTPUT_DIR";
// set to non-zero to confirm that dumping data files for all nodes is acceptable
constexpr const char* kDumpingDataToFilesForAllNodesIsOk =
    "ORT_DEBUG_NODE_IO_DUMPING_DATA_TO_FILES_FOR_ALL_NODES_IS_OK";
}  // namespace debug_node_inputs_outputs_env_vars

struct NodeDumpOptions {
  enum DumpFlags {
    ShapeOnly = 0,
    InputData = 1 << 0,
    OutputData = 1 << 1,
    AllData = InputData | OutputData,
  };

  // specifies the information to dump per node
  // see DumpFlags
  // Note: When dumping every node, dumping both input and output may be redundant.
  // Doing that may be more useful when dumping a subset of all nodes.
  int dump_flags{DumpFlags::ShapeOnly};

  // filters the nodes that are dumped
  // currently, dumped nodes are nodes whose name contains this value
  // if empty, dump all nodes
  std::string node_name_filter{};

  // the destination for dumped data
  enum class DataDestination {
    // write to stdout
    StdOut,
    // write to one file per tensor input/output as a TensorProto
    TensorProtoFiles,
  } data_destination{DataDestination::StdOut};

  // the output directory for dumped data files
  Path output_dir;
};

// gets NodeDumpOptions instance configured from environment variable values
const NodeDumpOptions& NodeDumpOptionsFromEnvironmentVariables();

// dumps inputs for a node
void DumpNodeInputs(
    const NodeDumpOptions& dump_options,
    const OpKernelContext& context, const Node& node, const SessionState& session_state);

void DumpNodeInputs(
    const OpKernelContext& context, const Node& node, const SessionState& session_state);

// dumps outputs for a node
void DumpNodeOutputs(
    const NodeDumpOptions& dump_options,
    OpKernelContext& context, const Node& node, const SessionState& session_state);

void DumpNodeOutputs(
    OpKernelContext& context, const Node& node, const SessionState& session_state);

}  // namespace utils
}  // namespace onnxruntime

#endif
