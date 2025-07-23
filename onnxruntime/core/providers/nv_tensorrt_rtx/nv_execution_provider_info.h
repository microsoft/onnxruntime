// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/framework/framework_provider_common.h"
#include "core/framework/library_handles.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/shared_library/provider_api.h"

#define TRT_DEFAULT_OPTIMIZER_LEVEL 3

namespace onnxruntime {
// Information needed to construct trt execution providers.
struct NvExecutionProviderInfo {
  int device_id{0};
  bool has_user_compute_stream{false};
  void* user_compute_stream{nullptr};
  int max_partition_iterations{1000};
  int min_subgraph_size{1};
  size_t max_workspace_size{0};
  size_t max_shared_mem_size{0};
  bool dump_subgraphs{false};
  std::string engine_cache_path{""};
  bool weight_stripped_engine_enable{false};
  std::string onnx_model_folder_path{""};
  const void* onnx_bytestream{nullptr};
  size_t onnx_bytestream_size{0};
  bool use_external_data_initializer{false};
  const void* external_data_bytestream{nullptr};
  size_t external_data_bytestream_size{0};
  bool engine_decryption_enable{false};
  std::string engine_decryption_lib_path{""};
  bool force_sequential_engine_build{false};
  std::string timing_cache_path{""};
  bool detailed_build_log{false};
  bool sparsity_enable{false};
  int auxiliary_streams{-1};
  std::string extra_plugin_lib_paths{""};
  std::string profile_min_shapes{""};
  std::string profile_max_shapes{""};
  std::string profile_opt_shapes{""};
  bool cuda_graph_enable{false};
  bool multi_profile_enable{false};
  bool dump_ep_context_model{false};
  std::string ep_context_file_path{""};
  int ep_context_embed_mode{0};
  std::string engine_cache_prefix{""};
  std::string op_types_to_exclude{""};

  static NvExecutionProviderInfo FromProviderOptions(const ProviderOptions& options,
                                                     const ConfigOptions& session_options);
  static ProviderOptions ToProviderOptions(const NvExecutionProviderInfo& info);
  std::vector<OrtCustomOpDomain*> custom_op_domain_list;
};
}  // namespace onnxruntime
