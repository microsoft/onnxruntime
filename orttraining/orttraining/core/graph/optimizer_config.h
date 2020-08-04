// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include "core/common/logging/logging.h"
#include "core/graph/node_arg.h"

namespace onnxruntime {
namespace training {

// This enum specifies different Adasum reduction algorithms.
// More will be added in the future based on the device, topology and etc.
enum AdasumReductionType {
  None,
  CpuReduction,
  GpuHierarchical,
};

// Configuration for the DeepSpeed ZeRO technique.  Currently only the stage
// setting is supported, and only with stages 0 (disabled) and 1 (optimizer
// state partitioning).

struct ZeROConfig {
  // Default configuration
  ZeROConfig() {
  }

  ZeROConfig(int s) : stage(s) {
  }

  int stage{0};
};

// configuration per optimizer node
struct OptimizerNodeConfig {
  std::string name{};
  const NodeArg* fp16_weight_arg{};
  std::string lr_feed_name{};
  std::unordered_map<std::string, float> attributes{};
  std::unordered_map<std::string, int64_t> int_attributes{};
  std::string loss_scale_input_name{};
  bool use_fp16_moments{false};
  bool update_weight{true};  // indicates whether Optimizer should do weight update, or output new gradient
  bool enabled{true};        // indicates whether this weight is included in the Optimizer
};

// configuration for optimizer portion of graph
struct OptimizerGraphConfig {
  int data_parallel_group_rank{0};
  int data_parallel_group_size{1};
  int local_rank{0};
  int local_size{1};
  bool use_mixed_precision{false};
  bool allreduce_in_fp16{false};
  bool use_nccl{false};
  ZeROConfig deepspeed_zero{0};
  int gradient_accumulation_steps{1};
  int64_t horovod_reduce_op{1};
  std::string loss_scale_input_name{};  // empty string means no loss scaling factor is applied
  AdasumReductionType adasum_reduction_type{AdasumReductionType::None};
  bool enable_grad_norm_clip{true};
};

}  // namespace training
}  // namespace onnxruntime
