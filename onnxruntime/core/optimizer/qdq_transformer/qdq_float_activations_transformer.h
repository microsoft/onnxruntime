// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/optimizer/graph_transformer.h"
#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

/**
@Class QDQFloatActivationsTransformer

Removes remaining activation Q->DQ pairs after compute-op QDQ fusions (e.g., QLinearMatMul, QLinearConv)
have been applied by QDQSelectorActionTransformer.

This is intended for fully QDQ models where we want unfused ops to run in float precision.
It works best when DropQDQNodesRules/SplitQDQRules are skipped (via session.qdq_float_activations="1"),
so that data-movement ops keep their Q/DQ wrappers, making all Q->DQ pairs directly adjacent.

Sub-passes:
  A) Remove all adjacent Q->DQ pairs where all of Q's consumers are DQ nodes with matching scale/zp.
  B) Fuse newly eligible DQ(blockwise)->MatMul patterns into MatMulNBits.
  C) Constant-fold remaining weight DQ nodes on constant initializers into float tensors.
*/
class QDQFloatActivationsTransformer : public GraphTransformer {
 public:
  QDQFloatActivationsTransformer(int64_t qdq_matmulnbits_accuracy_level,
                                 concurrency::ThreadPool* intra_op_thread_pool,
                                 const IExecutionProvider& cpu_execution_provider,
                                 const ConfigOptions& config_options)
      : GraphTransformer("QDQFloatActivationsTransformer"),
        qdq_matmulnbits_accuracy_level_(qdq_matmulnbits_accuracy_level),
        intra_op_thread_pool_(intra_op_thread_pool),
        cpu_execution_provider_(cpu_execution_provider),
        config_options_(config_options) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

  int64_t qdq_matmulnbits_accuracy_level_;
  concurrency::ThreadPool* intra_op_thread_pool_;
  const IExecutionProvider& cpu_execution_provider_;
  const ConfigOptions& config_options_;
};

}  // namespace onnxruntime
