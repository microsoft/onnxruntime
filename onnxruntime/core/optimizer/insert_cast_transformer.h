// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

/**
@Class InsertCastTransformer

Transformer to insert cast node that casts float16 to float for cpu nodes
*/
class InsertCastTransformer : public onnxruntime::GraphTransformer {
 public:
  /**
   * @brief Initializer
   * @param name                    for logging purpose
   * @param cpu_kernel_registry     used to query whether an op node can be safely created
   * @param enable_cpu_fp16         if true, allows CPU fp16 kernels to run without forcing fp32 casts
   * @param force_cpu_fp32          if true, applies the CPU fp16 fallback heuristic, which may keep selected
   *                                fp16 CPU nodes on the existing fp32 cast fallback path when native fp16 is
   *                                not expected to be profitable for the active MLAS backend
   * @param mlas_backend_kernel_selector_config
   *                                active MLAS backend selector config. Used by the CPU fp16 heuristic to avoid
   *                                preserving native fp16 paths that rely on backend-specific support, such as
   *                                native packed-B, when that backend is disabled or unavailable.
   */
  InsertCastTransformer(const std::string& name, const KernelRegistry* cpu_kernel_registry,
                        bool enable_cpu_fp16 = false, bool force_cpu_fp32 = true,
                        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config = nullptr)
      : onnxruntime::GraphTransformer(name),
        cpu_kernel_registries_(cpu_kernel_registry != nullptr ? InlinedVector<gsl::not_null<const KernelRegistry*>>{cpu_kernel_registry}
                                                              : InlinedVector<gsl::not_null<const KernelRegistry*>>{}),
        enable_cpu_fp16_(enable_cpu_fp16),
        force_cpu_fp32_(!cpu_kernel_registries_.empty() && force_cpu_fp32),
        mlas_backend_kernel_selector_config_(mlas_backend_kernel_selector_config != nullptr
                                                 ? *mlas_backend_kernel_selector_config
                                                 : MLAS_BACKEND_KERNEL_SELECTOR_CONFIG{}) {}

  InsertCastTransformer(const std::string& name,
                        InlinedVector<gsl::not_null<const KernelRegistry*>> cpu_kernel_registries,
                        bool enable_cpu_fp16 = false, bool force_cpu_fp32 = true,
                        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config = nullptr)
      : onnxruntime::GraphTransformer(name),
        cpu_kernel_registries_(std::move(cpu_kernel_registries)),
        enable_cpu_fp16_(enable_cpu_fp16),
        force_cpu_fp32_(!cpu_kernel_registries_.empty() && force_cpu_fp32),
        mlas_backend_kernel_selector_config_(mlas_backend_kernel_selector_config != nullptr
                                                 ? *mlas_backend_kernel_selector_config
                                                 : MLAS_BACKEND_KERNEL_SELECTOR_CONFIG{}) {}

 private:
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  bool NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input,
                      const logging::Logger& logger) const;

  const InlinedVector<gsl::not_null<const KernelRegistry*>> cpu_kernel_registries_;

  const bool enable_cpu_fp16_;

  // Some CPU fp16 kernels are only profitable for specific shapes and backend capabilities. A broader cost model would
  // be better; for now we use conservative checks for known slower cases and native packed-B availability.
  const bool force_cpu_fp32_;

  // Copied from session options so graph optimization makes the same backend-capability decisions that execution will.
  const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};
}  // namespace onnxruntime
