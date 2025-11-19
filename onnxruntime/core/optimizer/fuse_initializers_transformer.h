// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/optimizer/graph_transformer.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

/**
 * @class FuseInitializersTransformer
 *
 * A Transformer to fuse cast node that casts from init_type to cvt_type, back to their next/output nodes.
 * Below is the explanation on how this transforms works. It depends on "InsertCastTransforms" to produce the
 * intermediate representation from which it fuses the initializers (which are the cast node with zero input,
 * one initializer, and one output) back to the next/output node. After fusion, the link/edge between such
 * cast node to next/output node will then be removed.
 *
 *
 * ```
 *
 *         "Input Graph"                       "Intermediate Representation"               "Fusion Transforms"
 *
 *           --------                   --------        --------        --------                 --------
 *          | X_Fp16 |                 | X_Fp16 |      | W_Fp16 |      | B_Fp16 |               | X_Fp16 |
 *           --------                   --------        --------        --------                 --------
 *              |                          |               |               |                        |
 *              |                          |               |               |                        |
 *              |                          V               V               V                        V
 *              |                       | Cast |        | Cast |        | Cast |                 | Cast |
 *              |                       | Fp16 |        | Fp16 |        | Fp16 |                 | Fp16 |
 *              |                       |  To  |        |  To  |        |  To  |                 |  To  |
 *              |                       | Fp32 |        | Fp32 |        | Fp32 |                 | Fp32 |
 *              |                          |               |               |                        |
 *              |                          |               |               |                        |
 *              V                          V               V               V                        V
 *  ----------------------------       -----------------------------------------       ----------------------------
 * |        Conv_Fp16           |     |                                         |     |         Conv_Fp32          |
 * |        --W_Fp16--          | ==> |                Conv_Fp32                | ==> |         --W_Fp32--         |
 * |        --B_Fp16--          |     |                                         |     |         --B_Fp32--         |
 *  ----------------------------       -----------------------------------------       ----------------------------
 *              |                                          |                                        |
 *              |                                          |                                        |
 *              |                                          V                                        V
 *              |                                       | Cast |                                 | Cast |
 *              |                                       | Fp32 |                                 | Fp32 |
 *              |                                       |  To  |                                 |  To  |
 *              |                                       | Fp16 |                                 | Fp16 |
 *              |                                          |                                        |
 *              |                                          |                                        |
 *              V                                          V                                        V
 *           --------                                   --------                                 --------
 *          | Y_Fp16 |                                 | Y_Fp16 |                               | Y_Fp16 |
 *           --------                                   --------                                 --------
 *
 * ```
 *
 */
class FuseInitializersTransformer : public GraphTransformer {
 public:
  /**
   * @brief   Fuses Initializers to child node after conversion to child node kernel type
   *          to save compute on Cast Op during inference.
   *
   * This transforms must be applied after InsertCastTransformer. Currently only FP16 Initializers are fused with
   * nodes supporting FP32 Initializers, however, the code is designed to apply for any supported conversion/s.
   *
   * @param name          Name of the transforms, just for logging purpose.
   * @param init_type     The unsupported type for which cast nodes are inserted to convert
   *                      the initializers to supported type initializers expected by next/output nodes.
   * @param cvt_type      The supported type initializers expected by next/output nodes.
   * @param thread_pool   A pointer to thread pool to support conversion from init_type to cvt_type
   *                      with multithreading
   */
  FuseInitializersTransformer(const std::string& name,
                              const onnxruntime::MLDataType init_type,
                              const onnxruntime::MLDataType cvt_type,
                              onnxruntime::concurrency::ThreadPool* thread_pool = nullptr) : GraphTransformer(name, {}),
                                                                                             init_type_(init_type),
                                                                                             cvt_type_(cvt_type),
                                                                                             thread_pool_(thread_pool) {}

 private:
  const onnxruntime::MLDataType init_type_;
  const onnxruntime::MLDataType cvt_type_;
  onnxruntime::concurrency::ThreadPool* thread_pool_;
  Status ApplyImpl(
      Graph& graph,
      bool& modified,
      int graph_level,
      const logging::Logger& logger) const override;
};
}  // namespace onnxruntime
