// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Adapter-mode kernel registry for the CUDA plugin EP.
//
// When ORT_CUDA_PLUGIN_USE_EP_ADAPTER is defined, this file provides the real
// implementation of CreateCudaKernelRegistryFromOrtTables. It replaces the
// legacy path that depended on generated .inc files, using a hardcoded
// registration table and the existing GetCreateFnForOp dispatch.
//
// This translation unit is compiled as .cc (not .cu) and does not need NVCC.

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cuda/plugin/cuda_plugin_kernels.h"

namespace onnxruntime {
namespace cuda_plugin {

// ---------------------------------------------------------------------------
// Registration table entry.
// ---------------------------------------------------------------------------
struct AdapterKernelRegistration {
  const char* op_type;
  int since_version_start;
  int since_version_end;                      // 0 = open-ended
  const char* domain;                         // nullptr = kOnnxDomain
  const char* constraint_name;                // e.g. "T", "T1" — nullptr = unconstrained
  ONNXTensorElementDataType type_constraint;  // UNDEFINED = unconstrained
};

// ---------------------------------------------------------------------------
// Static registration table. Each entry here maps to a kernel that has a
// corresponding Create*Kernel function accessible via ResolvePluginKernelCreateFn.
//
// The table below covers the same ops as GetCreateFnForOp (the 61+ ops
// currently supported by the plugin). Type constraints are set broadly;
// the Create*Kernel functions handle type dispatching internally.
// ---------------------------------------------------------------------------

// Helper macros for concise table entries
#define REG(op, sv_start, sv_end) {#op, sv_start, sv_end, nullptr, nullptr, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED}
#define REG_T(op, sv_start, sv_end, T) {#op, sv_start, sv_end, nullptr, "T", ONNX_TENSOR_ELEMENT_DATA_TYPE_##T}
#define REG_MS(op, sv_start, sv_end) {#op, sv_start, sv_end, "com.microsoft", nullptr, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED}
#define REG_MS_T(op, sv_start, sv_end, T) {#op, sv_start, sv_end, "com.microsoft", "T", ONNX_TENSOR_ELEMENT_DATA_TYPE_##T}

// Activation ops — all types handled internally (float, half, double, bf16)
#define ACTIVATION_TYPED_VERSIONS(op, vs, ve) \
  REG_T(op, vs, ve, FLOAT),                   \
      REG_T(op, vs, ve, FLOAT16),             \
      REG_T(op, vs, ve, DOUBLE),              \
      REG_T(op, vs, ve, BFLOAT16)

#define ACTIVATION_TYPED_VERSIONS_HFD(op, vs, ve) \
  REG_T(op, vs, ve, FLOAT),                       \
      REG_T(op, vs, ve, FLOAT16),                 \
      REG_T(op, vs, ve, DOUBLE)

static constexpr AdapterKernelRegistration kAdapterRegistrations[] = {
    // =============== Activations ===============
    ACTIVATION_TYPED_VERSIONS(Relu, 6, 12),
    ACTIVATION_TYPED_VERSIONS(Relu, 13, 13),
    ACTIVATION_TYPED_VERSIONS(Relu, 14, 0),
    ACTIVATION_TYPED_VERSIONS(Elu, 6, 0),
    ACTIVATION_TYPED_VERSIONS_HFD(HardSigmoid, 6, 21),
    ACTIVATION_TYPED_VERSIONS(HardSigmoid, 22, 0),
    ACTIVATION_TYPED_VERSIONS(HardSwish, 14, 21),
    ACTIVATION_TYPED_VERSIONS(HardSwish, 22, 0),
    ACTIVATION_TYPED_VERSIONS_HFD(LeakyRelu, 6, 15),
    ACTIVATION_TYPED_VERSIONS(LeakyRelu, 16, 0),
    ACTIVATION_TYPED_VERSIONS(Selu, 6, 0),
    ACTIVATION_TYPED_VERSIONS(Sigmoid, 6, 12),
    ACTIVATION_TYPED_VERSIONS(Sigmoid, 13, 0),
    ACTIVATION_TYPED_VERSIONS(Softplus, 1, 0),
    ACTIVATION_TYPED_VERSIONS(Softsign, 1, 0),
    ACTIVATION_TYPED_VERSIONS(Tanh, 6, 12),
    ACTIVATION_TYPED_VERSIONS(Tanh, 13, 0),
    ACTIVATION_TYPED_VERSIONS(ThresholdedRelu, 10, 0),

    // =============== Unary elementwise ===============
    REG(Abs, 6, 12),
    REG(Abs, 13, 0),
    REG(Neg, 6, 12),
    REG(Neg, 13, 0),
    REG(Floor, 6, 12),
    REG(Floor, 13, 0),
    REG(Ceil, 6, 12),
    REG(Ceil, 13, 0),
    REG(Sqrt, 6, 12),
    REG(Sqrt, 13, 0),
    REG(Exp, 6, 12),
    REG(Exp, 13, 0),
    REG(Log, 6, 12),
    REG(Log, 13, 0),

    // =============== Binary elementwise ===============
    REG(Add, 7, 12),
    REG(Add, 13, 13),
    REG(Add, 14, 0),
    REG(Sub, 7, 12),
    REG(Sub, 13, 13),
    REG(Sub, 14, 0),
    REG(Mul, 7, 12),
    REG(Mul, 13, 13),
    REG(Mul, 14, 0),
    REG(Div, 7, 12),
    REG(Div, 13, 13),
    REG(Div, 14, 0),
    REG(Pow, 7, 11),
    REG(Pow, 12, 12),
    REG(Pow, 13, 14),
    REG(Pow, 15, 0),
    REG(Equal, 7, 10),
    REG(Equal, 11, 12),
    REG(Equal, 13, 0),
    REG(Greater, 7, 8),
    REG(Greater, 9, 12),
    REG(Greater, 13, 0),
    REG(Less, 7, 8),
    REG(Less, 9, 12),
    REG(Less, 13, 0),

    // =============== Math ===============
    REG(Clip, 6, 10),
    REG(Clip, 11, 11),
    REG(Clip, 12, 12),
    REG(Clip, 13, 0),
    REG(MatMul, 1, 12),
    REG(MatMul, 13, 0),
    REG(Gemm, 7, 8),
    REG(Gemm, 9, 10),
    REG(Gemm, 11, 12),
    REG(Gemm, 13, 0),
    REG(Conv, 1, 10),
    REG(Conv, 11, 0),
    REG(Softmax, 1, 10),
    REG(Softmax, 11, 12),
    REG(Softmax, 13, 0),
    REG(LogSoftmax, 1, 10),
    REG(LogSoftmax, 11, 12),
    REG(LogSoftmax, 13, 0),
    REG(Where, 9, 15),
    REG(Where, 16, 0),

    // =============== Reduction ===============
    REG(ArgMax, 1, 10),
    REG(ArgMax, 11, 12),
    REG(ArgMax, 13, 0),
    REG(ArgMin, 1, 10),
    REG(ArgMin, 11, 12),
    REG(ArgMin, 13, 0),
    REG(ReduceMax, 1, 10),
    REG(ReduceMax, 11, 11),
    REG(ReduceMax, 12, 12),
    REG(ReduceMax, 13, 17),
    REG(ReduceMax, 18, 19),
    REG(ReduceMax, 20, 0),
    REG(ReduceMean, 1, 10),
    REG(ReduceMean, 11, 12),
    REG(ReduceMean, 13, 17),
    REG(ReduceMean, 18, 0),
    REG(ReduceMin, 1, 10),
    REG(ReduceMin, 11, 11),
    REG(ReduceMin, 12, 12),
    REG(ReduceMin, 13, 17),
    REG(ReduceMin, 18, 19),
    REG(ReduceMin, 20, 0),
    REG(ReduceSum, 1, 10),
    REG(ReduceSum, 11, 12),
    REG(ReduceSum, 13, 0),

    // =============== Tensor ===============
    REG(Cast, 6, 8),
    REG(Cast, 9, 12),
    REG(Cast, 13, 18),
    REG(Cast, 19, 0),
    REG(Concat, 4, 10),
    REG(Concat, 11, 12),
    REG(Concat, 13, 0),
    REG(Split, 2, 10),
    REG(Split, 11, 12),
    REG(Split, 13, 17),
    REG(Split, 18, 0),
    REG(Gather, 1, 10),
    REG(Gather, 11, 12),
    REG(Gather, 13, 0),
    REG(Transpose, 1, 12),
    REG(Transpose, 13, 0),

    // =============== Shape-only ops ===============
    REG(Reshape, 5, 12),
    REG(Reshape, 13, 13),
    REG(Reshape, 14, 18),
    REG(Reshape, 19, 20),
    REG(Reshape, 21, 0),
    REG(Squeeze, 1, 10),
    REG(Squeeze, 11, 12),
    REG(Squeeze, 13, 0),
    REG(Unsqueeze, 1, 10),
    REG(Unsqueeze, 11, 12),
    REG(Unsqueeze, 13, 0),
    REG(Flatten, 1, 8),
    REG(Flatten, 9, 10),
    REG(Flatten, 11, 12),
    REG(Flatten, 13, 0),

    // =============== Contrib ops (com.microsoft) ===============
    REG_MS(Attention, 1, 0),
    REG_MS(DecoderMaskedMultiHeadAttention, 1, 0),
    REG_MS(EmbedLayerNormalization, 1, 0),
    REG_MS(FastGelu, 1, 0),
    REG_MS(GatherBlockQuantized, 1, 0),
    REG_MS(GemmaRotaryEmbedding, 1, 0),
    REG_MS(GroupQueryAttention, 1, 0),
    REG_MS(MatMulNBits, 1, 0),
    REG_MS(MoE, 1, 0),
    REG_MS(MultiHeadAttention, 1, 0),
    REG_MS(QMoE, 1, 0),
    REG_MS(RotaryEmbedding, 1, 0),
    REG_MS(SkipLayerNormalization, 1, 0),
    REG_MS(SkipSimplifiedLayerNormalization, 1, 0),
};

#undef REG
#undef REG_T
#undef REG_MS
#undef REG_MS_T
#undef ACTIVATION_TYPED_VERSIONS
#undef ACTIVATION_TYPED_VERSIONS_HFD

// ---------------------------------------------------------------------------
// CreateCudaKernelRegistryFromOrtTables — adapter-mode implementation
// ---------------------------------------------------------------------------
OrtStatus* CreateCudaKernelRegistryFromOrtTables(const OrtEpApi& ep_api,
                                                 const char* ep_name,
                                                 void* create_kernel_state,
                                                 OrtKernelRegistry** out_registry) {
  // create_kernel_state is unused in adapter mode; the adapter path resolves
  // kernel factories via PluginRegistry::Instance() instead.
  (void)create_kernel_state;
  *out_registry = nullptr;

  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelRegistry registry;

  // Group registrations by (op, domain, version_start, version_end) to build
  // kernel defs with proper per-constraint type lists.
  struct KernelDefKey {
    std::string op_type;
    std::string domain;
    int since_version_start;
    int since_version_end;

    bool operator<(const KernelDefKey& other) const {
      return std::tie(op_type, domain, since_version_start, since_version_end) <
             std::tie(other.op_type, other.domain, other.since_version_start, other.since_version_end);
    }
  };

  // Map: key -> { constraint_name -> set of type enums }
  std::map<KernelDefKey, std::map<std::string, std::set<ONNXTensorElementDataType>>> grouped;

  // Use PluginRegistry for O(log n) per-key lookup (keyed by op+domain+since+end).
  const auto& plugin_map = onnxruntime::cuda::PluginRegistry::Instance().AllEntries();

  for (const auto& reg : kAdapterRegistrations) {
    std::string domain = reg.domain ? reg.domain : "";
    onnxruntime::cuda::PluginRegistry::EntryKey lookup_key{
        reg.op_type, domain, reg.since_version_start, reg.since_version_end};
    if (plugin_map.find(lookup_key) == plugin_map.end()) {
      continue;  // Op not implemented via new registration — skip
    }

    KernelDefKey key{
        reg.op_type,
        domain,
        reg.since_version_start,
        reg.since_version_end,
    };

    if (reg.constraint_name && reg.constraint_name[0] != '\0' &&
        reg.type_constraint != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      grouped[key][reg.constraint_name].insert(reg.type_constraint);
    } else {
      // Unconstrained registration — ensure key exists
      grouped[key];
    }
  }

  // Build and register kernel defs from grouped data.
  for (const auto& [key, constraints] : grouped) {
    onnxruntime::cuda::PluginRegistry::EntryKey lookup_key{
        key.op_type, key.domain, key.since_version_start, key.since_version_end};
    auto it = plugin_map.find(lookup_key);
    if (it == plugin_map.end()) continue;

    Ort::KernelDefBuilder builder;
    builder.SetOperatorType(key.op_type.c_str());
    if (!key.domain.empty()) {
      builder.SetDomain(key.domain.c_str());
    }

    builder.SetSinceVersion(key.since_version_start,
                            key.since_version_end > 0 ? key.since_version_end : 2147483647);

    builder.SetExecutionProvider(ep_name);

    for (const auto& [cname, types] : constraints) {
      std::vector<const OrtDataType*> type_list;
      for (ONNXTensorElementDataType t : types) {
        const OrtDataType* dt = nullptr;
        RETURN_IF_ERROR(ep_api.GetTensorDataType(t, &dt));
        type_list.push_back(dt);
      }
      builder.AddTypeConstraint(cname.c_str(), type_list);
    }

    Ort::KernelDef kernel_def = builder.Build();
    // context is &it->second (const KernelFactory*). std::map values have stable
    // addresses. Casting object pointer to void* is valid C++ (no fn-ptr cast).
    RETURN_IF_ERROR(registry.AddKernel(kernel_def.release(), onnxruntime::cuda::GenericCreateKernel, (void*)(&it->second)));
  }

  *out_registry = registry.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
