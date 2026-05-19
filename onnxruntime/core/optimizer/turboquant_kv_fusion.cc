// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/turboquant_kv_fusion.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/float16.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_arg.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

namespace {

// ----------------------------------------------------------------------------
// Preset parsing.  Mirrors the python TQ_PRESETS dictionary used by the
// offline calibration tool, so the same string identifiers work for both.
// ----------------------------------------------------------------------------

struct TQPreset {
  int key_bits;
  int value_bits;
  bool norm_correction;
};

// Returns true and fills `out` if the preset string is recognised.  Empty,
// "none", "off" all return false (= disabled).
bool ParseTQPreset(const std::string& s, TQPreset* out) {
  if (s.empty() || s == "none" || s == "off" || s == "0") return false;
  if (s == "turboquant_4bit_nc") { *out = {4, 4, true}; return true; }
  if (s == "turboquant_k3v4_nc") { *out = {3, 4, true}; return true; }
  if (s == "turboquant_3bit_nc") { *out = {3, 3, true}; return true; }
  if (s == "turboquant_4bit")    { *out = {4, 4, false}; return true; }
  if (s == "turboquant_3bit")    { *out = {3, 3, false}; return true; }
  return false;
}

// ----------------------------------------------------------------------------
// Lloyd-Max centroids for N(0, 1/d).  Computed at runtime via the same
// fixed-point iteration used by the python reference (centroids.py); the
// resulting values are deterministic given (d, bits) and identical to what
// the offline rewriter injects.
// ----------------------------------------------------------------------------

double GaussianPdf(double x, double sigma2) {
  static constexpr double kInvSqrtTwoPi = 0.39894228040143267793994605993438;
  return (kInvSqrtTwoPi / std::sqrt(sigma2)) * std::exp(-x * x / (2.0 * sigma2));
}

double Trapz(double a, double b, int n, double sigma2,
             bool weighted_by_x) {
  if (n <= 0 || a >= b) return 0.0;
  const double h = (b - a) / static_cast<double>(n);
  auto eval = [&](double x) {
    double f = GaussianPdf(x, sigma2);
    return weighted_by_x ? x * f : f;
  };
  double acc = 0.5 * (eval(a) + eval(b));
  for (int i = 1; i < n; ++i) {
    acc += eval(a + static_cast<double>(i) * h);
  }
  return acc * h;
}

std::vector<float> SolveLloydMax(int d, int bits) {
  const int n_levels = 1 << bits;
  const double sigma2 = 1.0 / static_cast<double>(d);
  const double sigma = std::sqrt(sigma2);

  // Initial centroids: evenly spaced in [-3.5σ, 3.5σ].
  std::vector<double> centroids(n_levels);
  const double lo = -3.5 * sigma;
  const double hi = 3.5 * sigma;
  for (int i = 0; i < n_levels; ++i) {
    centroids[i] = lo + (hi - lo) * (static_cast<double>(i) + 0.5) /
                            static_cast<double>(n_levels);
  }

  constexpr int kMaxIter = 200;
  constexpr double kTol = 1e-10;
  constexpr int kIntegN = 200;
  for (int iter = 0; iter < kMaxIter; ++iter) {
    // Boundaries = midpoints between consecutive centroids.
    std::vector<double> bounds(n_levels + 1);
    bounds.front() = -10.0 * sigma;
    bounds.back() = 10.0 * sigma;
    for (int i = 0; i < n_levels - 1; ++i) {
      bounds[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
    }
    // New centroids = E[X | b_{i-1} < X <= b_i] under N(0, sigma2).
    std::vector<double> new_centroids(n_levels);
    double max_drift = 0.0;
    for (int i = 0; i < n_levels; ++i) {
      const double num = Trapz(bounds[i], bounds[i + 1], kIntegN, sigma2, true);
      const double den = Trapz(bounds[i], bounds[i + 1], kIntegN, sigma2, false);
      new_centroids[i] = (den > 1e-30) ? (num / den) : centroids[i];
      max_drift = std::max(max_drift, std::abs(new_centroids[i] - centroids[i]));
    }
    centroids = std::move(new_centroids);
    if (max_drift < kTol) break;
  }

  std::vector<float> result(n_levels);
  for (int i = 0; i < n_levels; ++i) result[i] = static_cast<float>(centroids[i]);
  return result;
}

// ----------------------------------------------------------------------------
// Walsh-Hadamard matrix (Sylvester construction) of order d, normalised so
// H * H^T = I.  d must be a power of two.
// ----------------------------------------------------------------------------

std::vector<float> BuildWalshHadamard(int d) {
  // Recursively build via Kronecker product with [[1, 1], [1, -1]].
  std::vector<float> H(static_cast<size_t>(d) * d, 1.0f);
  for (int n = 1; n < d; n *= 2) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const float a = H[static_cast<size_t>(i) * d + j];
        H[static_cast<size_t>(i) * d + (j + n)] = a;
        H[static_cast<size_t>(i + n) * d + j] = a;
        H[static_cast<size_t>(i + n) * d + (j + n)] = -a;
      }
    }
  }
  // Normalise by 1/sqrt(d) so H is orthonormal.
  const float inv = 1.0f / std::sqrt(static_cast<float>(d));
  for (auto& v : H) v *= inv;
  return H;
}

// ----------------------------------------------------------------------------
// Helpers for adding initializers and modifying nodes / IO type info.
// ----------------------------------------------------------------------------

NodeArg& AddFp16Initializer(Graph& graph,
                            const std::string& name,
                            const std::vector<float>& fp32_data,
                            const std::vector<int64_t>& shape) {
  // Convert fp32 -> fp16.
  std::vector<MLFloat16> fp16_data(fp32_data.size());
  for (size_t i = 0; i < fp32_data.size(); ++i) {
    fp16_data[i] = MLFloat16(fp32_data[i]);
  }
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (auto d : shape) tp.add_dims(d);
  tp.set_raw_data(fp16_data.data(), fp16_data.size() * sizeof(MLFloat16));
  return graph_utils::AddInitializer(graph, tp);
}

// Set a string attribute on a node, replacing any existing value of that name.
void SetStringAttr(Node& node, const std::string& name, const std::string& value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr.set_s(value);
  node.AddAttributeProto(std::move(attr));
}

void SetIntAttr(Node& node, const std::string& name, int64_t value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(value);
  node.AddAttributeProto(std::move(attr));
}

// Slot byte sizes for the packed cache layout.
//   K slot: ceil(D * key_bits / 8) bytes + 2 bytes of fp16 vec_norm.
//   V slot: ceil(D * value_bits / 8) bytes + 4 bytes (v_scale fp16 + v_zero fp16).
int SlotBytes(int head_dim, int bits, bool is_value) {
  return ((head_dim * bits) + 7) / 8 + (is_value ? 4 : 2);
}

// Mutate a NodeArg's TypeProto to (uint8, [..., new_last_dim]).  Used to
// rewrite the past_key/past_value/present_key/present_value tensor shapes
// to the packed cache layout.  We can't call NodeArg::SetType directly (it's
// private to Graph), so we go through UpdateTypeAndShape with
// override_types=true — that's the public path optimizer transforms use to
// change a graph value's element type.
Status RewriteCacheNodeArg(NodeArg& arg, int64_t new_last_dim,
                           const logging::Logger& logger) {
  const auto* existing = arg.TypeAsProto();
  ONNX_NAMESPACE::TypeProto rewritten;
  if (existing != nullptr) {
    rewritten = *existing;
  }
  auto* tt = rewritten.mutable_tensor_type();
  tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  if (tt->has_shape() && tt->shape().dim_size() > 0) {
    auto* last_dim = tt->mutable_shape()->mutable_dim(tt->shape().dim_size() - 1);
    last_dim->clear_dim_param();
    last_dim->set_dim_value(new_last_dim);
  }
  return arg.UpdateTypeAndShape(rewritten, /*strict=*/false,
                                /*override_types=*/true, logger);
}

// Try to read head_dim from the past_key NodeArg's shape.  Returns -1 if
// shape information is missing.
int InferHeadDimFromPastKey(const NodeArg* past_key_arg) {
  if (past_key_arg == nullptr) return -1;
  const auto* tp = past_key_arg->TypeAsProto();
  if (tp == nullptr || !tp->has_tensor_type()) return -1;
  const auto& tt = tp->tensor_type();
  if (!tt.has_shape() || tt.shape().dim_size() < 4) return -1;
  // past_key shape: (batch, num_kv_heads, seq, head_size).  Use the last dim.
  const auto& last = tt.shape().dim(tt.shape().dim_size() - 1);
  if (!last.has_dim_value()) return -1;
  return static_cast<int>(last.dim_value());
}

// Cache: one shared codebook + Hadamard initializer per (head_dim, key_bits).
// Avoids duplicating the same 16-fp16 / 64*64-fp16 tensor for every layer.
struct InitCache {
  // Stable names so repeated runs of this transformer (e.g. session re-create)
  // collide on the same initializer instead of accumulating duplicates.
  static std::string CodebookName(int head_dim, int key_bits) {
    return "__turboquant_kcodebook__hd" + std::to_string(head_dim) +
           "_b" + std::to_string(key_bits);
  }
  static std::string HadamardName(int head_dim) {
    return "__turboquant_hadamard__hd" + std::to_string(head_dim);
  }

  NodeArg* GetOrCreateCodebook(Graph& graph, int head_dim, int key_bits) {
    const std::string name = CodebookName(head_dim, key_bits);
    auto it = codebook_args_.find(name);
    if (it != codebook_args_.end()) return it->second;
    const ONNX_NAMESPACE::TensorProto* existing_init = nullptr;
    if (graph.GetInitializedTensor(name, existing_init)) {
      // Already exists in the graph (e.g. user pre-converted), reuse the NodeArg.
      NodeArg* existing = graph.GetNodeArg(name);
      codebook_args_[name] = existing;
      return existing;
    }
    auto values = SolveLloydMax(head_dim, key_bits);
    NodeArg& arg = AddFp16Initializer(graph, name, values,
                                      {static_cast<int64_t>(values.size())});
    codebook_args_[name] = &arg;
    return &arg;
  }

  NodeArg* GetOrCreateHadamard(Graph& graph, int head_dim) {
    const std::string name = HadamardName(head_dim);
    auto it = hadamard_args_.find(name);
    if (it != hadamard_args_.end()) return it->second;
    const ONNX_NAMESPACE::TensorProto* existing_init = nullptr;
    if (graph.GetInitializedTensor(name, existing_init)) {
      NodeArg* existing = graph.GetNodeArg(name);
      hadamard_args_[name] = existing;
      return existing;
    }
    auto values = BuildWalshHadamard(head_dim);
    NodeArg& arg = AddFp16Initializer(graph, name, values,
                                      {head_dim, head_dim});
    hadamard_args_[name] = &arg;
    return &arg;
  }

 private:
  std::unordered_map<std::string, NodeArg*> codebook_args_;
  std::unordered_map<std::string, NodeArg*> hadamard_args_;
};

}  // namespace

Status TurboQuantKVFusion::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                     const logging::Logger& logger) const {
  modified = false;

  // Read session option that gates this transformer.  No option => skip.
  TQPreset preset{};
  if (!ParseTQPreset(preset_, &preset)) {
    return Status::OK();
  }
  // Boundary protection (number of first/last GQA layers to leave at fp16).
  int boundary_n = boundary_n_;

  // First pass: find all GroupQueryAttention nodes (com.microsoft) in order.
  std::vector<Node*> gqa_nodes;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "GroupQueryAttention" &&
        (node.Domain() == "com.microsoft" || node.Domain().empty())) {
      gqa_nodes.push_back(&node);
    }
  }
  if (gqa_nodes.empty()) {
    return Status::OK();
  }

  const int n_layers = static_cast<int>(gqa_nodes.size());
  const int skip_lo = std::min(boundary_n, n_layers);
  const int skip_hi = std::max(0, n_layers - boundary_n);

  // Try to infer head_dim from the first node that has a usable past_key shape.
  int head_dim = -1;
  for (Node* node : gqa_nodes) {
    if (node->InputDefs().size() > 3) {
      head_dim = InferHeadDimFromPastKey(node->InputDefs()[3]);
      if (head_dim > 0) break;
    }
  }
  if (head_dim <= 0) {
    LOGS(logger, WARNING) << "TurboQuantKVFusion: could not infer head_dim from any "
                          << "GroupQueryAttention past_key shape; skipping rewrite";
    return Status::OK();
  }
  // Sanity: TurboQuant kernels are dispatched on power-of-two head_dim ∈ {64, 128, 256}.
  if (head_dim & (head_dim - 1)) {
    LOGS(logger, WARNING) << "TurboQuantKVFusion: head_dim=" << head_dim
                          << " is not a power of two; skipping";
    return Status::OK();
  }

  const int k_slot = SlotBytes(head_dim, preset.key_bits, false);
  const int v_slot = SlotBytes(head_dim, preset.value_bits, true);
  const int cache_last_dim = std::max(k_slot, v_slot);

  InitCache init_cache;
  NodeArg* codebook_arg = init_cache.GetOrCreateCodebook(graph, head_dim, preset.key_bits);
  NodeArg* hadamard_arg = init_cache.GetOrCreateHadamard(graph, head_dim);

  // Empty NodeArg, used when we need to pad the input list to slot 14 / 15.
  NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);

  int n_rewritten = 0;
  for (int idx = 0; idx < n_layers; ++idx) {
    if (idx < skip_lo || idx >= skip_hi) {
      LOGS(logger, INFO) << "TurboQuantKVFusion: skipping layer " << idx
                         << " (boundary protection)";
      continue;
    }
    Node& node = *gqa_nodes[idx];

    // Set / replace attributes.
    SetStringAttr(node, "kv_quant_method", "turboquant");
    SetIntAttr(node, "key_quant_bits", preset.key_bits);
    SetIntAttr(node, "value_quant_bits", preset.value_bits);
    SetIntAttr(node, "norm_correction", preset.norm_correction ? 1 : 0);

    // Pad input list to length 16 with empty NodeArgs and wire codebook / hadamard.
    auto& input_defs = node.MutableInputDefs();
    while (input_defs.size() < 14) {
      input_defs.push_back(&empty_arg);
    }
    if (input_defs.size() == 14) {
      input_defs.push_back(codebook_arg);
    } else {
      input_defs[14] = codebook_arg;
    }
    if (input_defs.size() == 15) {
      input_defs.push_back(hadamard_arg);
    } else {
      input_defs[15] = hadamard_arg;
    }

    // Keep MutableInputArgsCount in sync with the new input count.  ORT
    // requires sum(args_count) == size(input_defs); since GroupQueryAttention
    // is all-singleton (no variadic inputs), set every slot to 1 — empty
    // optional inputs use empty NodeArg sentinels but still consume one slot.
    auto& args_count = node.MutableInputArgsCount();
    args_count.assign(input_defs.size(), 1);

    // Rewrite past_key (3), past_value (4), present_key (1), present_value (2)
    // to (uint8, [..., cache_last_dim]) by mutating the underlying NodeArg.
    auto rewrite_idx = [&](size_t slot, bool is_input) -> Status {
      const auto& defs = is_input ? node.InputDefs() : node.OutputDefs();
      if (slot < defs.size() && defs[slot] != nullptr && defs[slot]->Exists()) {
        // Look up the canonical NodeArg in the graph and rewrite it.
        NodeArg* canonical = graph.GetNodeArg(defs[slot]->Name());
        if (canonical != nullptr) {
          ORT_RETURN_IF_ERROR(RewriteCacheNodeArg(*canonical, cache_last_dim, logger));
        }
      }
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(rewrite_idx(3, /*is_input=*/true));
    ORT_RETURN_IF_ERROR(rewrite_idx(4, /*is_input=*/true));
    ORT_RETURN_IF_ERROR(rewrite_idx(1, /*is_input=*/false));
    ORT_RETURN_IF_ERROR(rewrite_idx(2, /*is_input=*/false));

    ++n_rewritten;
  }

  if (n_rewritten > 0) {
    LOGS(logger, INFO) << "TurboQuantKVFusion: rewrote " << n_rewritten
                       << " / " << n_layers << " GQA nodes for preset '"
                       << preset_ << "' (k_slot=" << k_slot
                       << " v_slot=" << v_slot
                       << " cache_last_dim=" << cache_last_dim << ")";
    graph.SetGraphResolveNeeded();
    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
