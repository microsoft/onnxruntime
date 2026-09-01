// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph/model.h"
#include "core/optimizer/gqa_value_layout_transformer.h"
#include "core/session/IOBinding.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

// Geometry kept small, but with max_sequence_length != head_size so that a transpose which fails to
// swap the last two dimensions is caught by the shape assertions rather than passing silently.
constexpr int64_t kBatch = 1;
constexpr int64_t kSeq = 1;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kKvNumHeads = 1;
constexpr int64_t kHeadSize = 16;
constexpr int64_t kMaxSeq = 8;
constexpr int64_t kPastSeq = 3;  // valid entries in the past cache when live_past_cache is set
constexpr int64_t kQHidden = kNumHeads * kHeadSize;
constexpr int64_t kKvHidden = kKvNumHeads * kHeadSize;

// A pattern that varies along both of the swapped dimensions, so transposing it is observable.
std::vector<MLFloat16> CachePattern(int64_t seq_len, int64_t head_size, float offset) {
  std::vector<MLFloat16> data(static_cast<size_t>(seq_len * head_size));
  for (int64_t s = 0; s < seq_len; ++s) {
    for (int64_t h = 0; h < head_size; ++h) {
      data[static_cast<size_t>(s * head_size + h)] =
          MLFloat16(offset + static_cast<float>(s) * 0.25f - static_cast<float>(h) * 0.03125f);
    }
  }
  return data;
}

struct BuildOptions {
  // Feed past_value through an Identity so it is no longer a graph input.
  bool past_value_behind_identity = false;
  // Route present_value through an Identity so it is no longer a graph output.
  bool present_value_behind_identity = false;
  // Omit the past cache inputs entirely (prefill-only model). GQA type inference requires past_key
  // and past_value to be present or absent together, so both are dropped.
  bool no_past_kv = false;
  // Omit the present_value output entirely.
  bool no_present_value = false;
  // Configure a 4-bit quantized Value cache, which cannot be transposed byte-wise.
  bool four_bit_value_cache = false;
  // Add a second GQA node that consumes the same past_key/past_value graph inputs. Transforming
  // either node would mutate a boundary NodeArg the other still reads as BNSH.
  bool second_gqa_sharing_past_kv = false;
  // Keep present_value as a graph output but also feed it to an Identity inside the graph. That
  // internal consumer expects BNSH and would silently receive BNHS.
  bool present_value_also_consumed_internally = false;
  // Feed past_value through a value-layout Transpose from a BNHS graph input while leaving
  // present_value as a plain BNSH graph output, i.e. a half-converted node.
  bool partially_transformed = false;
  // Wire both Value operands through value-layout Transposes to BNHS graph boundaries, i.e. a model
  // that already carries the conversion, as one saved via session.optimized_model_filepath would.
  bool already_transformed = false;

  bool TransposedPastValue() const { return partially_transformed || already_transformed; }

  // Fill the past caches with a pattern that varies along both sequence_length and head_size, and
  // set the sequence lengths so the kernel actually reads them. Without this the caches are zero and
  // unread, which would make a numerical parity test pass even with a broken transpose.
  bool live_past_cache = false;

  int32_t SeqLensK() const { return live_past_cache ? kPastSeq : 0; }
  int32_t TotalSequenceLength() const { return live_past_cache ? static_cast<int32_t>(kPastSeq + kSeq) : 1; }

  // Length of the present cache. With a past cache the model shares one max_sequence_length buffer;
  // without one, GQA infers a present cache holding just the new tokens.
  int64_t PresentCacheLength() const { return no_past_kv ? kSeq : kMaxSeq; }
};

void BuildGqaModel(ModelTestBuilder& builder, const BuildOptions& opts) {
  NodeArg& empty_arg = builder.graph_.GetOrCreateNodeArg("", nullptr);

  NodeArg* query = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kQHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));
  NodeArg* key = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));
  NodeArg* value = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));

  NodeArg* past_key = &empty_arg;
  NodeArg* past_value = &empty_arg;
  if (!opts.no_past_kv) {
    const std::vector<int64_t> cache_shape{kBatch, kKvNumHeads, kMaxSeq, kHeadSize};
    if (opts.live_past_cache) {
      past_key = builder.MakeInput<MLFloat16>(cache_shape, CachePattern(kMaxSeq, kHeadSize, 0.5f));
      past_value = builder.MakeInput<MLFloat16>(cache_shape, CachePattern(kMaxSeq, kHeadSize, -0.25f));
    } else {
      past_key = builder.MakeInput<MLFloat16>(cache_shape, MLFloat16(0.0f), MLFloat16(0.0f));
      past_value = builder.MakeInput<MLFloat16>(cache_shape, MLFloat16(0.0f), MLFloat16(0.0f));
    }

    if (opts.past_value_behind_identity) {
      NodeArg* forwarded = builder.MakeIntermediate<MLFloat16>(
          std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize});
      builder.AddNode("Identity", {past_value}, {forwarded});
      past_value = forwarded;
    }

    if (opts.TransposedPastValue()) {
      // past_value already arrives BNHS through a value-layout Transpose. With
      // already_transformed the present side is converted to match; with partially_transformed it
      // is left as a plain BNSH graph output, giving a half-converted node. The original past_value
      // graph input is left dangling, which is legal and irrelevant here.
      NodeArg* bnhs_input = builder.MakeInput<MLFloat16>(
          std::vector<int64_t>{kBatch, kKvNumHeads, kHeadSize, kMaxSeq}, MLFloat16(0.0f), MLFloat16(0.0f));
      NodeArg* bnsh = builder.MakeIntermediate<MLFloat16>(cache_shape);
      Node& transpose = builder.AddNode("Transpose", {bnhs_input}, {bnsh});
      transpose.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 2});
      past_value = bnsh;
    }
  }

  NodeArg* seqlens_k =
      builder.MakeInput<int32_t>(std::vector<int64_t>{kBatch}, std::vector<int32_t>{opts.SeqLensK()});
  NodeArg* total_seq_len =
      builder.MakeInput<int32_t>(std::vector<int64_t>{1}, std::vector<int32_t>{opts.TotalSequenceLength()});

  const std::vector<int64_t> present_shape{kBatch, kKvNumHeads, opts.PresentCacheLength(), kHeadSize};

  NodeArg* gqa_out = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{kBatch, kSeq, kQHidden});
  NodeArg* present_key = builder.MakeOutput<MLFloat16>(present_shape);

  // present_value is either the graph output directly, or an intermediate that an Identity forwards
  // to the graph output.
  NodeArg* present_value = &empty_arg;
  NodeArg* identity_target = nullptr;
  NodeArg* bnhs_present_target = nullptr;
  if (!opts.no_present_value) {
    if (opts.present_value_behind_identity) {
      present_value = builder.MakeIntermediate<MLFloat16>(present_shape);
      identity_target = builder.MakeOutput<MLFloat16>(present_shape);
    } else if (opts.already_transformed) {
      present_value = builder.MakeIntermediate<MLFloat16>(present_shape);
      bnhs_present_target = builder.MakeOutput<MLFloat16>(
          std::vector<int64_t>{kBatch, kKvNumHeads, kHeadSize, opts.PresentCacheLength()});
    } else {
      present_value = builder.MakeOutput<MLFloat16>(present_shape);
    }
  }

  std::vector<NodeArg*> gqa_inputs{query, key, value, past_key, past_value, seqlens_k, total_seq_len};

  Node& gqa = builder.AddNode("GroupQueryAttention",
                              gqa_inputs,
                              {gqa_out, present_key, present_value},
                              kMSDomain);
  gqa.AddAttribute("num_heads", static_cast<int64_t>(kNumHeads));
  gqa.AddAttribute("kv_num_heads", static_cast<int64_t>(kKvNumHeads));

  if (opts.four_bit_value_cache) {
    gqa.AddAttribute("v_quant_type", std::string("PER_CHANNEL"));
    gqa.AddAttribute("kv_cache_bit_width", static_cast<int64_t>(4));
  }

  if (bnhs_present_target != nullptr) {
    Node& transpose = builder.AddNode("Transpose", {present_value}, {bnhs_present_target});
    transpose.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 2});
  }

  if (opts.present_value_also_consumed_internally) {
    NodeArg* extra_output = builder.MakeOutput<MLFloat16>(present_shape);
    builder.AddNode("Identity", {present_value}, {extra_output});
  }

  if (opts.second_gqa_sharing_past_kv) {
    NodeArg* second_out = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{kBatch, kSeq, kQHidden});
    NodeArg* second_present_key = builder.MakeOutput<MLFloat16>(present_shape);
    NodeArg* second_present_value = builder.MakeOutput<MLFloat16>(present_shape);

    Node& second_gqa = builder.AddNode("GroupQueryAttention",
                                       gqa_inputs,
                                       {second_out, second_present_key, second_present_value},
                                       kMSDomain);
    second_gqa.AddAttribute("num_heads", static_cast<int64_t>(kNumHeads));
    second_gqa.AddAttribute("kv_num_heads", static_cast<int64_t>(kKvNumHeads));
  }

  if (identity_target != nullptr) {
    builder.AddNode("Identity", {present_value}, {identity_target});
  }
}

std::unique_ptr<GraphTransformer> MakeTransformer() {
  return std::make_unique<GqaValueLayoutTransformer>();
}

const std::vector<int64_t> kBnsh{kBatch, kKvNumHeads, kMaxSeq, kHeadSize};
const std::vector<int64_t> kBnhs{kBatch, kKvNumHeads, kHeadSize, kMaxSeq};

// ModelTestBuilder generates positional names ("input_3", "output_2"), so the checkers navigate the
// graph structurally instead of by name.
const Node* FindGqa(const Graph& graph) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "GroupQueryAttention" && node.Domain() == kMSDomain) {
      return &node;
    }
  }
  return nullptr;
}

bool IsValueLayoutTranspose(const Node& node) {
  if (node.OpType() != "Transpose" || node.Domain() != kOnnxDomain) {
    return false;
  }
  const auto& attrs = node.GetAttributes();
  auto perm = attrs.find("perm");
  if (perm == attrs.end() || perm->second.ints_size() != 4) {
    return false;
  }
  return perm->second.ints(0) == 0 && perm->second.ints(1) == 1 &&
         perm->second.ints(2) == 3 && perm->second.ints(3) == 2;
}

Status ExpectShape(const NodeArg* arg, const std::vector<int64_t>& expected, const std::string& what) {
  ORT_RETURN_IF(arg == nullptr, what, " not found.");

  const auto* shape = arg->Shape();
  ORT_RETURN_IF(shape == nullptr, what, " ('", arg->Name(), "') has no shape.");
  ORT_RETURN_IF_NOT(static_cast<size_t>(shape->dim_size()) == expected.size(),
                    what, " ('", arg->Name(), "') has rank ", shape->dim_size(), ", expected ", expected.size(), ".");

  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& dim = shape->dim(static_cast<int>(i));
    ORT_RETURN_IF_NOT(dim.has_dim_value() && dim.dim_value() == expected[i],
                      what, " ('", arg->Name(), "') dimension ", i, " is ",
                      dim.has_dim_value() ? std::to_string(dim.dim_value()) : dim.dim_param(),
                      ", expected ", expected[i], ".");
  }

  return Status::OK();
}

Status ExpectTransposeCount(const Graph& graph, int expected, int expected_gqa = 1) {
  const auto op_to_count = CountOpsInGraph(graph);
  const int actual = OpCount(op_to_count, "Transpose");
  ORT_RETURN_IF_NOT(actual == expected, "Expected ", expected, " Transpose nodes, found ", actual, ".");

  const int actual_gqa = OpCount(op_to_count, "com.microsoft.GroupQueryAttention");
  ORT_RETURN_IF_NOT(actual_gqa == expected_gqa,
                    "Expected ", expected_gqa, " GroupQueryAttention nodes to be preserved, found ", actual_gqa, ".");
  return Status::OK();
}

Status ExpectNoTransposes(const Graph& graph, int expected_gqa = 1) {
  return ExpectTransposeCount(graph, 0, expected_gqa);
}

// Walks GQA input 4 back through the inserted Transpose to the graph input, asserting the operand
// stayed BNSH and the boundary became BNHS.
Status ExpectBnhsPastValue(const Graph& graph, const Node& gqa) {
  const NodeArg* operand = gqa.InputDefs()[4];
  ORT_RETURN_IF_ERROR(ExpectShape(operand, kBnsh, "GQA past_value operand"));

  const Node* transpose = graph.GetProducerNode(operand->Name());
  ORT_RETURN_IF(transpose == nullptr || !IsValueLayoutTranspose(*transpose),
                "GQA past_value is not produced by a Transpose(perm=[0,1,3,2]).");

  const NodeArg* boundary = transpose->InputDefs()[0];
  ORT_RETURN_IF_NOT(graph.IsInputsIncludingInitializers(boundary),
                    "past_value ('", boundary->Name(), "') must remain a graph input.");
  return ExpectShape(boundary, kBnhs, "past_value graph input");
}

// Mirror of the above for GQA output 2. cache_len differs from kMaxSeq for a prefill-only model,
// where GQA infers a present cache holding just the new tokens.
Status ExpectBnhsPresentValue(const Graph& graph, const Node& gqa, int64_t cache_len = kMaxSeq) {
  const std::vector<int64_t> bnsh{kBatch, kKvNumHeads, cache_len, kHeadSize};
  const std::vector<int64_t> bnhs{kBatch, kKvNumHeads, kHeadSize, cache_len};

  const NodeArg* operand = gqa.OutputDefs()[2];
  ORT_RETURN_IF_ERROR(ExpectShape(operand, bnsh, "GQA present_value operand"));

  const auto consumers = graph.GetConsumerNodes(operand->Name());
  ORT_RETURN_IF(consumers.size() != 1 || consumers[0] == nullptr || !IsValueLayoutTranspose(*consumers[0]),
                "GQA present_value is not consumed by exactly one Transpose(perm=[0,1,3,2]).");

  const NodeArg* boundary = consumers[0]->OutputDefs()[0];
  ORT_RETURN_IF_NOT(graph.IsOutput(boundary),
                    "present_value ('", boundary->Name(), "') must remain a graph output.");
  return ExpectShape(boundary, bnhs, "present_value graph output");
}

Status ExpectBnhsBoundary(Graph& graph) {
  ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 2));

  const Node* gqa = FindGqa(graph);
  ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");

  ORT_RETURN_IF_ERROR(ExpectBnhsPastValue(graph, *gqa));
  ORT_RETURN_IF_ERROR(ExpectBnhsPresentValue(graph, *gqa));

  // The Key cache must be untouched: still wired straight to the graph boundary, still BNSH.
  ORT_RETURN_IF_NOT(graph.IsInputsIncludingInitializers(gqa->InputDefs()[3]),
                    "past_key must remain wired directly to the graph input.");
  ORT_RETURN_IF_ERROR(ExpectShape(gqa->InputDefs()[3], kBnsh, "past_key graph input"));
  ORT_RETURN_IF_NOT(graph.IsOutput(gqa->OutputDefs()[1]),
                    "present_key must remain wired directly to the graph output.");
  ORT_RETURN_IF_ERROR(ExpectShape(gqa->OutputDefs()[1], kBnsh, "present_key graph output"));

  return Status::OK();
}

// Serializes the default GQA model so an InferenceSession can load it, which is the only way to
// exercise the session option plumbing and the optimization-level behaviour.
Status BuildSerializedGqaModel(const logging::Logger& logger, std::string& model_bytes) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 21;
  domain_to_version[kMSDomain] = 1;

  Model model("GqaValueLayoutTest", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();

  ModelTestBuilder helper(graph);
  BuildGqaModel(helper, BuildOptions{});
  helper.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());

  ORT_RETURN_IF_NOT(model.ToProto().SerializeToString(&model_bytes), "Failed to serialize the test model.");
  return Status::OK();
}

// Everything the runtime tests need to drive a session: the serialized model, a full set of BNSH
// feeds, and the boundary tensor names (which ModelTestBuilder generates, so they are read back off
// the built graph rather than assumed).
struct RuntimeGqaModel {
  std::string bytes;
  NameMLValMap bnsh_feeds;
  std::string past_value_name;
  std::string present_value_name;
  std::string attention_output_name;
  std::vector<std::string> output_names;
};

Status BuildRuntimeGqaModel(const logging::Logger& logger, RuntimeGqaModel& out) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 21;
  domain_to_version[kMSDomain] = 1;

  Model model("GqaValueLayoutRuntimeTest", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();

  BuildOptions opts;
  opts.live_past_cache = true;

  ModelTestBuilder helper(graph);
  BuildGqaModel(helper, opts);
  helper.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());

  const Node* gqa = FindGqa(graph);
  ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");

  out.past_value_name = gqa->InputDefs()[4]->Name();
  out.present_value_name = gqa->OutputDefs()[2]->Name();
  out.attention_output_name = gqa->OutputDefs()[0]->Name();
  out.bnsh_feeds = helper.feeds_;
  for (const auto* output : graph.GetOutputs()) {
    out.output_names.push_back(output->Name());
  }

  ORT_RETURN_IF_NOT(model.ToProto().SerializeToString(&out.bytes), "Failed to serialize the test model.");
  return Status::OK();
}

AllocatorPtr CpuAllocator() {
  return TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
}

// Physically transposes the last two dimensions of a rank-4 MLFloat16 tensor. Used to convert the
// BNSH feed into the BNHS one, and to convert a BNHS result back for comparison.
Status TransposeLastTwoDims(const OrtValue& src, OrtValue& dst) {
  const Tensor& src_tensor = src.Get<Tensor>();
  const auto& src_dims = src_tensor.Shape().GetDims();
  ORT_RETURN_IF_NOT(src_dims.size() == 4, "Expected a rank-4 tensor, got rank ", src_dims.size(), ".");

  const int64_t outer = src_dims[0] * src_dims[1];
  const int64_t rows = src_dims[2];
  const int64_t cols = src_dims[3];

  const std::vector<int64_t> dst_dims{src_dims[0], src_dims[1], cols, rows};
  std::vector<MLFloat16> dst_data(static_cast<size_t>(outer * rows * cols));

  const MLFloat16* src_data = src_tensor.Data<MLFloat16>();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        dst_data[static_cast<size_t>((o * cols + c) * rows + r)] =
            src_data[static_cast<size_t>((o * rows + r) * cols + c)];
      }
    }
  }

  CreateMLValue<MLFloat16>(CpuAllocator(), dst_dims, dst_data, &dst);
  return Status::OK();
}

OrtValue CloneTensor(const OrtValue& src) {
  const Tensor& src_tensor = src.Get<Tensor>();
  const std::vector<int64_t> dims{src_tensor.Shape().GetDims().begin(), src_tensor.Shape().GetDims().end()};
  const std::vector<MLFloat16> data{src_tensor.Data<MLFloat16>(),
                                    src_tensor.Data<MLFloat16>() + src_tensor.Shape().Size()};
  OrtValue copy;
  CreateMLValue<MLFloat16>(CpuAllocator(), dims, data, &copy);
  return copy;
}

// Bit-exact comparison. Both sessions run the same kernel over the same values; the only difference
// is a permutation applied before and after, so any discrepancy is a real defect rather than drift.
Status ExpectTensorsEqual(const OrtValue& expected, const OrtValue& actual, const std::string& what) {
  const Tensor& e = expected.Get<Tensor>();
  const Tensor& a = actual.Get<Tensor>();

  ORT_RETURN_IF_NOT(e.Shape() == a.Shape(), what, ": shape mismatch, expected ", e.Shape().ToString(),
                    " got ", a.Shape().ToString(), ".");

  const MLFloat16* e_data = e.Data<MLFloat16>();
  const MLFloat16* a_data = a.Data<MLFloat16>();
  for (int64_t i = 0; i < e.Shape().Size(); ++i) {
    ORT_RETURN_IF_NOT(e_data[i].val == a_data[i].val, what, ": element ", i, " differs (expected ",
                      e_data[i].ToFloat(), ", got ", a_data[i].ToFloat(), ").");
  }
  return Status::OK();
}

// Compares two BNSH caches over the region the operator defines. Entries past
// total_sequence_length are unspecified: the shared-buffer path leaves the caller's stale data
// there, while a freshly allocated present_value need not.
Status ExpectCacheRegionEqual(const OrtValue& expected, const OrtValue& actual, int64_t valid_seq,
                              const std::string& what) {
  const Tensor& e = expected.Get<Tensor>();
  const Tensor& a = actual.Get<Tensor>();
  ORT_RETURN_IF_NOT(e.Shape() == a.Shape(), what, ": shape mismatch, expected ", e.Shape().ToString(),
                    " got ", a.Shape().ToString(), ".");

  const auto& dims = e.Shape().GetDims();
  ORT_RETURN_IF_NOT(dims.size() == 4, what, ": expected a rank-4 tensor.");
  const int64_t outer = dims[0] * dims[1];
  const int64_t seq = dims[2];
  const int64_t head_size = dims[3];
  ORT_RETURN_IF_NOT(valid_seq <= seq, what, ": valid_seq ", valid_seq, " exceeds the cache length ", seq, ".");

  const MLFloat16* e_data = e.Data<MLFloat16>();
  const MLFloat16* a_data = a.Data<MLFloat16>();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t s = 0; s < valid_seq; ++s) {
      for (int64_t h = 0; h < head_size; ++h) {
        const size_t i = static_cast<size_t>((o * seq + s) * head_size + h);
        ORT_RETURN_IF_NOT(e_data[i].val == a_data[i].val, what, ": entry (", o, ", ", s, ", ", h,
                          ") differs (expected ", e_data[i].ToFloat(), ", got ", a_data[i].ToFloat(), ").");
      }
    }
  }
  return Status::OK();
}

Status ExpectTensorsClose(const OrtValue& expected, const OrtValue& actual, float tolerance,
                          const std::string& what) {
  const Tensor& e = expected.Get<Tensor>();
  const Tensor& a = actual.Get<Tensor>();
  ORT_RETURN_IF_NOT(e.Shape() == a.Shape(), what, ": shape mismatch, expected ", e.Shape().ToString(),
                    " got ", a.Shape().ToString(), ".");

  const MLFloat16* e_data = e.Data<MLFloat16>();
  const MLFloat16* a_data = a.Data<MLFloat16>();
  for (int64_t i = 0; i < e.Shape().Size(); ++i) {
    const float diff = std::abs(e_data[i].ToFloat() - a_data[i].ToFloat());
    ORT_RETURN_IF_NOT(diff <= tolerance, what, ": element ", i, " differs by ", diff, " (expected ",
                      e_data[i].ToFloat(), ", got ", a_data[i].ToFloat(), ", tolerance ", tolerance, ").");
  }
  return Status::OK();
}

// Do two tensors hold the same elements in the same memory order, ignoring shape? Used to assert
// that a transpose actually rearranges data. Comparing with shapes included would be useless here:
// the two tensors are deliberately BNSH [1,1,8,16] against BNHS [1,1,16,8], so a shape-aware
// comparison always reports a difference and establishes nothing about the data.
bool FlatDataIsIdentical(const OrtValue& a, const OrtValue& b) {
  const Tensor& ta = a.Get<Tensor>();
  const Tensor& tb = b.Get<Tensor>();
  if (ta.Shape().Size() != tb.Shape().Size()) {
    return false;
  }

  const MLFloat16* a_data = ta.Data<MLFloat16>();
  const MLFloat16* b_data = tb.Data<MLFloat16>();
  for (int64_t i = 0; i < ta.Shape().Size(); ++i) {
    if (a_data[i].val != b_data[i].val) {
      return false;
    }
  }
  return true;
}

// Guards against a parity test that would pass on degenerate data: if a tensor were all zeros, or
// identical under a transpose, comparing it would prove nothing about the layout conversion.
Status ExpectNonDegenerate(const OrtValue& value, const std::string& what) {
  const Tensor& tensor = value.Get<Tensor>();
  const MLFloat16* data = tensor.Data<MLFloat16>();
  const int64_t count = tensor.Shape().Size();

  bool any_nonzero = false;
  bool any_variation = false;
  for (int64_t i = 0; i < count; ++i) {
    any_nonzero = any_nonzero || data[i].ToFloat() != 0.0f;
    any_variation = any_variation || data[i].val != data[0].val;
  }

  ORT_RETURN_IF_NOT(any_nonzero, what, " is all zeros, so comparing it proves nothing.");
  ORT_RETURN_IF_NOT(any_variation, what, " is constant, so comparing it proves nothing.");
  return Status::OK();
}

size_t IndexOfOutput(const RuntimeGqaModel& model, const std::string& name) {
  for (size_t i = 0; i < model.output_names.size(); ++i) {
    if (model.output_names[i] == name) {
      return i;
    }
  }
  return model.output_names.size();
}

SessionOptions MakeSessionOptions(const char* value_layout) {
  SessionOptions session_options;
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  if (value_layout != nullptr) {
    ORT_ENFORCE(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, value_layout).IsOK());
  }
  return session_options;
}

}  // namespace

class GqaValueLayoutTransformerTest : public GraphTransformationTests {};

TEST_F(GqaValueLayoutTransformerTest, InsertsTransposesAndSwapsBoundaryShapes) {
  auto build = [](ModelTestBuilder& builder) { BuildGqaModel(builder, BuildOptions{}); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectBnhsBoundary(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, IsIdempotent) {
  auto build = [](ModelTestBuilder& builder) { BuildGqaModel(builder, BuildOptions{}); };

  // steps=2 runs the transformer twice. A second insertion would produce four Transposes and swap
  // the boundary shapes back to BNSH, so ExpectBnhsBoundary catches a missing idempotency guard.
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/2,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectBnhsBoundary(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, OutputSideOnlyWhenPastValueIsAbsent) {
  BuildOptions opts;
  opts.no_past_kv = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 1));
        const Node* gqa = FindGqa(graph);
        ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");
        return ExpectBnhsPresentValue(graph, *gqa, /*cache_len=*/kSeq);
      }));
}

TEST_F(GqaValueLayoutTransformerTest, InputSideOnlyWhenPresentValueIsAbsent) {
  BuildOptions opts;
  opts.no_present_value = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 1));
        const Node* gqa = FindGqa(graph);
        ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");
        return ExpectBnhsPastValue(graph, *gqa);
      }));
}

// The two operands are in scope independently. past_value arrives from an Identity, so it is not
// application bound and keeps BNSH; present_value is still a graph output, so it must be converted.
// Skipping the whole node would leave an application-visible output in BNSH after the session
// accepted BNHS.
TEST_F(GqaValueLayoutTransformerTest, ConvertsPresentValueWhenOnlyPastValueIsInternal) {
  BuildOptions opts;
  opts.past_value_behind_identity = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/2,  // twice: the mixed case must stay idempotent
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 1));
        const Node* gqa = FindGqa(graph);
        ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");
        // The internal past_value operand is untouched and still BNSH.
        ORT_RETURN_IF_ERROR(ExpectShape(gqa->InputDefs()[4], kBnsh, "GQA past_value operand"));
        return ExpectBnhsPresentValue(graph, *gqa);
      }));
}

// Mirror image: present_value is consumed by an Identity so it is not application read, while
// past_value is still a graph input and must be converted.
TEST_F(GqaValueLayoutTransformerTest, ConvertsPastValueWhenOnlyPresentValueIsInternal) {
  BuildOptions opts;
  opts.present_value_behind_identity = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/2,  // twice: the mixed case must stay idempotent
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) {
        ORT_RETURN_IF_ERROR(ExpectTransposeCount(graph, 1));
        const Node* gqa = FindGqa(graph);
        ORT_RETURN_IF(gqa == nullptr, "GroupQueryAttention node is missing.");
        // The internal present_value operand is untouched and still BNSH.
        ORT_RETURN_IF_ERROR(ExpectShape(gqa->OutputDefs()[2], kBnsh, "GQA present_value operand"));
        return ExpectBnhsPastValue(graph, *gqa);
      }));
}

// A past_value that is neither a graph input nor bindable at all: nothing to convert on that side,
// and present_value is absent, so the node is left alone.
TEST_F(GqaValueLayoutTransformerTest, SkipsWhenNeitherOperandIsApplicationVisible) {
  BuildOptions opts;
  opts.past_value_behind_identity = true;
  opts.no_present_value = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectNoTransposes(graph); },
      [](Graph& graph) { return ExpectNoTransposes(graph); }));
}

// Boundary NodeArgs are shared. Swapping a shared past_value's declared shape while rewiring only
// one of its consumers would leave the other reading a BNHS tensor as BNSH, and processing the
// second node would swap the declared shape back to BNSH and undo the first. The boundary is
// application visible, so the option cannot be honored and initialization must fail.
TEST_F(GqaValueLayoutTransformerTest, RejectsPastValueSharedByTwoGqaNodes) {
  BuildOptions opts;
  opts.second_gqa_sharing_past_kv = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      TestGraphTransformer(build, /*opset_version=*/21, *logger_, MakeTransformer(),
                           TransformerLevel::Level1, /*steps=*/1, nullptr, nullptr),
      "requires this node to be its only consumer");
}

// An internal consumer of the present_value graph output expects BNSH, so retargeting the GQA
// output through a Transpose would silently hand it BNHS.
TEST_F(GqaValueLayoutTransformerTest, RejectsPresentValueAlsoConsumedInternally) {
  BuildOptions opts;
  opts.present_value_also_consumed_internally = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      TestGraphTransformer(build, /*opset_version=*/21, *logger_, MakeTransformer(),
                           TransformerLevel::Level1, /*steps=*/1, nullptr, nullptr),
      "requires it to have no internal consumers");
}

// An initializer that is also a graph input can be overridden by a feed, so the application may bind
// it, but its baked-in data stays BNSH no matter what happens to the declared shape. Swapping the
// shape alone would either fail Graph::Resolve on the mismatch or, when the feed is omitted, hand the
// default BNSH buffer to a Transpose that reads it as BNHS.
TEST_F(GqaValueLayoutTransformerTest, RejectsOverridableInitializerPastValue) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 21;
  domain_to_version[kMSDomain] = 1;

  Model model("GqaValueLayoutOverridableInitializer", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, *logger_);
  Graph& graph = model.MainGraph();

  ModelTestBuilder helper(graph);
  BuildGqaModel(helper, BuildOptions{});
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  const Node* gqa = FindGqa(graph);
  ASSERT_NE(gqa, nullptr);
  const std::string past_value_name = gqa->InputDefs()[4]->Name();

  // Back past_value with an initializer while keeping it in the declared input list. That
  // combination is what ORT reports as an overridable initializer.
  const std::vector<const NodeArg*> declared_inputs = graph.GetInputsIncludingInitializers();

  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(past_value_name);
  initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (const int64_t dim : {kBatch, kKvNumHeads, kMaxSeq, kHeadSize}) {
    initializer.add_dims(dim);
  }
  // FLOAT16 initializer data lives in int32_data, two bytes per element.
  initializer.mutable_int32_data()->Resize(static_cast<int>(kBatch * kKvNumHeads * kMaxSeq * kHeadSize), 0);
  graph.AddInitializedTensor(initializer);

  graph.SetInputs(declared_inputs);
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_FALSE(graph.GetOverridableInitializers().empty()) << "test setup did not produce an overridable initializer";

  GqaValueLayoutTransformer transformer;
  bool modified = false;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(transformer.Apply(graph, modified, *logger_),
                                      "overridable initializer");
  EXPECT_FALSE(modified);
}

// A rejection must leave the graph exactly as it was loaded. Each model here holds two independent
// GQA nodes, one convertible and one with an internally consumed present_value that fails
// validation. A transformer that converted as it walked the graph would rewire the convertible node
// before reaching the other one, leaving a half-converted, unresolved graph behind.
//
// Both build orders are covered because GetNodesInTopologicalOrder() does not necessarily follow
// insertion order for independent nodes: whichever way it sorts, one of these two models presents
// the convertible node first and so catches a transformer that mutates as it validates.
TEST_F(GqaValueLayoutTransformerTest, LeavesTheGraphUntouchedWhenValidationFails) {
  for (const bool convertible_first : {true, false}) {
    SCOPED_TRACE(convertible_first ? "convertible node built first" : "invalid node built first");

    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 21;
    domain_to_version[kMSDomain] = 1;

    Model model("GqaValueLayoutValidationFailure", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, *logger_);
    Graph& graph = model.MainGraph();

    BuildOptions invalid;
    invalid.present_value_also_consumed_internally = true;

    ModelTestBuilder helper(graph);
    if (convertible_first) {
      BuildGqaModel(helper, BuildOptions{});
      BuildGqaModel(helper, invalid);
    } else {
      BuildGqaModel(helper, invalid);
      BuildGqaModel(helper, BuildOptions{});
    }
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());

    GqaValueLayoutTransformer transformer;
    bool modified = false;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(transformer.Apply(graph, modified, *logger_),
                                        "requires it to have no internal consumers");

    EXPECT_FALSE(modified);
    ASSERT_STATUS_OK(ExpectNoTransposes(graph, /*expected_gqa=*/2));
  }
}

// The transformer converts both operands together, so a node with only one side converted means the
// graph was edited by hand. Converting the rest cannot repair it, so fail rather than proceed.
TEST_F(GqaValueLayoutTransformerTest, RejectsPartiallyTransformedNode) {
  BuildOptions opts;
  opts.partially_transformed = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      TestGraphTransformer(build, /*opset_version=*/21, *logger_, MakeTransformer(),
                           TransformerLevel::Level1, /*steps=*/1, nullptr, nullptr),
      "applied to only one of past_value / present_value");
}

// A model saved after the transform was applied is left alone on reload.
TEST_F(GqaValueLayoutTransformerTest, SkipsAnAlreadyTransformedModel) {
  BuildOptions opts;
  opts.already_transformed = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeTransformer(),
      TransformerLevel::Level1, /*steps=*/1,
      [](Graph& graph) { return ExpectTransposeCount(graph, 2); },
      // Still exactly the two Transposes the model arrived with: no second pair was added.
      [](Graph& graph) { return ExpectBnhsBoundary(graph); }));
}

TEST_F(GqaValueLayoutTransformerTest, RejectsFourBitValueCache) {
  BuildOptions opts;
  opts.four_bit_value_cache = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  // Two 4-bit values are packed per byte along head_size, so a byte-wise Transpose cannot express
  // the layout change. Failing loudly beats silently producing wrong results on a non-fusing EP.
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      TestGraphTransformer(build, /*opset_version=*/21, *logger_, MakeTransformer(),
                           TransformerLevel::Level1, /*steps=*/1, nullptr, nullptr),
      "4-bit quantized Value cache");
}

// The same rejection must apply to a model that already carries the Transposes. Classifying it as
// already-converted and returning early would let a 4-bit model initialize and then execute the
// invalid byte-wise transpose on any EP that does not fuse it.
TEST_F(GqaValueLayoutTransformerTest, RejectsFourBitValueCacheOnAnAlreadyTransformedModel) {
  BuildOptions opts;
  opts.four_bit_value_cache = true;
  opts.already_transformed = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildGqaModel(builder, opts); };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      TestGraphTransformer(build, /*opset_version=*/21, *logger_, MakeTransformer(),
                           TransformerLevel::Level1, /*steps=*/1, nullptr, nullptr),
      "4-bit quantized Value cache");
}

// The transform changes the layout the session expects at its own inputs and outputs, so it is
// applied directly by TransformGraph rather than registered as a level 1 optimizer. This test pins
// that down: registered optimizers are skipped entirely at ORT_DISABLE_ALL.
TEST_F(GqaValueLayoutTransformerTest, AppliedWhenOptimizationsAreDisabled) {
  std::string model_bytes;
  ASSERT_STATUS_OK(BuildSerializedGqaModel(*logger_, model_bytes));

  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Default;  // ORT_DISABLE_ALL
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, "BNHS"));

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_STATUS_OK(ExpectBnhsBoundary(session.GetMutableGraph()));
}

TEST_F(GqaValueLayoutTransformerTest, NotAppliedForTheDefaultLayout) {
  std::string model_bytes;
  ASSERT_STATUS_OK(BuildSerializedGqaModel(*logger_, model_bytes));

  SessionOptions session_options;
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, "BNSH"));

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_STATUS_OK(ExpectNoTransposes(session.GetGraph()));
}

TEST_F(GqaValueLayoutTransformerTest, RejectsAnInvalidLayoutValue) {
  std::string model_bytes;
  ASSERT_STATUS_OK(BuildSerializedGqaModel(*logger_, model_bytes));

  SessionOptions session_options;
  session_options.session_logid = "GqaValueLayoutTransformerTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsGqaValueLayout, "NHWC"));

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));

  // An unrecognized option value is a caller error, so the code must be INVALID_ARGUMENT rather than
  // the generic FAIL. A model that cannot satisfy a recognized value reports FAIL instead, and
  // applications distinguish the two to decide whether falling back to BNSH is worth trying.
  const Status status = session.Initialize();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT) << status.ErrorMessage();
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Invalid value for session option"));
}

namespace {

// Builds boundary -> Transpose -> Identity -> Transpose -> boundary, i.e. the shape the graph is left
// in when a compiling EP claims the GroupQueryAttention node and leaves the flanking Transposes
// behind. Identity stands in for the EP's fused node. With keep_transposes=false the boundaries
// connect straight to Identity, which is what fusing the whole sequence looks like.
Status BuildPostPartitionGraph(Graph& graph, bool keep_transposes, GqaValueLayoutBoundaries& boundaries) {
  const std::vector<int64_t> bnhs{kBatch, kKvNumHeads, kHeadSize, kMaxSeq};
  const std::vector<int64_t> bnsh{kBatch, kKvNumHeads, kMaxSeq, kHeadSize};

  // Both boundaries are BNHS either way; only what sits between them changes.
  ModelTestBuilder builder(graph);
  NodeArg* boundary_in = builder.MakeInput<MLFloat16>(bnhs, MLFloat16(0.0f), MLFloat16(0.0f));
  NodeArg* boundary_out = builder.MakeOutput<MLFloat16>(bnhs);

  if (keep_transposes) {
    NodeArg* fused_in = builder.MakeIntermediate<MLFloat16>(bnsh);
    NodeArg* fused_out = builder.MakeIntermediate<MLFloat16>(bnsh);

    Node& in_transpose = builder.AddNode("Transpose", {boundary_in}, {fused_in});
    in_transpose.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 2});

    builder.AddNode("Identity", {fused_in}, {fused_out});

    Node& out_transpose = builder.AddNode("Transpose", {fused_out}, {boundary_out});
    out_transpose.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 2});
  } else {
    builder.AddNode("Identity", {boundary_in}, {boundary_out});
  }

  builder.SetGraphOutputs();
  ORT_RETURN_IF_ERROR(graph.Resolve());

  boundaries.past_value_inputs.push_back(boundary_in->Name());
  boundaries.present_value_outputs.push_back(boundary_out->Name());
  return Status::OK();
}

Model MakePostPartitionModel(const logging::Logger& logger) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 21;
  domain_to_version[kMSDomain] = 1;
  return Model("GqaValueLayoutPostPartition", false, ModelMetaData(), PathString(),
               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
}

}  // namespace

// A compiling EP may claim the GQA node and replace it with a fused node while leaving the flanking
// Transposes in the graph. Both full-cache copies still execute, so the diagnostic must not depend on
// finding a GroupQueryAttention node to search from.
TEST_F(GqaValueLayoutTransformerTest, ReportsUnfusedTransposesWhenTheGqaNodeWasReplaced) {
  Model model = MakePostPartitionModel(*logger_);
  GqaValueLayoutBoundaries boundaries;
  ASSERT_STATUS_OK(BuildPostPartitionGraph(model.MainGraph(), /*keep_transposes=*/true, boundaries));

  ASSERT_EQ(FindGqa(model.MainGraph()), nullptr)
      << "the fixture must not contain a GQA node, otherwise it cannot catch the regression";

  const auto unfused = ReportUnfusedGqaValueLayoutTransposes(model.MainGraph(), boundaries, *logger_);
  EXPECT_THAT(unfused, ::testing::UnorderedElementsAre(boundaries.past_value_inputs[0],
                                                       boundaries.present_value_outputs[0]));
}

// The other half of the contract: when the provider did absorb the Transposes, nothing is reported.
TEST_F(GqaValueLayoutTransformerTest, ReportsNothingWhenTheTransposesWereFused) {
  Model model = MakePostPartitionModel(*logger_);
  GqaValueLayoutBoundaries boundaries;
  ASSERT_STATUS_OK(BuildPostPartitionGraph(model.MainGraph(), /*keep_transposes=*/false, boundaries));

  const auto unfused = ReportUnfusedGqaValueLayoutTransposes(model.MainGraph(), boundaries, *logger_);
  EXPECT_TRUE(unfused.empty());
}

// The design accepts that a non-fusing EP executes the inserted transposes. That fallback is only
// acceptable if it is numerically correct, so verify it on the CPU EP rather than only checking
// graph structure: the BNHS session fed a transposed cache must match the BNSH session exactly.
TEST_F(GqaValueLayoutTransformerTest, BnhsMatchesBnshOnCpu) {
  RuntimeGqaModel model;
  ASSERT_STATUS_OK(BuildRuntimeGqaModel(*logger_, model));

  const size_t present_value_index = IndexOfOutput(model, model.present_value_name);
  const size_t attention_output_index = IndexOfOutput(model, model.attention_output_name);
  ASSERT_LT(present_value_index, model.output_names.size());
  ASSERT_LT(attention_output_index, model.output_names.size());

  // Baseline: the default BNSH layout, no transposes in the graph.
  std::vector<OrtValue> bnsh_fetches;
  {
    SessionOptions session_options = MakeSessionOptions(nullptr);
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model.bytes.data(), static_cast<int>(model.bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(ExpectNoTransposes(session.GetGraph()));
    ASSERT_STATUS_OK(session.Run(RunOptions{}, model.bnsh_feeds, model.output_names, &bnsh_fetches));
  }

  // BNHS: same model, same values, but the Value cache is handed over transposed.
  std::vector<OrtValue> bnhs_fetches;
  {
    NameMLValMap bnhs_feeds = model.bnsh_feeds;
    OrtValue past_value_bnhs;
    ASSERT_STATUS_OK(TransposeLastTwoDims(model.bnsh_feeds.at(model.past_value_name), past_value_bnhs));
    bnhs_feeds[model.past_value_name] = past_value_bnhs;

    SessionOptions session_options = MakeSessionOptions(kGqaValueLayoutBNHS);
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model.bytes.data(), static_cast<int>(model.bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(ExpectBnhsBoundary(session.GetMutableGraph()));
    ASSERT_STATUS_OK(session.Run(RunOptions{}, bnhs_feeds, model.output_names, &bnhs_fetches));
  }

  // Confirm the comparison is meaningful before making it.
  ASSERT_STATUS_OK(ExpectNonDegenerate(bnsh_fetches[attention_output_index], "attention output"));
  ASSERT_STATUS_OK(ExpectNonDegenerate(bnsh_fetches[present_value_index], "present_value"));
  ASSERT_STATUS_OK(ExpectNonDegenerate(bnhs_fetches[present_value_index], "BNHS present_value"));
  // A transpose-invariant present_value would hide a broken conversion. Compare the raw element
  // sequences, ignoring the (deliberately different) shapes.
  ASSERT_FALSE(FlatDataIsIdentical(bnsh_fetches[present_value_index], bnhs_fetches[present_value_index]))
      << "BNSH and BNHS present_value hold the same elements in the same order, so the transpose moved "
         "nothing and this test cannot detect a layout bug.";

  // The attention output is layout independent and must match directly.
  ASSERT_STATUS_OK(ExpectTensorsEqual(bnsh_fetches[attention_output_index],
                                      bnhs_fetches[attention_output_index], "attention output"));

  // present_value comes back BNHS; transposing it must reproduce the BNSH result exactly.
  OrtValue present_value_bnsh;
  ASSERT_STATUS_OK(TransposeLastTwoDims(bnhs_fetches[present_value_index], present_value_bnsh));
  ASSERT_STATUS_OK(ExpectTensorsEqual(bnsh_fetches[present_value_index], present_value_bnsh, "present_value"));
}

// The same check with one buffer bound to both past_value and present_value, which is how a decode
// loop actually drives the model. The two inserted transposes decouple the aliased boundary buffer
// from the GQA operands, so the data dependency Transpose -> GQA -> Transpose keeps this well
// defined even though the CPU EP does not fuse them.
//
// The reference here is the same BNHS model driven with separate input and output buffers, not the
// BNSH session. Binding one buffer to both sides in BNSH hands the CPU kernel an aliased past and
// present, so it takes its shared-buffer path; under BNHS the operands are the transpose
// intermediates, so it cannot. Comparing across those two paths would be comparing two different
// kernel implementations. BnhsMatchesBnshOnCpu already establishes that BNHS with separate buffers
// matches BNSH exactly, so chaining the two tests covers the whole claim.
TEST_F(GqaValueLayoutTransformerTest, BnhsWithAliasedCacheBufferMatchesSeparateBuffersOnCpu) {
  RuntimeGqaModel model;
  ASSERT_STATUS_OK(BuildRuntimeGqaModel(*logger_, model));

  const size_t attention_output_index = IndexOfOutput(model, model.attention_output_name);
  const size_t present_value_index = IndexOfOutput(model, model.present_value_name);
  ASSERT_LT(attention_output_index, model.output_names.size());
  ASSERT_LT(present_value_index, model.output_names.size());

  OrtValue past_value_bnhs;
  ASSERT_STATUS_OK(TransposeLastTwoDims(model.bnsh_feeds.at(model.past_value_name), past_value_bnhs));

  // Reference: separate buffers.
  std::vector<OrtValue> reference_fetches;
  {
    NameMLValMap bnhs_feeds = model.bnsh_feeds;
    bnhs_feeds[model.past_value_name] = past_value_bnhs;

    SessionOptions session_options = MakeSessionOptions(kGqaValueLayoutBNHS);
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model.bytes.data(), static_cast<int>(model.bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(RunOptions{}, bnhs_feeds, model.output_names, &reference_fetches));
  }

  // Aliased: one buffer bound to both past_value and present_value, as a decode loop would.
  OrtValue cache = CloneTensor(past_value_bnhs);
  OrtValue aliased_attention_output;
  {
    SessionOptions session_options = MakeSessionOptions(kGqaValueLayoutBNHS);
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model.bytes.data(), static_cast<int>(model.bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(ExpectBnhsBoundary(session.GetMutableGraph()));

    std::unique_ptr<IOBinding> binding;
    ASSERT_STATUS_OK(session.NewIOBinding(&binding));

    for (const auto& [name, value] : model.bnsh_feeds) {
      if (name != model.past_value_name) {
        ASSERT_STATUS_OK(binding->BindInput(name, value));
      }
    }
    ASSERT_STATUS_OK(binding->BindInput(model.past_value_name, cache));

    for (const auto& name : model.output_names) {
      if (name == model.present_value_name) {
        ASSERT_STATUS_OK(binding->BindOutput(name, cache));
      } else {
        ASSERT_STATUS_OK(binding->BindOutput(name));
      }
    }

    ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));

    const auto& outputs = binding->GetOutputs();
    for (size_t i = 0; i < model.output_names.size(); ++i) {
      if (model.output_names[i] == model.attention_output_name) {
        aliased_attention_output = outputs[i];
      }
    }
  }

  ASSERT_STATUS_OK(ExpectNonDegenerate(aliased_attention_output, "attention output"));
  ASSERT_STATUS_OK(ExpectNonDegenerate(cache, "aliased cache buffer"));

  // The session wrote the caller's buffer rather than leaving the input untouched.
  ASSERT_FALSE(ExpectTensorsEqual(past_value_bnhs, cache, "aliased cache buffer").IsOK())
      << "The aliased buffer is unchanged, so this test is not exercising the in-place update.";

  ASSERT_STATUS_OK(ExpectTensorsEqual(reference_fetches[attention_output_index], aliased_attention_output,
                                      "attention output, aliased vs separate buffers"));

  // The buffer holds BNHS, so transpose both sides into BNSH before comparing the defined region.
  OrtValue cache_as_bnsh;
  OrtValue reference_present_as_bnsh;
  ASSERT_STATUS_OK(TransposeLastTwoDims(cache, cache_as_bnsh));
  ASSERT_STATUS_OK(TransposeLastTwoDims(reference_fetches[present_value_index], reference_present_as_bnsh));
  ASSERT_STATUS_OK(ExpectCacheRegionEqual(reference_present_as_bnsh, cache_as_bnsh, kPastSeq + kSeq,
                                          "aliased cache buffer"));
}

// The ORT format load path does not run TransformGraph, so the option cannot be honored there.
// Silently ignoring it would leave the session expecting BNSH while the application supplies BNHS.
TEST_F(GqaValueLayoutTransformerTest, RejectsOrtFormatModel) {
  SessionOptions session_options = MakeSessionOptions(kGqaValueLayoutBNHS);

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(ORT_TSTR("testdata/mnist.basic.ort")));

  // Also a caller error: the option is valid, but not for this model format.
  const Status status = session.Initialize();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT) << status.ErrorMessage();
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("is not supported for ORT format models"));
}

TEST_F(GqaValueLayoutTransformerTest, AllowsOrtFormatModelWithTheDefaultLayout) {
  SessionOptions session_options = MakeSessionOptions(kGqaValueLayoutBNSH);

  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(ORT_TSTR("testdata/mnist.basic.ort")));
  ASSERT_STATUS_OK(session.Initialize());
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
