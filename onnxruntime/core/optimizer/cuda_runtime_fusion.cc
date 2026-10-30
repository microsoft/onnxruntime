// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/cuda_runtime_fusion.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string_view>

#include "contrib_ops/cuda/bert/causal_conv_with_state_validation.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/pattern_rewriter.h"

namespace onnxruntime::cuda_runtime_fusion {
namespace {
using namespace rewrite;
constexpr int kInt64 = ONNX_NAMESPACE::TensorProto_DataType_INT64;

enum class BiasAlternative { Unfolded,
                             Folded,
                             Convolved };

struct Contract {
  int64_t channels;
  int64_t kernel;
  int64_t dilation;
  int64_t pad;
  int type;
  BiasAlternative bias_alternative;
  bool normalized_zero_tail;
  InlinedVector<std::string> bindings;

  bool operator==(const Contract& other) const {
    return channels == other.channels && kernel == other.kernel && dilation == other.dilation &&
           pad == other.pad && type == other.type && bias_alternative == other.bias_alternative &&
           normalized_zero_tail == other.normalized_zero_tail && bindings == other.bindings;
  }
};

ONNX_NAMESPACE::AttributeProto DilationAttribute(const Contract& contract) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name("dilation");
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(contract.dilation);
  return attr;
}

constexpr char kPattern[] = R"(
  interface BiasedConv(c,w,bias) -> (a)
  pattern unfolded implements BiasedConv: (bias_shape) -> {
    z=Conv(c,w); bias3d=Reshape(bias,bias_shape); sum=Add(z,bias3d); return (a=sum)
  }
  pattern folded implements BiasedConv {
    z=Conv(c,w); sum=Add(z,bias); return (a=sum)
  }
  pattern convolved implements BiasedConv {
    z=Conv(c,w,bias); return (a=z)
  }
  pattern BiasForms = unfolded | folded | convolved
  interface HistoryTail(c,index,vector_shape,axes) -> (state)
  pattern original implements HistoryTail: (window) -> {
    shape=Shape(c); length=Gather(shape,index);
    start=Sub(length,window); start1=Reshape(start,vector_shape);
    end1=Reshape(length,vector_shape); sliced=Slice(c,start1,end1,axes);
    return (state=sliced)
  }
  pattern zero implements HistoryTail {
    shape=Shape(c); length=Gather(shape,index); end1=Reshape(length,vector_shape);
    sliced=Slice(c,end1,end1,axes); return (state=sliced)
  }
  pattern TailForms = original | zero
  pattern Mobius(x,h,w,bias,index,vector_shape,axes) {
    c=Concat(h,x); tail=TailForms(c,index,vector_shape,axes);
    bias_add=BiasForms(c,w,bias);
    y=Swish(bias_add.a); return y,tail.state
  })";

constexpr char kReplacement[] = R"(
  (x,w,bias,h,dilation)=>{
    y,state=com.microsoft.CausalConvWithState(x,w,bias,h,
      ndim=1,activation="silu",channels_last=0,dilation=dilation,state_window=0);
    return y,state
  })";

bool Int(const ONNX_NAMESPACE::AttributeProto& attr, int64_t expected) {
  return attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT && attr.i() == expected;
}

bool Ints(const ONNX_NAMESPACE::AttributeProto& attr, std::initializer_list<int64_t> expected) {
  return attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INTS &&
         static_cast<size_t>(attr.ints_size()) == expected.size() &&
         std::equal(attr.ints().begin(), attr.ints().end(), expected.begin());
}

// Inspect only operators already captured by Rama; never search for an alternate structure.
bool Attributes(const Match& match, const Contract& contract) {
  for (const auto* node : match.nodes) {
    if (node->Domain() != kOnnxDomain) return false;
    const auto& op = node->OpType();
    const auto& attrs = node->GetAttributes();
    if (op == "Slice" && node->InputDefs().size() != 4) return false;
    if (op == "Conv" &&
        node->InputDefs().size() != (contract.bias_alternative == BiasAlternative::Convolved ? 3u : 2u)) return false;
    if (op == "Concat" && (attrs.count("axis") != 1 || !Int(attrs.at("axis"), 2))) return false;
    if (op == "Conv" && (attrs.count("group") != 1 || !Int(attrs.at("group"), contract.channels))) return false;
    for (const auto& [name, attr] : attrs) {
      bool valid = false;
      if (op == "Concat") valid = name == "axis" && Int(attr, 2);
      if (op == "Gather") valid = name == "axis" && Int(attr, 0);
      if (op == "Shape") {
        valid = (name == "start" && Int(attr, 0)) ||
                (name == "end" && Int(attr, std::numeric_limits<int64_t>::max()));
      }
      if (op == "Reshape") valid = name == "allowzero" && Int(attr, 0);
      if (op == "Conv") {
        valid = (name == "group" && Int(attr, contract.channels)) ||
                (name == "kernel_shape" && Ints(attr, {contract.kernel})) ||
                (name == "strides" && Ints(attr, {1})) ||
                (name == "dilations" && Ints(attr, {contract.dilation})) ||
                (name == "pads" && Ints(attr, {0, 0})) ||
                (name == "auto_pad" && attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING &&
                 attr.s() == "NOTSET");
      }
      if (op == "Swish") {
        valid = name == "alpha" && attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT &&
                std::isfinite(attr.f()) && attr.f() == 1.0f;
      }
      if (!valid) return false;
    }
  }
  return true;
}

bool Header(const Graph& graph, const NodeArg* value, int type, std::initializer_list<int64_t> dims) {
  if (!value) return false;
#if !defined(DISABLE_SPARSE_TENSORS)
  if (graph.IsSparseInitializer(value->Name())) return false;
#endif
  const auto* tensor = graph.GetConstantInitializer(value->Name(), false);
  const auto* metadata = value->TypeAsProto();
  if (metadata && (!metadata->has_tensor_type() ||
                   (metadata->tensor_type().elem_type() != 0 && metadata->tensor_type().elem_type() != type))) return false;
  if (const auto* shape = value->Shape()) {
    if (static_cast<size_t>(shape->dim_size()) != dims.size()) return false;
    size_t axis = 0;
    for (auto dim : dims) {
      const auto& actual = shape->dim(static_cast<int>(axis++));
      if (actual.has_dim_value() && actual.dim_value() != dim) return false;
    }
  }
  return tensor && tensor->data_type() == type && static_cast<size_t>(tensor->dims_size()) == dims.size() &&
         std::equal(tensor->dims().begin(), tensor->dims().end(), dims.begin());
}

// No tensor data is read here. In-memory markers must refer to graph-owned CPU storage,
// not just a plausible TensorProto address. File-backed storage is not opened until acceptance.
bool LearnedStorage(const Graph& graph, const NodeArg* value, size_t elements) {
  const auto* proto = graph.GetConstantInitializer(value->Name(), false);
  if (!proto) return false;
  const bool fp32 = proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  const size_t element_bytes = fp32 ? sizeof(float) : sizeof(uint16_t);
  if (elements > std::numeric_limits<size_t>::max() / element_bytes ||
      elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / element_bytes ||
      proto->has_segment() || (fp32 ? proto->int32_data_size() : proto->float_data_size()) ||
      proto->double_data_size() ||
      proto->int64_data_size() || proto->uint64_data_size() || proto->string_data_size()) return false;
  OrtValue owned;
  const bool has_owned = graph.GetOrtValueInitializer(value->Name(), owned, false);
  if (!utils::HasExternalData(*proto)) {
    return proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_DEFAULT &&
           !has_owned && proto->external_data().empty() &&
           (proto->raw_data().empty() || (proto->int32_data_size() == 0 && proto->float_data_size() == 0)) &&
           utils::ValidateEmbeddedTensorProtoDataSizeAndShape(*proto).IsOK();
  }
  if (!proto->raw_data().empty() || proto->int32_data_size() || proto->int64_data_size() ||
      proto->float_data_size() || proto->double_data_size() || proto->uint64_data_size() ||
      proto->string_data_size()) return false;
  std::unique_ptr<ExternalDataInfo> external;
  if (!ExternalDataInfo::Create(proto->external_data(), external).IsOK() || external->GetOffset() < 0 ||
      (external->GetLength() != 0 && external->GetLength() != elements * element_bytes)) return false;
  if (!utils::HasExternalDataInMemory(*proto)) return !has_owned;
  if (!has_owned || !owned.IsTensor()) return false;
  const auto& tensor = owned.Get<Tensor>();
  return tensor.GetElementType() == proto->data_type() && tensor.Location().device.Type() == OrtDevice::CPU &&
         tensor.Shape() == TensorShape(proto->dims()) &&
         graph_utils::CheckInMemoryDataMatch(*proto, tensor);
}

bool BiasHeader(const Graph& graph, const NodeArg* value, const Contract& contract, bool folded) {
  const bool valid_header = folded ? Header(graph, value, contract.type, {1, contract.channels, 1})
                                   : Header(graph, value, contract.type, {contract.channels});
  return valid_header && LearnedStorage(graph, value, static_cast<size_t>(contract.channels));
}

BiasAlternative SelectedBiasAlternative(const Match& match) {
  const auto form = match.instance("bias_add").alternative();
  if (form == "unfolded") return BiasAlternative::Unfolded;
  if (form == "folded") return BiasAlternative::Folded;
  ORT_ENFORCE(form == "convolved", "Unexpected CUDA runtime fusion bias alternative.");
  return BiasAlternative::Convolved;
}

bool NormalizedZeroTail(const Match& match) {
  const auto form = match.instance("tail").alternative();
  ORT_ENFORCE(form == "original" || form == "zero", "Unexpected CUDA runtime fusion tail alternative.");
  return form == "zero";
}

bool SmallConstant(const Graph& graph, const NodeArg* value,
                   std::initializer_list<int64_t> dims, std::initializer_list<int64_t> expected) {
  if (!Header(graph, value, kInt64, dims)) return false;
  // The generic reader can load external storage; discovery only permits embedded payloads.
  if (utils::HasExternalData(*graph.GetConstantInitializer(value->Name(), false))) return false;
  InlinedVector<int64_t> data;
  bool available = false;
  // An unavailable payload declines. A corrupt payload is a load/adapter error, not a match.
  ORT_THROW_IF_ERROR(ReadConstantTensor(graph, value, data, available));
  return available && data.size() == expected.size() && std::equal(data.begin(), data.end(), expected.begin());
}

bool Type(const NodeArg* value, int expected, bool required) {
  const auto* type = value->TypeAsProto();
  if (!type || !type->has_tensor_type()) return !required && !type;
  const int element = type->tensor_type().elem_type();
  return element == expected || (!required && element == 0);
}

bool MergeKnown(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim, std::optional<int64_t>& known) {
  if (!dim.has_dim_value()) return true;
  if (known && *known != dim.dim_value()) return false;
  known = dim.dim_value();
  return true;
}

bool Shape(const NodeArg* value, int type, std::initializer_list<int64_t> expected, bool required = false) {
  if (!Type(value, type, required)) return false;
  const auto* shape = value->Shape();
  if (!shape) return !required;
  if (static_cast<size_t>(shape->dim_size()) != expected.size()) return false;
  size_t axis = 0;
  for (auto dim : expected) {
    const auto& actual = shape->dim(static_cast<int>(axis++));
    if (actual.has_dim_value() && actual.dim_value() < 0) return false;
    if (dim >= 0 && ((required && !actual.has_dim_value()) ||
                     (actual.has_dim_value() && actual.dim_value() != dim))) return false;
  }
  return true;
}

bool Semantics(const Graph& graph, Match& match, Contract* extracted = nullptr) {
  const auto bias_alternative = SelectedBiasAlternative(match);
  const bool folded = bias_alternative == BiasAlternative::Folded;
  const bool unfolded = bias_alternative == BiasAlternative::Unfolded;
  const bool zero_tail = NormalizedZeroTail(match);
  const auto bias_form = match.instance("bias_add").alternativeView();
  const auto tail = match.instance("tail").alternativeView();
  const auto* state = match.instance("tail").valueOf("state");
  const auto* weight = graph.GetConstantInitializer(match.valueOf("w")->Name(), false);
  if (!weight || weight->dims_size() != 3 || weight->dims(1) != 1) return false;
  const int type = weight->data_type();
  if (type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) return false;
  int64_t dilation = 1;
  for (const auto* node : match.nodes) {
    if (node->OpType() != "Conv") continue;
    const auto it = node->GetAttributes().find("dilations");
    if (it != node->GetAttributes().end()) {
      if (it->second.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
          it->second.ints_size() != 1) return false;
      dilation = it->second.ints(0);
    }
  }
  const size_t element_bytes = type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ? sizeof(float) : sizeof(uint16_t);
  if (!contrib::cuda::ValidateCausalConvNativeParameters(std::nullopt, weight->dims(0), std::nullopt,
                                                         weight->dims(2), dilation, 1, element_bytes)
           .IsOK()) return false;
  Contract contract{weight->dims(0), weight->dims(2), dilation, (weight->dims(2) - 1) * dilation, type, bias_alternative, zero_tail, {}};
  const auto channels = contract.channels;
  const auto kernel = contract.kernel;
  const auto pad = contract.pad;
  if ((zero_tail && (kernel != 1 || pad != 0)) ||
      !Attributes(match, contract) ||
      !SmallConstant(graph, match.valueOf("index"), {}, {2}) ||
      (!zero_tail && !SmallConstant(graph, tail.valueOf("window"), {}, {pad})) ||
      !SmallConstant(graph, match.valueOf("vector_shape"), {1}, {1}) ||
      !SmallConstant(graph, match.valueOf("axes"), {1}, {2}) ||
      (unfolded && !SmallConstant(graph, bias_form.valueOf("bias_shape"), {3}, {1, -1, 1})) ||
      !Header(graph, match.valueOf("w"), type, {channels, 1, kernel}) ||
      !LearnedStorage(graph, match.valueOf("w"), static_cast<size_t>(kernel * channels)) ||
      !BiasHeader(graph, match.valueOf("bias"), contract, folded) ||
      !Shape(match.valueOf("w"), type, {channels, 1, kernel}, true) ||
      !Shape(match.valueOf("bias"), type,
             folded ? std::initializer_list<int64_t>{1, channels, 1} : std::initializer_list<int64_t>{channels}, true) ||
      !Shape(match.valueOf("x"), type, {-1, channels, -1}, true) ||
      !Shape(match.valueOf("h"), type, {-1, channels, pad}, true) ||
      (zero_tail && !Shape(state, type, {-1, channels, 0}, true))) {
    return false;
  }
  std::optional<int64_t> batch;
  std::optional<int64_t> sequence;
  for (const auto* value : {match.valueOf("x"), match.valueOf("h"), match.valueOf("c"), bias_form.valueOf("z"),
                            match.instance("bias_add").valueOf("a"), match.valueOf("y"), state}) {
    const bool history = value == match.valueOf("h") || value == state;
    if (!Shape(value, type, {-1, channels, history ? pad : -1})) return false;
    const auto* shape = value->Shape();
    if (!shape) continue;
    if (!MergeKnown(shape->dim(0), batch)) return false;
    if (history) continue;
    auto length = shape->dim(2);
    if (value == match.valueOf("c") && length.has_dim_value()) {
      if (length.dim_value() < pad) return false;
      length.set_dim_value(length.dim_value() - pad);
    }
    if (!MergeKnown(length, sequence)) return false;
  }
  if (!contrib::cuda::ValidateCausalConvNativeParameters(batch, channels, sequence, kernel, dilation, 1, element_bytes)
           .IsOK()) return false;
  const bool valid = Shape(tail.valueOf("shape"), kInt64, {3}) &&
                     Shape(tail.valueOf("length"), kInt64, {}) &&
                     (zero_tail || (Shape(tail.valueOf("start"), kInt64, {}) &&
                                    Shape(tail.valueOf("start1"), kInt64, {1}))) &&
                     Shape(tail.valueOf("end1"), kInt64, {1}) &&
                     (!unfolded || Shape(bias_form.valueOf("bias3d"), type, {1, channels, 1}));
  if (!valid || !match.BindAttribute("dilation", DilationAttribute(contract))) return false;
  // Capture identities and metadata by value, not graph pointers or borrowed Match views.
  for (const auto* value : match.values) {
    contract.bindings.push_back(value->Name());
    contract.bindings.push_back(value->TypeAsProto() ? value->TypeAsProto()->SerializeAsString() : "");
  }
  if (extracted) *extracted = std::move(contract);
  return true;
}

Status ProbeTarget(Graph& graph, const Replacement& recipe, const Replacement::TensorBindings& bindings,
                   const NodeAttributes& attributes, size_t output_count,
                   const IExecutionProvider::IKernelLookup& lookup, bool& available) {
  Replacement::Instantiation instantiated;
  ORT_RETURN_IF_ERROR(recipe.Instantiate(graph, bindings, attributes, instantiated));
  ORT_RETURN_IF_NOT(instantiated.nodes.size() == 1 && instantiated.outputs.size() == output_count,
                    "CUDA runtime fusion target recipe must emit one node with the expected outputs.");
  InlinedVector<const NodeArg*> outputs(instantiated.outputs.begin(), instantiated.outputs.end());
  graph.SetOutputs(outputs);
  ORT_RETURN_IF_ERROR(graph.Resolve());
  available = lookup.LookUpKernel(*graph.GetNode(instantiated.nodes.front())) != nullptr;
  return Status::OK();
}

Status TargetAvailable(const Match& match, const Contract& contract, const Replacement& recipe,
                       const IExecutionProvider::IKernelLookup& lookup, const logging::Logger& logger,
                       bool& available) {
  Model scratch("cuda_runtime_fusion_probe", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, logger);
  auto& graph = scratch.MainGraph();
  Replacement::TensorBindings bindings;
  InlinedVector<const NodeArg*> inputs;
  for (const char* name : {"x", "w", "bias", "h"}) {
    const auto* value = match.valueOf(name);
    auto type = *value->TypeAsProto();
    if (std::string_view(name) == "bias" && contract.bias_alternative == BiasAlternative::Folded) {
      auto* shape = type.mutable_tensor_type()->mutable_shape();
      shape->clear_dim();
      shape->add_dim()->set_dim_value(contract.channels);
    }
    auto& input = graph.GetOrCreateNodeArg(name, &type);
    bindings.emplace(name, &input);
    inputs.push_back(&input);
  }
  graph.SetInputs(inputs);
  return ProbeTarget(graph, recipe, bindings, {{"dilation", DilationAttribute(contract)}}, 2, lookup, available);
}

Status AddFlatBiasCopy(Graph& graph, const NodeArg& source, const Contract& contract, NodeArg*& flat_bias) {
  ORT_RETURN_IF_NOT(BiasHeader(graph, &source, contract, true), "Folded CUDA runtime fusion bias storage changed.");
  const auto* proto = graph.GetConstantInitializer(source.Name(), false);
  // Initializer owns a bitwise copy, including for graph-owned in-memory source storage.
  // Never reshape a copied OrtValue: its Tensor would still be shared with the source.
  Initializer copy(graph, *proto, graph.ModelPath(), false);
  ONNX_NAMESPACE::TensorProto flat_proto;
  OrtValue owned;
  copy.ToProtoWithOrtValue(flat_proto, owned);
  flat_proto.set_name(graph.GenerateNodeArgName("cuda_runtime_fusion_flat_bias"));
  flat_proto.clear_dims();
  flat_proto.add_dims(contract.channels);
  if (owned.IsAllocated()) owned.GetMutable<Tensor>()->Reshape(TensorShape({contract.channels}));
  auto& added = graph_utils::AddInitializerWithOrtValue(graph, flat_proto, std::move(owned));
  flat_bias = &added;
  return Status::OK();
}

std::unique_ptr<ComputeCapability> Offer(const Match& match, NodeIndex anchor, Contract contract,
                                         const Pattern& pattern, const Replacement& recipe) {
  Options options;
  options.compatible_execution_providers.clear();
  Rules rules;
  rules.emplace_back(kMobiusBf16Rule, pattern, recipe, [contract](const Graph& graph, Match& candidate) {
                       Contract actual;
                       return Semantics(graph, candidate, &actual) && actual == contract; }, InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}});
  auto rewriter = std::make_shared<const PatternRewriter>(std::move(rules), std::move(options));
  auto subgraph = std::make_unique<IndexedSubGraph>();
  for (const auto* node : match.nodes) subgraph->nodes.push_back(node->Index());
  const auto source_ids = subgraph->nodes;
  const auto y = match.valueOf("y")->Name();
  const auto state = match.instance("tail").valueOf("state")->Name();
  const auto bias = match.valueOf("bias")->Name();
  const bool folded = contract.bias_alternative == BiasAlternative::Folded;
  auto parent = std::make_unique<ComputeCapability>(std::move(subgraph));
  auto optimization = std::make_unique<ComputeCapability>(std::make_unique<IndexedSubGraph>());
  optimization->sub_graph->nodes = source_ids;
  optimization->optimization_func =
      [rewriter = std::move(rewriter), anchor, source_ids, y, state, bias, folded, contract](
          Graph& graph, const ComputeCapability& selected, ComputeCapability& capability,
          const GraphOptimizerRegistry&) -> Status {
    ORT_RETURN_IF_NOT(selected.sub_graph && capability.sub_graph, "Missing CUDA runtime fusion region.");
    InlinedHashSet<NodeIndex> region(selected.sub_graph->nodes.begin(), selected.sub_graph->nodes.end());
    const InlinedHashSet<NodeIndex> expected(source_ids.begin(), source_ids.end());
    ORT_RETURN_IF_NOT(region == expected && selected.sub_graph->nodes.size() == source_ids.size(),
                      "CUDA runtime fusion selected region changed.");
    for (auto id : source_ids) {
      const auto* node = graph.GetNode(id);
      ORT_RETURN_IF_NOT(node && node->GetExecutionProviderType().empty(),
                        "CUDA runtime fusion source is missing or already assigned.");
    }
    const auto* original_y = graph.GetNodeArg(y);
    const auto* original_state = graph.GetNodeArg(state);
    ORT_RETURN_IF_NOT(original_y && original_state, "Missing CUDA runtime fusion output.");
    const auto* original_bias = graph.GetNodeArg(bias);
    ORT_RETURN_IF_NOT(BiasHeader(graph, original_bias, contract, folded), "CUDA runtime fusion bias changed.");
    Statistics statistics;
    ORT_RETURN_IF_ERROR(rewriter->RunOnce(graph, statistics, region, anchor));
    ORT_RETURN_IF_NOT(statistics.rewrites == 1 && region.size() == 1,
                      "CUDA runtime fusion must rewrite exactly once to one node.");
    auto* target = graph.GetNode(*region.begin());
    ORT_RETURN_IF_NOT(target && target->Domain() == kMSDomain && target->OpType() == "CausalConvWithState" &&
                          target->InputDefs().size() == 4 && target->InputDefs()[2] == original_bias &&
                          target->OutputDefs().size() == 2 && target->OutputDefs()[0] == original_y &&
                          target->OutputDefs()[1] == original_state,
                      "CUDA runtime fusion did not preserve the target output identities.");
    if (folded) {
      NodeArg* flat_bias = nullptr;
      ORT_RETURN_IF_ERROR(AddFlatBiasCopy(graph, *original_bias, contract, flat_bias));
      graph_utils::ReplaceNodeInput(*target, 2, *flat_bias);
    }
    ORT_RETURN_IF_ERROR(graph.Resolve());
    ORT_RETURN_IF_NOT(BiasHeader(graph, target->InputDefs()[2], contract, false) &&
                          target->OutputDefs()[0] == original_y && target->OutputDefs()[1] == original_state,
                      "CUDA runtime fusion final bias or output identities changed.");
    capability.sub_graph->nodes.assign(region.begin(), region.end());
    return Status::OK();
  };
  parent->nodes_to_optimize.push_back(std::move(optimization));
  return parent;
}

constexpr char kUnitSwishPattern[] = R"(pattern UnitSwish(x) { y=Swish(x); return y })";
constexpr char kUnitSwishReplacement[] = R"((x)=>{ y=com.microsoft.ScaledSiLU(x,alpha=1.0); return y })";

// Own metadata values. Graph-owned pointers are identity tokens, never dereferenced.
// Acceptance must use the original live graph and fetch its current source by index.
struct UnitSwishSnapshot {
  const Node* node = nullptr;
  const NodeArg* input = nullptr;
  const NodeArg* output = nullptr;
  std::string input_name;
  std::string output_name;
  std::string name;
  std::string domain;
  std::string overload;
  int version = 0;
  std::string input_type;
  std::string output_type;
  InlinedVector<std::pair<std::string, std::string>> attributes;

  bool operator==(const UnitSwishSnapshot& other) const {
    return node == other.node && input == other.input && output == other.output &&
           input_name == other.input_name && output_name == other.output_name &&
           name == other.name && domain == other.domain && overload == other.overload && version == other.version &&
           input_type == other.input_type && output_type == other.output_type && attributes == other.attributes;
  }
};

// No payload inspection, including for initializer inputs. Zero extents annihilate the
// product before multiplication; unknown extents do not invent a runtime size.
bool UnitSwishShape(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (!shape) return true;
  bool unknown = false;
  bool zero = false;
  for (const auto& dim : shape->dim()) {
    if (!dim.has_dim_value())
      unknown = true;
    else if (dim.dim_value() < 0)
      return false;
    else if (dim.dim_value() == 0)
      zero = true;
  }
  if (zero || unknown) return true;
  // Mirrors LaunchScaledSiLU in scaled_silu_impl.cu: 256 threads and an int block count.
  constexpr int64_t max_count = int64_t{256} * std::numeric_limits<int>::max();
  int64_t count = 1;
  for (const auto& dim : shape->dim()) {
    if (dim.dim_value() > max_count / count) return false;
    count *= dim.dim_value();
  }
  return true;
}

// Read live metadata only; discovery and acceptance must use the same complete check.
bool UnitSwishSemantics(const Graph& graph, const Match& match, UnitSwishSnapshot* snapshot) {
  const auto& imports = graph.DomainToVersionMap();
  if (!imports.count(kOnnxDomain) || imports.at(kOnnxDomain) != 24 ||
      !imports.count(kMSDomain) || imports.at(kMSDomain) != 1 || match.nodes.size() != 1) return false;
  const auto* node = *match.nodes.begin();
  if (node->OpType() != "Swish" || node->Domain() != kOnnxDomain || node->SinceVersion() != 24 ||
      !node->Overload().empty() || !node->GetExecutionProviderType().empty() || node->InputDefs().size() != 1 ||
      node->OutputDefs().size() != 1) return false;
  const auto* input = match.valueOf("x");
  const auto* output = match.valueOf("y");
  if (!input || !output || !input->Exists() || !output->Exists()) return false;
  const int type = Type(input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, true)
                       ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT
                       : ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  if (!Type(input, type, true) || !Type(output, type, true) ||
      !UnitSwishShape(input->Shape()) || !UnitSwishShape(output->Shape())) return false;
#if !defined(DISABLE_SPARSE_TENSORS)
  if (graph.IsSparseInitializer(input->Name())) return false;
#endif
  if (const auto* tensor = graph.GetConstantInitializer(input->Name(), false)) {
    if (tensor->data_type() != type) return false;
    ONNX_NAMESPACE::TensorShapeProto initializer_shape;
    for (auto dim : tensor->dims()) initializer_shape.add_dim()->set_dim_value(dim);
    if (!UnitSwishShape(&initializer_shape)) return false;
    for (const auto* value : {input, output}) {
      if (const auto* shape = value->Shape()) {
        if (shape->dim_size() != tensor->dims_size()) return false;
        for (int i = 0; i < shape->dim_size(); ++i) {
          if (shape->dim(i).has_dim_value() && shape->dim(i).dim_value() != tensor->dims(i)) return false;
        }
      }
    }
  }
  const auto* in_shape = input->Shape();
  const auto* out_shape = output->Shape();
  if (in_shape && out_shape) {
    if (in_shape->dim_size() != out_shape->dim_size()) return false;
    for (int i = 0; i < in_shape->dim_size(); ++i) {
      const auto& a = in_shape->dim(i);
      const auto& b = out_shape->dim(i);
      if (a.has_dim_value() && b.has_dim_value() && a.dim_value() != b.dim_value()) return false;
    }
  }
  UnitSwishSnapshot actual;
  for (const auto& [name, attr] : node->GetAttributes()) {
    if (name != "alpha" || attr.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT ||
        !std::isfinite(attr.f()) || attr.f() != 1.0f) return false;
    actual.attributes.emplace_back(name, attr.SerializeAsString());
  }
  std::sort(actual.attributes.begin(), actual.attributes.end());
  actual.node = node;
  actual.input = input;
  actual.output = output;
  actual.input_name = input->Name();
  actual.output_name = output->Name();
  actual.name = node->Name();
  actual.domain = node->Domain();
  actual.overload = node->Overload();
  actual.version = node->SinceVersion();
  actual.input_type = input->TypeAsProto()->SerializeAsString();
  actual.output_type = output->TypeAsProto()->SerializeAsString();
  if (snapshot) *snapshot = std::move(actual);
  return true;
}

Status UnitSwishTargetAvailable(const Match& match, const Replacement& recipe,
                                const IExecutionProvider::IKernelLookup& lookup,
                                const logging::Logger& logger, bool& available) {
  Model scratch("cuda_unit_swish_probe", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, logger);
  auto& graph = scratch.MainGraph();
  auto& input = graph.GetOrCreateNodeArg("x", match.valueOf("x")->TypeAsProto());
  graph.SetInputs({&input});
  return ProbeTarget(graph, recipe, {{"x", &input}}, {}, 1, lookup, available);
}

std::unique_ptr<ComputeCapability> UnitSwishOffer(NodeIndex anchor, UnitSwishSnapshot snapshot,
                                                  const Pattern& pattern, const Replacement& recipe) {
  Options options;
  options.compatible_execution_providers.clear();
  Rules rules;
  auto predicate = [snapshot](const Graph& graph, Match& candidate) {
    UnitSwishSnapshot actual;
    return UnitSwishSemantics(graph, candidate, &actual) && actual == snapshot;
  };
  rules.emplace_back(kUnitSwishRule, pattern, recipe, std::move(predicate),
                     InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}});
  auto rewriter = std::make_shared<const PatternRewriter>(std::move(rules), std::move(options));
  auto region = std::make_unique<IndexedSubGraph>();
  region->nodes.push_back(anchor);
  auto parent = std::make_unique<ComputeCapability>(std::move(region));
  auto optimization = std::make_unique<ComputeCapability>(std::make_unique<IndexedSubGraph>());
  optimization->sub_graph->nodes.push_back(anchor);
  optimization->optimization_func =
      [rewriter, anchor, snapshot](Graph& graph, const ComputeCapability& selected, ComputeCapability& capability,
                                   const GraphOptimizerRegistry&) -> Status {
    ORT_RETURN_IF_NOT(selected.sub_graph && capability.sub_graph &&
                          selected.sub_graph->nodes.size() == 1 && selected.sub_graph->nodes[0] == anchor &&
                          capability.sub_graph->nodes.size() == 1 && capability.sub_graph->nodes[0] == anchor,
                      "Unit Swish selected region changed.");
    const auto* source = graph.GetNode(anchor);
    ORT_RETURN_IF_NOT(source && source == snapshot.node && source->GetExecutionProviderType().empty(),
                      "Unit Swish source is missing or already assigned.");
    InlinedHashSet<NodeIndex> region{anchor};
    Statistics statistics;
    ORT_RETURN_IF_ERROR(rewriter->Run(graph, statistics, region));
    ORT_RETURN_IF_NOT(statistics.rewrites == 1 && region.size() == 1,
                      "Unit Swish must rewrite exactly once to one node.");
    const auto* target = graph.GetNode(*region.begin());
    ORT_RETURN_IF_NOT(target && target->Domain() == kMSDomain && target->OpType() == "ScaledSiLU" &&
                          target->InputDefs().size() == 1 && target->InputDefs()[0] == snapshot.input &&
                          target->OutputDefs().size() == 1 && target->OutputDefs()[0] == snapshot.output,
                      "Unit Swish did not preserve input/output identities.");
    ORT_RETURN_IF_ERROR(graph.Resolve());
    ORT_RETURN_IF_NOT(target->InputDefs()[0] == snapshot.input && target->OutputDefs()[0] == snapshot.output &&
                          graph.GetNodeArg(snapshot.output_name) == snapshot.output,
                      "Unit Swish final input/output identities changed.");
    capability.sub_graph->nodes.assign(region.begin(), region.end());
    return Status::OK();
  };
  parent->nodes_to_optimize.push_back(std::move(optimization));
  return parent;
}

Status SelectUnitSwish(const GraphViewer& viewer, const IExecutionProvider::IKernelLookup& lookup,
                       const logging::Logger& logger,
                       std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  const auto& graph = viewer.GetGraph();
  const Pattern pattern = PatternLibrary(kUnitSwishPattern).GetPattern("UnitSwish");
  const Replacement recipe(kUnitSwishReplacement);
  std::vector<std::unique_ptr<ComputeCapability>> pending;
  for (auto anchor : viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph.GetNode(anchor);
    if (!node || node->OpType() != "Swish") continue;
    Status probe_status;
    ORT_RETURN_IF_ERROR(pattern.VisitMatches(graph, anchor, [&](Match& match) {
      UnitSwishSnapshot snapshot;
      if (!UnitSwishSemantics(graph, match, &snapshot)) return true;
      bool available = false;
      probe_status = UnitSwishTargetAvailable(match, recipe, lookup, logger, available);
      if (!probe_status.IsOK()) return false;
      if (available) pending.push_back(UnitSwishOffer(anchor, std::move(snapshot), pattern, recipe));
      return true;
    }));
    ORT_RETURN_IF_ERROR(probe_status);
  }
  capabilities = std::move(pending);
  return Status::OK();
}
}  // namespace

Status Select(const GraphViewer& viewer, const IExecutionProvider::IKernelLookup& kernel_lookup,
              const GraphOptimizerRegistry& registry, const std::string& rule_id,
              std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  ORT_RETURN_IF_NOT(capabilities.empty(), "CUDA runtime fusion requires initially empty output.");
  ORT_RETURN_IF_NOT(rule_id == kMobiusBf16Rule || rule_id == kUnitSwishRule,
                    "Unknown CUDA runtime fusion rule: ", rule_id);
  const auto* config = rule_id == kUnitSwishRule ? kUnitSwishConfig : kCausalConvConfig;
  const auto setting = registry.GetSessionOptions().config_options.GetConfigOrDefault(config, "0");
  ORT_RETURN_IF_NOT(setting == "0" || setting == "1", config, " must be exactly 0 or 1.");
  if (!registry.SupportsCapabilityOptimizations()) return Status::OK();
  if (registry.GetSessionOptions().graph_optimization_level < TransformerLevel::Level2) return Status::OK();
  if (setting == "0" || viewer.GetFilterInfo() || viewer.IsSubgraph()) return Status::OK();
  const auto& graph = viewer.GetGraph();
  const auto& imports = graph.DomainToVersionMap();
  if (!imports.count(kOnnxDomain) || imports.at(kOnnxDomain) != 24 ||
      !imports.count(kMSDomain) || imports.at(kMSDomain) != 1) return Status::OK();
  ORT_RETURN_IF(graph.GraphResolveNeeded(), "CUDA runtime fusion requires a resolved graph.");
  if (rule_id == kUnitSwishRule) {
    return SelectUnitSwish(viewer, kernel_lookup, *registry.GetLogger(), capabilities);
  }
  const Replacement recipe(kReplacement);
  const Pattern pattern = PatternLibrary(kPattern).GetPattern("Mobius");
  std::vector<std::unique_ptr<ComputeCapability>> pending;
  InlinedHashSet<NodeIndex> offered;
  for (auto anchor : viewer.GetNodesInTopologicalOrder()) {
    const auto* node = graph.GetNode(anchor);
    if (!node || node->Domain() != kOnnxDomain || node->OpType() != "Swish" ||
        !node->GetExecutionProviderType().empty() || offered.count(anchor)) continue;
    Status probe_status;
    ORT_RETURN_IF_ERROR(pattern.VisitMatches(graph, anchor, [&](Match& match) {
      for (const auto* node : match.nodes) {
        if (!node->GetExecutionProviderType().empty() || offered.count(node->Index())) return true;
      }
      Contract contract;
      if (!Semantics(graph, match, &contract)) return true;
      bool available = false;
      probe_status = TargetAvailable(match, contract, recipe, kernel_lookup, *registry.GetLogger(), available);
      if (!probe_status.IsOK()) return false;
      if (!available) return true;
      pending.push_back(Offer(match, anchor, std::move(contract), pattern, recipe));
      for (const auto* node : match.nodes) offered.insert(node->Index());
      return true;
    }));
    ORT_RETURN_IF_ERROR(probe_status);
  }
  capabilities = std::move(pending);
  return Status::OK();
}
}  // namespace onnxruntime::cuda_runtime_fusion

#endif
