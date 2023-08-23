// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocation_planner.h"
#include <list>
#include <algorithm>
#include <deque>
#include <sstream>
#include <ctime>
#include <iomanip>
#include "core/common/exceptions.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"
#include "core/platform/env.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_steps.h"
#include "core/framework/stream_execution_context.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"

#ifdef ORT_ENABLE_STREAM
#include "nlohmann/json.hpp"
using json = nlohmann::json;
#endif

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

namespace NestedSubgraphInfoDetails {

// Used to compose a unique key to identify a nested subgraph
// relative to a current graph level (which in turn is identified using a "base")
std::string ComposeNestedSubgraphInfoKeyHelper(const std::string& base,
                                               size_t graph_depth,
                                               NodeIndex node_index,
                                               const std::string& attr_name) {
  // key = base + graph depth + current graph node index + attr name corresponding to the subgraph
  return ::onnxruntime::MakeString(base, graph_depth, node_index, attr_name);
}

}  // namespace NestedSubgraphInfoDetails

std::ostream& operator<<(std::ostream& out, AllocKind alloc_kind) {
  switch (alloc_kind) {
    case AllocKind::kAllocate:
      out << "Allocate";
      break;
    case AllocKind::kAllocateStatically:
      out << "AllocateStatically";
      break;
    case AllocKind::kPreExisting:
      out << "PreExisting";
      break;
    case AllocKind::kReuse:
      out << "Reuse";
      break;
    case AllocKind::kAllocateOutput:
      out << "AllocateOutput";
      break;
    case AllocKind::kShare:
      out << "Share";
      break;
    case AllocKind::kAllocatedExternally:
      out << "AllocatedExternally";
      break;
    case AllocKind::kNotSet:
      out << "NotSet";
      break;
  }
  return out;
}

// Output details of an execution plan:
std::ostream& operator<<(std::ostream& out, std::pair<const SequentialExecutionPlan*, const SessionState*> planinfo) {
  const SequentialExecutionPlan& plan = *planinfo.first;
  const SessionState& session_state = *planinfo.second;

  const auto& name_idx_map = session_state.GetOrtValueNameIdxMap();
  std::map<int, std::string_view> index_to_name;  // order by Node_Arg index by default

  out << "Allocation Plan:\n";
  out << "(ort_value_idx) output_name : <allocation plan>\n";
  auto plan_size = plan.allocation_plan.size();
  for (auto& name_index : name_idx_map) {
    index_to_name[name_index.second] = name_index.first;
  }
  for (auto it = index_to_name.begin(); it != index_to_name.end(); it++) {
    int index = it->first;
    out << "(" << index << ")" << it->second << " : ";
    if (0 <= index && static_cast<size_t>(index) < plan_size) {
      auto& elt_plan = plan.allocation_plan[index];
      out << elt_plan.alloc_kind;
      if (elt_plan.alloc_kind == AllocKind::kReuse) out << " " << elt_plan.reused_buffer;
      auto& loc = elt_plan.location;
      out << ", " << loc.ToString();
    } else {
      out << "Index out-of-range!";
    }
    out << std::endl;
  }

  out << "\nExecution Plan:\n";
  for (size_t i = 0; i < plan.execution_plan.size(); ++i) {
    auto& execution_plan = plan.execution_plan[i];
    out << "Start logic stream: " << i << " on device: " << std::to_string(execution_plan->device_.Type())
        << std::endl;
    for (auto& step : execution_plan->steps_) {
      out << step->ToString() << std::endl;
    }
    out << "End logic stream : " << i << std::endl;
  }

  return out;
}

static const KernelCreateInfo& GetKernelCreateInfo(
    const KernelCreateInfoMap& kernel_create_info_map,
    NodeIndex node_index) {
  auto entry = kernel_create_info_map.find(node_index);
  ORT_ENFORCE(entry != kernel_create_info_map.cend(),
              "SessionState should have saved the KernelCreateInfo prior to this running. NodeIndex:", node_index);

  return *entry->second;
}

class PlannerImpl {
 public:
  PlannerImpl(const Node* parent_node, const onnxruntime::GraphViewer& graph_viewer,
              gsl::span<const NodeArg* const> outer_scope_node_args, const ExecutionProviders& providers,
              const KernelCreateInfoMap& kernel_create_info_map,
              const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps,
              const InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_node_arg_to_location_map,
              const OrtValueNameIdxMap& ort_value_name_idx_map,
              const ISequentialPlannerContext& context, SequentialExecutionPlan& plan)
      : context_(&context),
        plan_(plan),
        parent_node_(parent_node),
        graph_viewer_(graph_viewer),
        outer_scope_node_args_(outer_scope_node_args),
        execution_providers_(providers),
        kernel_create_info_map_(kernel_create_info_map),
        subgraphs_kernel_create_info_maps_(subgraphs_kernel_create_info_maps),
        outer_scope_node_arg_to_location_map_(outer_scope_node_arg_to_location_map),
        ort_value_name_idx_map_(ort_value_name_idx_map) {}

  Status CreatePlan(
#ifdef ORT_ENABLE_STREAM
      const IStreamCommandHandleRegistry& stream_handle_registry,
#endif
      const PathString& partition_config_file,
      const logging::Logger& logger);

 private:
  gsl::not_null<const ISequentialPlannerContext*> context_;
  SequentialExecutionPlan& plan_;

  const Node* parent_node_;
  const onnxruntime::GraphViewer& graph_viewer_;
  gsl::span<const NodeArg* const> outer_scope_node_args_;
  const ExecutionProviders& execution_providers_;

  const KernelCreateInfoMap& kernel_create_info_map_;
  const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps_;

  const InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_node_arg_to_location_map_;

  const OrtValueNameIdxMap& ort_value_name_idx_map_;

  size_t num_logic_streams_{0};
  std::vector<InlinedVector<NodeIndex>> stream_nodes_;
  InlinedVector<size_t> node_stream_map_;

  // dependence_graph_ keeps the dependencies combining model graph and logic streams
  // e.g. dependence_graph_[downstream_node] = [upstream_node_0, upstream_node_1, upstream_node_2 ...]
  // upstream_node_0 and upstream_node_1 are the immmediate upstream nodes of downstream_node
  // upstream_node_2 is the immediate nodes ahead of downstream_node in the same logic stream
  InlinedHashMap<onnxruntime::NodeIndex, InlinedHashSet<onnxruntime::NodeIndex>> dependence_graph_;
  InlinedHashMap<onnxruntime::OrtValueIndex, InlinedHashSet<onnxruntime::NodeIndex>> value_consumer_map_;
  InlinedHashMap<onnxruntime::OrtValueIndex, onnxruntime::NodeIndex> value_node_map_;

  // OrtValueInfo: Auxiliary information about an OrtValue used only during plan-generation:
  struct OrtValueInfo {
    const onnxruntime::NodeArg* p_def_site;  // the (unique) NodeArg corresponding to the MLValue
    int usecount = 0;                        // static reference-count

    // This is initialized to -1 to ensure that if ProcessDef is somehow not called, planning
    // will fail more cleanly.  This is also used as a temporary workaround to detect the
    // case that the DML provider has removed initilizers from the graph during partitioning.
    // Removing initializers is a temporary measure needed to limit the number of copies of
    // tensors in GPU memory.
    OrtValueIndex reused_buffer_index = -1;  // index of original buffer to reuse
    bool is_inplace_reuse = false;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    OrtValueIndex inplace_reused_buffer_index = -1;  // index of original buffer to reuse inplace
#endif
  };

  // ort_value_info_ is indexed by an OrtValueIndex
  std::vector<OrtValueInfo> ort_value_info_;

  // FreeBufferInfo is used to track information about ml-values whose buffers are
  // free to be reused.
  struct FreeBufferInfo {
    OrtValueIndex ml_value;
    // deallocate_point is an index into the execution-plan; thus, ml_value becomes free after
    // this step in the execution-plan is completed.
    size_t deallocate_point;
    FreeBufferInfo(OrtValueIndex ort_value, size_t dealloc_point)
        : ml_value(ort_value), deallocate_point(dealloc_point) {}
  };
  // freelist_ : a list of ml-values whose buffers are free to be reused, sorted by when
  // they became free (more recently freed earlier in the list).
  std::list<FreeBufferInfo> freelist_;

  OrtValueIndex Index(const OrtValueName& name) {
    OrtValueIndex result;
    auto status = ort_value_name_idx_map_.GetIdx(name, result);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return result;
  }

  int& UseCount(OrtValueIndex n) {
    ORT_ENFORCE(n >= 0 && static_cast<size_t>(n) < ort_value_info_.size(), "invalid value index: ", n, " against size ", ort_value_info_.size());
    return ort_value_info_[n].usecount;
  }
  int& UseCount(const OrtValueName& name) { return UseCount(Index(name)); }

  int DecrementUseCount(OrtValueIndex n) {
    int& use_count = --UseCount(n);
    assert(use_count >= 0);
    return use_count;
  }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  OrtValueIndex& InplaceBuffer(OrtValueIndex n) {
    ORT_ENFORCE(n >= 0 && static_cast<size_t>(n) < ort_value_info_.size());
    return ort_value_info_[n].inplace_reused_buffer_index;
  }
#endif

  OrtValueIndex& Buffer(OrtValueIndex n) {
    ORT_ENFORCE(n >= 0 && static_cast<size_t>(n) < ort_value_info_.size());
    return ort_value_info_[n].reused_buffer_index;
  }

  AllocPlanPerValue& AllocPlan(OrtValueIndex n) {
    ORT_ENFORCE(n >= 0 && static_cast<size_t>(n) < plan_.allocation_plan.size());
    return plan_.allocation_plan[static_cast<size_t>(n)];
  }

  AllocPlanPerValue& AllocPlan(const OrtValueName& name) { return AllocPlan(Index(name)); }

  // Initialize state for a given ml-value at its definition site:
  void ProcessDef(OrtValueIndex id, const onnxruntime::NodeArg* p_def_site) {
    ORT_ENFORCE(id >= 0 && static_cast<size_t>(id) < ort_value_info_.size());
    OrtValueInfo& info = ort_value_info_[id];
    info.usecount = 0;
    info.reused_buffer_index = id;  // initially, no reuse; the ml-value uses its own buffer
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    info.inplace_reused_buffer_index = id;  // initially, no reuse; the ml-value uses its own buffer
#endif

    info.p_def_site = p_def_site;
  }

  // Reuse/Alias/Share between two OrtValue indexes
  void Reuse(OrtValueIndex reused, OrtValueIndex reused_for, AllocKind alloc_kind) {
    ORT_ENFORCE(reused != reused_for);
    // find original buffer underlying ml-value we want to reuse:
    OrtValueIndex original = Buffer(reused);
    // record that the new buffer will reuse that original buffer
    Buffer(reused_for) = original;
    // adjust original buffer's usecount
    UseCount(original) += UseCount(reused_for);

    // update allocation plan (for use at execution-time)
    auto& symplan = AllocPlan(reused_for);
    symplan.alloc_kind = alloc_kind;
    symplan.reused_buffer = original;
  }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  void InplaceReuse(OrtValueIndex reused, OrtValueIndex reused_for) {
    ORT_ENFORCE(reused != reused_for);
    OrtValueIndex original = InplaceBuffer(reused);
    InplaceBuffer(reused_for) = original;
    AllocPlan(reused_for).inplace_reuse = original;
  }
#endif

  // Find if there exists some input tensor that we can use in-place for output_arg_num-th input in the node.
  bool FindReusableInput(const onnxruntime::Node& node, int output_arg_num, OrtValueIndex* reusable_input,
                         bool* is_strided_tensor) {
    *is_strided_tensor = false;
#ifdef ENABLE_TRAINING
    // Inputs of Yields are essentially the outputs for FW partial subgraph
    // These tensors will be passed back to pytorch, thus cannot share the buffer with other tensors

    // Unhandled corner case:
    // If FW output tensor is consumed by BW graph, and pytorch performs an inplace operation on th returned tensor,
    // we will run into a buffer corruption problem.
    // One potential fix is returning a copy of output tensor, if it has downstream dependency
    auto p_next_node = node.OutputNodesBegin();
    if (p_next_node != node.OutputNodesEnd() && p_next_node->OpType() == "YieldOp") {
      return false;
    }
#endif  // ENABLE_TRAINING

    auto p_output_arg = node.OutputDefs()[output_arg_num];
    const KernelCreateInfo& ci = GetKernelCreateInfo(kernel_create_info_map_, node.Index());

    if (ci.kernel_def == nullptr) {
      return false;
    }

    const auto& alias_map = ci.kernel_def->Alias();
    auto input_args = node.InputDefs();
    for (auto& pair : alias_map) {
      if (pair.second == output_arg_num) {
        // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for reshape)
        if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
          auto p_input_arg = input_args[pair.first];
          if (p_input_arg->Exists()) {
            *reusable_input = Index(p_input_arg->Name());
            return true;
          }
        }
      }
    }

    const auto& variadic_alias_offsets = ci.kernel_def->VariadicAlias();
    if (variadic_alias_offsets.has_value()) {
      int input_offset = variadic_alias_offsets->first;
      int output_offset = variadic_alias_offsets->second;
      // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for AllReduce)
      int alias_input_index = output_arg_num - output_offset + input_offset;
      if (alias_input_index >= 0 && static_cast<size_t>(alias_input_index) < input_args.size()) {
        auto p_input_arg = input_args[alias_input_index];
        if (p_input_arg->Exists()) {
          *reusable_input = Index(p_input_arg->Name());
          return true;
        }
      }
    }

    const auto& inplace_map = ci.kernel_def->MayInplace();
    for (auto& pair : inplace_map) {
      if (pair.second == output_arg_num) {
        if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
          auto p_input_arg = input_args[pair.first];
          if (p_input_arg->Exists()) {
            auto input_arg_index = Index(p_input_arg->Name());
            auto original = Buffer(input_arg_index);
            if (1 == UseCount(original)) {
              if (SameSize(*p_input_arg, *p_output_arg)) {
                // we can reuse this input since it is its last use and permitted for in-place update
                *reusable_input = input_arg_index;  // or original; both should be okay
                return true;
              }
            }
          }
        }
      }
    }

#ifdef ENABLE_STRIDED_TENSORS
    // If any output of the kernel can support strided tensor, and all its consumers' inputs also support
    // strided tensors at the corresponding position, this output will generate a strided tensor
    // and share the data from the corresponding input specified in MayStridedOutputsMap.
    const auto& may_strided_outputs_map = ci.kernel_def->MayStridedOutput();
    for (auto& pair : may_strided_outputs_map) {
      if (pair.second == output_arg_num && pair.first >= 0 && static_cast<size_t>(pair.first) < input_args.size() &&
          input_args[pair.first]->Exists()) {
        bool can_strided = true;
        for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
          const KernelCreateInfo& output_node_ci = GetKernelCreateInfo(kernel_create_info_map_, it->Index());
          if (!output_node_ci.kernel_def) {
            can_strided = false;
            break;
          }
          const auto& may_strided_inputs = output_node_ci.kernel_def->MayStridedInput();
          for (size_t i = 0; i < it->InputDefs().size(); ++i) {
            if (it->InputDefs()[i] == p_output_arg && std::find(may_strided_inputs.begin(), may_strided_inputs.end(),
                                                                static_cast<int>(i)) == may_strided_inputs.end()) {
              can_strided = false;
              break;
            }
          }
          if (!can_strided) {
            break;
          }
        }
        if (can_strided) {
          *reusable_input = Index(input_args[pair.first]->Name());
          *is_strided_tensor = true;
          return true;
        }
      }
    }
#endif

    return false;
  }

  static bool SameShape(const TensorShapeProto& shape1, const TensorShapeProto& shape2) {
    // TODO: This should probably be defined to be the equality operator on TensorShapeProto.
    namespace on = ONNX_NAMESPACE;
    int rank1 = shape1.dim_size();
    if (shape2.dim_size() != rank1) return false;
    for (int i = 0; i < rank1; i++) {
      const auto& val1 = shape1.dim(i);
      const auto& val2 = shape2.dim(i);
      if (utils::HasDimValue(val1) && utils::HasDimValue(val2) &&
          (val1.dim_value() == val2.dim_value()))
        continue;  // same known dimension
      if (utils::HasDimParam(val1) && utils::HasDimParam(val2)) {
        const auto& val1_param = val1.dim_param();
        if (val1_param == val2.dim_param() && !val1_param.empty())
          continue;  // same unknown dimension
      }
      return false;
    }
    return true;
  }

  /*! \brief Given a tensor-type, return the size of an element of the tensor.
   */
  static size_t GetElementSize(const DataType& tensor_type) {
    const TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(tensor_type);
    MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
    const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
    ORT_ENFORCE(nullptr != tensor_type_base);
    MLDataType elt_type = tensor_type_base->GetElementType();
    return elt_type->Size();
  }

  static bool SameSize(const TensorShapeProto& shape1, const onnxruntime::NodeArg& arg1,
                       const TensorShapeProto& shape2, const onnxruntime::NodeArg& arg2) {
    const auto& ptype1 = arg1.Type();
    const auto& ptype2 = arg2.Type();
    auto type1_size = GetElementSize(ptype1);
    auto type2_size = GetElementSize(ptype2);
    bool is_type1_string = arg1.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING;
    bool is_type2_string = arg2.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING;

    // sizeof(std::string) = sizeof(double) on gcc 4.8.x on CentOS. This causes the allocation planner to reuse
    // a tensor of type double. This won't work for string tensors since they need to be placement new'ed.
    // If either of the tensors is a string, don't treat them the same. Moreover, reusing a string tensor for a string
    // tensor without releasing the previous memory can cause memory leaks; hence we don't allow reuse across string
    // tensors as well.
    return !(is_type1_string || is_type2_string) && (type1_size == type2_size) && SameShape(shape1, shape2);

    /* TODO: we can improve this if the concrete shapes are known for both as below.
       Unclear whether this is worthwhile though.
    if (KnownSize(p_shape1) && KnownSize(p_shape2)) {
      // Comparison of statically-known size
      auto size1 = NumElements(p_shape1) * EltSize(ptype1);
      auto size2 = NumElements(p_shape2) * EltSize(ptype2);
      return size1 == size2;
    } else {
      // Comparison of statically-unknown size buffers
      return SameElementSize(ptype1, ptype2) && SameShape(shape1, shape2);
    }
    */
  }

  bool SameSize(const onnxruntime::NodeArg& arg1, const onnxruntime::NodeArg& arg2) {
    if ((!arg1.Exists()) || (!arg2.Exists())) return false;
    auto p_shape1 = context_->GetShape(arg1);
    auto p_shape2 = context_->GetShape(arg2);
    // If the shapes are unknown, we conservatively assume they may be of different size.
    if ((nullptr == p_shape1) || (nullptr == p_shape2)) return false;
    return SameSize(*p_shape1, arg1, *p_shape2, arg2);
  }

  // Find if freelist contains a buffer of the same size as output_arg
  bool FindReusableTensor(const onnxruntime::NodeArg& output_arg, OrtValueIndex* reusable_tensor) {
    if (!context_->GetEnableMemoryReuse()) {
      return false;
    }
    auto p_required_buffer_shape = context_->GetShape(output_arg);
    if (nullptr == p_required_buffer_shape || p_required_buffer_shape->dim_size() == 0) return false;
    auto& required_memory_info = AllocPlan(output_arg.Name()).location;

    for (auto it = freelist_.begin(); it != freelist_.end(); ++it) {
      size_t reusable = static_cast<size_t>(it->ml_value);
      const onnxruntime::NodeArg* p_node_arg = ort_value_info_.at(reusable).p_def_site;
      if (!p_node_arg) {
        // TODO this should be an error case, needs more investigation
        continue;
      }

#if !defined(DISABLE_OPTIONAL_TYPE)
      // Make sure optional types are not up for re-use as we aren't quite
      // sure if the re-used tensor will be a None or otherwise. This cannot
      // be determined statically.
      if (IsOptionalType(*p_node_arg)) {
        continue;
      }
#endif

      auto& available_memory_info = AllocPlan(p_node_arg->Name()).location;
      if (!(available_memory_info == required_memory_info)) continue;
      auto p_available_buffer_shape = context_->GetShape(*p_node_arg);
      if (nullptr != p_available_buffer_shape) {
        if (SameSize(*p_available_buffer_shape, *p_node_arg,
                     *p_required_buffer_shape, output_arg)) {
          *reusable_tensor = it->ml_value;
          freelist_.erase(it);
          return true;
        }
      }
    }
    return false;
  }

  void Initialize(size_t num_ml_values) {
    // All ml-value indices must be in range 0 .. num_ml_values-1
    ort_value_info_.resize(num_ml_values);

    // Initialize execution plan:
    plan_.execution_plan.reserve(num_logic_streams_);

    // Initialize allocation plan:
    plan_.allocation_plan.resize(num_ml_values);
  }

  bool HasExternalOutputs(const Node& node) const {
    const KernelCreateInfo& ci = GetKernelCreateInfo(kernel_create_info_map_, node.Index());
    if (ci.kernel_def == nullptr) {
      return false;
    }

    return ci.kernel_def->HasExternalOutputs();
  }

  Status ComputePlanForInputsAndWeights() {
    auto setup_preexisting = [this](const NodeArg* node_arg) {
      auto input_index = Index(node_arg->Name());
      AllocPlanPerValue& thisplan = AllocPlan(input_index);
      thisplan.alloc_kind = AllocKind::kPreExisting;
      thisplan.value_type = utils::GetMLDataType(*node_arg);
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
      size_t max_pc = plan_.execution_plan.size();
      thisplan.life_interval = std::pair<size_t, size_t>(0, max_pc);
#endif
    };

    // inputs of the graph:
    // An input ml-value's data is owned by the caller (of InferenceSession::Run())
    // It must be allocated by the caller, and will not be reused during inference.
    for (auto graph_input : graph_viewer_.GetInputs()) {
      setup_preexisting(graph_input);
    }

    // outer scope node args are treated the same as graph inputs
    for (auto outer_scope_node_arg : outer_scope_node_args_) {
      setup_preexisting(outer_scope_node_arg);
    }

    // set AllocationInfo for each weight
    return GeneratePlanForWeights();
  }

  Status ComputeReuseCount() {
    for (auto graph_input : graph_viewer_.GetInputs()) {
      OrtValueIndex index = Index(graph_input->Name());
      UseCount(index)++;  // Models caller's usage post-inference; ensures it will not be reused.
    }

    for (auto node_arg : outer_scope_node_args_) {
      OrtValueIndex index = Index(node_arg->Name());
      UseCount(index)++;  // ensure will not be re-used as this graph does not own the buffer
    }

    // All initializers should be treated as input
    for (const auto& pair : graph_viewer_.GetAllInitializedTensors()) {
      const auto& initializer_name = pair.first;
      UseCount(initializer_name)++;
    }

    for (auto& stream_execution_order : stream_nodes_) {
      for (NodeIndex node_index : stream_execution_order) {
        auto pnode = graph_viewer_.GetNode(node_index);
        if (pnode == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the node ", node_index);
        }

        auto process_input = [this](const NodeArg& input, size_t /*arg_idx*/) {
          const auto& name = input.Name();
          UseCount(name)++;
          return Status::OK();
        };

        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(pnode->InputDefs(), process_input));

        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(pnode->ImplicitInputDefs(), process_input));

        auto outputs = pnode->OutputDefs();
        auto num_outputs = outputs.size();
        bool has_external_outputs = HasExternalOutputs(*pnode);
        for (size_t i = 0; i < num_outputs; ++i) {
          auto* node_output = outputs[i];
          if (!node_output->Exists()) continue;
          OrtValueIndex index = Index(node_output->Name());
          // Ensures external outputs will not be reused.
          UseCount(index) += (has_external_outputs ? 2 : 1);
        }
      }
    }

    for (auto graph_output : graph_viewer_.GetOutputs()) {
      UseCount(graph_output->Name())++;  // Models caller's usage post-inference; ensures it will not be reused.
    }
    return Status::OK();
  }

  void ClearUseCount() {
    for (auto& value_info : ort_value_info_) {
      value_info.usecount = 0;
    }
  }

  Status ComputeValueLocation() {
    // Note: for every ml-value, its definition must appear before all its uses in a topological sort of a valid model
    using GraphInputsSet = InlinedHashSet<std::string_view>;
    const auto& graph_inputs_nodes = graph_viewer_.GetInputsIncludingInitializers();
    GraphInputsSet graph_inputs;
    graph_inputs.reserve(graph_inputs_nodes.size());
    for (auto& graph_input : graph_inputs_nodes) {
      graph_inputs.insert(graph_input->Name());
    }

    for (auto graph_input : graph_viewer_.GetInputs()) {
      OrtValueIndex index = Index(graph_input->Name());
      ProcessDef(index, graph_input);
    }

    for (auto node_arg : outer_scope_node_args_) {
      OrtValueIndex index = Index(node_arg->Name());
      ProcessDef(index, node_arg);
    }

    // All initializers should be treated as input
    for (const auto& pair : graph_viewer_.GetAllInitializedTensors()) {
      const auto& initializer_name = pair.first;
      OrtValueIndex index = Index(initializer_name);
      ProcessDef(index, graph_viewer_.GetNodeArg(pair.first));
    }

    InlinedHashSet<OrtValueIndex> set_node_arg_has_explicit_consumer;

    InlinedHashMap<OrtValueIndex, const IExecutionProvider*> map_implicitly_consumed_node_arg_to_ep;
    InlinedHashSet<OrtValueIndex> set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers;

    for (auto& stream_execution_order : stream_nodes_) {
      for (NodeIndex node_index : stream_execution_order) {
        auto pnode = graph_viewer_.GetNode(node_index);
        if (pnode == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the node ", node_index);
        }

        // Identify where each output of this node should be allocated.
        // This is determined by the OpKernel bound to the node.
        const KernelCreateInfo& kernel_create_info = GetKernelCreateInfo(kernel_create_info_map_, pnode->Index());

        const auto* p_kernel_def = kernel_create_info.kernel_def.get();

        ORT_ENFORCE(p_kernel_def, "Should not have entry in kernel create info with nullptr for kernel_def");

        auto exec_provider = execution_providers_.Get(*pnode);
        if (exec_provider == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the execution provider ",
                                 pnode->GetExecutionProviderType());
        }

        bool is_implicit_input = false;

        // Add location information if applicable for the provided input def
        auto process_input = [&graph_inputs, &exec_provider, &p_kernel_def, &is_implicit_input,
                              &set_node_arg_has_explicit_consumer,
                              &map_implicitly_consumed_node_arg_to_ep,
                              &set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers,
                              this](const NodeArg& input, size_t arg_idx) {
          const auto& name = input.Name();

          bool is_graph_input = (graph_inputs.find(name) != graph_inputs.cend());
          bool is_outer_scope_arg = std::find_if(outer_scope_node_args_.begin(), outer_scope_node_args_.end(),
                                                 [&name](const NodeArg* value) {
                                                   return value && value->Name() == name;
                                                 }) != outer_scope_node_args_.end();
          bool is_subgraph = (parent_node_ != nullptr);

          // If it's a graph input or outer scope node arg, set its plan.
          // NOTE: Copy nodes should have already been added if a graph input is fed as input
          // to nodes assigned to different providers.

          if (is_graph_input || is_outer_scope_arg) {
            OrtValueIndex index = Index(name);

            if (!is_implicit_input) {
              OrtMemType mem_type = p_kernel_def->InputMemoryType(arg_idx);
              plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetOrtDeviceByMemType(mem_type));
              set_node_arg_has_explicit_consumer.insert(index);
            } else {  // implicit input
              // Only process an implicit input if there are explicit consumers at this graph level
              // If there is an explicit consumer, the location MUST be where it is consumed
              // and not where it is located in the outer scope.
              // It is okay if we process a node consuming this arg as an implicit input
              // ahead of a node that is an explicit consumer, because we will just reset
              // this location in the 'if' branch above.

              // CASE 1: We see an implicit input without explicit consumers in a subgraph (pass-through subgraph inputs),
              // then set its location to be its corresponding location in the outer scope.
              // This is so that the subgraph copying mechanism doesn't trigger an unnecessary copy and any copying
              // decisions are deferred till there is an explicit consumer of the subgraph input in nested subgraphs.
              if (is_subgraph && set_node_arg_has_explicit_consumer.count(index) == 0) {
                auto iter = outer_scope_node_arg_to_location_map_.find(name);
                bool found_in_outer_scope_location_map = (iter != outer_scope_node_arg_to_location_map_.end());

                if (!is_graph_input) {
                  // Failing this enforce for an implicit subgraph input points to an internal error somewhere.
                  // For certain older opsets (Scan-8), we may not have added explicit subgraph inputs
                  // to the outer scope location map. See explanation in IsNodeWhereNodeInputsAreSameAsExplicitSubgraphInputs()
                  // called in FinalizeSessionStateImpl() in SessionState.
                  ORT_ENFORCE(found_in_outer_scope_location_map,
                              "There is no location for this node arg in the outer scope location map");
                }

                if (found_in_outer_scope_location_map) {
                  plan_.SetLocation(static_cast<size_t>(index), iter->second);
                }
              } else if (set_node_arg_has_explicit_consumer.count(index) == 0) {
                // CASE 2: We see an implicit input without explicit consumers in the main graph,
                // then set its location to be the device corresponding to the EP that the subgraph
                // holding node has been partitioned to.

                // The "ideal" solution is to set the location of its first "explicit" usage which may occur
                // in any nested subgraph of the node, but that is potentially too costly to
                // get at this stage (TODO: Investigate feasibility of this, see TODO in FinalizeSessionStateImpl() around this)

                // Instead, we take a "less than ideal" route which is to set the location to be the device
                // corresponding to the EP that the node is partitioned to. The hypothesis is that it is "most likely"
                // that the implicit input will eventually be consumed on that device in a nested subgraph.

                // The previous behavior was to default to CPU which will cause unnecessary copies when
                // (1) The user invokes Run() with an OrtValue backed by non-CPU memory (eg CUDA) and
                // the node in the subgraph that consumes the subgraph's implicit input is on a non-CPU device
                // in the subgraph
                // (2) The user tries to IOBind implicitly consumed graph inputs (GH Issue 11254) and
                // the node in the subgraph that consumes the subgraph's implicit input is on
                // a non-CPU device in the subgraph

                // Even if the user provides an input on CPU and the node in the subgraph that consumes the subgraph's
                // implicit input is on a non-CPU device, instead of the subgraph copying mechanism taking it to the device,
                // all we will do is "front-load" this copy in utils::CopyInputsAcrossDevices() with this approach.

                // NOTE 1: The only case this will be sub-optimal is when a node containing a subgraph is partitioned to a
                // non-CPU EP and the user provides an input (or tries to IOBind the input) AND it will eventually be
                // explicitly consumed on CPU - this scenario should be very rare and we forgo performance in this case
                // (the subgraph copying mechanism will make the copy to CPU eventually) in favor of optimizing for the
                // common case (which is that we expect the implicit input to be consumed on the non-CPU device corresponding
                // to the non-CPU EP).

                // NOTE 2: If the implicit input is consumed by multiple nodes (as implicit inputs in all of them) and
                // all of them are partitioned to the same EP, then we go ahead with the above stated logic.
                // If there are multiple EPs involved, we default the location to just CPU as there is ambiguity involved
                // as to which non-CPU device is "most optimal" for the implicit input.

                if (set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers.count(index) == 0) {
                  auto already_seen_ep_for_node_arg = map_implicitly_consumed_node_arg_to_ep.find(index);

                  if (already_seen_ep_for_node_arg == map_implicitly_consumed_node_arg_to_ep.end()) {
                    // First time we are encountering this implicitly consumed input at this graph level (or)
                    plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault));
                    map_implicitly_consumed_node_arg_to_ep.insert({index, exec_provider});
                  } else if (already_seen_ep_for_node_arg->second == exec_provider) {
                    // The EP that we previously seen for this implicit input is the same one as the current EP
                    // we have seen
                    plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault));
                  } else {
                    // Default the location to CPU
                    plan_.SetLocation(static_cast<size_t>(index),
                                      execution_providers_.Get(CPU)->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault));
                    set_implicitly_consumed_node_arg_has_heterogenous_ep_consumers.insert(index);
                  }
                }
              }
            }
          }

          return Status::OK();
        };

        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(pnode->InputDefs(), process_input));

        is_implicit_input = true;
        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(pnode->ImplicitInputDefs(), process_input));

        auto outputs = pnode->OutputDefs();
        auto num_outputs = outputs.size();
        for (size_t i = 0; i < num_outputs; ++i) {
          auto* node_output = outputs[i];
          if (!node_output->Exists()) continue;
          OrtValueIndex index = Index(node_output->Name());
          ProcessDef(index, node_output);
          plan_.SetLocation(static_cast<size_t>(index), exec_provider->GetOrtDeviceByMemType(p_kernel_def->OutputMemoryType(i)));
        }
      }
    }

    return Status::OK();
  }

  OrtDevice GetLocationForNodeInput(size_t input_index, const Node& node, const KernelCreateInfoMap& kernel_create_info_map) {
    auto* p_provider = execution_providers_.Get(node);
    ORT_ENFORCE(p_provider);

    const KernelCreateInfo& kernel_create_info = GetKernelCreateInfo(kernel_create_info_map, node.Index());

    // weights are not output from any node, so it's OK to put its location on CPU provider
    return p_provider->GetOrtDeviceByMemType(utils::IsInputOnCpu(node, &kernel_create_info, input_index) ? OrtMemTypeCPUInput : OrtMemTypeDefault);
  }

  void GeneratePlanForWeightsHelper(const GraphViewer& graph_viewer,
                                    const InitializedTensorSet& weights,
                                    const KernelCreateInfoMap& kernel_create_info_map,
                                    const std::string& subgraph_kernel_create_info_map_key_base,
                                    size_t graph_depth,
                                    /*out*/ std::vector<std::vector<OrtDevice>>& locations) {
    // Iterate over nodes in current level firstly to record location of usages
    // in current graph
    for (const auto& node : graph_viewer.Nodes()) {
      const auto& input_node_args = node.InputDefs();
      size_t num_node_inputs = input_node_args.size();

      for (size_t node_input_index = 0; node_input_index < num_node_inputs; ++node_input_index) {
        auto input_node_arg = input_node_args[node_input_index];

        // Skip processing missing optional inputs
        if (!input_node_arg->Exists()) {
          continue;
        }

        auto& def_name = input_node_arg->Name();

        // This node input doesn't correspond to any of the weights
        if (!weights.count(def_name)) {
          continue;
        }

        // While processing subgraphs, if we don't see an entry in the implicit
        // inputs of the node containing the subgraph, it is a shadow value.
        auto is_shadow_value_in_subgraph = [](const Node& subgraph_parent_node,
                                              const std::string& def_name) -> bool {
          bool is_shadow_value_in_subgraph = true;
          for (const auto& implicit_input : subgraph_parent_node.ImplicitInputDefs()) {
            if (implicit_input->Name() == def_name) {
              is_shadow_value_in_subgraph = false;
              break;
            }
          }

          return is_shadow_value_in_subgraph;
        };

        // Skip processing shadow values in subgraphs
        if (graph_depth > 0) {
          // We are processing a subgraph if we enter this
          const auto* parent_node = graph_viewer.ParentNode();

          // Skip processing if it is a shadow value
          if (is_shadow_value_in_subgraph(*parent_node, def_name)) {
            continue;
          }
        }

        auto wt_index = Index(def_name);
        // TODO: Identify error cases where-in an initializer is used on different
        // devices within the same graph level.
        // If we ever encounter that, it means that there is a severe bug in Memcpy
        // transformer and the model will crash while running. The Memcpy transformer
        // is supposed to duplicate initializers being used on different devices within
        // the same graph level and hence we should never see an initializer being used
        // on different devices here.
        // The same initializer being used on different devices across graph levels
        // (subgraphs) is okay and utils::CopyInputsAcrossDevices() will take it to
        // the right device before subgraph execution.
        locations[wt_index].emplace_back(
            GetLocationForNodeInput(node_input_index, node, kernel_create_info_map));
      }
    }

    // Iterate over nodes in current graph with subgraphs and recurse.
    for (const auto& node : graph_viewer.Nodes()) {
      // If the node has subgraphs (i.e.) control flow nodes,
      // walk the nodes in those subgraphs as well to best determine
      // the location for the OrtValue corresponding to the weights
      // (i.e.) do a recursion
      if (node.ContainsSubgraph()) {
        // A node may contain multiple subgraphs - so iterate through all of them
        for (auto& name_to_subgraph : node.GetAttributeNameToSubgraphMap()) {
          GraphViewer subgraph_viewer(*name_to_subgraph.second);

          const auto& local_subgraph_kernel_create_info_map_key =
              NestedSubgraphInfoDetails::ComposeNestedSubgraphInfoKeyHelper(subgraph_kernel_create_info_map_key_base,
                                                                            graph_depth, node.Index(), name_to_subgraph.first);

          auto specific_subgraph_kernel_create_info_map = subgraphs_kernel_create_info_maps_.find(local_subgraph_kernel_create_info_map_key);
          ORT_ENFORCE(specific_subgraph_kernel_create_info_map != subgraphs_kernel_create_info_maps_.end());

          GeneratePlanForWeightsHelper(subgraph_viewer,
                                       weights,
                                       specific_subgraph_kernel_create_info_map->second,
                                       local_subgraph_kernel_create_info_map_key,
                                       graph_depth + 1,
                                       locations);
        }
      }
    }
  }

  Status GeneratePlanForWeights() {
    // TODO: Move away from usage of vector of `OrtMemoryInfo`s per weight (initializer)
    // We do not need to maintain a vector of locations that a weight is used in.
    // We only need to know the location of its first usage according to the nodes
    // iteration rule in GeneratePlanForWeightsHelper() because:
    // (1) If the initializer is used in the graph level it is introduced in, then it can
    // only be used on one device as the Memcpy transformer will duplicate the initializer
    // (with a different name) in case it is used on multiple devices.
    // If the initializer is also additionally used in one of the subgraphs, we rely
    // on the utils::CopyInputsAcrossDevices() to copy it over to the appropriate device
    // before the subgraphs are executed.
    // (2) If the initializer is NOT used in the level it is introduced in and only used
    // in subgraphs, even then knowing its first usage location is enough as it can't be
    // used on different devices within the same graph level (see (1) for reason), and for
    // nested subgraphs, we can rely on the utils::CopyInputsAcrossDevices() to copy it
    // over to the appropriate device before the subgraphs are executed.
    std::vector<std::vector<OrtDevice>> locations(plan_.allocation_plan.size());

    GeneratePlanForWeightsHelper(graph_viewer_, graph_viewer_.GetAllInitializedTensors(),
                                 kernel_create_info_map_, "", 0, locations);

    for (size_t i = 0; i != locations.size(); ++i) {
      const std::vector<OrtDevice>& loc = locations[i];
      if (loc.empty()) continue;
      plan_.allocation_plan[i].alloc_kind = AllocKind::kAllocateStatically;
      // The planned location for an initializer is the location of its first usage.
      plan_.allocation_plan[i].location = loc[0];
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
      size_t max_pc = plan_.execution_plan.size();
      std::string node_arg_name;
      ORT_RETURN_IF_ERROR(ort_value_name_idx_map_.GetName(static_cast<int>(i), node_arg_name));
      auto node_arg = graph_viewer_.GetNodeArg(node_arg_name);
      plan_.allocation_plan[i].value_type = utils::GetMLDataType(*node_arg);
      plan_.allocation_plan[i].life_interval = std::pair<size_t, size_t>(0, max_pc);
#endif
    }
    return Status::OK();
  }

  bool IsSingleStream() {
    // if each device only have 1 logic stream
    // we can safely reuse the existing memory sharing algorithm
    InlinedHashSet<OrtDevice::DeviceType> stream_device_set;
    stream_device_set.reserve(num_logic_streams_);
    for (size_t i = 0; i < num_logic_streams_; ++i) {
      auto& stream = stream_nodes_[i];
      if (!stream.empty()) {
        auto device_type = plan_.execution_plan[i]->device_.Type();
        if (!stream_device_set.insert(device_type).second) {
          return false;
        }
      }
    }
    return true;
  }

#ifdef ORT_ENABLE_STREAM
  // assume we already have a baseline reuse plan (no memory reuse at all)
  // this funciton will optimize the plan by building a reuse plan with stream safety.
  Status OptimizeReusePlanForMultiStream() {
    InlinedHashMap<NodeIndex, int> dependent_counter;
    for (const auto& it : dependence_graph_) {
      for (NodeIndex node_index : it.second) {
        dependent_counter[node_index]++;
      }
    }
    std::deque<NodeIndex> que;
    for (const auto& it : dependence_graph_) {
      if (dependent_counter[it.first] == 0) {
        que.push_back(it.first);
      }
    }

    // fetch_all_dependents will collect all dependent nodes for "node_index"
    std::function<std::set<NodeIndex>(NodeIndex)> fetch_all_dependents = [&](NodeIndex node_index) {
      std::set<NodeIndex> dependents;

      std::function<void(NodeIndex)> dfs = [&](NodeIndex curr) {
        if (dependents.find(curr) == dependents.end()) {
          dependents.insert(curr);
          for (NodeIndex dep : dependence_graph_[curr]) {
            dfs(dep);
          }
        }
      };

      dfs(node_index);
      return dependents;
    };

    // waiting_list keeps all values who want to reuse some upstream values' memory
    std::map<OrtDevice, std::map<size_t, typename std::map<const onnxruntime::NodeArg* const, std::set<NodeIndex>*>>> waiting_list;

    // for each node, dependents_map keeps all its dependent upstream nodes that are sure to be completed ahead
    std::map<NodeIndex, std::set<NodeIndex>> dependents_map;

    std::map<OrtValueIndex, std::set<OrtValueIndex>> input_output_map;

    std::set<OrtValueIndex> reused;

    const auto& graph_viewer = graph_viewer_;
    const auto& value_map = ort_value_name_idx_map_;
    const auto& kernel_create_info_map = kernel_create_info_map_;
    auto& allocation_plan = plan_.allocation_plan;

    // build the consumer list for each value
    int num_ml_values = ort_value_name_idx_map_.MaxIdx() + 1;
    value_consumer_map_.reserve(num_ml_values);

    // iterate each stream from back, so the first element is the last consumer in single stream case
    for (auto& stream : stream_nodes_) {
      for (auto it = stream.rbegin(), end = stream.rend(); it != end; ++it) {
        NodeIndex node_index = *it;
        auto* node = graph_viewer_.GetNode(node_index);

        auto process_input = [&](const NodeArg& input, size_t /*arg_idx*/) {
          if (input.Exists()) {
            const auto& name = input.Name();
            int value_idx;
            ORT_RETURN_IF_ERROR(ort_value_name_idx_map_.GetIdx(name, value_idx));
            auto origin = Buffer(value_idx);
            if (origin != -1 && plan_.allocation_plan[origin].alloc_kind == AllocKind::kAllocate) {
              // add current node as consumer for origin buffer
              value_consumer_map_[origin].insert(node_index);
            }
          }
          return Status::OK();
        };

        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(node->InputDefs(), process_input));
        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(node->ImplicitInputDefs(), process_input));
      }
    }

    std::function<void(NodeIndex)> TryReuseInput = [&](NodeIndex node_index) {
      auto* node = graph_viewer.GetNode(node_index);

      for (size_t output_arg_num = 0; output_arg_num < node->OutputDefs().size(); output_arg_num++) {
        auto p_output_arg = node->OutputDefs()[output_arg_num];
        OrtValueIndex output_idx_global{};

        if (!value_map.GetIdx(p_output_arg->Name(), output_idx_global).IsOK() ||
            allocation_plan[output_idx_global].alloc_kind != AllocKind::kAllocate) {
          continue;
        }

        auto kci_it = kernel_create_info_map.find(node_index);
        if (kci_it == kernel_create_info_map.end()) {
          continue;
        }

        const KernelCreateInfo& ci = *kci_it->second;
        if (ci.kernel_def == nullptr) {
          continue;
        }

        bool found_reusable = false;
        const auto& alias_map = ci.kernel_def->Alias();
        auto input_args = node->InputDefs();
        for (auto* input_arg : input_args) {
          OrtValueIndex input_idx_global{};
          if (value_map.GetIdx(input_arg->Name(), input_idx_global).IsOK()) {
            input_output_map[input_idx_global].insert(output_idx_global);
          }
        }

        for (auto& pair : alias_map) {
          size_t alias_map_second = (size_t)pair.second;
          if (alias_map_second == output_arg_num) {
            // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for reshape)
            if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
              auto p_input_arg = input_args[pair.first];
              if (p_input_arg->Exists()) {
                OrtValueIndex reusable_input{};
                if (value_map.GetIdx(p_input_arg->Name(), reusable_input).IsOK() /*&&
                    allocation_plan[reusable_input].alloc_kind == AllocKind::kAllocate*/
                ) {
                  std::cout << p_input_arg->Name() << " reused by " << p_output_arg->Name() << " as input" << std::endl;
                  allocation_plan[output_idx_global].alloc_kind = AllocKind::kReuse;
                  allocation_plan[output_idx_global].reused_buffer = reusable_input;
                  value_consumer_map_[reusable_input].insert(value_consumer_map_[output_idx_global].begin(),
                                                             value_consumer_map_[output_idx_global].end());
                  reused.insert(reusable_input);
                  found_reusable = true;
                  break;
                }
              }
            }
          }
        }

        if (found_reusable) {
          continue;
        }

        const auto& variadic_alias_offsets = ci.kernel_def->VariadicAlias();
        if (variadic_alias_offsets.has_value()) {
          int input_offset = variadic_alias_offsets->first;
          int output_offset = variadic_alias_offsets->second;
          size_t alias_input_index = output_arg_num - output_offset + input_offset;

          if (alias_input_index < input_args.size()) {
            auto p_input_arg = input_args[alias_input_index];

            if (p_input_arg->Exists()) {
              OrtValueIndex reusable_input{};
              if (value_map.GetIdx(p_input_arg->Name(), reusable_input).IsOK() &&
                  allocation_plan[reusable_input].alloc_kind == AllocKind::kAllocate) {
                allocation_plan[output_idx_global].alloc_kind = AllocKind::kReuse;
                allocation_plan[output_idx_global].reused_buffer = reusable_input;
                value_consumer_map_[reusable_input].insert(value_consumer_map_[output_idx_global].begin(),
                                                           value_consumer_map_[output_idx_global].end());
                reused.insert(reusable_input);
                continue;
              }  // if
            }    // if
          }
        }

        const auto& inplace_map = ci.kernel_def->MayInplace();
        for (auto& pair : inplace_map) {
          size_t inplace_map_second = (size_t)pair.second;
          if (inplace_map_second == output_arg_num) {
            if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
              auto p_input_arg = input_args[pair.first];
              if (p_input_arg->Exists()) {
                OrtValueIndex input_arg_index{};
                if (value_map.GetIdx(p_input_arg->Name(), input_arg_index).IsOK() &&
                    allocation_plan[input_arg_index].alloc_kind == AllocKind::kAllocate) {
                  if (value_consumer_map_[input_arg_index].size() == 1 && SameSize(*p_input_arg, *p_output_arg)) {
                    allocation_plan[output_idx_global].alloc_kind = AllocKind::kReuse;
                    allocation_plan[output_idx_global].reused_buffer = input_arg_index;
                    value_consumer_map_[input_arg_index].insert(value_consumer_map_[output_idx_global].begin(),
                                                                value_consumer_map_[output_idx_global].end());
                    reused.insert(input_arg_index);
                  }
                }
              }
            }
          }
        }
      }
    };  // TryReuseInput

    // go over the outputs of "node_index" and try to reuse its memory
    std::function<void(NodeIndex)> TryReuseOutput = [&](NodeIndex node_index) {
      dependents_map[node_index] = fetch_all_dependents(node_index);
      auto* node = graph_viewer.GetNode(node_index);
      const auto& output_defs = node->OutputDefs();

      for (size_t output_idx_local = 0; output_idx_local < output_defs.size(); ++output_idx_local) {
        const auto& node_output = output_defs[output_idx_local];
        if (!node_output->Exists()) continue;
        OrtValueIndex output_idx_global{};

        if (value_map.GetIdx(node_output->Name(), output_idx_global).IsOK()) {
          if (reused.find(output_idx_global) != reused.end() ||
              allocation_plan[output_idx_global].alloc_kind != AllocKind::kAllocate) {
            continue;  // skip when it is already reused
          }

          const auto* shape = context_->GetShape(*node_output);
          if (!shape) continue;
          size_t size_in_bytes = shape->ByteSizeLong();

          const auto& location = allocation_plan[output_idx_global].location;
          auto local_iter = waiting_list.find(location);
          if (local_iter == waiting_list.end()) {
            waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
            continue;
          }

          auto size_iter = local_iter->second.find(size_in_bytes);
          if (size_iter == local_iter->second.end()) {
            waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
            continue;
          }

          bool get_reused = false;
          for (auto node_iter = size_iter->second.begin(); node_iter != size_iter->second.end();) {
            const onnxruntime::NodeArg* const downstream_arg = node_iter->first;
            OrtValueIndex downstream_value{};

            if (!value_map.GetIdx(downstream_arg->Name(), downstream_value).IsOK()) {
              node_iter = next(node_iter);
              continue;
            }

            // skip if it is a pair of input and output
            if (input_output_map[output_idx_global].find(downstream_value) != input_output_map[output_idx_global].end()) {
              node_iter = next(node_iter);
              continue;
            }

            const auto* downstream_shape = context_->GetShape(*downstream_arg);
            if (!SameSize(*downstream_shape, *downstream_arg, *shape, *node_output)) {
              node_iter = next(node_iter);
              continue;
            }

            auto* deps = node_iter->second;

            if (deps->find(node_index) == deps->end()) {
              node_iter = next(node_iter);
              continue;
            }

            bool all_covered = true;
            for (auto consumer : value_consumer_map_[output_idx_global]) {
              if (deps->find(consumer) == deps->end()) {
                all_covered = false;
                break;
              }
            }
            if (all_covered) {
              allocation_plan[downstream_value].alloc_kind = AllocKind::kReuse;
              allocation_plan[downstream_value].reused_buffer = output_idx_global;
              get_reused = true;
              // add new consumer for the value to be reused
              value_consumer_map_[output_idx_global].insert(value_node_map_[downstream_value]);
              value_consumer_map_[output_idx_global].insert(value_consumer_map_[downstream_value].begin(),
                                                            value_consumer_map_[downstream_value].end());
              node_iter = size_iter->second.erase(node_iter);
              if (size_iter->second.empty()) {
                local_iter->second.erase(size_iter);
              }
              break;  // only resued once
            } else {
              // dependents not fully covered, cannot reuse, try next one in waiting_list
              node_iter = next(node_iter);
            }
          }  // for
          if (get_reused) {
            reused.insert(output_idx_global);
          } else {
            // if not getting reused, add to waiting
            waiting_list[location][size_in_bytes][node_output] = &dependents_map[node_index];
          }
        }
      }
    };  // TryReuseOutput

    // topological traverse of the dependency graph
    InlinedHashSet<NodeIndex> visited;
    while (!que.empty()) {
      NodeIndex node_index = que.front();
      visited.insert(node_index);
      TryReuseInput(node_index);   // try reuse node's inputs as its outputs
      TryReuseOutput(node_index);  // try reuse node's outputs for downstream nodes
      que.pop_front();
      for (NodeIndex next_node_index : dependence_graph_[node_index]) {
        if (--dependent_counter[next_node_index] == 0) {
          que.push_back(next_node_index);
        }
      }
    }

    for (size_t value_index = 0; value_index < allocation_plan.size(); ++value_index) {
      if (allocation_plan[value_index].alloc_kind == AllocKind::kReuse) {
        while (allocation_plan[allocation_plan[value_index].reused_buffer].alloc_kind == AllocKind::kReuse &&
               allocation_plan[value_index].reused_buffer != allocation_plan[allocation_plan[value_index].reused_buffer].reused_buffer) {
          allocation_plan[value_index].reused_buffer = allocation_plan[allocation_plan[value_index].reused_buffer].reused_buffer;
        }
        ort_value_info_[value_index].reused_buffer_index = allocation_plan[value_index].reused_buffer;
      }
    }

    return Status::OK();
  }
#endif

  Status ComputeReusePlan() {
    gsl::not_null<const ISequentialPlannerContext*> backup_context = context_;
    SequentialPlannerContext no_mem_reuse_context(ExecutionMode::ORT_PARALLEL, ExecutionOrder::DEFAULT, false);
    if (!IsSingleStream()) {
      // use parallel execution context to generate a baseline first (no memory sharing)
      context_ = gsl::not_null<const ISequentialPlannerContext*>(&no_mem_reuse_context);
    }
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    // copy the use counts to a vector, before computing reuse
    std::vector<int> ort_value_usecount;
    ort_value_usecount.reserve(ort_value_info_.size());
#endif
    for (size_t i = 0; i < stream_nodes_.size(); ++i) {
      // compute use count first
      ORT_RETURN_IF_ERROR(ComputeReuseCount());
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
      if (i == 0) {
        for (auto ort_value_info : ort_value_info_) {
          ort_value_usecount.push_back(ort_value_info.usecount);
        }
      }
#endif
      ORT_RETURN_IF_ERROR(ComputeSingleStreamReusePlan(i));
      ClearUseCount();
      freelist_.clear();  // DONOT share freelist across streams
    }
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    CalculateLifetime(ort_value_usecount);
#endif
    if (IsSingleStream())
      return Status::OK();

    // restore context
    context_ = backup_context;

#ifdef ORT_ENABLE_STREAM
    ORT_RETURN_IF_ERROR(OptimizeReusePlanForMultiStream());
#endif

    return Status::OK();
  }

  // Should only be used after ProcessDef()
  Status ComputeSingleStreamReusePlan(size_t stream_index) {
    auto& execution_plan = stream_nodes_[stream_index];
    // Cached graph outputs.
    const auto& graph_outputs = graph_viewer_.GetOutputs();
    for (size_t program_counter = 0; program_counter < execution_plan.size(); ++program_counter) {
      auto node_index = execution_plan[program_counter];
      // the node (aka operator) which carries the considered program (aka computation).
      const auto* pnode = graph_viewer_.GetNode(node_index);
      // node outputs.
      const auto& output_defs = pnode->OutputDefs();
      // External outputs flag.
      bool has_external_outputs = HasExternalOutputs(*pnode);
      // output_arg_def_index is the index of ArgDefs in pnode's output list.
      // At the i-th iteration, we build the allocation plan for the i-th
      // NodeArg in pnode's output list. Allocation plan remains untouched for
      // optional-missing outputs (aka values with empty names).
      for (size_t output_arg_def_index = 0, end = output_defs.size(); output_arg_def_index < end; ++output_arg_def_index) {
        const auto& node_output = output_defs[output_arg_def_index];
        if (!node_output->Exists()) continue;
        // OrtValue index of the considered output NodeArg.
        const auto current = Index(node_output->Name());
        AllocPlan(current).value_type = utils::GetMLDataType(*node_output);
        // Declare OrtValue index of the reused buffer.
        // The the OrtValue indexed by current may reuse the memory in the OrtValue indexed by reused.
        OrtValueIndex reused;
        bool is_strided_tensor = false;
        if (has_external_outputs) {
          ORT_ENFORCE(!IsNonTensor(*node_output), "Only tensors are supported for external outputs for now.");
          AllocPlan(current).alloc_kind = AllocKind::kAllocatedExternally;
        } else if (std::find(graph_outputs.begin(), graph_outputs.end(), node_output) != graph_outputs.end()) {
          // node_output is graph's output, so we can't reuse intermediate buffer
          AllocPlan(current).alloc_kind = AllocKind::kAllocateOutput;

          // hacky perf optimization to not copy a pre-existing value to an output if this is a Loop subgraph and
          // the value is not being changed in the subgraph.
          //
          // this usage of a loop state variable has been seen in two scenarios. both have better alternatives now.
          // we maintain the optimization for existing models.
          //
          // 1. a loop state variable was being provided due to ONNX not supporting empty variadic inputs.
          //    a dummy loop state variable was required in this case.
          //    ONNX now supports empty variadic inputs, so a new model should not add a dummy loop state variable.
          //
          // 2. a loop state variable was being used to explicitly pass in an outer scope value to the subgraph.
          //    this sort of usage is automatically handled via implicit inputs and there's no need to add a
          //    loop state variable in order to access the outer scope value.
          if (parent_node_ && pnode->OpType() == "Identity" && parent_node_->OpType() == "Loop") {
            const NodeArg* input = pnode->InputDefs()[0];

            // first input to the Loop subgraph is the iteration number.
            bool input_is_loop_iteration_number = input == graph_viewer_.GetInputs()[0];
            if (input_is_loop_iteration_number) {
              // as the value inside the OrtValue gets changed by the Loop implementation on each iteration
              // (so it can re-use the OrtValue instance) if it is also a subgraph output it must be allocated
              // so a copy of the current value is returned, so leave alloc_kind as kAllocateOutput
            } else {
              const auto& input_name = input->Name();
              const auto input_index = Index(input_name);

              const auto& alloc_plan = AllocPlan(input_index);
              if (alloc_plan.alloc_kind == AllocKind::kPreExisting) {
                Reuse(input_index, current, AllocKind::kShare);
              }
            }
          }
        } else if (!context_->IsParallelExecutionEnabled() &&
                   FindReusableInput(*pnode, static_cast<int>(output_arg_def_index), &reused, &is_strided_tensor)) {
          // Re-using inputs is applicable for tensors, sequence tensors,
          // and optional types if the kernel has marked certain inputs as
          // possible candidates for re-use
          Reuse(reused, current, AllocKind::kReuse);
          ort_value_info_[current].is_inplace_reuse = true;
#ifdef ENABLE_STRIDED_TENSORS
          if (is_strided_tensor) AllocPlan(current).is_strided_tensor = true;
#else
          ORT_ENFORCE(!is_strided_tensor, "Strided tensor is not supported in non-training build for now.");
#endif  // ENABLE_STRIDED_TENSORS
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
          InplaceReuse(reused, current);
#endif
        } else if (IsNonTensor(*node_output)) {
          AllocPlan(current).alloc_kind = AllocKind::kAllocate;
        } else if (!context_->IsParallelExecutionEnabled() &&
                   FindReusableTensor(*node_output, &reused)) {
          // Reuse an available (dead) buffer for this output, this is only for sequential execution.
          Reuse(reused, current, AllocKind::kReuse);
        } else {
          // otherwise: allocate a new buffer for this output
          AllocPlan(current).alloc_kind = AllocKind::kAllocate;
        }
      }

      // determine if inputs of *pnode can be freed:
      for (auto node_input : pnode->InputDefs()) {
        if (node_input->Exists()) {
          auto& sym = node_input->Name();
          auto original = Buffer(Index(sym));
          // The index will be -1 if it's an initializer that was removed as part of a temporary workaround.
          // See comments in the OrtValueInfo definition.
          if ((original != -1) && (0 == DecrementUseCount(original))) {
            freelist_.push_front(FreeBufferInfo(original, program_counter));
          }
        }
      }

      for (auto node_input : pnode->ImplicitInputDefs()) {
        if (node_input->Exists()) {
          auto& sym = node_input->Name();
          auto original = Buffer(Index(sym));
          // The index will be -1 if it's an initializer that was removed as part of a temporary workaround.
          // See comments in the OrtValueInfo definition.
          if ((original != -1) && (0 == DecrementUseCount(original))) {
            freelist_.push_front(FreeBufferInfo(original, program_counter));
          }
        }
      }

      // determine if any outputs of *pnode are unused and can be freed:
      for (auto node_output : pnode->OutputDefs()) {
        if (node_output->Exists()) {
          auto& sym = node_output->Name();
          auto original = Buffer(Index(sym));
          // The index will be -1 if it's an initializer that was removed as part of a temporary workaround.
          // See comments in the OrtValueInfo definition.
          if (0 == DecrementUseCount(original)) {
            freelist_.push_front(FreeBufferInfo(original, program_counter));
          }
        }
      }
    }
    return Status::OK();
  }

#ifdef ENABLE_TRAINING
  Status CalculateProgramCounter() {
    ClearUseCount();
    ORT_RETURN_IF_ERROR(ComputeReuseCount());
    auto& execution_plan = plan_.node_execution_order_in_training;
    for (size_t program_counter = 0; program_counter < execution_plan.size(); ++program_counter) {
      auto node_index = execution_plan[program_counter];
      // the node (aka operator) which carries the considered program (aka computation).
      const auto* pnode = graph_viewer_.GetNode(node_index);
      // node outputs.
      const auto& output_defs = pnode->OutputDefs();
      for (size_t output_arg_def_index = 0, end = output_defs.size(); output_arg_def_index < end; ++output_arg_def_index) {
        const auto& node_output = output_defs[output_arg_def_index];
        if (!node_output->Exists()) continue;
        // OrtValue index of the considered output NodeArg.
        const auto current = Index(node_output->Name());
        if (AllocPlan(current).alloc_kind == AllocKind::kAllocate) {
          AllocPlan(current).program_counter.AddStart(program_counter);
        }
      }

      auto& node_release_action = plan_.node_release_list[node_index];
      for (auto& action_idx : node_release_action) {
        if (plan_.release_actions[action_idx].ref_count == 1) {
          int value_idx = static_cast<OrtValueIndex>(plan_.release_actions[action_idx].value_index);
          AllocPlan(value_idx).program_counter.AddEnd(program_counter);
        } else {
          // if the releaase action depends on multiple nodes,
          // we can't have a fixed lifetime for it.
          // leave it empty and we will assign it to the lifetime of the whole program at line 1698
        }
      }
    }

    // there are some corner case that an node's output is not graph output but has no consumer
    // currently we didn't generate deallocation plan for those values.
    // so manually fix the PC here.
    // TODO: fix the deallocation plan
    for (auto& alloc_plan : plan_.allocation_plan) {
      if ((alloc_plan.program_counter.Starts().size() - alloc_plan.program_counter.Ends().size()) == 1) {
        alloc_plan.program_counter.AddEnd(execution_plan.size());
      }
    }

    return Status::OK();
  }
#endif

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  void CalculateLifetime(std::vector<int>& ort_value_usecount) {
    auto& execution_plan = graph_viewer_.GetNodesInTopologicalOrder(context_->GetExecutionOrder());
    for (size_t program_counter = 0; program_counter < execution_plan.size(); ++program_counter) {
      auto node_index = execution_plan[program_counter];
      // the node (aka operator) which carries the considered program (aka computation).
      const auto* pnode = graph_viewer_.GetNode(node_index);
      // node outputs.
      const auto& output_defs = pnode->OutputDefs();
      // External outputs flag.
      for (size_t output_arg_def_index = 0, end = output_defs.size(); output_arg_def_index < end; ++output_arg_def_index) {
        const auto& node_output = output_defs[output_arg_def_index];
        if (!node_output->Exists()) continue;
        // OrtValue index of the considered output NodeArg.
        const auto current = Index(node_output->Name());
        AllocPlan(current).life_interval.first = program_counter;
        if (AllocPlan(current).alloc_kind == AllocKind::kAllocatedExternally ||
            AllocPlan(current).alloc_kind == AllocKind::kAllocateOutput) {
          AllocPlan(current).life_interval.second = execution_plan.size();
        }
        // determine if inputs of *pnode can be freed:
        for (auto node_input : pnode->InputDefs()) {
          if (node_input->Exists()) {
            auto& sym = node_input->Name();
            // Compute lifetime
            auto current2 = Index(sym);
            if ((current2 != -1) && (0 == --ort_value_usecount[current2])) {
              AllocPlan(current2).life_interval.second = program_counter;
            }
          }
        }

        for (auto node_input : pnode->ImplicitInputDefs()) {
          if (node_input->Exists()) {
            auto& sym = node_input->Name();
            // Compute lifetime
            auto current2 = Index(sym);
            if ((current2 != -1) && (0 == --ort_value_usecount[current2])) {
              AllocPlan(current2).life_interval.second = program_counter;
            }
          }
        }

        // determine if any outputs of *pnode are unused and can be freed:
        for (auto node_output2 : pnode->OutputDefs()) {
          if (node_output2->Exists()) {
            auto& sym = node_output2->Name();
            // The index will be -1 if it's an initializer that was removed as part of a temporary workaround.
            // See comments in the OrtValueInfo definition.
            auto current2 = Index(sym);
            if ((current2 != -1) && (0 == --ort_value_usecount[current2])) {
              AllocPlan(current2).life_interval.second = program_counter;
            }
          }
        }
      }
    }
  }
#endif

#ifdef ENABLE_TRAINING_CORE
  bool AllocateInputsContiguously(const Node& node) const {
    const KernelCreateInfo& ci = GetKernelCreateInfo(kernel_create_info_map_, node.Index());
    if (ci.kernel_def == nullptr) {
      return false;
    }

    return ci.kernel_def->AllocateInputsContiguously();
  }

  // Compute allocation order for tensors that are required to be allocated contiguously.
  Status ComputeAllocationOrder() {
    for (auto& stream : stream_nodes_) {
      std::vector<OrtValueIndex>& initializer_allocation_order(plan_.initializer_allocation_order);
      std::vector<OrtValueIndex>& activation_allocation_order(plan_.activation_allocation_order);
      for (auto& step : stream) {
        const auto* pnode = graph_viewer_.GetNode(step);
        if (pnode == nullptr) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cannot find the node ", step);
        if (!AllocateInputsContiguously(*pnode)) continue;
        // This node has requested inputs be allocated contiguously.
        const auto& input_defs = pnode->InputDefs();
        onnxruntime::AllocKind input_kind = AllocKind::kAllocateStatically;
        bool set_input_kind = true;
        for (const auto& node_input : input_defs) {
          if (!node_input->Exists()) continue;
          const auto current_idx = Index(node_input->Name());
          const auto& current_plan = AllocPlan(current_idx);
          const auto actual_idx = current_plan.alloc_kind == AllocKind::kReuse ? current_plan.reused_buffer : current_idx;
          const auto& actual_plan = AllocPlan(actual_idx);
          if (set_input_kind) {
            input_kind = actual_plan.alloc_kind;
            set_input_kind = false;
          }

          if ((actual_plan.alloc_kind == AllocKind::kAllocateStatically) && (input_kind != AllocKind::kAllocateStatically))
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "AllocateInputsContiguously() requires all inputs to be initializers, or all inputs to be non-initializers.");

          if (actual_plan.alloc_kind == AllocKind::kAllocateStatically) {
            if (std::find(initializer_allocation_order.begin(), initializer_allocation_order.end(), actual_idx) == initializer_allocation_order.end())
              initializer_allocation_order.push_back(actual_idx);
          } else {
            if (std::find(activation_allocation_order.begin(), activation_allocation_order.end(), actual_idx) == activation_allocation_order.end())
              activation_allocation_order.push_back(actual_idx);
          }
        }
      }
    }
    return Status::OK();
  }
#endif

  void VerifyMemoryTimeSchedule() {
    size_t idx = 0;
    for (const auto& entry : plan_.allocation_plan) {
      if (entry.alloc_kind == AllocKind::kAllocate) {
        ORT_ENFORCE(entry.program_counter.HasValidEntries(), "Invalid program_counter entries at index ", idx);
      }

      ++idx;
    }
  }

  // Convert information in execution plan and memory reuse plan into release plan
  Status GenerateDeallocationPlan() {
    // 1. build the consumer list for each value
    std::vector<InlinedVector<NodeIndex>> value_consumers;
    int num_ml_values = ort_value_name_idx_map_.MaxIdx() + 1;
    value_consumers.resize(num_ml_values);

    // iterate each stream from back, so the first element is the last consumer in single stream case
    for (auto& stream : stream_nodes_) {
      for (auto it = stream.rbegin(), end = stream.rend(); it != end; ++it) {
        NodeIndex node_index = *it;
        auto* node = graph_viewer_.GetNode(node_index);

        auto process_input = [&](const NodeArg& input, size_t /*arg_idx*/) {
          if (input.Exists()) {
            const auto& name = input.Name();
            int value_idx;
            ORT_RETURN_IF_ERROR(ort_value_name_idx_map_.GetIdx(name, value_idx));
            auto origin = Buffer(value_idx);
            if (origin != -1 && plan_.allocation_plan[origin].alloc_kind == AllocKind::kAllocate) {
              // add current node as consumer for origin buffer
              value_consumers[origin].push_back(node_index);
            }
          }
          return Status::OK();
        };

        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(node->InputDefs(), process_input));
        ORT_RETURN_IF_ERROR(Node::ForEachWithIndex(node->ImplicitInputDefs(), process_input));
      }
    }
    // 2. build the release actions and fill into node's release list
    auto process_consumer = [&](size_t release_action_idx, NodeIndex node_index) {
      plan_.release_actions[release_action_idx].ref_count++;
      plan_.node_release_list[node_index].push_back(release_action_idx);
    };
    plan_.node_release_list.resize(SafeInt<size_t>(graph_viewer_.MaxNodeIndex()) + 1);
    for (size_t i = 0; i < value_consumers.size(); ++i) {
      if (!value_consumers[i].empty()) {
        plan_.release_actions.push_back(SequentialExecutionPlan::ReleaseAction{i, 0});
        auto release_action_idx = plan_.release_actions.size() - 1;
        // check whether we can static determine where to release.
        // TODO: here we use a temporary simple solution is only static release when all the consumers are on the same stream
        // we actually can do better if all the consumers depends on the last consumer.
        // will optimize it later
        bool is_all_consumer_same_stream = true;
        auto stream_idx = node_stream_map_[value_consumers[i][0]];
        for (size_t j = 1; j < value_consumers[i].size(); ++j) {
          if (node_stream_map_[value_consumers[i][j]] != stream_idx) {
            is_all_consumer_same_stream = false;
            break;
          }
        }
        if (is_all_consumer_same_stream) {
          // all the consumers are on the same stream, so the first element is the last consumer int the stream.
          process_consumer(release_action_idx, value_consumers[i][0]);
        } else {
          // can't static determin, add all the consumers, we will use ref count in release action
          for (auto node_index : value_consumers[i]) {
            process_consumer(release_action_idx, node_index);
          }
        }
      }
    }
    return Status::OK();
  }

#ifndef ORT_ENABLE_STREAM
  void PartitionIntoStreams(const logging::Logger& /*logger*/,
                            const ExecutionProviders& /*execution_providers*/,
                            const PathString& /*partition_config_file*/) {
    if (graph_viewer_.NumberOfNodes() > 0) {
      stream_nodes_.push_back({});
      node_stream_map_.resize(SafeInt<size_t>(graph_viewer_.MaxNodeIndex()) + 1);
      for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder()) {
        stream_nodes_[0].push_back(node_index);
        node_stream_map_[node_index] = 0;
      }
      num_logic_streams_ = 1;
    }
  }

  Status BuildExecutionPlan(const ExecutionProviders& execution_providers) {
    // 1. create logic stream instance
    auto& execution_plan = plan_.execution_plan;

    if (graph_viewer_.NumberOfNodes() > 0) {
      ORT_ENFORCE(num_logic_streams_ == 1 && !stream_nodes_[0].empty());
      execution_plan.reserve(1);
      auto first_node_index = stream_nodes_[0][0];
      auto* node = graph_viewer_.GetNode(first_node_index);
      onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
      const IExecutionProvider* ep = execution_providers.Get(exec_provider_name);
      ORT_ENFORCE(ep);
      auto node_device_mem_location = ep->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault);
      execution_plan.emplace_back(std::make_unique<SequentialExecutionPlan::LogicStream>(node_device_mem_location));
      // 2. add steps to the execution plan
      for (auto node_index : stream_nodes_[0]) {
        execution_plan[0]->steps_.emplace_back(std::make_unique<LaunchKernelStep>(node_index));
      }
    } else {
      // graph with no nodes. e.g. subgraph of If might return the input as-is or a constant value from an initializer
    }

    return Status::OK();
  }

#else

  void
  PartitionIntoStreams(const logging::Logger& logger, const ExecutionProviders& execution_providers,
                       const PathString& partition_config_file) {
    auto partitioner = IGraphPartitioner::CreateGraphPartitioner(logger, partition_config_file);
    auto status = partitioner->PartitionGraph(graph_viewer_, execution_providers, stream_nodes_, context_->GetExecutionOrder());
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    node_stream_map_.resize(SafeInt<size_t>(graph_viewer_.MaxNodeIndex()) + 1);
    for (size_t i = 0; i < stream_nodes_.size(); ++i) {
      for (auto node_index : stream_nodes_[i]) {
        node_stream_map_[node_index] = i;
      }
    }
    num_logic_streams_ = stream_nodes_.size();
  }

  // build each logic streams
  Status BuildExecutionPlan(const ExecutionProviders& execution_providers,
                            const IStreamCommandHandleRegistry& stream_handle_registry) {
    // 1. create logic stream instance
    auto& execution_plan = plan_.execution_plan;
    execution_plan.reserve(num_logic_streams_);
    for (size_t i = 0; i < num_logic_streams_; ++i) {
      if (!stream_nodes_[i].empty()) {
        // get device from first node
        auto& node_index = stream_nodes_[i][0];
        auto* node = graph_viewer_.GetNode(node_index);
        onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
        const IExecutionProvider* ep = execution_providers.Get(exec_provider_name);
        ORT_ENFORCE(ep);
        auto node_device_mem_location = ep->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault);
        execution_plan.emplace_back(std::make_unique<SequentialExecutionPlan::LogicStream>(node_device_mem_location));
      } else {
        execution_plan.emplace_back(nullptr);
      }
    }
    // 2. Determining following things:
    //    a. which node needs to generate the notification
    //    b. which node needs to trigger downstream
#ifdef ENABLE_TRAINING
    // We will leverage the topological order for the training scenario.
    // The nodes before yieldOp in topo-order will be executed in RunForward() and nodes after will be executed in RunBackward()
    // This partition may not be exactly the same as forward model/gradient model, for example, some nodes in gradient model are
    // before yieldOp thus will be executed in RunForward()
    // But the final result is still correct, as long as all the nodes will be executed in either RunForward() or RunBackward()
    // and no dependency conflict during the execution.
    const std::vector<NodeIndex>& topo_sort = graph_viewer_.GetNodesInTopologicalOrder(context_->GetExecutionOrder());
    plan_.node_index_2_toposort_index.reserve(topo_sort.size());
    size_t yieldOp_index_in_toposort = topo_sort.size();
    for (size_t i = 0; i < topo_sort.size(); i++) {
      plan_.node_index_2_toposort_index[topo_sort[i]] = i;
      const Node* node = graph_viewer_.GetNode(topo_sort[i]);
      if (node->OpType() == "YieldOp") {
        ORT_ENFORCE(yieldOp_index_in_toposort == topo_sort.size(), "Two YieldOp in the graph");
        yieldOp_index_in_toposort = i;
      }
    }

    auto AreNodesSeparatedByYield = [&](NodeIndex producer, NodeIndex consumer) {
      size_t producer_topoindex = plan_.node_index_2_toposort_index[producer];
      size_t consumer_topoindex = plan_.node_index_2_toposort_index[consumer];
      return producer_topoindex < yieldOp_index_in_toposort && yieldOp_index_in_toposort < consumer_topoindex;
    };
#endif
    size_t num_trigger_points = 0;
    InlinedHashMap<NodeIndex, size_t> node_to_trigger_points;
    InlinedHashMap<NodeIndex, NotificationIndex> node_to_notification;
    std::map<NodeIndex, std::map<NodeIndex, WaitNotificationFn>> node_to_wait;
    for (size_t i = 0; i < num_logic_streams_; ++i) {
      for (auto node_index : stream_nodes_[i]) {
        auto* node = graph_viewer_.GetNode(node_index);
        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
          // if the output node is not in the same stream, generate a trigger point
          if (node_stream_map_[it->Index()] != i
#ifdef ENABLE_TRAINING
              // Do not insert Barrier/TriggerDownStream step if the producer and consumer are in different sides of yieldOp
              // As in this case producer will surely be ready before the consumer is running.
              && !AreNodesSeparatedByYield(node_index, it->Index())
#endif
          ) {
            node_to_trigger_points[node_index] = num_trigger_points++;
            break;
          }
        }
      }
    }
    for (size_t i = 0; i < num_logic_streams_; ++i) {
      for (auto node_index : stream_nodes_[i]) {
        auto* node = graph_viewer_.GetNode(node_index);
        auto stream_device = execution_plan[i]->device_.Type();
        // Neither trigger ActivateNotification/WaitOnEPStep for Shape op (whose output is ready for all the EPs), nor
        // upstream is on CPU device (As currently we never invoke RegisterWaitFn(CPU, ...) for all kinds of EP, thus no wait_handle can be retrieved for this case)
        if (node->OpType() != "Shape" && stream_device != OrtDevice::CPU) {
          for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
            bool output_consumed_in_subgraph = true;
            for (auto* output : node->OutputDefs()) {
              if (output->Exists()) {
                if (std::find(it->InputDefs().begin(), it->InputDefs().end(), output) != it->InputDefs().end()) {
                  output_consumed_in_subgraph = false;  // output direclty consumed in current graph
                  OrtValueIndex output_arg_idx;
                  ORT_THROW_IF_ERROR(ort_value_name_idx_map_.GetIdx(output->Name(), output_arg_idx));
                  // there are two cases we need notification:
                  // 1. the consumer is not in the same stream
                  // 2. the consumer is in the same stream(non-cpu device), but it consumes a CPU tensor from an non-shape op.
                  //    for example, a resize cuda kernel consumer a tensor from MemCpyToHost cuda kernel on the same stream.
                  //    in this case, the FIFO can't guarantee the cpu tensor is ready when resize kernel is launching
                  OrtDevice::DeviceType output_arg_device = plan_.allocation_plan[output_arg_idx].location.Type();
                  WaitNotificationFn wait_handle = stream_handle_registry.GetWaitHandle(stream_device, output_arg_device);
                  if ((node_stream_map_[it->Index()] != i || output_arg_device == OrtDevice::CPU) && wait_handle != nullptr) {
                    if (node_to_notification.find(node_index) == node_to_notification.end()) {
                      node_to_notification[node_index] = plan_.notification_owners.size();
                      plan_.notification_owners.push_back(i);
                    }
                    // if node_index is already in the map, it will NOT be overwritten by insert()
                    node_to_wait[it->Index()].insert({node_index, wait_handle});
                  }
                }
              }  // output->Exists
            }    // for each output
            if (output_consumed_in_subgraph) {
              const auto downstream = node_stream_map_[it->Index()];
              if (downstream != i) {
                auto downstream_device = execution_plan[downstream]->device_.Type();
                WaitNotificationFn wait_handle = stream_handle_registry.GetWaitHandle(stream_device, downstream_device);
                if (wait_handle) {
                  if (node_to_notification.find(node_index) == node_to_notification.end()) {
                    node_to_notification[node_index] = plan_.notification_owners.size();
                    plan_.notification_owners.push_back(i);
                  }
                  node_to_wait[it->Index()].insert({node_index, wait_handle});
                }
              }
            }
          }
        }
      }
    }

    // 3. Check the nodes in each logical stream, confirm it aligned with the device in the logic stream;
    for (size_t i = 0; i < num_logic_streams_; ++i) {
      std::set<const IExecutionProvider*> providers;
      for (auto node_index : stream_nodes_[i]) {
        auto* node = graph_viewer_.GetNode(node_index);
        onnxruntime::ProviderType exec_provider_name = node->GetExecutionProviderType();
        const IExecutionProvider* ep = execution_providers.Get(exec_provider_name);
        auto node_device_mem_location = ep->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault);
        ORT_ENFORCE(execution_plan[node_stream_map_[node_index]]->device_.Type() == node_device_mem_location.Type());
      }
    }

    // 4. add commands to logic queue
    for (size_t i = 0; i < num_logic_streams_; ++i) {
      for (size_t j = 0; j < stream_nodes_[i].size(); ++j) {
        auto node_index = stream_nodes_[i][j];
        if (j > 0) {
          // add dependency for current logic stream
          dependence_graph_[node_index].insert(stream_nodes_[i][j - 1]);
        }
        auto* node = graph_viewer_.GetNode(node_index);
        std::unordered_set<NodeIndex> visited;  // TODO(leca): See the bug description in PlannerTest.MultiStreamMultiOutput. Can remove this variable once this bug is fixed
        for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
          if (visited.find(it->Index()) != visited.end()) {
            continue;
          }
          visited.insert(it->Index());
          //  check whether we need to add barrier
          if (std::find(stream_nodes_[i].begin(), stream_nodes_[i].end(), it->Index()) == stream_nodes_[i].end()
#ifdef ENABLE_TRAINING
              && !AreNodesSeparatedByYield(it->Index(), node_index)
#endif
          ) {
            // find the trigger_point_id
            auto trigger_point_it = node_to_trigger_points.find(it->Index());
            ORT_ENFORCE(trigger_point_it != node_to_trigger_points.end());
            size_t trigger_point_index = trigger_point_it->second;
            // push a barrier
            size_t barrier_id = plan_.num_barriers++;
            plan_.downstream_map[trigger_point_index].push_back({i,
                                                                 static_cast<int>(execution_plan[i]->steps_.size())});
            execution_plan[i]->steps_.emplace_back(std::make_unique<BarrierStep>(barrier_id, node_index));
          }
        }

        auto wait_it = node_to_wait.find(node_index);
        if (wait_it != node_to_wait.end()) {
          for (auto wait_param : wait_it->second) {
            execution_plan[i]->steps_.emplace_back(std::make_unique<WaitOnEPStep>(wait_param.second,
                                                                                  node_to_notification[wait_param.first], node_index));
          }
        }

        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
          // add dependency for model graph
          dependence_graph_[it->Index()].insert(node_index);
        }
        // push launch kernel command
        execution_plan[i]->steps_.emplace_back(std::make_unique<LaunchKernelStep>(node_index));
        // check if any notification generated by this node, if yes, push a activate
        auto notification_it = node_to_notification.find(node_index);
        if (notification_it != node_to_notification.end()) {
          NotificationIndex notification_index = notification_it->second;
          execution_plan[i]->steps_.emplace_back(std::make_unique<ActivateNotificationStep>(notification_index, node_index));
        }
        // check if any trigger point generated by this node, if yes, push a trigger
        auto trigger_point_it = node_to_trigger_points.find(node_index);
        if (trigger_point_it != node_to_trigger_points.end()) {
          // notify downstreams
          execution_plan[i]->steps_.emplace_back(std::make_unique<TriggerDownstreamStep>(trigger_point_it->second, node_index));
        }
      }
    }

    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder(context_->GetExecutionOrder())) {
      auto* node = graph_viewer_.GetNode(node_index);
      const auto& output_defs = node->OutputDefs();
      for (size_t output_idx_local = 0; output_idx_local < output_defs.size(); ++output_idx_local) {
        const auto& node_output = output_defs[output_idx_local];
        if (!node_output->Exists()) continue;
        OrtValueIndex output_idx_global;
        ORT_THROW_IF_ERROR(ort_value_name_idx_map_.GetIdx(node_output->Name(), output_idx_global));
        plan_.value_to_stream_map[output_idx_global] = node_stream_map_[node_index];
        value_node_map_[output_idx_global] = node_index;
      }
    }
#ifdef ENABLE_TRAINING
    // 5. build the node_execution_order_in_training
    //  the training memory optimization rely on a stable order how kernel get launched to calculate memory pattern
    //  so we limit training scenario to run with single stream and single thread mode
    //  the code below will simulate the execution and get the stable execution order
    InlinedVector<int> execution_offsets(num_logic_streams_, -1);
    InlinedHashSet<OrtValueIndex> produced_values;

    for (auto graph_input : graph_viewer_.GetInputs()) {
      OrtValueIndex index = Index(graph_input->Name());
      produced_values.insert(index);
    }

    for (auto out_scope_arg : graph_viewer_.GetOuterScopeNodeArgNames()) {
      OrtValueIndex index = Index(out_scope_arg);
      produced_values.insert(index);
    }

    for (const auto& pair : graph_viewer_.GetAllInitializedTensors()) {
      const auto& initializer_name = pair.first;
      OrtValueIndex index = Index(initializer_name);
      produced_values.insert(index);
    }

    InlinedHashSet<OrtValueIndex> producable_values;
    for (auto node_index : graph_viewer_.GetNodesInTopologicalOrder(context_->GetExecutionOrder())) {
      auto* node = graph_viewer_.GetNode(node_index);
      // add the output to produce nodes list
      for (auto* output_def : node->OutputDefs()) {
        if (!output_def->Exists())
          continue;
        OrtValueIndex index = Index(output_def->Name());
        producable_values.insert(index);
      }
    }

    std::function<void(size_t, int)> process_stream;
    process_stream = [&](size_t i, int node_offset) {
      if (node_offset > execution_offsets[i])
        return;
      while (execution_offsets[i] < static_cast<int>(stream_nodes_[i].size())) {
        if (execution_offsets[i] == -1) {
          execution_offsets[i]++;
          continue;
        }
        NodeIndex node_index = stream_nodes_[i][execution_offsets[i]];
        auto* node = graph_viewer_.GetNode(node_index);
        // check whether the node is ready:
        bool input_ready = true;
        for (auto* input_def : node->InputDefs()) {
          if (!input_def->Exists())
            continue;
          OrtValueIndex index = Index(input_def->Name());
          if (produced_values.find(index) == produced_values.end() &&
              producable_values.find(index) != producable_values.end()) {
            input_ready = false;
            break;
          }
        }
        if (!input_ready)
          break;
        // trace the execution of this node
        plan_.node_execution_order_in_training.push_back(node_index);
        // add the output to produce nodes list
        for (auto* output_def : node->OutputDefs()) {
          if (!output_def->Exists())
            continue;
          OrtValueIndex index = Index(output_def->Name());
          produced_values.insert(index);
        }
        // trigger downstream
        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
          auto stream_idx = node_stream_map_[it->Index()];
          if (stream_idx != i) {
            auto node_it = std::find(stream_nodes_[stream_idx].begin(), stream_nodes_[stream_idx].end(), it->Index());
            int offset = static_cast<int>(std::distance(stream_nodes_[stream_idx].begin(), node_it));
            process_stream(stream_idx, offset);
          }
        }
        // move_to_next
        execution_offsets[i]++;
      }
    };

    auto num_of_nodes = graph_viewer_.GetNodesInTopologicalOrder(context_->GetExecutionOrder()).size();
    plan_.node_execution_order_in_training.reserve(num_of_nodes);
    for (size_t i = 0; i < stream_nodes_.size(); ++i) {
      process_stream(i, -1);
    }
    ORT_ENFORCE(plan_.node_execution_order_in_training.size() == num_of_nodes);
#endif

    return Status::OK();
  }
#endif

  static bool IsNonTensor(const onnxruntime::NodeArg& nodearg) {
    // TODO: unclear why we should go through a string-representation of type
    auto ptype = nodearg.Type();
    auto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(ptype);
    return !utils::HasTensorType(type_proto);
  }

#if !defined(DISABLE_OPTIONAL_TYPE)
  static bool IsOptionalType(const onnxruntime::NodeArg& nodearg) {
    const auto* type_proto = nodearg.TypeAsProto();
    return type_proto->value_case() == ONNX_NAMESPACE::TypeProto::kOptionalType;
  }
#endif

// For in-place reuse tensors, the lifetime is the union of all the tensors that tensors that use that buffer
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  void AdjustInplaceLifeIntervals() {
    InlinedHashMap<OrtValueIndex, InlinedVector<OrtValueIndex>> inplace_reuse_buffer;
    inplace_reuse_buffer.reserve(ort_value_info_.size());
    for (size_t i = 0; i < ort_value_info_.size(); ++i) {
      if (AllocPlan(OrtValueIndex(i)).inplace_reuse != OrtValueIndex(i)) {
        inplace_reuse_buffer[ort_value_info_[i].inplace_reused_buffer_index].push_back(OrtValueIndex(i));
      }
    }
    for (const auto& item : inplace_reuse_buffer) {
      IntervalT& lifetime = AllocPlan(item.first).life_interval;
      for (const auto& value : item.second) {
        auto start = AllocPlan(value).life_interval.first;
        auto end = AllocPlan(value).life_interval.second;
        lifetime.first = lifetime.first < start ? lifetime.first : start;
        lifetime.second = lifetime.second > end ? lifetime.second : end;
      }
      for (const auto& value : item.second) {
        AllocPlan(value).life_interval = lifetime;
      }
    }
  }
#endif
};

Status PlannerImpl::CreatePlan(
#ifdef ORT_ENABLE_STREAM
    const IStreamCommandHandleRegistry& stream_handle_registry,
#endif
    const PathString& partition_config_file,
    const logging::Logger& logger) {
  // 1. partition graph into streams
  PartitionIntoStreams(logger, execution_providers_, this->parent_node_ ? PathString{} : partition_config_file);

  // 2. initialize the plan based on stream partition result
  int num_ml_values = ort_value_name_idx_map_.MaxIdx() + 1;

  Initialize(static_cast<size_t>(num_ml_values));

  // compute value location
  ORT_RETURN_IF_ERROR(ComputeValueLocation());
  ORT_RETURN_IF_ERROR(ComputePlanForInputsAndWeights());

  // build execution plan
#ifdef ORT_ENABLE_STREAM
  ORT_RETURN_IF_ERROR(BuildExecutionPlan(execution_providers_, stream_handle_registry));
#else
  ORT_RETURN_IF_ERROR(BuildExecutionPlan(execution_providers_));
#endif

  // determine sharing/reuse among ml-values
  ORT_RETURN_IF_ERROR(ComputeReusePlan());

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  // Adjust the allocate and lifetime intervals for all ml-values, based on their allocation kind.
  AdjustInplaceLifeIntervals();
#endif

#ifdef ENABLE_TRAINING_CORE
  // Determine allocation order for weights and activations. This needs to be done after ComputeReusePlan.
  ORT_RETURN_IF_ERROR(ComputeAllocationOrder());
#endif

  // convert information in the freelist_ into a deallocation plan in required format
  ORT_RETURN_IF_ERROR(GenerateDeallocationPlan());

  // generate program counter
#ifdef ENABLE_TRAINING
  ORT_RETURN_IF_ERROR(CalculateProgramCounter());
#endif

  // Ensure Memory-Time schedule is valid. This should be called at the end because memory start/end timestamps
  // are updated until GenerateDeallocationPlan is finished.
  // TODO: enable verification
  // VerifyMemoryTimeSchedule();

  return Status::OK();
}

Status SequentialPlanner::CreatePlan(
    const Node* parent_node,
    const onnxruntime::GraphViewer& graph_viewer,
    gsl::span<const NodeArg* const> outer_scope_node_args,
    const ExecutionProviders& providers,
    const KernelCreateInfoMap& kernel_create_info_map,
    const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps,
    const InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_node_arg_to_location_map,
    const OrtValueNameIdxMap& ort_value_name_idx_map,
    const ISequentialPlannerContext& context,
#ifdef ORT_ENABLE_STREAM
    const IStreamCommandHandleRegistry& stream_handle_registry,
#endif
    const PathString& partition_config_file,
    const logging::Logger& logger,
    std::optional<SequentialExecutionPlan>& plan) {
  // allocate/reset here so we know it's clean
  plan.emplace();

  PlannerImpl planner(parent_node, graph_viewer, outer_scope_node_args, providers,
                      kernel_create_info_map, subgraphs_kernel_create_info_maps,
                      outer_scope_node_arg_to_location_map,
                      ort_value_name_idx_map, context, *plan);

  return planner.CreatePlan(
#ifdef ORT_ENABLE_STREAM
      stream_handle_registry,
#endif
      partition_config_file,
      logger);
}

#ifdef ORT_ENABLE_STREAM
/*
DeviceBasedPartitioner stores config in json format:
------------------------------------------------------
{
"type":"DeviceBasedPartitioner",
"streams":[
           ["node_1","node_7"],
           ["node_2","node_4","node_5"],
           ["node_3","node_6"],
          ]
"devices":["0","0","1"]
}
------------------------------------------------------
"streams" specifies streams of nodes;
"devices" specifies the type of device of each stream.
Pls check definition of OrtDevice for more detail on device type.
*/
class DeviceBasedPartitioner : public IGraphPartitioner {
 public:
  DeviceBasedPartitioner(const logging::Logger& logger,
                         const PathString& config_file) : IGraphPartitioner(logger, config_file) {
    Initialize();
  }

  ~DeviceBasedPartitioner() {
    if (need_save_) {
      SaveConfig();
    }
  }

  void SaveConfig() const;
  Status PartitionGraph(const onnxruntime::GraphViewer& graph_viewer,
                        const ExecutionProviders& execution_providers,
                        std::vector<InlinedVector<NodeIndex>>& stream_nodes,
                        ExecutionOrder execution_order) override;

  const char* Type() const override { return "DeviceBasedPartitioner"; }
  size_t Streams() const override { return node_names_by_stream_.size(); }

 private:
  void Initialize();
  // device_types_[i] saves the device type for nodes in node_names_by_stream_[i]
  std::vector<OrtDevice::DeviceType> device_types_;
  std::vector<InlinedVector<std::string>> node_names_by_stream_;
  bool need_save_ = false;
};

#define EXIT_ON_ERR(warning)         \
  LOGS(logger_, WARNING) << warning; \
  node_names_by_stream_.clear();     \
  if_stream.close();                 \
  return;

Status DeviceBasedPartitioner::PartitionGraph(const onnxruntime::GraphViewer& graph_viewer,
                                              const ExecutionProviders& execution_providers,
                                              std::vector<InlinedVector<NodeIndex>>& stream_nodes,
                                              ExecutionOrder execution_order) {
  InlinedHashMap<std::string, int> op_type_counter;
  auto& p_graph_nodes = graph_viewer.GetNodesInTopologicalOrder(execution_order);

  if (node_names_by_stream_.empty()) {  // input configure empty, do it from scratch

    InlinedHashMap<OrtDevice::DeviceType, int> device_to_stream;

    for (auto node_index : p_graph_nodes) {
      // get device info of the node
      const auto* node = graph_viewer.GetNode(node_index);
      const auto& op_type = node->OpType();
      const auto& node_name = node->Name();
      auto* ep = execution_providers.Get(*node);
      auto device_type = ep->GetOrtDeviceByMemType(OrtMemType::OrtMemTypeDefault).Type();

      // log the device
      auto it = device_to_stream.find(device_type);
      if (it == device_to_stream.end()) {
        device_to_stream[device_type] = static_cast<int>(node_names_by_stream_.size());
        node_names_by_stream_.push_back({});
        device_types_.push_back(device_type);
        it = device_to_stream.find(device_type);
      }
      // put the node into the belonging stream
      if (node_name.empty()) {
        node_names_by_stream_[it->second].push_back(op_type + std::to_string(op_type_counter[op_type]++));
      } else {
        node_names_by_stream_[it->second].push_back(node_name);
      }
    }
  }
  InlinedHashMap<std::string, size_t> node_stream_map;
  node_stream_map.reserve(p_graph_nodes.size());
  for (size_t i = 0; i < node_names_by_stream_.size(); ++i) {
    for (const auto& node_name : node_names_by_stream_[i]) {
      node_stream_map[node_name] = i;
    }
  }
  op_type_counter.clear();
  stream_nodes.clear();
  stream_nodes.resize(node_names_by_stream_.size());
  for (auto node_index : p_graph_nodes) {
    const auto* node = graph_viewer.GetNode(node_index);
    const auto& op_type = node->OpType();
    auto node_name = node->Name();
    if (node_name.empty()) {
      node_name = op_type + std::to_string(op_type_counter[op_type]++);
    }
    auto iter = node_stream_map.find(node_name);
    ORT_ENFORCE(iter != node_stream_map.end(), "Failed to find node \"", node_name, "\" in node-stream map");
    stream_nodes[node_stream_map[node_name]].push_back(node_index);
  }
  return Status::OK();
}

void DeviceBasedPartitioner::Initialize() {
  if (config_file_.empty()) {
    return;
  }
  std::ifstream if_stream(config_file_);
  if (if_stream.is_open()) {
    try {
      json json_config = json::parse(if_stream);
      if (json_config["type"] != Type()) {
        EXIT_ON_ERR("Partitioner type is not DeviceBasedPartitioner");
      }
      for (const auto& node_stream : json_config["streams"]) {
        node_names_by_stream_.emplace_back();
        for (const auto& node_name : node_stream) {
          node_names_by_stream_.back().push_back(node_name);
        }
      }
      for (const auto& device_type : json_config["devices"]) {
        const std::string type_str = device_type;
        device_types_.push_back(static_cast<OrtDevice::DeviceType>(std::atoi(type_str.c_str())));
      }
    } catch (const std::exception& ex) {
      EXIT_ON_ERR(ex.what());
    }
    if_stream.close();
    ORT_ENFORCE(node_names_by_stream_.size() == device_types_.size(),
                "Number of streams does not equal to number of device types!");
  } else {
    // when config file specified but cannot be read, rewrite it.
    need_save_ = true;
  }
}

void DeviceBasedPartitioner::SaveConfig() const {
  ORT_TRY {
    json json_config;
    json_config["type"] = "DeviceBasedPartitioner";
    if (!node_names_by_stream_.empty()) {
      json_config["streams"] = json::array();
      for (const auto& node_stream : node_names_by_stream_) {
        auto node_array = json::array();
        for (const auto& node_name : node_stream) {
          node_array.insert(node_array.end(), node_name);
        }
        json_config["streams"].insert(json_config["streams"].end(), node_array);
      }
    }
    if (!device_types_.empty()) {
      json_config["devices"] = json::array();
      for (const auto& device_type : device_types_) {
        json_config["devices"].insert(json_config["devices"].end(), std::to_string(device_type));
      }
    }
    std::ofstream of_stream(config_file_);
    if (of_stream.is_open()) {
      of_stream << json_config.dump();
      of_stream.close();
    }
  }
  ORT_CATCH(const std::exception& ex) {
    LOGS(logger_, WARNING) << "Caught exception during saving DeviceBasedPartitioner config: " << ex.what();
  }
}

std::unique_ptr<IGraphPartitioner> IGraphPartitioner::CreateGraphPartitioner(const logging::Logger& logger,
                                                                             const PathString& config_file) {
  // use device based partitioner by default
  IGraphPartitioner::GraphPartitioningStrategy partitioner_type =
      IGraphPartitioner::GraphPartitioningStrategy::DeviceBasedPartition;
  if (!config_file.empty()) {
    std::ifstream f(config_file);
    if (f.is_open()) {
      try {
        json json_config = json::parse(f);
        if (json_config.contains("type")) {
          auto type = json_config["type"];
          if (type == "DeviceBasedPartitioner") {
            partitioner_type = IGraphPartitioner::GraphPartitioningStrategy::DeviceBasedPartition;
          }
        }
      } catch (const std::exception& ex) {
        LOGS(logger, WARNING) << "Caught exception when reading partition config file: " << ex.what();
      }
      f.close();
    }
  }
  if (partitioner_type == IGraphPartitioner::GraphPartitioningStrategy::DeviceBasedPartition) {
    LOGS(logger, INFO) << "Use DeviceBasedPartition as default";
    return std::make_unique<DeviceBasedPartitioner>(logger, config_file);
  }  // else if other partitioner types ...
  ORT_THROW("Failed to create partitioner");
}

#endif

}  // namespace onnxruntime
