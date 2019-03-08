// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocation_planner.h"
#include <list>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include "core/common/exceptions.h"
#include "core/platform/env.h"
#include "core/framework/data_types.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

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
  }
  return out;
}

// Output details of an execution plan:
std::ostream& operator<<(std::ostream& out, std::pair<const SequentialExecutionPlan*, const SessionState*> planinfo) {
  const SequentialExecutionPlan& plan = *planinfo.first;
  const SessionState& session_state = *planinfo.second;
  auto& graph = *session_state.GetGraphViewer();
  std::unordered_map<int, std::string> index_to_name;

  out << "Allocation Plan:\n";
  auto plan_size = plan.allocation_plan.size();

  for (auto& name_index : session_state.GetMLValueNameIdxMap()) {
    auto index = name_index.second;
    index_to_name[index] = name_index.first;
    out << "(" << index << ") " << name_index.first << " : ";
    if (0 <= index && static_cast<size_t>(index) < plan_size) {
      auto& elt_plan = plan.allocation_plan[index];
      out << elt_plan.alloc_kind;
      if (elt_plan.alloc_kind == AllocKind::kReuse) out << " " << elt_plan.reused_buffer;

      auto& loc = elt_plan.location;
      out << ", " << loc.ToString();

      if (elt_plan.create_fence_if_async) out << ", use fence when async";

    } else {
      out << "Index out-of-range!";
    }

    out << std::endl;
  }

  out << "\nExecution Plan:\n";
  for (size_t i = 0; i < plan.execution_plan.size(); ++i) {
    auto& step = plan.execution_plan[i];
    auto node = graph.GetNode(step.node_index);
    ORT_ENFORCE(nullptr != node);
    out << "[" << i << "] ";
    out << node->OpType() << " (" << node->Name() << ")" << std::endl;
    if (step.free_from_index <= step.free_to_index) {
      out << "Free ml-values: ";
      std::string sep;
      for (int j = step.free_from_index; j <= step.free_to_index; ++j) {
        auto freed_value_index = plan.to_be_freed[j];
        auto name_iter = index_to_name.find(freed_value_index);
        auto name = (name_iter == index_to_name.end()) ? "INVALID INDEX" : name_iter->second;
        out << sep << "(" << freed_value_index << ") " << name;
        sep = ", ";
      }
      out << std::endl;
    }
  }

  return out;
}

class PlannerImpl {
 public:
  PlannerImpl(const Node* parent_node,
              const onnxruntime::GraphViewer& graph_viewer,
              const std::vector<const NodeArg*>& outer_scope_node_args,
              const ExecutionProviders& providers,
              const KernelRegistryManager& kernel_registry,
              const MLValueNameIdxMap& mlvalue_name_idx_map,
              const ISequentialPlannerContext& context,
              SequentialExecutionPlan& plan)
      : context_{context},
        plan_{plan},
        parent_node_{parent_node},
        graph_viewer_{graph_viewer},
        outer_scope_node_args_{outer_scope_node_args},
        execution_providers_{providers},
        kernel_registry_{kernel_registry},
        mlvalue_name_idx_map_{mlvalue_name_idx_map} {
  }

  Status CreatePlan();

 private:
  const ISequentialPlannerContext& context_;
  SequentialExecutionPlan& plan_;

  const Node* parent_node_;
  const onnxruntime::GraphViewer& graph_viewer_;
  const std::vector<const NodeArg*>& outer_scope_node_args_;
  const ExecutionProviders& execution_providers_;

  const KernelRegistryManager& kernel_registry_;
  const MLValueNameIdxMap& mlvalue_name_idx_map_;

  // MLValueInfo: Auxiliary information about an MLValue used only during plan-generation:
  struct MLValueInfo {
    const onnxruntime::NodeArg* p_def_site;  // the (unique) NodeArg corresponding to the MLValue
    int usecount = 0;                        // static reference-count
    MLValueIndex reused_buffer_index;        // index of original buffer to reuse
  };

  // ml_value_info_ is indexed by an MLValueIndex
  std::vector<MLValueInfo> ml_value_info_;

  // FreeBufferInfo is used to track information about ml-values whose buffers are
  // free to be reused.
  struct FreeBufferInfo {
    MLValueIndex ml_value;
    // deallocate_point is an index into the execution-plan; thus, ml_value becomes free after
    // this step in the execution-plan is completed.
    size_t deallocate_point;
    FreeBufferInfo(MLValueIndex mlvalue, size_t dealloc_point) : ml_value(mlvalue), deallocate_point(dealloc_point) {}
  };
  // freelist_ : a list of ml-values whose buffers are free to be reused, sorted by when
  // they became free (more recently freed earlier in the list).
  std::list<FreeBufferInfo> freelist_;

  MLValueIndex Index(const MLValueName& name) {
    MLValueIndex result;
    auto status = mlvalue_name_idx_map_.GetIdx(name, result);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return result;
  }

  int& UseCount(MLValueIndex n) { return ml_value_info_.at(n).usecount; }
  int& UseCount(const MLValueName& name) { return UseCount(Index(name)); }

  MLValueIndex& Buffer(MLValueIndex n) { return ml_value_info_.at(n).reused_buffer_index; }

  AllocPlanPerValue& AllocPlan(MLValueIndex n) {
    return plan_.allocation_plan.at(n);
  }

  AllocPlanPerValue& AllocPlan(const MLValueName& name) {
    return AllocPlan(Index(name));
  }

  // Initialize state for a given ml-value at its definition site:
  void ProcessDef(MLValueIndex id, const onnxruntime::NodeArg* p_def_site) {
    MLValueInfo& info = ml_value_info_.at(id);
    info.usecount = 0;
    info.reused_buffer_index = id;  // initially, no reuse; the ml-value uses its own buffer
    info.p_def_site = p_def_site;
  }

  void Reuse(MLValueIndex reused, MLValueIndex reused_for) {
    ORT_ENFORCE(reused != reused_for);
    // find original buffer underlying ml-value we want to reuse:
    MLValueIndex original = Buffer(reused);
    // record that the new buffer will reuse that original buffer
    Buffer(reused_for) = original;
    // adjust original buffer's usecount
    UseCount(original) += UseCount(reused_for);

    // update allocation plan (for use at execution-time)
    auto& symplan = AllocPlan(reused_for);
    symplan.alloc_kind = AllocKind::kReuse;
    symplan.reused_buffer = original;
  }

  // Find if there exists some input tensor that we can use in-place for output_arg
  bool FindAliasInput(const onnxruntime::Node& node, int output_arg_num, MLValueIndex* alias_input) {
    const KernelCreateInfo* ci;
    Status st = kernel_registry_.SearchKernelRegistry(node, &ci);
    if (!st.IsOK() || ci == nullptr || ci->kernel_def == nullptr) {
      return false;
    }

    const std::vector<std::pair<int, int>>& alias_map = ci->kernel_def->Alias();
    auto& input_args = node.InputDefs();
    for (auto pair : alias_map) {
      if (pair.second == output_arg_num) {
        // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for reshape)
        if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
          auto p_input_arg = input_args[pair.first];
          if (p_input_arg->Exists()) {
            *alias_input = Index(p_input_arg->Name());
            return true;
          }
        }
      }
    }

    return false;
  }

  bool FindInplaceInput(const onnxruntime::Node& node, int output_arg_num, MLValueIndex* inplace_input) {
    auto p_output_arg = node.OutputDefs()[output_arg_num];
    const KernelCreateInfo* ci;
    Status st = kernel_registry_.SearchKernelRegistry(node, &ci);
    if (!st.IsOK() || ci == nullptr || ci->kernel_def == nullptr) {
      return false;
    }

    const std::vector<std::pair<int, int>>& inplace_map = ci->kernel_def->MayInplace();
    auto& input_args = node.InputDefs();
    for (auto pair : inplace_map) {
      if (pair.second == output_arg_num) {
        if ((0 <= pair.first) && (static_cast<size_t>(pair.first) < input_args.size())) {
          auto p_input_arg = input_args[pair.first];
          if (p_input_arg->Exists()) {
            auto input_arg_index = Index(p_input_arg->Name());
            auto original = Buffer(input_arg_index);
            if (1 == UseCount(original)) {
              if (SameSize(*p_input_arg, *p_output_arg)) {
                // we can reuse this input since it is its last use and permitted for in-place update
                *inplace_input = input_arg_index;  // or original; both should be okay
                return true;
              }
            }
          }
        }
      }
    }

    return false;
  }

  bool FindReusableGraphInput(const onnxruntime::Node& node, int output_arg_num, MLValueIndex* reusable_input) {
    // skip if this is main graph
    if (parent_node_) {
      // only check a single layer. ie. an Identity/Dropout node connecting graph's input and output
      auto& graph_inputs = graph_viewer_.GetInputs();
      auto& graph_outputs = graph_viewer_.GetOutputs();

      if (FindAliasInput(node, output_arg_num, reusable_input)) {
        auto& arg = ml_value_info_.at(Buffer(*reusable_input)).p_def_site;
        auto it = std::find(graph_inputs.cbegin(), graph_inputs.cend(), arg);
        if (it != graph_inputs.end()) {
          auto graph_input_index = std::distance(graph_inputs.cbegin(), it);
          ORT_ENFORCE(graph_input_index >= 0 && static_cast<size_t>(graph_input_index) < graph_inputs.size());
          auto graph_output_index = std::distance(graph_outputs.cbegin(),
                                                  std::find(graph_outputs.cbegin(), graph_outputs.cend(), node.OutputDefs()[output_arg_num]));
          ORT_ENFORCE(graph_output_index >= 0 && static_cast<size_t>(graph_output_index) < graph_outputs.size());

          if (parent_node_->OpType() == "Loop") {
            // the matching carried parameters
            return graph_input_index >= 2;
          } else {
            // TODO: other operators like Scan
          }
        }
      }
    }

    return false;
  }

  bool SameShape(const TensorShapeProto& shape1, const TensorShapeProto& shape2) {
    // TODO: This should probably be defined to be the equality operator on TensorShapeProto.
    int rank1 = shape1.dim_size();
    if (shape2.dim_size() != rank1) return false;
    for (int i = 0; i < rank1; i++) {
      const auto& val1 = shape1.dim(i);
      const auto& val2 = shape2.dim(i);
      if (val1.has_dim_value() && val2.has_dim_value() && (val1.dim_value() == val2.dim_value()))
        continue;  // same known dimension
      if (val1.has_dim_param() && val2.has_dim_param() && (val1.dim_param() == val2.dim_param()))
        continue;  // same unknown dimension
      return false;
    }
    return true;
  }

  /*! \brief Given a tensor-type, return the size of an element of the tensor.
  */
  size_t GetElementSize(const DataType& tensor_type) {
    const TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(tensor_type);
    MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
    const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
    ORT_ENFORCE(nullptr != tensor_type_base);
    MLDataType elt_type = tensor_type_base->GetElementType();
    return elt_type->Size();
  }

  bool SameSize(const TensorShapeProto& shape1, const DataType& ptype1,
                const TensorShapeProto& shape2, const DataType& ptype2) {
    return (GetElementSize(ptype1) == GetElementSize(ptype2)) && SameShape(shape1, shape2);

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
    auto p_shape1 = context_.GetShape(arg1);
    auto p_shape2 = context_.GetShape(arg2);
    // If the shapes are unknown, we conservatively assume they may be of different size.
    if ((nullptr == p_shape1) || (nullptr == p_shape2)) return false;
    return SameSize(*p_shape1, arg1.Type(), *p_shape2, arg2.Type());
  }

  // Find if freelist contains a buffer of the same size as output_arg
  bool FindReusableTensor(const onnxruntime::NodeArg& output_arg, MLValueIndex* reusable_tensor) {
    auto p_required_buffer_shape = context_.GetShape(output_arg);
    if (nullptr == p_required_buffer_shape) return false;
    auto required_buffer_type = output_arg.Type();
    auto& required_allocator_info = AllocPlan(output_arg.Name()).location;

    for (auto it = freelist_.begin(); it != freelist_.end(); ++it) {
      auto reusable = it->ml_value;
      auto p_node_arg = ml_value_info_.at(reusable).p_def_site;
      auto& available_allocator_info = AllocPlan(p_node_arg->Name()).location;
      if (!(available_allocator_info == required_allocator_info)) continue;
      auto p_available_buffer_shape = context_.GetShape(*p_node_arg);
      if (nullptr != p_available_buffer_shape) {
        auto available_buffer_type = p_node_arg->Type();
        if (SameSize(*p_available_buffer_shape, available_buffer_type,
                     *p_required_buffer_shape, required_buffer_type)) {
          *reusable_tensor = reusable;
          freelist_.erase(it);
          return true;
        }
      }
    }
    return false;
  }

  void Initialize(size_t num_graph_nodes, size_t num_ml_values) {
    // All ml-value indices must be in range 0 .. num_ml_values-1
    ml_value_info_.resize(num_ml_values);

    // Initialize execution plan:
    plan_.execution_plan.reserve(num_graph_nodes);

    // Initialize allocation plan:
    plan_.allocation_plan.resize(num_ml_values);
  }

  Status ComputeUseCounts() {
    // Note: for every ml-value, its definition must appear before all its uses in a topological sort of a valid model

    for (auto graph_input : graph_viewer_.GetInputs()) {
      MLValueIndex index = Index(graph_input->Name());
      ProcessDef(index, graph_input);
      UseCount(index)++;  // Models caller's usage post-inference; ensures it will not be reused.
    }

    for (auto node_arg : outer_scope_node_args_) {
      MLValueIndex index = Index(node_arg->Name());
      ProcessDef(index, node_arg);
      UseCount(index)++;  // ensure will not be re-used as this graph does not own the buffer
    }

    // All initializers should be treated as input
    for (const auto& pair : graph_viewer_.GetAllInitializedTensors()) {
      const auto& initializer_name = pair.first;
      MLValueIndex index = Index(initializer_name);
      ProcessDef(index, graph_viewer_.GetNodeArg(pair.first));
      UseCount(initializer_name)++;
    }

    for (SequentialExecutionPlan::NodeExecutionPlan& step : plan_.execution_plan) {
      auto pnode = graph_viewer_.GetNode(step.node_index);
      if (pnode == nullptr) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the node ", step.node_index);
      for (auto node_input : pnode->InputDefs()) {
        if (node_input->Exists())
          UseCount(node_input->Name())++;
      }

      for (auto node_input : pnode->ImplicitInputDefs()) {
        if (node_input->Exists())
          UseCount(node_input->Name())++;
      }
      // Identify where each output of this node should be allocated.
      // This is determined by the opkernel bound to the node.
      const KernelCreateInfo* kernel_create_info = nullptr;
      ORT_RETURN_IF_ERROR(kernel_registry_.SearchKernelRegistry(*pnode, &kernel_create_info));
      auto p_kernelDef = kernel_create_info->kernel_def.get();
      if (nullptr == p_kernelDef) {
        std::ostringstream errormsg;
        errormsg << "No suitable kernel definition found for op " << pnode->OpType();
        if (pnode->Op() != nullptr) {
          errormsg << "(" << pnode->Op()->since_version() << ")";
        }
        if (!pnode->Name().empty()) errormsg << " (node " << pnode->Name() << ")";
        return Status(ONNXRUNTIME, FAIL, errormsg.str());
      }

      auto exec_provider = execution_providers_.Get(*pnode);
      if (exec_provider == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Can not find the execution provider ",
                               pnode->GetExecutionProviderType());
      }

      auto& default_allocator_info = exec_provider->GetAllocator(0, OrtMemTypeDefault)->Info();
      auto& outputs = pnode->OutputDefs();
      auto num_outputs = outputs.size();

      for (size_t i = 0; i < num_outputs; ++i) {
        auto* node_output = outputs[i];
        if (node_output->Exists()) {
          MLValueIndex index = Index(node_output->Name());
          ProcessDef(index, node_output);
          ++UseCount(index);
          if (strcmp(default_allocator_info.name, CPU) != 0) {
            // By default, outputs of this node are allocated on the default device allocator,
            // except for outputs marked for allocation in MemoryType:
            auto memory_type = p_kernelDef->OutputMemoryType(i);
            if (memory_type == OrtMemTypeDefault) {
              AllocPlan(index).location = default_allocator_info;
            } else {
              AllocPlan(index).location = exec_provider->GetAllocator(0, memory_type)->Info();
            }
          }
        }
      }
      // if sync is needed, mark allocation plan as create_fence_if_async=true
      // note that the input arg may come from an execution provider (i.e. CPU) that does not support async,
      // in which case create_fence_if_async would be ignored when creating MLValue
      if (p_kernelDef->ExecQueueId() != 0) {
        pnode->ForEachDef([this](const onnxruntime::NodeArg& arg, bool /*is_input*/) {
          MLValueIndex index = Index(arg.Name());
          AllocPlan(index).create_fence_if_async = true;
        });
      }
    }

    for (auto graph_output : graph_viewer_.GetOutputs()) {
      UseCount(graph_output->Name())++;  // Models caller's usage post-inference; ensures it will not be reused.
    }

    return Status::OK();
  }

  // TODO: Don't generate plan for CPU tensors, which may get its memory from 'mmap(2)'
  Status GeneratePlanForWeights() {
    auto& weights = graph_viewer_.GetAllInitializedTensors();

    for (auto& node : graph_viewer_.Nodes()) {
      ORT_RETURN_IF_ERROR(onnxruntime::Node::ForEachWithIndex(node.InputDefs(), [this, &node, &weights](
                                                                                    const onnxruntime::NodeArg& def,
                                                                                    size_t index) {
        auto& def_name = def.Name();
        if (!weights.count(def_name)) return Status::OK();

        auto wt_index = Index(def_name);
        AllocPlanPerValue& thisplan = AllocPlan(wt_index);
        auto* p_provider = execution_providers_.Get(node);
        ORT_ENFORCE(p_provider);

        thisplan.alloc_kind = AllocKind::kAllocateStatically;
        const KernelCreateInfo* kernel_create_info;
        ORT_RETURN_IF_ERROR(kernel_registry_.SearchKernelRegistry(node, &kernel_create_info));
        if (kernel_create_info == nullptr || kernel_create_info->kernel_def == nullptr)
          return Status(ONNXRUNTIME, FAIL, "search kernel failed");  // shouldn't reach here
        if (MemTypeOnCpuExplicitly(kernel_create_info->kernel_def->InputMemoryType(index)))
          // weights are not output from any node, so it's OK to put its location on CPU provider
          thisplan.location =
              execution_providers_.Get(onnxruntime::kCpuExecutionProvider)->GetAllocator(0, OrtMemTypeDefault)->Info();
        else
          thisplan.location = p_provider->GetAllocator(0, OrtMemTypeDefault)->Info();

        return Status::OK();
      }));
    }
    return Status::OK();
  }

  Status ComputeReusePlan() {
    std::vector<SequentialExecutionPlan::NodeExecutionPlan>& execution_plan{plan_.execution_plan};

    // Identify allocation/deallocation plan for every ml-value

    auto setup_preexisting = [this](const NodeArg* node_arg) {
      auto input_index = Index(node_arg->Name());
      AllocPlanPerValue& thisplan = AllocPlan(input_index);
      thisplan.alloc_kind = AllocKind::kPreExisting;
      thisplan.value_type = utils::GetMLDataType(*node_arg);
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

    ORT_RETURN_IF_ERROR(GeneratePlanForWeights());

    for (size_t program_counter = 0; program_counter < execution_plan.size(); ++program_counter) {
      SequentialExecutionPlan::NodeExecutionPlan step = execution_plan[program_counter];
      auto pnode = graph_viewer_.GetNode(step.node_index);
      // graph outputs
      auto& graph_outputs = graph_viewer_.GetOutputs();
      // determine allocation for outputs of pnode
      int output_arg_num = 0;
      for (auto node_output : pnode->OutputDefs()) {
        if (!node_output->Exists()) continue;
        auto current = Index(node_output->Name());
        AllocPlan(current).value_type = utils::GetMLDataType(*node_output);
        MLValueIndex reused;
        if (std::find(graph_outputs.begin(), graph_outputs.end(), node_output) != graph_outputs.end() &&
            !FindReusableGraphInput(*pnode, output_arg_num, &reused)) {
          // node_output is graph's output, so we can't reuse intermedia buffer
          AllocPlan(current).alloc_kind = AllocKind::kAllocateOutput;
        } else if (IsNonTensor(*node_output)) {
          // we do not try sharing-optimization for non-tensors
          AllocPlan(current).alloc_kind = AllocKind::kAllocate;
        } else if (FindAliasInput(*pnode, output_arg_num, &reused) ||
                   FindInplaceInput(*pnode, output_arg_num, &reused)) {
          // Reuse one of this node's input buffers as the output buffer (for in-place update)
          Reuse(reused, current);
        } else if (!context_.EnableParallelExecution() && FindReusableTensor(*node_output, &reused)) {
          // Reuse an available (dead) buffer for this output, this is only for sequential execution.
          Reuse(reused, current);
        } else {
          // otherwise: allocate a new buffer for this output
          AllocPlan(current).alloc_kind = AllocKind::kAllocate;
        }
        output_arg_num++;
      }
      // determine if inputs of *pnode can be freed:
      for (auto node_input : pnode->InputDefs()) {
        if (node_input->Exists()) {
          auto& sym = node_input->Name();
          auto original = Buffer(Index(sym));
          if (0 == --UseCount(original))
            freelist_.push_front(FreeBufferInfo(original, program_counter));
        }
      }

      for (auto node_input : pnode->ImplicitInputDefs()) {
        if (node_input->Exists()) {
          auto& sym = node_input->Name();
          auto original = Buffer(Index(sym));
          if (0 == --UseCount(original))
            freelist_.push_front(FreeBufferInfo(original, program_counter));
        }
      }

      // determine if any outputs of *pnode are unused and can be freed:
      for (auto node_output : pnode->OutputDefs()) {
        if (node_output->Exists()) {
          auto& sym = node_output->Name();
          auto original = Buffer(Index(sym));
          if (0 == --UseCount(original))
            freelist_.push_front(FreeBufferInfo(original, program_counter));
        }
      }
    }
    return Status::OK();
  }

  // Convert information in a freelist (about which ml-value becomes free when) into
  // a deallocation plan in the format required in an ExecutionPlan
  void GenerateDeallocationPlan() {
    // Store (indices of) ml-values to be freed in plan->to_be_freed
    // Set plan->execution_plan[n].free_from_index/free_to_index for every n that must free some ml-value.

    plan_.to_be_freed.reserve(freelist_.size());
    bool has_prev_dealloc_point = false;
    size_t prev_dealloc_point = 0;
    //TODO: should be size_t
    int current = 0;  // current index into the to_be_freed vector

    // Copy all items from freelist to to_be_freed in reverse order
    for (auto it = freelist_.rbegin(), end = freelist_.rend(); it != end; ++it) {
      plan_.to_be_freed.push_back(it->ml_value);
      //
      if (it->deallocate_point != prev_dealloc_point) {
        if (has_prev_dealloc_point)
          plan_.execution_plan[prev_dealloc_point].free_to_index = current - 1;
        prev_dealloc_point = it->deallocate_point;
        has_prev_dealloc_point = true;
        plan_.execution_plan[prev_dealloc_point].free_from_index = current;
      }
      current++;
    }

    if (has_prev_dealloc_point)
      plan_.execution_plan[prev_dealloc_point].free_to_index = current - 1;
  }

  bool IsNonTensor(const onnxruntime::NodeArg& nodearg) {
    // TODO: unclear why we should go through a string-representation of type
    auto ptype = nodearg.Type();
    auto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(ptype);
    return !type_proto.has_tensor_type();
  }
};  // namespace onnxruntime

Status PlannerImpl::CreatePlan() {
  auto& p_graph_nodes = graph_viewer_.GetNodesInTopologicalOrder();

  auto num_ml_values = mlvalue_name_idx_map_.MaxIdx() + 1;

  Initialize(p_graph_nodes.size(), num_ml_values);

  // Determine execution order: we use the default topological sort order for now. We can later
  // explore more efficient orderings (from a memory usage perspective).
  for (auto n : p_graph_nodes) {
    plan_.execution_plan.emplace_back(n);
  }

  // compute use counts for all ml-values
  ORT_RETURN_IF_ERROR(ComputeUseCounts());

  // determine sharing/reuse among ml-values
  ORT_RETURN_IF_ERROR(ComputeReusePlan());

  // convert information in the freelist_ into a deallocation plan in required format
  GenerateDeallocationPlan();

  return Status::OK();
}

Status SequentialPlanner::CreatePlan(const Node* parent_node,
                                     const onnxruntime::GraphViewer& graph_viewer,
                                     const std::vector<const NodeArg*>& outer_scope_node_args,
                                     const ExecutionProviders& providers,
                                     const KernelRegistryManager& kernel_registry,
                                     const MLValueNameIdxMap& mlvalue_name_idx_map,
                                     const ISequentialPlannerContext& context,
                                     std::unique_ptr<SequentialExecutionPlan>& plan) {
  // allocate/reset here so we know it's clean
  plan = std::make_unique<SequentialExecutionPlan>();

  PlannerImpl planner(parent_node, graph_viewer, outer_scope_node_args,
                      providers, kernel_registry, mlvalue_name_idx_map, context, *plan);

  return planner.CreatePlan();
}

}  // namespace onnxruntime
