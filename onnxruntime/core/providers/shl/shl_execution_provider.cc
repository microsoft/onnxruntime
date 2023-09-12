// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "onnx_converter.h"
#include "shl_execution_provider.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/memcpy.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

constexpr const char* SHL = "SHL";
constexpr const char* SHL_CPU = "SHLCpu";

namespace shl_ep {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kShlExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kShlExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kShlExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kShlExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

Status RegisterShlKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kShlExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kShlExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetShlKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterShlKernels(*kernel_registry));

  return kernel_registry;
}
}  // namespace shl_ep

struct ShlFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  std::string node_name;
};

ShlExecutionProvider::ShlExecutionProvider(const std::unordered_map<std::string, std::string>& config)
    : IExecutionProvider{onnxruntime::kShlExecutionProvider}, config_(config) {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(SHL, OrtAllocatorType::OrtDeviceAllocator));
      },
      0};

  CreateAllocator(default_memory_info);

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(SHL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }};

  CreateAllocator(cpu_memory_info);
}

std::vector<std::vector<int>> ShlExecutionProvider::GetSupportedNodes(
    const onnxruntime::GraphViewer& graph_viewer) const {
  auto all_fusible_nodes = shl_ep::GetAllFusionNode(shl_ep::MarkfusibleNodes(graph_viewer));
  return shl_ep::GetSupportedNodes(graph_viewer, all_fusible_nodes);
}

std::vector<std::unique_ptr<ComputeCapability>>
ShlExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/) const {
  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Handle If and Loop operators
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() &&
        tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "Shl: Initializers with external data"
                               " location are not currently supported";
      return result;
    }
  }

  std::set<const NodeArg*> all_node_inputs;
  for (const auto& node : graph_viewer.Nodes()) {
    for (const auto input : node.InputDefs()) {
      all_node_inputs.insert(input);
    }
  }
  const auto graph_outputs = graph_viewer.GetOutputs();

  const auto supported_nodes_vector = GetSupportedNodes(graph_viewer);

  int counter = 0;

  for (const auto& group : supported_nodes_vector) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(index);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph =
          std::make_unique<IndexedSubGraph>();
      // Find inputs and outputs of the subgraph
      std::unordered_map<const NodeArg*, int>
          fused_inputs, fused_outputs, fused_outputs_to_add;
      std::unordered_set<const NodeArg*> erased;
      int input_order = 0;
      int output_order = 0;

      for (const auto& index : group) {
        sub_graph->nodes.push_back(index);
        const auto& node = graph_viewer.GetNode(index);

        for (const auto& input : node->InputDefs()) {
          const auto& it = fused_outputs.find(input);

          if (it != fused_outputs.end()) {
            fused_outputs.erase(it);
            erased.insert(input);
          }
          // only when input is neither in output list nor erased list, add the
          // input to input list
          else if (erased.find(input) == erased.end()) {
            fused_inputs[input] = input_order++;
          }
        }

        // For output searching, there is a special case:
        // If node's OutputEdges are more than its outputs, meaning certain
        // output is used more than once,
        // if the output is connected to nodes that don't belong to the
        // subgraph, the output need to be added to the output list
        if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
          for (auto it = node->OutputEdgesBegin(),
                    end = node->OutputEdgesEnd();
               it != end; ++it) {
            const auto& node_idx = it->GetNode().Index();
            const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

            if (node_set.find(node_idx) != node_set.end()) {
              const auto& iter = fused_inputs.find(output);

              if (iter != fused_inputs.end()) {
                fused_inputs.erase(iter);
                erased.insert(output);
              } else if (erased.find(output) == erased.end()) {
                fused_outputs[output] = output_order++;
              }
            } else {
              fused_outputs_to_add[output] = output_order++;
            }
          }
        } else {
          for (const auto& output : node->OutputDefs()) {
            const auto& it = fused_inputs.find(output);

            if (it != fused_inputs.end()) {
              fused_inputs.erase(it);
              erased.insert(output);
            }
            // only when output is neither in input list nor erased list,
            // add the output to output list
            else if (erased.find(output) == erased.end()) {
              fused_outputs[output] = output_order++;
            }
          }
        }
      }

      fused_outputs.insert(
          fused_outputs_to_add.begin(), fused_outputs_to_add.end());

      // Sort inputs and outputs by the order they were added
      std::map<int, const NodeArg*> inputs, outputs;

      for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
        if (it->first->Type() != nullptr)
          inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      for (auto it = fused_outputs.begin(),
                end = fused_outputs.end();
           it != end; ++it) {
        for (const auto& x : all_node_inputs) {
          if (x->Name() == it->first->Name()) {
            outputs.insert(
                std::pair<int, const NodeArg*>(it->second, it->first));
            break;
          }
        }
        if (std::find(graph_outputs.begin(),
                      graph_outputs.end(), it->first) != graph_outputs.end()) {
          outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
        }
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def =
          std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
      meta_def->name = "SHL_" + std::to_string(counter++);
      meta_def->domain = kMSDomain;

      for (const auto& input : inputs) {
        meta_def->inputs.push_back(input.second->Name());
      }

      for (const auto& output : outputs) {
        meta_def->outputs.push_back(output.second->Name());
      }

      meta_def->since_version = 1;
      sub_graph->SetMetaDef(std::move(meta_def));

      result.push_back(
          std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

Status ShlExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                     std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_view = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    // build shl tensor and graph
    csinn_session* current_sess = csinn_alloc_session();
    shl_ep::OnnxToShlConverter converter(current_sess, config_);
    converter.Convert(graph_view);

    std::unordered_map<std::string, size_t> names2index;
    const auto& input_defs = fused_node.InputDefs();
    names2index.reserve(input_defs.size());
    for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
      names2index[input_defs[i]->Name()] = i;
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context,
                                         FunctionState* state) {
      std::unique_ptr<ShlFuncState> p = std::make_unique<ShlFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, context->node_name};
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [current_sess](FunctionState state) {
      if (state) {
        ShlFuncState* p = static_cast<ShlFuncState*>(state);
        csinn_session_deinit(current_sess);
        csinn_free_session(current_sess);
        delete p;
      }
    };

    compute_info.compute_func = [=](FunctionState state,
                                    const OrtApi* /*api*/,
                                    OrtKernelContext* context) {
      auto get_shape_from_shl_tensor = [](const csinn_tensor* tensor) -> std::vector<int64_t> {
        std::vector<int64_t> shape(tensor->dim_count);
        std::transform(tensor->dim, tensor->dim + tensor->dim_count, shape.begin(), [](int32_t val) -> int64_t {
          return static_cast<int64_t>(val);
        });
        return shape;
      };

      Ort::KernelContext ctx(context);
      const size_t n_outputs = ctx.GetOutputCount();
      int input_num = csinn_get_input_number(current_sess);

      for (int i = 0; i < input_num; i++) {
        csinn_tensor* shl_input = csinn_alloc_tensor(current_sess);
        csinn_get_input(i, shl_input, current_sess);
        size_t index = names2index.at(shl_input->name);
        auto input_tensor = ctx.GetInput(index);
        std::vector<int64_t> shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (shl_input->dim_count == 0) {
          current_sess->dynamic_shape = true;
        }
        if (current_sess->dynamic_shape) {
          shl_ep::UpdateShlTensorDim(shl_input, shape);
        }
        shl_input->data = const_cast<void*>(input_tensor.GetTensorRawData());
        csinn_update_input(i, shl_input, current_sess);
        csinn_free_tensor(shl_input);
      }

      csinn_session_run(current_sess);

      for (size_t i = 0; i < n_outputs; i++) {
        csinn_tensor* shl_output = csinn_alloc_tensor(current_sess);
        csinn_get_output(i, shl_output, current_sess);
        auto output_shape = get_shape_from_shl_tensor(shl_output);
        const auto output_tensor = ctx.GetOutput(i, output_shape);
        void* output_buf = const_cast<void*>(output_tensor.GetTensorRawData());
        int out_size = csinn_tensor_byte_size(shl_output);
        memcpy(output_buf, shl_output->data, out_size);
        shl_mem_free(shl_output->data);
        csinn_free_tensor(shl_output);
      }
      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> ShlExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = shl_ep::GetShlKernelRegistry();
  return kernel_registry;
}

}  // namespace onnxruntime
