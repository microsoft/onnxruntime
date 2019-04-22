// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include "mkldnn_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/memcpy.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/mkldnn/subgraph/mkldnn_custom_op.h"
#include "mkldnn_fwd.h"

namespace onnxruntime {

constexpr const char* MKLDNN = "MklDnn";
constexpr const char* MKLDNN_CPU = "MklDnnCpu";

namespace mkl_dnn {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().OutputMemoryType<OrtMemTypeCPUOutput>(0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

}  // namespace mkl_dnn

MKLDNNExecutionProvider::MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& /*info*/)
    : IExecutionProvider{onnxruntime::kMklDnnExecutionProvider} {
  DeviceAllocatorRegistrationInfo default_allocator_info({OrtMemTypeDefault,
                                                          [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(MKLDNN, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault)); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(default_allocator_info));

  DeviceAllocatorRegistrationInfo cpu_allocator_info({OrtMemTypeCPUOutput,
                                                      [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(MKLDNN_CPU, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeCPUOutput)); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(cpu_allocator_info));
}

MKLDNNExecutionProvider::~MKLDNNExecutionProvider() {
}

Status MKLDNNExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  // Support CPU <-> MKLDNN for now
  if (!(strcmp(src.Location().name, MKLDNN) == 0 && strcmp(dst.Location().name, CPU) == 0) &&
      !(strcmp(src.Location().name, CPU) == 0 && strcmp(dst.Location().name, MKLDNN) == 0) &&
      !(strcmp(src.Location().name, MKLDNN) == 0 && strcmp(dst.Location().name, MKLDNN_CPU) == 0)) {
    ORT_NOT_IMPLEMENTED(src.Location().name, " copy to ", dst.Location().name, " is not implemented");
  }

  // Todo: Copy for now. May optimize later to avoid copy.
  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  memcpy(dst_data, src_data, bytes);

  return Status::OK();
}

namespace mkl_dnn {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Sum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, 8, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 8, 8, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, float, LRN);

void RegisterMKLDNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, 8, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 8, 8, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, float, LRN)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetMklDnnKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterMKLDNNKernels(*kernel_registry);
  return kernel_registry;
}
}  // namespace mkl_dnn

std::vector<std::unique_ptr<ComputeCapability>> MKLDNNExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // switch between mkldnn-vanilla and mkldnn-subgraph implementation using
  // MKLDNN_SUBGRAPH environment variable
  const char* env = getenv("ORT_MKLDNN_SUBGRAPH");
  int use_subgraph = 0;
  if (env != nullptr)
    use_subgraph = atoi(env);

  if (use_subgraph == 0) {
    for (auto& node : graph_viewer.Nodes()) {
      for (auto registry : kernel_registries) {
        if (registry->TryFindKernel(node, Type()) != nullptr) {
          std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
          sub_graph->nodes.push_back(node.Index());
          result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
          break;
        }
      }
    }
    return result;
  }

  // use sub-graph implementation
  std::vector<std::string> mkl_ops = {"Conv", "BatchNormalization", "Relu", "Sum",
                                      "AveragePool", "GlobalMaxPool", "GlobalAveragePool", "MaxPool", "LRN"};
  SubgraphVariables sub_var;
  // output name to node index map. Using it to find sub-graph end nodes
  // if output of a node is not an input to any node in a sub-graph is end node
  std::map<std::string, int> output_to_source_node_map;
  NodeAttributes subgraph_attributes;
  int node_index = 0;

  while (node_index < graph_viewer.MaxNodeIndex()) {
    auto node = graph_viewer.GetNode(node_index);
    std::vector<std::string>::iterator it = std::find(
        mkl_ops.begin(), mkl_ops.end(), node->OpType());

    if (it != mkl_ops.end()) {
      sub_var.subgraph_node_indexes.push_back(node->Index());

      auto& node_inputs = node->InputDefs();
      sub_var.outputs.push_back(node->OutputDefs()[0]->Name());

      // can we fuse (at mkldnn level) nodes?
      bool fused = false;
      if (sub_var.subgraph_node_indexes.size() > 1 && node->OpType() == "Relu") {
        if (sub_var.subgraph_ptr->mklnodes.back().name == "BatchNormalization" || sub_var.subgraph_ptr->mklnodes.back().name == "Conv") {
          sub_var.subgraph_ptr->mklnodes.back().name += "-Relu";
          fused = true;
        }
      }

      if (!fused) {
        MklNode mklnode;
        mklnode.name = node->OpType();
        mklnode.num_inputs = static_cast<int>(node->InputDefs().size());
        mklnode.input_start_index = static_cast<int>(sub_var.inputs.size()) - 1;
        mklnode.node_index = static_cast<int>(sub_var.subgraph_ptr->mklnodes.size()) + 1;
        auto& node_outputs = node->OutputDefs();
        mklnode.output_name = node_outputs[0]->Name();
        if (node->OpType() == "Conv") {
          mklnode.weight_name = node->InputDefs()[1]->Name();
        }
        for (int i = 0; i < node_inputs.size(); i++) {
          auto iter = output_to_source_node_map.find(node_inputs[i]->Name());
          if (iter != output_to_source_node_map.end())
            mklnode.parent_nodes.push_back(iter->second);
        }
        sub_var.subgraph_ptr->mklnodes.push_back(mklnode);
        output_to_source_node_map.insert(std::make_pair(node_outputs[0]->Name(), static_cast<int>(sub_var.subgraph_ptr->mklnodes.size() - 1)));
      } else {
        auto& node_outputs = node->OutputDefs();
        output_to_source_node_map.erase(sub_var.subgraph_ptr->mklnodes.back().output_name);
        sub_var.subgraph_ptr->mklnodes.back().output_name = node_outputs[0]->Name();
        output_to_source_node_map.insert(std::make_pair(node_outputs[0]->Name(), static_cast<int>(sub_var.subgraph_ptr->mklnodes.size() - 1)));
      }

      // Add inputs which are not in the outputs vector.
      for (int i = 0; i < node_inputs.size(); i++) {
        auto itr = std::find(sub_var.outputs.begin(), sub_var.outputs.end(), node_inputs[i]->Name());
        if (itr == sub_var.outputs.end()) {
          sub_var.inputs.push_back(node_inputs[i]->Name());
        } else {
          // Vector of node outputs, which is input to other node
          // if node output is not input to any other node, then it's the end node
          // which we will find later
          sub_var.outputs_as_input_other_node.push_back(node_inputs[i]->Name());
        }
      }

      NodeAttributes attributes = node->GetAttributes();
      if (attributes.size() > 0) {
        int index = static_cast<int>(sub_var.subgraph_ptr->mklnodes.size());

        for (auto attIt = attributes.begin(); attIt != attributes.end(); ++attIt) {
          std::string key = node->OpType() + "-" + std::to_string(index) + "-" + attIt->first;
          std::pair<std::string, ONNX_NAMESPACE::AttributeProto> att(key, attIt->second);
          subgraph_attributes[key] = attIt->second;
        }
      }

      auto temp_index = node_index + 1;
      if (temp_index < graph_viewer.MaxNodeIndex()) {
        if (sub_var.subgraph_node_indexes.size() > 0) {
          // if next node is mkldnn node and if it's input is not output of current node
          //   if next node input is output of any of the nodes in sub-graph continue
          // else
          //   break and create sub-graph
          auto next_node = graph_viewer.GetNode(temp_index);
          std::vector<std::string>::iterator sub_it = std::find(
              mkl_ops.begin(), mkl_ops.end(), next_node->OpType());
          if (sub_it != mkl_ops.end()) {
            auto& next_node_inputs = next_node->InputDefs();
            bool input_from_subgraph = true;
            int inputs_count = 1;
            if (next_node->OpType() == "Sum")
              inputs_count = static_cast<int>(next_node_inputs.size());
            for (int i = 0; i < inputs_count; i++) {
              auto in = next_node_inputs[i];
              auto itr = std::find(sub_var.outputs.begin(), sub_var.outputs.end(), in->Name());
              if (itr == sub_var.outputs.end()) {
                input_from_subgraph = false;
              }
            }
            if (input_from_subgraph == false) {
              CreateMetaDef(sub_var, subgraph_attributes, result);
              subgraph_attributes.clear();
              output_to_source_node_map.clear();
            }
          }
        }
        if (sub_var.subgraph_node_indexes.size() > 0) {
          if (static_cast<int>(node->GetOutputEdgesCount()) > 1) {
            // If current node has branches
            //    iterate and see if all nodes are mkldnn ops OR
            //      it ends in node with same number of input edges (mkldnn node or cpu node)
            //      create sub-graph
            bool create_subgraph = false;
            bool break_loop = false;
            while (!break_loop) {
              if (temp_index > graph_viewer.MaxNodeIndex())
                break_loop = true;

              auto next_node = graph_viewer.GetNode(temp_index);
              if (next_node->GetInputEdgesCount() == node->GetOutputEdgesCount()) {
                // if all nodes in the branch loop are mkldnn nodes
                // then continue with adding nodes to sub-graph
                break_loop = true;
              }
              // inner nodes. if inner nodes are not  mkldnn nodes
              // create subgraph (inception v2)
              std::vector<std::string>::iterator sub_it = std::find(
                  mkl_ops.begin(), mkl_ops.end(), next_node->OpType());
              if (sub_it == mkl_ops.end()) {
                // break and create a sub-graph
                break_loop = true;
                create_subgraph = true;
              }
              temp_index++;
            }
            if (create_subgraph) {
              CreateMetaDef(sub_var, subgraph_attributes, result);
              subgraph_attributes.clear();
              output_to_source_node_map.clear();
            }
          }
        }
      }
    } else {
      if (sub_var.subgraph_node_indexes.size() >= 1) {
        CreateMetaDef(sub_var, subgraph_attributes, result);
        subgraph_attributes.clear();
      }
    }
    node_index++;
  }  // graph_viewer node iterator ends
  if (sub_var.subgraph_node_indexes.size() >= 1) {
    CreateMetaDef(sub_var, subgraph_attributes, result);
    subgraph_attributes.clear();
  }
  return result;
}

void MKLDNNExecutionProvider::CreateMetaDef(SubgraphVariables& sub_var, const NodeAttributes& subgraph_attributes,
                                            std::vector<std::unique_ptr<ComputeCapability>>& result) const {
  std::string graph_fused_nodes;
  std::string node_list;
  std::string subgraph_id = std::to_string(sub_var.subgraph_index);
  sub_var.subgraph_index++;

  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "MkldnnCustomOp" + std::to_string(sub_var.subgraph_index);
  meta_def->domain = kMklDnnExecutionProvider;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = sub_var.inputs;
  meta_def->attributes.insert(subgraph_attributes.begin(), subgraph_attributes.end());

  // Find the end nodes
  for (auto& mklnode : sub_var.subgraph_ptr->mklnodes) {
    auto itr = std::find(sub_var.outputs_as_input_other_node.begin(),
                         sub_var.outputs_as_input_other_node.end(), mklnode.output_name);
    if (itr == sub_var.outputs_as_input_other_node.end()) {
      meta_def->outputs.push_back(mklnode.output_name);
      mklnode.output_index = static_cast<int>(meta_def->outputs.size()) - 1;
    }
  }

  ONNX_NAMESPACE::AttributeProto ap;
  ap.set_s(subgraph_id);
  // ap.add_strings(node_list.c_str());
  meta_def->attributes["subgraph_id"] = ap;
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  sub_graph->nodes = sub_var.subgraph_node_indexes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  mkl_subgraphs_.insert(std::make_pair(subgraph_id, sub_var.subgraph_ptr));

  // Reset subgraph and meta_Def
  sub_var.Reset();
}

Status MKLDNNExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto* fused_node : fused_nodes) {
    auto attributes = fused_node->GetAttributes();
    NodeComputeInfo compute_info;

    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      auto* p = new onnxruntime::mkl_dnn::MkldnnCustomOp<float>(context, attributes, this);
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<onnxruntime::mkl_dnn::MkldnnCustomOp<float>*>(state);
    };

    compute_info.compute_func = [](FunctionState state, ONNXRunTimeTensor* input_tensors,
                                   size_t num_inputs, ONNXRunTimeTensor* output_tensors, size_t num_outputs) {
      onnxruntime::mkl_dnn::MkldnnCustomOp<float>* custom_op = reinterpret_cast<mkl_dnn::MkldnnCustomOp<float>*>(state);
      custom_op->Compute(input_tensors, num_inputs, output_tensors, num_outputs);

      return 0;
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> MKLDNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::mkl_dnn::GetMklDnnKernelRegistry();
  return kernel_registry;
}
}  // namespace onnxruntime
