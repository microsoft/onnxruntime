// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/memcpy.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "migraphx_inc.h"
#include "migraphx_execution_provider.h"
#include "hip_allocator.h"
#include "gpu_data_transfer.h"
#include <fstream>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {


ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMiGraphXExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .ExecQueueId(kHipStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMiGraphXExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .ExecQueueId(kHipStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMiGraphXExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMiGraphXExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static void RegisterMiGraphXKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMiGraphXExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMiGraphXExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetMiGraphXKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterMiGraphXKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry> MiGraphXExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::GetMiGraphXKernelRegistry();
  return kernel_registry;
}

MiGraphXExecutionProvider::MiGraphXExecutionProvider(const MiGraphXExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMiGraphXExecutionProvider} {

  // Set GPU device to be used
  hipSetDevice(info.device_id);

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault, [](int id) { return onnxruntime::make_unique<HIPAllocator>(id, MIGRAPHX); }, std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id_);
  InsertAllocator(allocator_);


  DeviceAllocatorRegistrationInfo pinned_memory_info(
      {OrtMemTypeCPUOutput, [](int) { return onnxruntime::make_unique<HIPPinnedAllocator>(0, MIGRAPHX_PINNED); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(pinned_memory_info, device_id_));


  // create the target based on the device_id
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device_id_);
  std::set<std::string> valid_targets = {"gpu", "cpu"};
  if (valid_targets.count(info.target_device) == 0)
  {
    LOGS_DEFAULT(FATAL) << "Device " << info.target_device << " are not supported";    
  }

  t_ = migraphx::target(info.target_device.c_str());
}

AllocatorPtr MiGraphXExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

std::unique_ptr<onnxruntime::IDataTransfer> MiGraphXExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<onnxruntime::GPUDataTransfer>();
}

static bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      return true;
    default:
      return false;
  }
}

static bool get_migraphx_type(ONNXTensorElementDataType type, 
                              migraphx_shape_datatype_t &mgx_type)
{
  mgx_type = migraphx_shape_float_type;
  switch(type) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
      mgx_type = migraphx_shape_half_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
      mgx_type = migraphx_shape_float_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
      mgx_type = migraphx_shape_double_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
      mgx_type = migraphx_shape_int8_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
      mgx_type = migraphx_shape_int16_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
      mgx_type = migraphx_shape_int32_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
      mgx_type = migraphx_shape_int64_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
      mgx_type = migraphx_shape_uint8_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
      mgx_type = migraphx_shape_uint16_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
      mgx_type = migraphx_shape_uint32_type;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      mgx_type = migraphx_shape_uint64_type;
      break;
    default:
      LOGS_DEFAULT(WARNING) << "MiGraphx: unsupported data type " << type << ", fallback to CPU";
      LOGS_DEFAULT(WARNING) << "implementation" << std::endl;
      return false;
  }

  return true;
}

// Returns true only if op is in a mode that is not currently supported
static bool IsUnsupportedOpMode(const Node* node, const onnxruntime::GraphViewer& graph_viewer) {
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  if (optype == "MaxPool") {
    //MaxPool "indices" output is not currently supported.
    if (node->OutputDefs().size() > 1) {
      return true;
    }

    // ceil_mode and dilations attrs are not supported in MiGraphX
    const auto& attributes = node->GetAttributes();
    const auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() and ceil_attr->second.i() != 0) {
      return true;
    }

    auto dila_attr = attributes.find("dilations");
    if (dila_attr != attributes.end()) {
      auto dilas = dila_attr->second.ints();
      bool ret = std::all_of(dilas.begin(), dilas.end(), [](auto i) { return i == 1;});
      if (ret == false)
      {
        return true;
      }
    }

    // storage order 1 (column major format) is not supported
    const auto storage_order_attr = attributes.find("storage_order");
    if (storage_order_attr != attributes.end() and storage_order_attr->second.i() != 0)
    {
      return true;
    }

    // input can only have 4 dims
    const auto input_shape = node->InputDefs()[0]->Shape();
    if (input_shape->dim_size() != 4)
    {
      return true;
    }
  } else if (optype == "Pad") {
    // Pad is only supported only up to opset 10 (in opset 11 more inputs were added)
    if (node->InputDefs().size() > 1) {
      return true;
    }

    const auto& attributes = node->GetAttributes();
    // Pad only support constant mode
    const auto mode_attr = attributes.find("mode");
    if(mode_attr != attributes.end())
    {
      const auto mode = mode_attr->second.s();
      static const std::set<std::string> allowed_modes = {"constant"};

      return allowed_modes.count(mode) == 0;
    }
  } else if (optype == "Slice") {
    //Slice in opset 10 is currently not supported.
    //unsupported inputs: starts, ends, axes, steps
    if (node->InputDefs().size() > 1) {
      return true;
    }
    //MiGraphX does not properly handle the situation where any 
    //value of the "starts" attribute is higher than a corresponding 
    // value in the "ends"
    const auto& attributes = node->GetAttributes();
    if (attributes.count("starts") == 0 || attributes.count("ends") == 0) {
      return true;
    }

    const auto& starts = attributes.find("starts")->second.ints();
    const auto& ends = attributes.find("ends")->second.ints();
    for (int i = 0; i < starts.size(); ++i) {
      if (starts.Get(i) > ends.Get(i)) {
        return true;
      }
    }
  } else if (optype == "AveragePool") {
    // ceil_mode attribute is not supported in MiGraphX
    const auto& attributes = node->GetAttributes();
    const auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second.i() != 0) {
      return true;
    }

    // input can only have 4 dims
    const auto input_shape = node->InputDefs()[0]->Shape();
    if (input_shape->dim_size() != 4)
    {
      return true;
    }

    // migraphx does not support count_include_pad to be 1
    const auto cip_attr = attributes.find("count_include_pad");
    if (cip_attr != attributes.end() && cip_attr->second.i() != 0)
    {
      return true;
    }

    const auto ap_attr = attributes.find("auto_pad");
    static const std::set<std::string> allowed_pad_modes = {"SAME_UPPER", "NOTSET"};
    if (ap_attr != attributes.end())
    {
      // explicit pad should be symmetric in migraphx
      auto s_pad = ap_attr->second.s();
      auto pads_attr = attributes.find("pads");
      if (s_pad == "NOTSET")
      {
        if (pads_attr != attributes.end())
        {
          auto pads = pads_attr->second.ints();
          if (pads.size() != 4)
          {
            return true;
          }

          if ((pads[0] != pads[2]) || (pads[1] != pads[3]))
          {
            return true;
          }
        }
      }
      // either SAME_UPPER or SAME_LOWER
      else if (s_pad.find("SAME") != std::string::npos)
      {
        // pads cannot exist when auto_pad is same_upper or same_lower
        if (pads_attr != attributes.end())
        {
          return true;
        }

        // compute the padding size to see whether they are symmetric
        std::vector<int> strides = {1, 1};
        auto stride_attr = attributes.find("strides");
        if (stride_attr != attributes.end())
        {
          auto attr_strides = stride_attr->second.ints();
          std::copy(attr_strides.begin(), attr_strides.end(), strides.begin());
        }

        std::vector<int> kernel_lens = {1, 1};
        auto kernel_attr = attributes.find("kernel_shape");
        if (kernel_attr != attributes.end())
        {
          auto attr_k = kernel_attr->second.ints();
          std::copy(attr_k.begin(), attr_k.end(), kernel_lens.begin());
        }

        auto tensor_dims = input_shape->dim();
        std::vector<int> in_lens;
        std::transform(tensor_dims.begin(),
                       tensor_dims.end(),
                       std::back_inserter(in_lens),
                       [&](auto&& d) -> std::size_t {
                           if(d.has_dim_value())
                           {
                               return d.dim_value();
                           }
                           return 1;
                       });

        std::vector<int> out_lens(2);
        out_lens[0]  = (in_lens[2] + strides[0] - 1) / strides[0];
        out_lens[1]  = (in_lens[3] + strides[1] - 1) / strides[1];
        std::vector<int> explicit_pads(2);
        explicit_pads[0] = (out_lens[0] - 1) * strides[0] + kernel_lens[0] - in_lens[2];
        explicit_pads[1] = (out_lens[1] - 1) * strides[1] + kernel_lens[1] - in_lens[3];
        if ((explicit_pads[0] & 1) != 0 or (explicit_pads[1] & 1) != 0)
        {
          return true;
        }
      }
    }
  } else if (optype == "Expand") {
    // MiGraphX only supports constant shape input values
    const auto& shape_input = node->InputDefs()[1];
    return !graph_viewer.IsConstantInitializer(shape_input->Name(), true);
  } else if (optype == "Clip") {
    // MiGraphX only support opset6 with 1 input
    return (node->InputDefs().size() != 1);
  } else if (optype == "Reshape") {
    // MiGraphX only support opset6 with 1 input
    const auto& shape_arg = node->InputDefs()[1];
    return initializers.find(shape_arg->Name()) == initializers.end();
  } else if (optype == "Conv") {
    // input can only have 4 dims
    const auto input_shape = node->InputDefs()[0]->Shape();
    if (input_shape->dim_size() != 4)
    {
      return true;
    }
  } else if (optype == "ConstantOfShape") {
    const auto shape_arg = node->InputDefs()[0];
    return initializers.find(shape_arg->Name()) == initializers.end();
  }
  else if (optype == "ConvInteger") {
    if (node->InputDefs()[0]->Shape()->dim_size() != 4)
    {
      return true;
    }

    // migraphx can handle only two inputs
    if (node->InputDefs().size() != 2)
    {
      return true;
    }

    // only support int8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr)
    {
      return true;
    }

    if (input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8)
    {
      return true;
    }
  }
  else if (optype == "MatMulInteger") {
    // migraphx can handle only two inputs
    if (node->InputDefs().size() != 2)
    {
      return true;
    }

    // only support int8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr)
    {
      return true;
    }

    if (input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8)
    {
      return true;
    }
  }

  //Op doesn't fall into known any of unsupported modes.
  return false;
}

static bool IsNodeSupported(const std::set<std::string>& op_set,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeIndex node_idx) {
  const auto& node = graph_viewer.GetNode(node_idx);
  const auto& optype = node->OpType();
  const auto& domain = node->Domain();

  // Three types of checking:
  // 1. Check input and output data types are supported.
  // 2. Check op_type is implemented in migraphx
  // 3. Check the mode is implemented in migraphx
  // if 3. failed, call the constant folding capability in migraphx
  // to see whether some input parameters can be calculated statically

  // check data type
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
    are_types_supported &= IsTypeSupported(&node_arg);
  });

  if (!are_types_supported) {
    return false;
  }

  // whether an operator implemented in migraphx
  if (op_set.count(optype) == 0) {
    return false;
  }

  // check that some modes might not be supported in migraphx for some operators
  if (domain == kOnnxDomain && IsUnsupportedOpMode(node, graph_viewer)) {
    // not supported, then check the constant folding capability of migraphx
    // to see whether it is supported
    return false;
  }

  return true;
}

static void AppendNodesToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "MIGraphX_" + std::to_string(++op_counter);
  meta_def->domain = kMiGraphXDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  sub_graph->nodes = nodes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
}

static std::vector<NodeIndex>
GetUnsupportedNodeIndices(const GraphViewer& graph_viewer, /*out*/ std::unordered_set<std::string>& mgx_required_initializers) {
  // const auto mgx_supported_ops = GetMiGraphXSupportedOps();
  static std::set<std::string> mgx_supported_ops = {"Abs", "Acos", "Add", "ArgMax", "ArgMin", "Asin", "Atan",
      "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip", "Concat", "Constant", "ConstantFill",
      "ConstantOfShape", "Conv", "Cos", "Cosh", "Div", "Dropout", "Elu", "Erf", "Exp", "Expand", 
      "Flatten", "Floor", "GRU", "Gather", "Gemm", "GlobalAveragePool", "GlobalMaxPool", "Identity", "ImageScaler", 
      "InstanceNormalization", "LRN", "LSTM", "LeakyRelu", "Log", "LogSoftmax", "MatMul", "Max", "MaxPool", "Min", 
      "Mul", "Pad", "Pow", "RNN", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", 
      "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape", "Round", "Shape", 
      "Sigmoid", "Sign", "Sin", "Sinh", "Slice", "Softmax", "Sqrt", "Squeeze", "Sub", "Sum", "Tan", "Tanh", 
      "Transpose", "Unsqueeze"};

  std::vector<NodeIndex> unsupported_nodes_idx;
  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    if (IsNodeSupported(mgx_supported_ops, graph_viewer, node_idx)) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&mgx_required_initializers, &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                mgx_required_initializers.insert(node_arg.Name());
              } }, true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

// Returns a vector clusters(or node_idx). For each unsupported node, the graph
// is split into 3 parts. supported_cluster + (UNsupported_node + rest_of_the_graph). 
// This functions returns vector of all supported_subgraphx by amdmigraphx
static std::vector<std::vector<NodeIndex>>
GetPartitionedSubgraphs(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> mgx_subgraphx;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx) 
    // and append it to return list.
    std::vector<NodeIndex> this_subgraph{prev, it};
    if (!this_subgraph.empty()) {
      mgx_subgraphx.push_back(std::move(this_subgraph));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  // Tail
  std::vector<NodeIndex> this_subgraph{prev, topological_order.end()};
  if (!this_subgraph.empty()) {
    mgx_subgraphx.push_back(std::move(this_subgraph));
  }

  return mgx_subgraphx;
}

static void GetInputsOutputsOfSubgraph(const GraphViewer& graph_viewer,
                                       const std::vector<NodeIndex>& nodes,
                                       const std::unordered_set<std::string>& mgx_required_initializers,
                                       std::vector<std::string>& nodes_inputs,
                                       std::vector<std::string>& nodes_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;

  for (const auto& node_idx : nodes) {
    const auto& node = graph_viewer.GetNode(node_idx);  

    // Collect all inputs and outputs
    node->ForEachDef(
        [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg, bool is_input) {
          if (is_input) {
            if (!input_args.count(node_arg.Name())) {
              ordered_input_args.push_back(node_arg.Name());
            }
            input_args.insert(node_arg.Name());
          } else {
            output_args.insert(node_arg.Name());
          }
        },
        true);

    // Check if output of this node is used by nodes outside 
    // subgraph. If yes add this to cluster outputs
    for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
      const auto& ext_node = graph_viewer.GetNode((*it).Index());

      if (std::find(nodes.begin(), nodes.end(), ext_node->Index()) == nodes.end()) {
        // Node is external to subgraph. Search through its 
        // inputs to find the output that is generated by subgraph.
        std::set<std::string> ext_node_inputs;
        ext_node->ForEachDef(
            [&ext_node_inputs](const onnxruntime::NodeArg& arg, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(arg.Name());
              }
            },
            true);

        for (const auto& out_def : node->OutputDefs()) {
          if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
            external_output_args.insert(out_def->Name());
          }
        }
      }
    }
  }

  //Extract initializers used by subgraph.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  std::vector<std::string> const_inputs;
  for (const auto& in_arg : ordered_input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        mgx_required_initializers.count(in_arg)) {
      const_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : ordered_input_args) {
    if (!output_args.count(in_arg) &&
        !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        mgx_required_initializers.count(in_arg))) {
      nodes_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : const_inputs) {
    nodes_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(nodes_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      nodes_outputs.push_back(name);
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
MiGraphXExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {

  std::vector<std::unique_ptr<ComputeCapability>> result;
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() && tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "MIGraphX: Initializers with external data location are not currently supported";
      return result;
    }
  }

  // Construct modelproto from graph
  onnxruntime::Model model(graph_viewer.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(),
          graph_viewer.DomainToVersionMap(), std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
  onnxruntime::Graph& graph_build = model.MainGraph();
  for (const auto& node : graph_viewer.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }

  auto status = graph_build.Resolve();

  //Add initializer to graph
  std::size_t init_tensor_num = 0;
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    init_tensor_num++;
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(status.IsOK(), status);
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // migraphx now can only support on output. if there are multiple
  // outputs, we cannot support this model
  std::size_t num_outputs = model_proto.graph().output().size();
  if (num_outputs > 1)
  {
      LOGS_DEFAULT(WARNING) << "MIGraphX can support only one output, but input model";
      LOGS_DEFAULT(WARNING) << "has " << num_outputs << " outputs, so fall back to";
      LOGS_DEFAULT(WARNING) << "default CPU implementation!";

      return result;
  }

  // migraphx now cannot support inputs with dynamic shape
  std::size_t num_inputs = model_proto.graph().input_size();
  for (std::size_t in_index = 0; in_index < num_inputs; ++in_index)
  {
    auto in_node = model_proto.graph().input(in_index);
    const NodeArg* node_arg = graph_viewer.GetNodeArg(in_node.name());
    if (node_arg == nullptr) continue;
    auto&& type_as_proto = node_arg->TypeAsProto();
    auto& dims = type_as_proto->tensor_type().shape().dim();
    for (auto&& d : dims)
    {
      if (not d.has_dim_value())
      {
        LOGS_DEFAULT(WARNING) << "MiGraphX, model input " << in_node.name(); 
        LOGS_DEFAULT(WARNING) << "is dynamic shape, not supported. Fallback";
        LOGS_DEFAULT(WARNING) << "to default CPU execution!" << std::endl;

        return result;
      }
    }
  }

  std::string onnx_string_buffer;
  model_proto.SerializeToString(&onnx_string_buffer);

  // This is a list of initializers that migraphx considers as constants. 
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> mgx_required_initializers;
  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, mgx_required_initializers);

  //If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    //Fill inputs with names
    std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    // In scenarios, when there are no inputs or all inputs being initializers,
    // ConstantFolding optimization in onnxruntime pre-computes the value.
    if (inputs.empty()) {
      return result;
    }

    // Initializers need to be part of meta_def->inputs
    std::for_each(mgx_required_initializers.begin(), mgx_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    // Fill outputs with names
    std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Create and add this graph to result.
    AppendNodesToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), inputs, outputs, result);

  } else {  // unsupported_nodes_idx.empty()
    const auto mgx_clusters = GetPartitionedSubgraphs(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    for (const auto& this_cluster : mgx_clusters) {
      std::vector<std::string> cluster_inputs, cluster_outputs;
      GetInputsOutputsOfSubgraph(graph_viewer, this_cluster, mgx_required_initializers, cluster_inputs, cluster_outputs);

      if (!cluster_inputs.empty()) {
        AppendNodesToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
      }
    }
  }

  return result;
}

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node, 
        const logging::Logger& logger) {
  const auto* node_function = fused_node->GetFunctionBody();

  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ", fused_node->Name());

  const Graph& node_subgraph = node_function->Body();
  onnxruntime::Model model{node_subgraph.Name(), true, ModelMetaData{},
                           IOnnxRuntimeOpSchemaRegistryList{}, node_subgraph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger};

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  //model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_subgraph.ToGraphProto();

  auto opset = model_proto.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(node_subgraph.DomainToVersionMap().at(kOnnxDomain));

  return model_proto;
}

Status MiGraphXExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  // std::size_t fused_node_index = 0;
  for (const auto& fused_node : fused_nodes) {
    // map parameter input name to index
    std::unordered_map<std::string, std::size_t> input_name_index;
    const auto& input_defs = fused_node->InputDefs();
    input_name_index.reserve(input_defs.size());
    for (std::size_t i = 0; i < input_defs.size(); ++i) {
      input_name_index[input_defs[i]->Name()] = i;
    }

    // record name of each output
    std::unordered_map<std::string, std::size_t> output_name_index;
    const auto& output_defs = fused_node->OutputDefs();
    output_name_index.reserve(output_defs.size());
    for (std::size_t i = 0; i < output_defs.size(); ++i) {
      output_name_index[output_defs[i]->Name()] = i;
    }

    // FIXME later, hack
    output_name_index.clear();
    output_name_index["output"] = 0;

    // reconstruct the subgraph proto from fused nodes
    onnx::ModelProto model_proto = GetModelProtoFromFusedNode(fused_node, *GetLogger());
    std::string onnx_string_buffer;
    model_proto.SerializeToString(&onnx_string_buffer);

    // by parsing the model_proto, create a program corresponding to
    // the input fused_node
    migraphx::program prog = migraphx::parse_onnx_buffer(onnx_string_buffer.data(), onnx_string_buffer.size());

    // compile the program
    prog.compile(t_);
    map_progs_[fused_node->Name()] = prog;

    std::unordered_map<std::size_t, std::size_t> input_index_map;
    std::unordered_map<std::size_t, std::size_t> output_index_map;
    migraphx::program_parameter_shapes param_shapes = prog.get_parameter_shapes();
    std::size_t param_index = 0;
    for(auto&& name : param_shapes.names())
    {
      // process the input
      auto iit = input_name_index.find(name);
      if (iit != input_name_index.end())
      {
        input_index_map[param_index] = iit->second;
      }

      // process the output
      auto oit = output_name_index.find(name);
      if (oit != output_name_index.end())
      {
        output_index_map[param_index] = oit->second;
      }
      ++param_index;
    }

    map_input_index_[fused_node->Name()] = input_index_map;
    map_output_index_[fused_node->Name()] = output_index_map;

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<MiGraphXFuncState> p = onnxruntime::make_unique<MiGraphXFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, map_progs_[context->node_name], t_,
            map_input_index_[context->node_name], map_output_index_[context->node_name], &mgx_mu_};
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<MiGraphXFuncState*>(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      MiGraphXFuncState* mgx_state = reinterpret_cast<MiGraphXFuncState*>(state);
      std::unordered_map<std::size_t, std::size_t>& map_input_index = mgx_state->input_indexes;
      std::unordered_map<std::size_t, std::size_t>& map_output_index = mgx_state->output_indexes;
      migraphx::target t = mgx_state->t;
      migraphx::program& prog = mgx_state->prog;

      migraphx::program_parameter_shapes param_shapes = prog.get_parameter_shapes();
      migraphx::program_parameters m;

      std::size_t param_index = 0;
      for (auto&& name : param_shapes.names())
      {
        if (map_input_index.count(param_index) > 0)
        {
          const OrtValue* input_tensor = ort.KernelContext_GetInput(context, map_input_index[param_index]);
          auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
          const auto& tensor_shape = ort.GetTensorShape(tensor_info);
          auto tensor_type = ort.GetTensorElementType(tensor_info);
          ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

          migraphx_shape_datatype_t mgx_type;
          get_migraphx_type(tensor_type, mgx_type);
          auto mgx_s = param_shapes[name];

          if (mgx_type != mgx_s.type())
          {
            LOGS_DEFAULT(FATAL) << "MIGraphX: param type mismatch";
          }
          m.add(name, migraphx::argument(param_shapes[name], const_cast<void*>(ort.GetTensorData<void>(input_tensor))));
        }

        if (map_output_index.count(param_index) > 0)
        {
          std::size_t output_index = map_output_index.begin()->second;
          migraphx::shapes res_shapes = prog.get_output_shapes();
          auto lens = res_shapes[0].lengths();
          std::vector<int64_t> ort_shape(lens.begin(), lens.end());
          OrtValue* output_tensor = ort.KernelContext_GetOutput(context, output_index, ort_shape.data(), ort_shape.size());
          void* output_data = ort.GetTensorMutableData<void>(output_tensor);
          m.add("output", migraphx::argument(param_shapes["output"], output_data));
        } 
        param_index++;
      }

      {
        // lock to avoid race condition
        std::lock_guard<OrtMutex> lock(*(mgx_state->mgx_mu_ptr));
        auto prog_outputs = prog.eval(m);
        hipDeviceSynchronize();

        // there is no output in parameter, we need to explicitly copy
        // data from input buffer to output buffer
        auto&& param_names = param_shapes.names();
        if (std::find(param_names.begin(), param_names.end(), std::string("output")) == param_names.end())
        {
          auto gpu_res = prog_outputs[0];
          migraphx::shape res_shape = gpu_res.get_shape();
          auto res_lens = res_shape.lengths();
          std::vector<int64_t> ort_shape{res_lens.begin(), res_lens.end()};
          OrtValue* output_tensor = ort.KernelContext_GetOutput(context, 0, ort_shape.data(), ort_shape.size());
          void* output_data = ort.GetTensorMutableData<void>(output_tensor);
          hipMemcpy(output_data, gpu_res.data(), res_shape.bytes(), hipMemcpyDeviceToDevice);
        }
      } 

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

}  // namespace onnxruntime
