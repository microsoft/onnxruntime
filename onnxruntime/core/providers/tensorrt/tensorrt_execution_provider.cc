// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorrt_execution_provider.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/framework/memcpy.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_fence.h"
#include "core/platform/env.h"
#include "core/common/status.h"
#include "onnx/shape_inference/implementation.h"
#include "cuda_runtime_api.h"
#include "gsl/gsl"
#include "core/graph/model.h"
#include "core/providers/cuda/gpu_data_transfer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .ExecQueueId(kCudaStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .ExecQueueId(kCudaStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static void RegisterTensorrtKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetTensorrtKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterTensorrtKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry> TensorrtExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::GetTensorrtKernelRegistry();
  return kernel_registry;
}

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& GetTensorrtLogger() {
  static TensorrtLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
  return trt_logger;
}

TensorrtExecutionProvider::TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTensorrtExecutionProvider}, device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));

  DeviceAllocatorRegistrationInfo default_allocator_info(
      {OrtMemTypeDefault, [](int id) { return std::make_unique<CUDAAllocator>(id, TRT); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(default_allocator_info, device_id_));

  DeviceAllocatorRegistrationInfo pinned_allocator_info(
      {OrtMemTypeCPUOutput, [](int) { return std::make_unique<CUDAPinnedAllocator>(0, TRT_PINNED); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(pinned_allocator_info, device_id_));
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {}

std::unique_ptr<IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index, const onnxruntime::GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.first.size());
  for (const auto& index : graph_nodes_index.first) {
    node_set.insert(node_index[index]);
  }
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

  // Find inputs and outputs of the subgraph
  std::unordered_map<const NodeArg *, int> fused_inputs, fused_outputs, fused_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  for (const auto& index : graph_nodes_index.first) {
    sub_graph->nodes.push_back(node_index[index]);
    const auto& node = graph.GetNode(node_index[index]);
    for (const auto& input : node->InputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        //only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    // For output searching, there is a special case:
    // If node's OutputEdges are more than its outputs, meaning certain output is used more than once,
    // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
    // to the output list
    if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
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
        // only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end()) {
          fused_outputs[output] = output_order++;
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());

  // Sort inputs and outputs by the order they were added
  std::multimap<int, const NodeArg *> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // Assign inputs and outputs to subgraph's meta_def
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "TRTKernel_" + std::to_string(kernels_index++);
  meta_def->domain = kMSDomain;

  for (const auto& input : inputs) {
    meta_def->inputs.push_back(input.second->Name());
  }

  for (const auto& output : outputs) {
    meta_def->outputs.push_back(output.second->Name());
  }

  meta_def->since_version = 1;
  sub_graph->SetMetaDef(meta_def);

  return sub_graph;
}

SubGraphCollection_t TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input, int iterations, const int max_iterations,
                                                                 const onnxruntime::GraphViewer& graph, bool* early_termination) const {
  // Return if iterations are exceeding predefined number
  SubGraphCollection_t nodes_list_output;
  if (iterations > max_iterations) {
    *early_termination = true;
    return nodes_list_output;
  }

  iterations++;
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  int counter = 0;
  for (const auto& group : nodes_vector_input) {
    //construct subgraph
    if (!group.first.empty()) {
      std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);

      if (group.second) {
        nodes_list_output.push_back(group);
      } else {
        onnxruntime::Model model_build(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
        onnxruntime::Graph& graph_build = model_build.MainGraph();

        //Add node and node args
        for (const auto& index : group.first) {
          const auto& node = graph.GetNode(node_index[index]);
          std::vector<onnxruntime::NodeArg *> inputs, outputs;
          for (auto input : node->InputDefs()) {
            auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
          }
          for (auto output : node->OutputDefs()) {
            auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
            outputs.push_back(&n_output);
          }
          graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
        }

        ORT_ENFORCE(graph_build.Resolve().IsOK());

        for (const auto& input : sub_graph->GetMetaDef()->inputs) {
          const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
          if (graph.GetInitializedTensor(input, initializer)) {
            graph_build.AddInitializedTensor(*initializer);
          }
        }

        // Serialize modelproto to string
        ONNX_NAMESPACE::ModelProto model_proto = model_build.ToProto();
        std::string string_buf;
        model_proto.SerializeToString(&string_buf);

        // Get supported node list recursively
        SubGraphCollection_t parser_nodes_list;
        TensorrtLogger& trt_logger = GetTensorrtLogger();
        auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));

        auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
        trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_list);

        SubGraphCollection_t next_nodes_list;
        const onnxruntime::GraphViewer graph_viewer(graph_build);
        next_nodes_list = GetSupportedList(parser_nodes_list, iterations, max_iterations, graph_viewer, early_termination);
        for (int i = 0, end = next_nodes_list.size(); i < end; ++i) {
          for (int j = 0, end = next_nodes_list[i].first.size(); j < end; ++j) {
            next_nodes_list[i].first[j] = group.first[next_nodes_list[i].first[j]];
          }
          nodes_list_output.push_back(next_nodes_list[i]);
        }
      }
    }
  }
  return nodes_list_output;
}

std::vector<std::unique_ptr<ComputeCapability>>
TensorrtExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                         const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Construct modelproto from graph
  onnxruntime::Model model(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
  onnxruntime::Graph& graph_build = model.MainGraph();
  for (const auto& node : graph.Nodes()) {
    std::vector<onnxruntime::NodeArg *> inputs, outputs;
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
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(status.IsOK(), status);
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize modelproto to string
  std::string string_buf;
  model_proto.SerializeToString(&string_buf);

  //save ModelProto to file
  int fd;
  Env::Default().FileOpenWr("trt_model_proto_getcap.onnx", fd);
  model_proto.SerializeToFileDescriptor(fd);

  //print out all nodes for debugging
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  int node_size = graph.NumberOfNodes();
  std::cout << "node size: " << node_size << std::endl;
  for (int index = 0; index < node_size; ++index){
      const auto& node = graph.GetNode(node_index[index]);
      std::cout << "node number: " << index << ", node_index[index]: " << node_index[index]
                << ", name: " << node->Name() << ", op type: " << node->OpType() << std::endl;
  }

  // Get supported node list
  SubGraphCollection_t parser_nodes_vector;
  TensorrtLogger& trt_logger = GetTensorrtLogger();
  auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));
  auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
  trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_vector);

  SubGraphCollection_t supported_nodes_vector;
  const char* batch_env = getenv("ORT_TENSORRT_MAX_PARSER_ITERATIONS");
  const int max_iterations = batch_env ? atoi(batch_env) : max_parser_iterations_;
  bool early_termination = false;
  supported_nodes_vector = GetSupportedList(parser_nodes_vector, 0, max_iterations, graph, &early_termination);
  if (early_termination) {
    supported_nodes_vector.clear();
  }

  //for static shape faster-rcnn: /home/steven/work/model/Faster-RCNN_fromYufeng/static_shapes_from_cpu/
  //supported_nodes_vector = {{{1, 2, 3, 4, 5, 6}, true}, {{ 8, 9, 10 }, true}, {{ 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 , 55 , 56 , 57 , 58 , 59 , 60 , 61 , 62 , 63 , 64 , 65 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 , 96 , 97 , 98 , 99 , 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213}, true}, {{ 216}, true}, {{ 218, 219, 220}, true}, {{ 222, 223, 224, 225}, true}, {{ 227, 228}, true}, {{ 230, 231, 232}, true}, {{ 234, 235, 236, 237, 238, 239, 240}, true}, {{ 242}, true}, {{ 244, 245, 246}, true}, {{ 248, 249, 250, 251}, true}, {{ 253, 254}, true}, {{ 256, 257, 258}, true}, {{ 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291}, true}, {{ 296}, true}, {{ 305, 306, 307, 308, 309}, true}, {{311, 312}, true}, {{ 314, 315}, true}, {{ 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339}, true}, {{ 342}, true}, {{ 344, 345, 346}, true}, {{ 348, 349, 350, 351}, true}, {{ 353, 354}, true}, {{ 356, 357, 358}, true}, {{ 360, 361, 362, 363, 364, 365, 366}, true}, {{ 368}, true}, {{ 370, 371, 372}, true}, {{ 374, 375, 376, 377}, true}, {{ 379, 380}, true}, {{ 382, 383, 384}, true}, {{ 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417}, true}, {{ 422}, true}, {{ 431, 432, 433, 434, 435}, true}, {{437, 438}, true}, {{ 440, 441}, true}, {{ 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465}, true}, {{ 468}, true}, {{ 470, 471, 472}, true}, {{ 474, 475, 476, 477}, true}, {{ 479, 480}, true}, {{ 482, 483, 484}, true}, {{ 486, 487, 488, 489, 490, 491, 492}, true}, {{ 494}, true}, {{ 496, 497, 498}, true}, {{ 500, 501, 502, 503}, true}, {{ 505, 506}, true}, {{ 508, 509, 510}, true}, {{ 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543}, true}, {{ 548}, true}, {{ 557, 558, 559, 560, 561}, true}, {{563, 564}, true}, {{ 566, 567}, true}, {{ 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591}, true}, {{ 594}, true}, {{ 596, 597, 598}, true}, {{ 600, 601, 602, 603}, true}, {{ 605, 606}, true}, {{ 608, 609, 610}, true}, {{ 612, 613, 614, 615, 616, 617, 618}, true}, {{ 620}, true}, {{ 622, 623, 624}, true}, {{ 626, 627, 628, 629}, true}, {{ 631, 632}, true}, {{ 634, 635, 636}, true}, {{ 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669}, true}, {{ 674}, true}, {{ 683, 684, 685, 686, 687}, true}, {{689, 690}, true}, {{ 692, 693}, true}, {{ 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717}, true}, {{ 720}, true}, {{ 722, 723, 724}, true}, {{ 726, 727, 728, 729}, true}, {{ 731, 732}, true}, {{ 734, 735, 736}, true}, {{ 738, 739, 740, 741, 742, 743, 744}, true}, {{ 746}, true}, {{ 748, 749, 750}, true}, {{ 752, 753, 754, 755}, true}, {{ 757, 758}, true}, {{ 760, 761, 762}, true}, {{ 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795}, true}, {{ 800}, true}, {{ 809, 810, 811, 812, 813}, true}, {{ 815, 816}, true}, {{ 818, 819}, true}, {{ 821, 822, 823}, true}, /*{{ 825}, true},*/ /*{{ 827, 828, 829, 830, 831}, true},*/ {{ 833}, true}, {{ 835}, true}, {{ 837}, true}, {{ 839, 840, 841}, true}, {{ 843}, true}, {{ 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856}, true}, {{ 858}, true}, /*{{ 863}, true},*/ {{ 876, 877, 878, 879, 880}, true}, {{ 882}, true}, {{ 888, 889, 890, 891, 892}, true}, {{ 894}, true}, {{ 900, 901, 902, 903, 904}, true}, {{ 906}, true}, {{ 908, 909, 910}, true}, {{ 912, 913, 914}, true}, {{ 919, 920}, true}, {{ 924, 925, 926}, true}, {{ 931, 932}, true}, {{ 936, 937, 938}, true}, {{ 943, 944}, true}, {{ 948, 949, 950}, true},{{961, 962, 963, 964, 965, 966}, true},{{968, 969, 970, 971, 972, 973}, true}, {{ 976}, true}, {{ 978, 979, 980}, true}, {{ 982, 983, 984, 985}, true}, {{ 987, 988}, true}, {{ 990, 991, 992}, true}, {{ 994, 995, 996, 997, 998, 999, 1000}, true}, {{ 1002}, true}, {{ 1004, 1005, 1006}, true}, {{ 1008, 1009, 1010, 1011}, true}, {{ 1013, 1014}, true}, {{ 1016, 1017, 1018}, true}, {{ 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032}, true}, {{1034}, true}, {{1036, 1037, 1038, 1039, 1040, 1041, 1042}, true}, {{ 1044, 1045, 1046, 1047}, true}, {{ 1049, 1050, 1051}, true}, {{ 1053, 1054}, true}, {{ 1062, 1063, 1064}, true}, {{ 1067, 1068, 1069, 1070, 1071}, true}, {{ 1073, 1074, 1075, 1076, 1077, 1078}, true}, {{ 1082, 1083, 1084}, true}, {{ 1086}, true}, {{ 1092, 1093, 1094, 1095, 1096}, true}};
  //supported_nodes_vector = {{{317}, true}};//317: gather_1001

  //opset10/mask-rcnn
  //supported_nodes_vector = {{{318}, true}};//reshape_829

  // Construct subgraph capability from node list
  std::vector<std::unique_ptr<ComputeCapability>> result;
  int counter = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

common::Status TensorrtExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
    std::vector<int> input_indexes;
    std::vector<int> input_dim_sizes;
    std::vector<int> output_indexes;
    std::vector<int> output_dim_sizes;
    std::unordered_map<int, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int> output_types;

    // Build map from input name to its index in input definitions
    std::unordered_map<std::string, int> input_map;
    const auto& input_defs = fused_node->InputDefs();
    input_map.reserve(input_defs.size());
    for (int i = 0, end = input_defs.size(); i < end; ++i) {
      input_map[input_defs[i]->Name()] = i;
    }

    // Build map from output name to its index in output definitions
    std::unordered_map<std::string, int> output_map;
    const auto& output_defs = fused_node->OutputDefs();
    output_map.reserve(output_defs.size());
    for (int i = 0, end = output_defs.size(); i < end; ++i) {
      output_map[output_defs[i]->Name()] = i;
    }

    // Reconstruct graph proto from fused node's function body
    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    *(model_proto.mutable_graph()) = graph_body.ToGraphProto();
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string string_buf;
    model_proto.SerializeToString(&string_buf);

    //save ModelProto to file
    int fd;
    Env::Default().FileOpenWr("trt_model_proto_Compile_" + fused_node->Name() + ".onnx", fd);
    model_proto.SerializeToFileDescriptor(fd);

    // Create TensorRT engine
    TensorrtLogger& trt_logger = GetTensorrtLogger();
    auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));
    auto trt_config = unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
    auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
    trt_parser->parse(string_buf.data(), string_buf.size());

    const char* batch_env = getenv("ORT_TENSORRT_MAX_BATCH_SIZE");
    if (batch_env) {
      const int max_batch_size = atoi(batch_env);
      SetMaxBatchSize(max_batch_size);
    }

    const char* workspace_env = getenv("ORT_TENSORRT_MAX_WORKSPACE_SIZE");
    if (workspace_env) {
      const size_t max_workspace_size = atoi(workspace_env);
      SetMaxWorkspaceSize(max_workspace_size);
    }

    trt_builder->setMaxBatchSize(max_batch_size_);

    trt_config->setMaxWorkspaceSize(max_workspace_size_);

    //Set optimization profile for dynamic shapes
    bool opt_profile = false;
    auto trt_profile = trt_builder->createOptimizationProfile();
    for (unsigned int i = 0, end = trt_network->getNbInputs(); i < end; ++i) {
      auto input = trt_network->getInput(i);
      nvinfer1::Dims dims = input->getDimensions();
      if (input->isShapeTensor()) {  // shape tensor
        std::cout << "Compile: Create optimization profile for shape tensor: " << input->getName() << std::endl;
        int shapes_size = dims.nbDims;  //e.g. int64[3], shapes_size=3
        std::vector<int32_t> shapes_min(shapes_size), shapes_opt(shapes_size), shapes_max(shapes_size);
        for (int j = 0, end = shapes_size; j < end; ++j) {
          shapes_min[j] = 1;
          shapes_opt[j] = 1000;
          shapes_max[j] = 1000;
        }
        trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shapes_size);
        trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shapes_size);
        trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shapes_size);
        opt_profile = true;
      } else {  // execution tensor
        bool dynamic_shape = false;
        nvinfer1::Dims dims_min = dims;
        nvinfer1::Dims dims_opt = dims;
        nvinfer1::Dims dims_max = dims;
        for (int j = 0, end = dims.nbDims; j < end; ++j) {
          if (dims.d[j] == -1) {
            dims_min.d[j] = 1;
            dims_opt.d[j] = 1;//1000 , large batch size will cause out-of-memory on GTX1080
            dims_max.d[j] = 1;
            dynamic_shape = true;
          }
        }

        if (dynamic_shape) {
          std::cout << "Compile: Create optimization profile for dynamic shape: " << input->getName() << std::endl;
          trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_min);
          trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
          trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_max);
          opt_profile = true;
        }
      }
    }
    if (opt_profile) {
      std::cout << "Compile: Set optimization profile" << std::endl;
      trt_config->addOptimizationProfile(trt_profile);
    }

    auto trt_engine = unique_pointer<nvinfer1::ICudaEngine>(trt_builder->buildEngineWithConfig(*trt_network, *trt_config));
    if (trt_engine == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build Engine for fused node: " + fused_node->Name());
    }

    // Build TensorRT context
    auto trt_context = unique_pointer<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (trt_context == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build Execution Context for fused node: " + fused_node->Name());
    }

    // Get input shape and binding index
    int num_inputs = trt_network->getNbInputs();
    input_indexes.resize(num_inputs);
    input_dim_sizes.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      auto input = trt_network->getInput(i);
      const std::string& name = input->getName();
      size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());
      nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(bindingIndex));
      auto iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[bindingIndex] = iter->second;
      }
      size_t dim_size = 1;
      if (input->isShapeTensor()) {  // shape tensor
        for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
          input_shape_ranges[bindingIndex][j] = std::make_pair(INT_MAX, INT_MIN);
        }
      } else {
        for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
          if (dimensions.d[j] == -1) {
            input_shape_ranges[bindingIndex][j] = std::make_pair(INT_MAX, INT_MIN);
          }
          dim_size *= dimensions.d[j];
        }
        input_dim_sizes[bindingIndex] = dim_size;  //note: input_dim_sizes is invalid for dynamic shape
      }
    }

    // Get output shape and binding index
    int num_outputs = trt_network->getNbOutputs();
    output_indexes.resize(num_outputs);
    output_dim_sizes.resize(num_outputs);
    output_shapes.resize(num_outputs);
    output_types.resize(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      const std::string& name = trt_network->getOutput(i)->getName();
      size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());
      nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(bindingIndex));
      bindingIndex -= num_inputs;
      auto iter = output_map.find(name);
      if (iter != output_map.end()) {
        output_indexes[bindingIndex] = iter->second;
      }
      size_t dim_size = 1;
      for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
        output_shapes[bindingIndex].push_back(dimensions.d[j]);
        dim_size *= dimensions.d[j];
      }
      output_dim_sizes[bindingIndex] = dim_size;

      const auto& graph_output = model_proto.graph().output();
      const auto& tensor_type = graph_output[i].type().tensor_type();
      output_types[bindingIndex] = tensor_type.elem_type();
    }

    ORT_ENFORCE(trt_engine->getNbBindings() == (num_inputs + num_outputs));

    // Save engine, context and input/output info to map
    parsers_.emplace(fused_node->Name(), std::move(trt_parser));
    engines_.emplace(fused_node->Name(), std::move(trt_engine));
    contexts_.emplace(fused_node->Name(), std::move(trt_context));
    builders_.emplace(fused_node->Name(), std::move(trt_builder));
    networks_.emplace(fused_node->Name(), std::move(trt_network));
    configs_.emplace(fused_node->Name(), std::move(trt_config));
    input_info_[fused_node->Name()].push_back(input_indexes);
    input_info_[fused_node->Name()].push_back(input_dim_sizes);
    output_info_[fused_node->Name()].push_back(output_indexes);
    output_info_[fused_node->Name()].push_back(output_dim_sizes);
    output_info_[fused_node->Name()].push_back(output_types);
    input_shape_ranges_[fused_node->Name()] = input_shape_ranges;
    output_shapes_[fused_node->Name()] = output_shapes;

    // Create function state
    // TODO: remove default capture
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<TensorrtFuncState> p = std::make_unique<TensorrtFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, parsers_[context->node_name].get(),
            engines_[context->node_name].get(), contexts_[context->node_name].get(), builders_[context->node_name].get(),
            networks_[context->node_name].get(), configs_[context->node_name].get(), input_info_[context->node_name],
            output_info_[context->node_name], input_shape_ranges_[context->node_name], output_shapes_[context->node_name], &tensorrt_mu_};
      *state = p.release();
      return 0;
    };

    // Release function state
    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<TensorrtFuncState*>(state);
    };

    // Create compute function
    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      ///std::cout << "TRT compute" << std::endl;
      Ort::CustomOpApi ort{*api};
      TensorrtFuncState* trt_state = reinterpret_cast<TensorrtFuncState*>(state);
      const std::vector<int>& input_indexes = (trt_state->input_info)[0];
      const std::vector<int>& output_indexes = (trt_state->output_info)[0];
      const std::vector<int>& output_types = (trt_state->output_info)[2];

      int num_binding_inputs = input_indexes.size();
      int num_binding_outputs = output_indexes.size();
      int total_bindings = num_binding_inputs + num_binding_outputs;
      std::vector<void*> buffers(total_bindings);

      bool dynamic_shape = false;
      if (!trt_state->context->allInputDimensionsSpecified()) {
        dynamic_shape = true;
        std::cout << "This run has dynamic shape inputs" << std::endl;
      }

      //TRT6: statistics of input dimensions
      bool dimension_update = false;
      auto trt_profile = trt_state->builder->createOptimizationProfile();
      for (int i = 0, end = num_binding_inputs; i < end; ++i) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_indexes[i]);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);
        if (trt_state->input_shape_ranges.find(i) != trt_state->input_shape_ranges.end()) {
          // Dynamic tensor
          auto input = trt_state->network->getInput(i);  //TODO: check if getInput indexing is same with binding index
          std::cout << "input name: " << input->getName() << std::endl;
          nvinfer1::Dims dims = input->getDimensions();
          nvinfer1::Dims dims_min = dims;
          nvinfer1::Dims dims_opt = dims;
          nvinfer1::Dims dims_max = dims;

          nvinfer1::Dims dimensions = trt_state->context->getEngine().getBindingDimensions(static_cast<int>(i));
          for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
            if (trt_state->input_shape_ranges[i].find(j) != trt_state->input_shape_ranges[i].end()) {
              if (tensor_shape[j] < trt_state->input_shape_ranges[i][j].first) {  //update minimum dimension
                trt_state->input_shape_ranges[i][j].first = tensor_shape[j];
                dims_min.d[j] = tensor_shape[j];
                dimension_update = true;
              }
              if (tensor_shape[j] > trt_state->input_shape_ranges[i][j].second) {  //update maximum dimension
                trt_state->input_shape_ranges[i][j].second = tensor_shape[j];
                dims_max.d[j] = tensor_shape[j];
                dims_opt.d[j] = tensor_shape[j];  //note: set opt profile to max dimension for now
                dimension_update = true;
              }

              if (dimension_update) {
                std::cout << "Shape range updated: Binding input " << i << ", dimension " << j
                          << ": min: " << trt_state->input_shape_ranges[i][j].first
                          << ", max: " << trt_state->input_shape_ranges[i][j].second << std::endl;
              }
            }
          }

          if (dimension_update) {
            if (trt_state->context->getEngine().isShapeBinding(i)) {
              std::cout << "Compute: update optimization profile for shape tensor: " << input->getName() << std::endl;
              int shapes_size = dims_min.nbDims;
              std::vector<int32_t> shapes_min(shapes_size), shapes_opt(shapes_size), shapes_max(shapes_size);
              for (int j = 0, end = shapes_size; j < end; ++j) {
                shapes_min[j] = dims_min.d[j];
                shapes_opt[j] = dims_opt.d[j];
                shapes_max[j] = dims_max.d[j];
              }
              trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shapes_size);
              trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shapes_size);
              trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shapes_size);
            } else {
              std::cout << "Compute: update optimization profile for dynamic shape: " << input->getName() << std::endl;
              trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_min);
              trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
              trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_max);
            }
          }
        }
      }

      //regenerate engine and context
      if (dimension_update) {
        std::cout << "Compute: create new engine" << std::endl;
        trt_state->config->addOptimizationProfile(trt_profile);
        trt_state->engine = trt_state->builder->buildEngineWithConfig(*trt_state->network, *trt_state->config);
        ORT_ENFORCE(trt_state->engine != nullptr);

        trt_state->context = trt_state->engine->createExecutionContext();
        ORT_ENFORCE(trt_state->context != nullptr);
      }

      // Get batch size and allocate cuda memory for inputs
      for (int i = 0, end = num_binding_inputs; i < end; ++i) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_indexes[i]);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);

        //Set dynamic shapes
        nvinfer1::Dims dimensions = trt_state->context->getEngine().getBindingDimensions(static_cast<int>(i));
        if (dynamic_shape) {
          for (int j = 0, end = tensor_shape.size(); j < end; ++j)
            dimensions.d[j] = tensor_shape[j];
          trt_state->context->setBindingDimensions(i, dimensions);
        }

        auto tensor_type = ort.GetTensorElementType(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i] = const_cast<float*>(ort.GetTensorData<float>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i] = const_cast<int8_t*>(ort.GetTensorData<int8_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i] = const_cast<int32_t*>(ort.GetTensorData<int32_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
          int input_dim_size = 1;
          for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
            input_dim_size *= tensor_shape[j];
          }
          CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[i], input_dim_size * sizeof(int32_t)));
          cuda::Impl_Cast<int64_t, int32_t>(ort.GetTensorData<int64_t>(input_tensor), reinterpret_cast<int32_t*>(buffers[i]), input_dim_size);
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.");
        }
      }

      // Allocate CUDA memory for outputs
      std::vector<int> output_dim_size(num_binding_outputs, 1);
      std::vector<OrtValue*> output_tensor(num_binding_outputs, nullptr);
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        // Set dynamic shapes
        nvinfer1::Dims dimensions = trt_state->context->getBindingDimensions(static_cast<int>(i + num_binding_inputs));
        for (int j = 0, end = trt_state->output_shapes[i].size(); j < end; ++j) {
          trt_state->output_shapes[i][j] = dimensions.d[j];
        }

        int output_index = output_indexes[i];
        output_tensor[i] = ort.KernelContext_GetOutput(context, output_index, trt_state->output_shapes[i].data(), trt_state->output_shapes[i].size());

        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<float>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int8_t>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int32_t>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Allocate INT32 CUDA memory for INT64 output type because TensorRT doesn't fully support INT64
          for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
            output_dim_size[i] *= dimensions.d[j];
          }
          CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[i + num_binding_inputs], output_dim_size[i] * sizeof(int32_t)));
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP output onnx tensor data type: " + std::to_string(output_types[i]) + " not supported.");
        }
      }

      // Run TRT inference
      std::lock_guard<OrtMutex> lock(*(trt_state->tensorrt_mu_ptr));
      //trt_state->context->enqueueV2(&buffers[0], nullptr, nullptr);
      if (!trt_state->context->enqueueV2(&buffers[0], nullptr, nullptr)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TensorRT EP Execution Context Enqueue Failed.");
      }

      // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          cuda::Impl_Cast<int32_t, int64_t>(reinterpret_cast<int32_t*>(buffers[i + num_binding_inputs]), ort.GetTensorMutableData<int64_t>(output_tensor[i]), output_dim_size[i]);
        }
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime
