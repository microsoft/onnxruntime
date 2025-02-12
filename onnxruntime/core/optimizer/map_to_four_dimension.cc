// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/optimizer/map_to_four_dimension.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime::common;

namespace onnxruntime {

MapToFourDimensions::MapToFourDimensions() noexcept
    : GraphTransformer("MapToFourDimensions") {
}

/**
 * Replace Gemm/MatMul with Transpose and 1x1 Conv
 * 
 *    
 * 
 *                       (Transpose) 
 *                            |
 *                            |
 *  (MatMul)     =>         (Conv) --- (Unsqueeze) --- (Transpose) --- Weight
 *                            |
 *                            |
 *                       (Transpose)
 */
Status AddConv(Graph& graph, Node& matmul) {
  const auto* input_0 = matmul.InputDefs()[0];

  // if 2D MatMul
  if (input_0->Shape()->dim_size() == 4) {
    /**
     * One Conv is added and three Transposes + one Unsqueeze are added for the Conv.
     * 
     * One Transpose for MatMul's input, another Transpose for MatMul's output and the last Transpose + Unsqueeze for MatMul's weight. 
     * (Note: This weight -> Transpose -> Unsqueeze will be constant folded in another transforming run when applying ConstantFold transformer)
     */ 
    
    std::string conv_node_name = graph.GenerateNodeName(matmul.Name() + "_conv");
    std::string transpose_node_0_name = graph.GenerateNodeName(matmul.Name() + "_transpose_input");
    std::string transpose_node_1_name = graph.GenerateNodeName(matmul.Name() + "_transpose_output");
    std::string transpose_node_2_name = graph.GenerateNodeName(matmul.Name() + "_transpose_weight");
    std::string unsqueeze_node_name = graph.GenerateNodeName(matmul.Name() + "_unsqueeze_weight");

    // Create Conv, Transpose and Unsqueeze node's output args.
    // The output type should be the same as the MatMul node's output (going to be removed) type.
    auto* conv_node_output_arg = &graph.GetOrCreateNodeArg(conv_node_name, matmul.OutputDefs()[0]->TypeAsProto());
    auto* transpose_node_0_output_arg = &graph.GetOrCreateNodeArg(transpose_node_0_name, matmul.OutputDefs()[0]->TypeAsProto());
    auto* transpose_node_2_output_arg = &graph.GetOrCreateNodeArg(transpose_node_2_name, matmul.OutputDefs()[0]->TypeAsProto());
    auto* unsqueeze_node_output_arg = &graph.GetOrCreateNodeArg(unsqueeze_node_name, matmul.OutputDefs()[0]->TypeAsProto());

    std::vector<onnxruntime::NodeArg*> conv_node_input_defs = {transpose_node_0_output_arg, unsqueeze_node_output_arg};
    std::vector<onnxruntime::NodeArg*> conv_node_output_defs = {conv_node_output_arg};
    std::vector<onnxruntime::NodeArg*> transpose_node_0_input_defs = {matmul.MutableInputDefs()[0]};
    std::vector<onnxruntime::NodeArg*> transpose_node_0_output_defs = {transpose_node_0_output_arg};
    std::vector<onnxruntime::NodeArg*> transpose_node_1_input_defs = {conv_node_output_arg};
    std::vector<onnxruntime::NodeArg*> transpose_node_1_output_defs = {matmul.MutableOutputDefs()[0]};
    std::vector<onnxruntime::NodeArg*> transpose_node_2_input_defs = {matmul.MutableInputDefs()[1]};
    std::vector<onnxruntime::NodeArg*> transpose_node_2_output_defs = {transpose_node_2_output_arg};

    // Create Conv
    auto& conv_node = graph.AddNode(conv_node_name, "Conv", "Mapping from MatMul",
                                    conv_node_input_defs, conv_node_output_defs);
    // Create Transposes
    auto& transpose_node_0 = graph.AddNode(transpose_node_0_name, "Transpose", "For Conv's input",
                                           transpose_node_0_input_defs, transpose_node_0_output_defs);
    auto& transpose_node_1 = graph.AddNode(transpose_node_1_name, "Transpose", "For Conv's output",
                                           transpose_node_1_input_defs, transpose_node_1_output_defs);
    auto& transpose_node_2 = graph.AddNode(transpose_node_2_name, "Transpose", "For Conv's weight",
                                           transpose_node_2_input_defs, transpose_node_2_output_defs);

    std::vector<int64_t> perm = {0, 3, 2, 1};
    transpose_node_0.AddAttribute("perm", perm);
    transpose_node_1.AddAttribute("perm", perm);

    // Create input as OrtValue for Unsqueeze, i.e. the "axes" inputs.
    // The input will become the initializer.
    OrtMemoryInfo* mem_info = nullptr;
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    auto status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    std::unique_ptr<OrtMemoryInfo, decltype(ort_api->ReleaseMemoryInfo)> rel_info(mem_info, ort_api->ReleaseMemoryInfo);

    const int input_data_cnt = 2;
    int64_t squeeze_input_data_1[input_data_cnt] = {2, 3};
    const size_t input_len = input_data_cnt * sizeof(int64_t);
    const int64_t input_shape[] = {1};
    const size_t shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

    OrtValue* unsqueeze_input_1 = nullptr;
    ort_api->CreateTensorWithDataAsOrtValue(mem_info, squeeze_input_data_1, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &unsqueeze_input_1);
    const Tensor& unsqueeze_input_tensor_1 = unsqueeze_input_1->Get<Tensor>();
    ONNX_NAMESPACE::TensorProto unsqueeze_tensorproto_1 = utils::TensorToTensorProto(unsqueeze_input_tensor_1, unsqueeze_node_name + "_axes");


    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(unsqueeze_tensorproto_1.data_type());
    auto* unsqueeze_arg_1 = &graph.GetOrCreateNodeArg(unsqueeze_node_name + "_axes", &t);

    graph.AddInitializedTensor(unsqueeze_tensorproto_1);

    // Create Unsqueeze
    std::vector<onnxruntime::NodeArg*> unsqueeze_node_input_defs = {transpose_node_2_output_arg, unsqueeze_arg_1};
    std::vector<onnxruntime::NodeArg*> unsqueeze_node_output_defs = {unsqueeze_node_output_arg};
    auto& unsqueeze_node = graph.AddNode(unsqueeze_node_name, "Unsqueeze", "For Conv's weight",
                                         unsqueeze_node_input_defs, unsqueeze_node_output_defs);
  }
  return Status::OK();
}

/**
 * Replace Reshape node and ReduceSum node with
 * two Slice nodes, two ReduceSum nodes and one Concat node.
 */
Status AddSliceReduceSumConcat(Graph& graph, Node& reshape, Node& reduce_sum) {
  // Create Slice node names
  std::string slice_node_0_name = graph.GenerateNodeName(reshape.Name() + "_slice_0");
  std::string slice_node_1_name = graph.GenerateNodeName(reshape.Name() + "_slice_1");

  // Create Slice node's output arg.
  // The Slice node's output type should be the same as the Reshape node's output (going to be removed) type.
  auto* slice_node_0_arg = &graph.GetOrCreateNodeArg(slice_node_0_name, reshape.OutputDefs()[0]->TypeAsProto());
  auto* slice_node_1_arg = &graph.GetOrCreateNodeArg(slice_node_1_name, reshape.OutputDefs()[0]->TypeAsProto());
  
  // Create inputs as OrtValue for Slice nodes, i.e. the "start", "ends" and "axes" inputs.
  // The inputs will become initializers.
  OrtMemoryInfo* mem_info = nullptr;
  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  auto status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
  std::unique_ptr<OrtMemoryInfo, decltype(ort_api->ReleaseMemoryInfo)> rel_info(mem_info, ort_api->ReleaseMemoryInfo);

  const int input_data_cnt = 1;
  int64_t slice_0_input_data_1[input_data_cnt] = {0};
  int64_t slice_0_input_data_2[input_data_cnt] = {4};
  int64_t slice_0_input_data_3[input_data_cnt] = {3};
  int64_t slice_1_input_data_1[input_data_cnt] = {4};
  int64_t slice_1_input_data_2[input_data_cnt] = {8};
  int64_t slice_1_input_data_3[input_data_cnt] = {3};
  const size_t input_len = input_data_cnt * sizeof(int64_t);
  const int64_t input_shape[] = {1};
  const size_t shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

  OrtValue* slice_0_input_1 = nullptr;
  OrtValue* slice_0_input_2 = nullptr;
  OrtValue* slice_0_input_3 = nullptr;
  OrtValue* slice_1_input_1 = nullptr;
  OrtValue* slice_1_input_2 = nullptr;
  OrtValue* slice_1_input_3 = nullptr;
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, slice_0_input_data_1, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &slice_0_input_1);
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, slice_0_input_data_2, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &slice_0_input_2);
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, slice_0_input_data_3, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &slice_0_input_3);
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, slice_1_input_data_1, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &slice_1_input_1);
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, slice_1_input_data_2, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &slice_1_input_2);
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, slice_1_input_data_3, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &slice_1_input_3);

  const Tensor& slice_0_input_tensor_1 = slice_0_input_1->Get<Tensor>();
  const Tensor& slice_0_input_tensor_2 = slice_0_input_2->Get<Tensor>();
  const Tensor& slice_0_input_tensor_3 = slice_0_input_3->Get<Tensor>();
  const Tensor& slice_1_input_tensor_1 = slice_1_input_1->Get<Tensor>();
  const Tensor& slice_1_input_tensor_2 = slice_1_input_2->Get<Tensor>();
  const Tensor& slice_1_input_tensor_3 = slice_1_input_3->Get<Tensor>();

  ONNX_NAMESPACE::TensorProto slice_0_tensorproto_1 = utils::TensorToTensorProto(slice_0_input_tensor_1, slice_node_0_name + "_starts");
  ONNX_NAMESPACE::TensorProto slice_0_tensorproto_2 = utils::TensorToTensorProto(slice_0_input_tensor_2, slice_node_0_name + "_ends");
  ONNX_NAMESPACE::TensorProto slice_0_tensorproto_3 = utils::TensorToTensorProto(slice_0_input_tensor_3, slice_node_0_name + "_axes");
  ONNX_NAMESPACE::TensorProto slice_1_tensorproto_1 = utils::TensorToTensorProto(slice_1_input_tensor_1, slice_node_1_name + "_starts");
  ONNX_NAMESPACE::TensorProto slice_1_tensorproto_2 = utils::TensorToTensorProto(slice_1_input_tensor_2, slice_node_1_name + "_ends");
  ONNX_NAMESPACE::TensorProto slice_1_tensorproto_3 = utils::TensorToTensorProto(slice_1_input_tensor_3, slice_node_1_name + "_axes");

  ONNX_NAMESPACE::TypeProto t;
  t.mutable_tensor_type()->set_elem_type(slice_0_tensorproto_1.data_type());
  auto* slice_node_0_arg_1 = &graph.GetOrCreateNodeArg(slice_node_0_name + "_starts", &t);
  auto* slice_node_0_arg_2 = &graph.GetOrCreateNodeArg(slice_node_0_name + "_ends", &t);
  auto* slice_node_0_arg_3 = &graph.GetOrCreateNodeArg(slice_node_0_name + "_axes", &t);
  auto* slice_node_1_arg_1 = &graph.GetOrCreateNodeArg(slice_node_1_name + "_starts", &t);
  auto* slice_node_1_arg_2 = &graph.GetOrCreateNodeArg(slice_node_1_name + "_ends", &t);
  auto* slice_node_1_arg_3 = &graph.GetOrCreateNodeArg(slice_node_1_name + "_axes", &t);

  graph.AddInitializedTensor(slice_0_tensorproto_1);
  graph.AddInitializedTensor(slice_0_tensorproto_2);
  graph.AddInitializedTensor(slice_0_tensorproto_3);
  graph.AddInitializedTensor(slice_1_tensorproto_1);
  graph.AddInitializedTensor(slice_1_tensorproto_2);
  graph.AddInitializedTensor(slice_1_tensorproto_3);

  std::vector<onnxruntime::NodeArg*> slice_node_0_input_defs = {reshape.MutableInputDefs()[0], slice_node_0_arg_1, slice_node_0_arg_2, slice_node_0_arg_3};
  std::vector<onnxruntime::NodeArg*> slice_node_1_input_defs = {reshape.MutableInputDefs()[0], slice_node_1_arg_1, slice_node_1_arg_2, slice_node_1_arg_3};
  std::vector<onnxruntime::NodeArg*> slice_node_0_output_defs = {slice_node_0_arg};
  std::vector<onnxruntime::NodeArg*> slice_node_1_output_defs = {slice_node_1_arg};

  // Create 2 Slice nodes
  auto& slice_node_0 = graph.AddNode(slice_node_0_name, "Slice", "Map 5D/6D to 4D",
                                     slice_node_0_input_defs, slice_node_0_output_defs);
  auto& slice_node_1 = graph.AddNode(slice_node_1_name, "Slice", "Map 5D/6D to 4D",
                                     slice_node_1_input_defs, slice_node_1_output_defs);

  // Create 2 ReduceSum nodes
  std::string reduce_sum_node_0_name = graph.GenerateNodeName(reduce_sum.Name() + "_0");
  std::string reduce_sum_node_1_name = graph.GenerateNodeName(reduce_sum.Name() + "_1");

  int64_t reduce_sum_0_input_data_1[input_data_cnt] = {3};
  int64_t reduce_sum_1_input_data_1[input_data_cnt] = {3};

  OrtValue* reduce_sum_0_input_1 = nullptr;
  OrtValue* reduce_sum_1_input_1 = nullptr;
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, reduce_sum_0_input_data_1, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &reduce_sum_0_input_1);
  ort_api->CreateTensorWithDataAsOrtValue(mem_info, reduce_sum_1_input_data_1, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &reduce_sum_1_input_1);

  const Tensor& reduce_sum_0_input_tensor_1 = reduce_sum_0_input_1->Get<Tensor>();
  const Tensor& reduce_sum_1_input_tensor_1 = reduce_sum_1_input_1->Get<Tensor>();

  ONNX_NAMESPACE::TensorProto reduce_sum_0_tensorproto_1 = utils::TensorToTensorProto(reduce_sum_0_input_tensor_1, reduce_sum_node_0_name + "_axes");
  ONNX_NAMESPACE::TensorProto reduce_sum_1_tensorproto_1 = utils::TensorToTensorProto(reduce_sum_1_input_tensor_1, reduce_sum_node_1_name + "_axes");

  ONNX_NAMESPACE::TypeProto t1;
  t1.mutable_tensor_type()->set_elem_type(reduce_sum_0_tensorproto_1.data_type());
  auto* reduce_sum_0_arg_1 = &graph.GetOrCreateNodeArg(reduce_sum_node_0_name + "_axes", &t1);
  auto* reduce_sum_1_arg_1 = &graph.GetOrCreateNodeArg(reduce_sum_node_1_name + "_axes", &t1);

  graph.AddInitializedTensor(reduce_sum_0_tensorproto_1);
  graph.AddInitializedTensor(reduce_sum_1_tensorproto_1);

  // Create ReduceSum node's output arg.
  // The ReduceSum node's output type should be the same as the original ReduceSum node's output (going to be removed) type.
  auto* reduce_sum_node_0_arg = &graph.GetOrCreateNodeArg(reduce_sum_node_0_name, reduce_sum.OutputDefs()[0]->TypeAsProto());
  auto* reduce_sum_node_1_arg = &graph.GetOrCreateNodeArg(reduce_sum_node_1_name, reduce_sum.OutputDefs()[0]->TypeAsProto());

  std::vector<onnxruntime::NodeArg*> reduce_sum_node_0_input_defs = {slice_node_0_output_defs[0], reduce_sum_0_arg_1};
  std::vector<onnxruntime::NodeArg*> reduce_sum_node_1_input_defs = {slice_node_1_output_defs[0], reduce_sum_1_arg_1};
  std::vector<onnxruntime::NodeArg*> reduce_sum_node_0_output_defs = {reduce_sum_node_0_arg};
  std::vector<onnxruntime::NodeArg*> reduce_sum_node_1_output_defs = {reduce_sum_node_1_arg};

  auto& reduce_sum_node_0 = graph.AddNode(reduce_sum_node_0_name, "ReduceSum", "Map 5D/6D to 4D",
                                          reduce_sum_node_0_input_defs, reduce_sum_node_0_output_defs);
  auto& reduce_sum_node_1 = graph.AddNode(reduce_sum_node_1_name, "ReduceSum", "Map 5D/6D to 4D",
                                          reduce_sum_node_1_input_defs, reduce_sum_node_1_output_defs);

  reduce_sum_node_0.AddAttribute("keepdims", (int64_t)(1));
  reduce_sum_node_1.AddAttribute("keepdims", (int64_t)(1));

  // Create 1 Concat node
  std::string concat_node_name = graph.GenerateNodeName(reduce_sum_node_0_name + "_concat");
  std::vector<onnxruntime::NodeArg*> concat_node_arg_input_defs = {reduce_sum_node_0_arg, reduce_sum_node_1_arg};
  auto& concat_node = graph.AddNode(concat_node_name, "Concat", "Map 5D/6D to 4D",
                                    concat_node_arg_input_defs, reduce_sum.MutableOutputDefs());
  concat_node.AddAttribute("axis", (int64_t)(3));

  return Status::OK();
}

Status MapToFourDimensions::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  bool have_updated_nodes = false;
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }
    
    bool map_tensor_to_4d = false;

    // Requirements:
    //   1. Map 2D/3D to 4D. Replace 2D Gemms with Transpose/Reshape and 1x1 Conv.
    //   2. Map 5D/6D to 4D. Replace Reshape and ReduceSum with two Slices, two ReduceSums and one Concat. 
    if (node->OpType() == "MatMul") {
      const auto* input_0 = node->InputDefs()[0];
      const auto* input_1 = node->InputDefs()[1];
      /*
      if ((input_0->Shape()->dim_size() == 2 || input_0->Shape()->dim_size() == 3) &&
          (input_1->Shape()->dim_size() == 2 || input_1->Shape()->dim_size() == 3)) {
        map_tensor_to_4d = true;
      }
      */
      if (input_0->Shape()->dim_size() == 4) {
        map_tensor_to_4d = true;
      }
    } else if (node->OpType() == "ReduceSum") {
      // Assume Reshape -> Q -> DQ -> ReduceSum since we don't remove Q/DQ for now
      // TODO: Make sure Reshape, Q and DQ does exist
      const Node& node_x = *node->InputNodesBegin();  // Q
      const Node& node_y = *node_x.InputNodesBegin(); // DQ
      const Node& node_z = *node_y.InputNodesBegin(); // Reshape
      if (node_z.OpType() == "Reshape") {
        const auto* output_0 = node_z.OutputDefs()[0];
        if (output_0->Shape()->dim_size() == 5) {
          map_tensor_to_4d = true;
        }
      }
    }

    if (!map_tensor_to_4d) {
      continue;
    }

    if (node->OpType() == "MatMul") {
      AddConv(graph, *node);
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    } else if (node->OpType() == "ReduceSum") {
      // assume Reshape -> Q -> DQ -> ReduceSum since we don't remove Q/DQ for now
      // TODO: Make sure Reshape, Q and DQ does exist
      const Node& q_node = *node->InputNodesBegin();      // Q
      const Node& dq_node = *q_node.InputNodesBegin();     // DQ
      const Node& const_reshape_node = *dq_node.InputNodesBegin();    // Reshape
      Node* reshape_node = graph.GetNode(const_reshape_node.Index());  // Mutable Reshape

      AddSliceReduceSumConcat(graph, *reshape_node, *node);

      // Remove original Reshape and ReduceSum
      //   - Remove the output edges of the constant node and then remove the node itself.
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
      // graph_utils::RemoveNodeOutputEdges(graph, *reshape_node);
      // graph.RemoveNode(reshape_node->Index());
    }
  }

  ORT_RETURN_IF_ERROR(graph.Resolve());

  return Status::OK();
}
}  // namespace onnxruntime
