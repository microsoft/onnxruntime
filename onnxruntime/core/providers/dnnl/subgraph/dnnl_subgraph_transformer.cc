// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_subgraph_transformer.h"
#include "core/providers/shared_library/provider_api.h"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <memory>

namespace onnxruntime {
namespace ort_dnnl {

// apply all transformation rules in order
void DnnlGraphTransformer::Apply(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer) {
  ConvRelu(subgraph);
  MatMulBinaryEltwise(subgraph);
  Gelu(subgraph, onnx_subgraph_viewer);
  FastGelu(subgraph, onnx_subgraph_viewer);
  RemoveMatMulIntegerZP(subgraph, onnx_subgraph_viewer);
  MatMulIntegerBinaryEltwise(subgraph);
}

// resolve a fusion by replacing old_indices nodes with a new_node
// unneeded tensors will be deleted, old news' edges will be cleared
// new_node will be set with new edges and inserted to subgraph
void DnnlGraphTransformer::ResolveFusion(DnnlSubgraph& subgraph, std::vector<size_t> old_indices, std::unique_ptr<DnnlNode> new_node) {
  // the tensors to keep
  std::unordered_set<std::string> keep_tensors;

  // get keep tensors from new_node
  // all tensors related to new_node needs to be kept
  for (auto input : new_node->Inputs()) {
    if (input && input->Exists()) {
      keep_tensors.insert(input->Name());
    }
  }

  for (auto output : new_node->Outputs()) {
    if (output && output->Exists()) {
      keep_tensors.insert(output->Name());
    }
  }

  // find out tensors to remove, cleanup tensor consumers and producer
  std::unordered_set<std::string> tensors_to_remove;
  for (auto index : old_indices) {
    auto cur_node = subgraph.GetDnnlNode(index);
    {
      int input_index = 0;
      for (auto input : cur_node->Inputs()) {
        if (input && input->Exists()) {
          input->RemoveConsumer(DnnlNodeArg(cur_node, input_index, false));
          if (!keep_tensors.count(input->Name())) {
            tensors_to_remove.insert(input->Name());
          }
        }
        input_index++;
      }
    }
    for (auto output : cur_node->Outputs()) {
      if (output && output->Exists()) {
        output->ResetProducer();
        if (!keep_tensors.count(output->Name())) {
          tensors_to_remove.insert(output->Name());
        }
      }
    }
  }

  // remove unused tensors
  for (const auto& tensor_name : tensors_to_remove) {
    auto tensor = subgraph.GetDnnlTensor(tensor_name);
    if (tensor) {
      // has consumer and producer
      if (tensor->GetConsumers().size() || tensor->GetProducer().Exists()) {
        continue;
      } else {
        subgraph.RemoveTensor(tensor_name);
      }
    }
    // subgraph.RemoveTensor(tensor_name);
  }
  // remove unused nodes
  for (auto index : old_indices) {
    subgraph.RemoveNode(index);
  }

  // reestablish producer and consumer for tensors related to new node
  // such tensors should not get deleted
  {
    size_t input_index = 0;
    for (auto input : new_node->Inputs()) {
      if (input) {
        input->AddConsumer(DnnlNodeArg(new_node.get(), input_index, false));
      }

      input_index++;
    }

    size_t output_index = 0;
    for (auto output : new_node->Outputs()) {
      if (output) {
        output->SetProducer(DnnlNodeArg(new_node.get(), output_index, true));
      }
      output_index++;
    }
  }

  // new node now has correct input output tensors as well as tensor connections
  // subgraph now owns the new node
  subgraph.AddNode(std::move(new_node));
}

// helper to determine whether a tensor acts as subgraph output
bool DnnlGraphTransformer::IsGraphOutput(DnnlSubgraph& subgraph, DnnlTensor& tensor) {
  auto graph_outputs = subgraph.GetDnnlOutputs();
  if (std::find(graph_outputs.cbegin(), graph_outputs.cend(), &tensor) != graph_outputs.cend()) {
    return true;
  }
  return false;
}

// helper to determien whether
bool DnnlGraphTransformer::ProduceGraphOutput(DnnlSubgraph& subgraph, DnnlNode& node) {
  auto graph_outputs = subgraph.GetDnnlOutputs();
  for (auto output : node.Outputs()) {
    if (output && output->Exists()) {
      if (IsGraphOutput(subgraph, *output)) {
        return true;
      }
    }
  }
  return false;
}

bool DnnlGraphTransformer::IsNodeFusable(DnnlSubgraph& subgraph, DnnlNode* node) const {
  if (node == nullptr) {
    return false;
  }
  // isSingleOutput(DnnlNode* node);
  if (node->OutputCount() != 1) {
    std::string s = "Invalid " + node->OpType() + " node";
    ORT_THROW(s);
  }
  // isConsumedBySingleNode(DnnlNode* node);
  if (node->Output(0).Exists() && node->Output(0).GetConsumers().size() != 1) {
    return false;
  }
  // isOutputPartOfSubgraph(DnnlSubgraph& subgraph, DnnlNode* node);
  auto graph_outputs = subgraph.GetDnnlOutputs();
  if (std::find(graph_outputs.cbegin(), graph_outputs.cend(), &node->Output(0)) != graph_outputs.cend()) {
    return false;
  }
  return true;
}

bool IsScalar(const DnnlTensor& input_arg) {
  auto dim = input_arg.Dim();
  auto dim_size = dim.size();
  return dim_size == 0 || (dim_size == 1 && dim[0] == 1);
}

bool DnnlGraphTransformer::IsInitilizedWithExpectedValue(const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlTensor& input_arg, float expected_value) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (!onnx_subgraph_viewer.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    return false;
  }

  if (tensor_proto == nullptr) {
    return false;
  }

  if (!tensor_proto->has_raw_data()) {
    return false;
  }

  const auto data_type = input_arg.Type();
  if (data_type == dnnl::memory::data_type::f32) {
    const float* val = reinterpret_cast<const float*>(tensor_proto->raw_data().data());
    if (std::isnan(val[0]) || std::isinf(val[0])) {
      if (std::isinf(val[0]) && std::isinf(expected_value) && (std::signbit(val[0]) == std::signbit(expected_value))) {
        return true;
      }
      return false;
    }

    const float atol = 1e-8f;
    const float rtol = 1e-5f;
    float diff = std::abs(val[0] - expected_value);
    if (diff > (atol + rtol * std::abs(expected_value))) {
      return false;
    }
  } else {
    // Not expected data types.
    return false;
  }

  return true;
}

DnnlNode* FirstParentByType(DnnlNode* node, const std::string& parent_type) {
  for (size_t i = 0; i < node->InputCount(); ++i) {
    auto prev_node = node->Input(static_cast<int>(i)).GetProducer().GetNode();
    if (prev_node != nullptr && prev_node->OpType() == parent_type) {
      return prev_node;
    }
  }
  return nullptr;
}

/*
     This function fuses subgraph like the following into one Gelu node.
     Subgraph pattern 1:
                   +-------Mul(0.5)---------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul ==>
                          (B=1.4142...)        (1)

      Subgraph pattern 2:
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                          (B=1.4142...)        (1)            (0.5)

       After Fusion:
                [root]--> Gelu ==>
*/
void DnnlGraphTransformer::Gelu(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer) {
  static int gelu_index = 0;
  // traverse with max index as there will be empty nodes due to fusion
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    auto div_node = subgraph.GetDnnlNode(index);
    std::vector<size_t> gelu_indices;
    //----------------------------
    if (div_node == nullptr || div_node->OpType() != "Div") {
      continue;
    }

    // Check second input is sqrt(2)
    // Some Bert models uses this approximation of SQRT2 in the Gelu function
    float approximated_sqrt_two = 1.4142099618911743f;
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, div_node->Input(1), approximated_sqrt_two) &&
        !IsInitilizedWithExpectedValue(onnx_subgraph_viewer, div_node->Input(1), static_cast<float>(M_SQRT2))) {
      continue;
    }

    if (!IsNodeFusable(subgraph, div_node)) {
      continue;
    }
    gelu_indices.push_back(div_node->Index());
    //----------------------------
    auto erf_node = div_node->Output(0).GetConsumers()[0].GetNode();
    if (erf_node == nullptr || erf_node->OpType() != "Erf") {
      continue;
    }

    if (!IsNodeFusable(subgraph, erf_node)) {
      continue;
    }
    gelu_indices.push_back(erf_node->Index());
    //----------------------------
    auto add_node = erf_node->Output(0).GetConsumers()[0].GetNode();
    if (add_node == nullptr || add_node->OpType() != "Add") {
      continue;
    }

    bool is_add_input0 = add_node->Input(0).Name() == erf_node->Output(0).Name();
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, add_node->Input(is_add_input0 ? 1 : 0), 1.0f)) {
      continue;
    }

    if (!IsNodeFusable(subgraph, add_node)) {
      continue;
    }
    gelu_indices.push_back(add_node->Index());
    //----------------------------
    auto mul1_node = add_node->Output(0).GetConsumers()[0].GetNode();
    if (mul1_node == nullptr || mul1_node->OpType() != "Mul") {
      continue;
    }

    // if (!IsNodeFusable(subgraph, mul1_node)) {
    //   continue;
    // }
    gelu_indices.push_back(mul1_node->Index());
    //----------------------------
    // look for Mul(0.5) using pattern 1 shown above
    bool is_pattern_1 = true;
    auto mul2_node = FirstParentByType(mul1_node, "Mul");
    if (mul2_node != nullptr) {
      // the input Div and Mul2 should share at least one input.
      bool is_mul2_input0 = div_node->Input(0).Name() == mul2_node->Input(0).Name();
      bool is_mul2_input1 = div_node->Input(0).Name() == mul2_node->Input(1).Name();
      if (!(is_mul2_input0 ^ is_mul2_input1)) {
        is_pattern_1 = false;
      }
      if (is_pattern_1 && !IsInitilizedWithExpectedValue(onnx_subgraph_viewer, mul2_node->Input(is_mul2_input0 ? 1 : 0), 0.5f)) {
        is_pattern_1 = false;
      }
      if (is_pattern_1 && !IsNodeFusable(subgraph, mul2_node)) {
        is_pattern_1 = false;
      }
    } else {
      is_pattern_1 = false;
    }

    // look for Mul(0.5) using pattern 2 shown above
    if (!is_pattern_1) {
      // We only need to check mul1_node IsNodeFusable for pattern 2
      if (!IsNodeFusable(subgraph, mul1_node)) {
        continue;
      }
      mul2_node = mul1_node->Output(0).GetConsumers()[0].GetNode();
      if (mul2_node == nullptr || mul2_node->OpType() != "Mul") {
        continue;
      }

      if (mul2_node->OutputCount() != 1) {
        ORT_THROW("Invalid Mul node");
      }
      bool is_mul2_first_input = mul2_node->Input(0).Name() == mul1_node->Output(0).Name();
      if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, mul2_node->Input(is_mul2_first_input ? 1 : 0), 0.5f)) {
        continue;
      }
    }
    gelu_indices.push_back(mul2_node->Index());

    // construct new node
    auto new_node = std::make_unique<DnnlNode>();
    new_node->Name() = div_node->Name() + "_Gelu_" + std::to_string(gelu_index++);
    new_node->OpType() = "Gelu";
    new_node->Inputs().push_back(div_node->Inputs()[0]);
    if (is_pattern_1) {
      for (auto def : mul1_node->Outputs()) {
        new_node->Outputs().push_back(def);
      }
    } else {
      for (auto def : mul2_node->Outputs()) {
        new_node->Outputs().push_back(def);
      }
    }
    // no attributes needed for Gelu if needed this can be updated
    // new_node->Attributes().insert(div_node->Attributes());

    // insert new node, remove original nodes, connect new edges
    ResolveFusion(subgraph, {gelu_indices}, std::move(new_node));
    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "Gelu fusion found [" << gelu_index << "]";
    }
  }
}

/*
Rewrite graph fusing Gelu activation subgraph to a single Gelu node.
The formula corresponding to Gelu activation subgraph :

    x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))) or
    x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))),

where x is the input.
*/
void DnnlGraphTransformer::FastGelu(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer) {
  static int fastgelu_index = 0;
  // traverse with max index as there will be empty nodes due to fusion
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    auto dnnl_node = subgraph.GetDnnlNode(index);
    if (!FastGeluFirstFormula(subgraph, onnx_subgraph_viewer, dnnl_node, fastgelu_index)) {
      FastGeluSecondFormula(subgraph, onnx_subgraph_viewer, dnnl_node, fastgelu_index);
    }
  }
}

/*
Rewrite graph fusing Gelu activation subgraph to a single Gelu node.
The formula corresponding to Gelu activation subgraph :

    x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))

where x is the input.
*/
bool DnnlGraphTransformer::FastGeluFirstFormula(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlNode* mul1_node, int& fastgelu_index) {
  std::vector<size_t> gelu_indices;
  //----------mul(0.44715)------------------
  if (mul1_node == nullptr || mul1_node->OpType() != "Mul") {
    return false;
  }
  int32_t mul1_input_index = -1;
  const float mul_val = 0.044715f;
  for (auto i = 0; i < 2; i++) {
    if (IsInitilizedWithExpectedValue(onnx_subgraph_viewer, mul1_node->Input(i), mul_val)) {
      mul1_input_index = i;
      break;
    }
  }
  if (mul1_input_index == -1) return false;

  if (!IsNodeFusable(subgraph, mul1_node)) {
    return false;
  }
  gelu_indices.push_back(mul1_node->Index());
  //-------------Mul---------------
  auto mul2_node = mul1_node->Output(0).GetConsumers()[0].GetNode();
  if (mul2_node == nullptr || mul2_node->OpType() != "Mul") {
    return false;
  }
  if (!IsNodeFusable(subgraph, mul2_node)) {
    return false;
  }
  gelu_indices.push_back(mul2_node->Index());
  //-------------Add(1.0)---------------
  auto add1_node = mul2_node->Output(0).GetConsumers()[0].GetNode();
  if (add1_node == nullptr || add1_node->OpType() != "Add") {
    return false;
  }
  bool is_add_input0 = mul2_node->Output(0).Name() == add1_node->Input(0).Name();
  if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, add1_node->Input(is_add_input0 ? 1 : 0), 1.0f)) {
    return false;
  }

  if (!IsNodeFusable(subgraph, add1_node)) {
    return false;
  }
  gelu_indices.push_back(add1_node->Index());
  //-------------Mul---------------
  auto mul3_node = add1_node->Output(0).GetConsumers()[0].GetNode();
  if (mul3_node == nullptr || mul3_node->OpType() != "Mul") {
    return false;
  }
  if (!IsNodeFusable(subgraph, mul3_node)) {
    return false;
  }
  gelu_indices.push_back(mul3_node->Index());
  //-------------Mul(0.7978845834732056f)---------------
  auto prev_mul4_node = FirstParentByType(mul3_node, "Mul");
  if (prev_mul4_node == nullptr) {
    return false;
  }

  int32_t mul4_input_index = -1;
  const float mul4_val = 0.7978845834732056f;
  for (auto i = 0; i < 2; i++) {
    if (IsInitilizedWithExpectedValue(onnx_subgraph_viewer, prev_mul4_node->Input(i), mul4_val)) {
      mul4_input_index = i;
      break;
    }
  }
  if (mul4_input_index == -1) return false;

  if (!IsNodeFusable(subgraph, prev_mul4_node)) {
    return false;
  }
  gelu_indices.push_back(prev_mul4_node->Index());

  auto tanh_node = mul3_node->Output(0).GetConsumers()[0].GetNode();
  int32_t x_input_index = (mul1_input_index == 0) ? 1 : 0;
  if (FastGeluFormulaCommon(subgraph, onnx_subgraph_viewer, mul1_node, x_input_index, tanh_node, gelu_indices, fastgelu_index)) {
    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "FastGelu fusion found [" << fastgelu_index << "] (first formula)";
    }
    return true;
  }
  return false;
}

/*
Rewrite graph fusing Gelu activation subgraph to a single Gelu node.
The formula corresponding to Gelu activation subgraph :

    x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))),

where x is the input.
*/
void DnnlGraphTransformer::FastGeluSecondFormula(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlNode* pow_node, int& fastgelu_index) {
  std::vector<size_t> gelu_indices;
  //---------Pow-------------------
  if (pow_node == nullptr || pow_node->OpType() != "Pow") {
    return;
  }

  auto& pow_exponent = pow_node->Input(1);
  if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, pow_exponent, 3.0f)) {
    return;
  }

  if (!IsNodeFusable(subgraph, pow_node)) {
    return;
  }
  gelu_indices.push_back(pow_node->Index());
  //----------Mul(0.044714998453855515f)------------------
  auto mul1_node = pow_node->Output(0).GetConsumers()[0].GetNode();
  if (mul1_node == nullptr || mul1_node->OpType() != "Mul") {
    return;
  }

  float fastgelu_muliplyer = 0.044714998453855515f;
  bool is_mul1_input0 = pow_node->Output(0).Name() == mul1_node->Input(0).Name();
  if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, mul1_node->Input(is_mul1_input0 ? 1 : 0), fastgelu_muliplyer)) {
    return;
  }
  if (!IsNodeFusable(subgraph, mul1_node)) {
    return;
  }
  gelu_indices.push_back(mul1_node->Index());
  //----------Add------------------
  auto add1_node = mul1_node->Output(0).GetConsumers()[0].GetNode();
  if (add1_node == nullptr || add1_node->OpType() != "Add") {
    return;
  }

  if (!IsNodeFusable(subgraph, add1_node)) {
    return;
  }
  gelu_indices.push_back(add1_node->Index());
  //----------Mul(sqrt(2/pi))------------------
  auto mul2_node = add1_node->Output(0).GetConsumers()[0].GetNode();
  if (mul2_node == nullptr || mul2_node->OpType() != "Mul") {
    return;
  }

  // constant is sqrt(2/pi)
  float fastgelu_sqrt_2_div_pi = 0.7978845834732056f;
  bool is_mul2_input0 = add1_node->Output(0).Name() == mul2_node->Input(0).Name();
  if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, mul2_node->Input(is_mul2_input0 ? 1 : 0), fastgelu_sqrt_2_div_pi)) {
    return;
  }

  if (!IsNodeFusable(subgraph, mul2_node)) {
    return;
  }
  gelu_indices.push_back(mul2_node->Index());

  //----------Tanh------------------
  auto tanh_node = mul2_node->Output(0).GetConsumers()[0].GetNode();
  // since the first node is pow the x_input_index is always 0
  if (FastGeluFormulaCommon(subgraph, onnx_subgraph_viewer, pow_node, 0, tanh_node, gelu_indices, fastgelu_index)) {
    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "FastGelu fusion found [" << fastgelu_index << "] (second formula)";
    }
  }
}

/*
 Looks for the part of FastGelu that is common to both formulas if the pattern is found
 return true otherwise return false.
    i.e. x * 0.5 * (1.0 + tanh(...))

    x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))) or
    x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))),
where x is the input.
*/
bool DnnlGraphTransformer::FastGeluFormulaCommon(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlNode* gelu_start_node, int32_t x_input_index, DnnlNode* tanh_node, std::vector<size_t>& gelu_indices, int& fastgelu_index) {
  //----------Tanh------------------
  if (tanh_node == nullptr || tanh_node->OpType() != "Tanh") {
    return false;
  }

  if (!IsNodeFusable(subgraph, tanh_node)) {
    return false;
  }
  gelu_indices.push_back(tanh_node->Index());
  //----------Add(1.0)------------------
  auto add2_node = tanh_node->Output(0).GetConsumers()[0].GetNode();
  if (add2_node == nullptr || add2_node->OpType() != "Add") {
    return false;
  }
  bool is_add2_input0 = tanh_node->Output(0).Name() == add2_node->Input(0).Name();
  if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, add2_node->Input(is_add2_input0 ? 1 : 0), 1.0f)) {
    return false;
  }

  if (!IsNodeFusable(subgraph, add2_node)) {
    return false;
  }
  gelu_indices.push_back(add2_node->Index());
  //----------Mul------------------
  auto mul3_node = add2_node->Output(0).GetConsumers()[0].GetNode();
  if (mul3_node == nullptr || mul3_node->OpType() != "Mul") {
    return false;
  }

  if (mul3_node->OutputCount() != 1) {
    ORT_THROW("Invalid Mul node");
  }

  gelu_indices.push_back(mul3_node->Index());
  //---------Mul(0.5)---------------------
  if (mul3_node->InputCount() != 2) {
    return false;
  }
  auto prev_mul4_node = FirstParentByType(mul3_node, "Mul");
  if (prev_mul4_node == nullptr) {
    return false;
  }
  bool is_mul_input0 = gelu_start_node->Input(x_input_index).Name() == prev_mul4_node->Input(0).Name();
  bool is_mul_input1 = gelu_start_node->Input(x_input_index).Name() == prev_mul4_node->Input(1).Name();
  if (!(is_mul_input0 ^ is_mul_input1)) {
    return false;
  }
  if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, prev_mul4_node->Input(is_mul_input0 ? 1 : 0), 0.5f)) {
    return false;
  }
  if (!IsNodeFusable(subgraph, prev_mul4_node)) {
    return false;
  }
  gelu_indices.push_back(prev_mul4_node->Index());

  // construct new node
  auto new_node = std::make_unique<DnnlNode>();
  new_node->Name() = "Dnnl_FastGelu_" + std::to_string(fastgelu_index++);
  new_node->OpType() = "FastGelu";
  new_node->Inputs().push_back(gelu_start_node->Inputs()[x_input_index]);
  for (auto def : mul3_node->Outputs()) {
    new_node->Outputs().push_back(def);
  }
  // No Attributes needed for FastGelu. If they are needed this can be added in.
  // new_node->Attributes().insert(gelu_start_node->Attributes());

  // insert new node, remove original nodes, connect new edges
  ResolveFusion(subgraph, {gelu_indices}, std::move(new_node));
  return true;
}

void DnnlGraphTransformer::ConvRelu(DnnlSubgraph& subgraph) {
  // global index of convrelu
  static int conv_relu_index = 0;

  // traverse with max index as there will be empty nodes due to fusion
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    auto dnnl_node = subgraph.GetDnnlNode(index);

    // look for conv relu pattern
    if (dnnl_node == nullptr) {
      continue;
    }

    if (dnnl_node->OpType() != "Conv") {
      continue;
    }

    if (!IsNodeFusable(subgraph, dnnl_node)) {
      continue;
    }

    auto next_dnnl_node = dnnl_node->Output(0).GetConsumers()[0].GetNode();
    if (next_dnnl_node == nullptr) {
      continue;
    }
    if (next_dnnl_node->OpType() != "Relu") {
      continue;
    }

    // construct new node
    auto new_node = std::make_unique<DnnlNode>();
    new_node->Name() = dnnl_node->Name() + "_ConvRelu_" + std::to_string(conv_relu_index++);
    new_node->OpType() = "ConvRelu";
    for (auto def : dnnl_node->Inputs()) {
      new_node->Inputs().push_back(def);
    }
    for (auto def : next_dnnl_node->Outputs()) {
      new_node->Outputs().push_back(def);
    }
    new_node->Attributes().insert(dnnl_node->Attributes());

    // insert new node, remove original nodes, connect new edges
    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "ConvRelu fusion of [" << dnnl_node->Name() << "] and [" << next_dnnl_node->Name() << "]";
    }
    ResolveFusion(subgraph, {dnnl_node->Index(), next_dnnl_node->Index()}, std::move(new_node));
  }
}

void DnnlGraphTransformer::MatMulBinaryEltwise(DnnlSubgraph& subgraph) {
  static int fused_index = 0;
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    std::vector<size_t> matmul_binary_eltwize_indices = {};
    auto dnnl_node = subgraph.GetDnnlNode(index);
    auto attr_node = dnnl_node;

    if (dnnl_node == nullptr || dnnl_node->OpType() != "MatMul") {
      continue;
    }

    if (!IsNodeFusable(subgraph, dnnl_node)) {
      continue;
    }
    auto fused_node_inputs = dnnl_node->Inputs();
    matmul_binary_eltwize_indices.push_back(dnnl_node->Index());

    dnnl_node = FuseBinaryEltwisePostOps(subgraph,
                                         dnnl_node,
                                         matmul_binary_eltwize_indices,
                                         fused_node_inputs,
                                         attr_node);

    if (!(matmul_binary_eltwize_indices.size() > 1)) {
      matmul_binary_eltwize_indices.clear();
      continue;
    }

    // construct new node
    auto fused_node = std::make_unique<DnnlNode>();
    fused_node->Name() = "MatMulPostOps_fusion" + std::to_string(fused_index++);
    std::string fused_node_name = "MatMulPostOps";
    for (size_t i : matmul_binary_eltwize_indices) {
      if (subgraph.GetDnnlNode(i)->OpType() != "MatMul") {
        fused_node->AppendPostOp(subgraph.GetDnnlNode(i)->OpType());
      }
    }
    fused_node->OpType() = fused_node_name;
    fused_node->Inputs() = fused_node_inputs;
    fused_node->Outputs() = {dnnl_node->Outputs()[0]};

    fused_node->Attributes().insert(attr_node->Attributes());

    if (debug_log_) {
      std::stringstream ss;
      for (size_t i : matmul_binary_eltwize_indices) {
        ss << subgraph.GetDnnlNode(i)->OpType() << "[" << subgraph.GetDnnlNode(i)->Name() << "] ";
      }
      LOGS_DEFAULT(ERROR) << fused_node->OpType() << "[" << fused_node->Name() << "] fusion of " << ss.str();
    }
    // insert new node, remove original nodes, connect new edges
    ResolveFusion(subgraph, matmul_binary_eltwize_indices, std::move(fused_node));
  }
}

void DnnlGraphTransformer::RemoveMatMulIntegerZP(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer) {
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    auto dnnl_node = subgraph.GetDnnlNode(index);

    // look for matmulint
    if (dnnl_node == nullptr || dnnl_node->OpType() != "MatMulInteger") {
      continue;
    }

    // if B zero point exists
    if (!(dnnl_node->InputCount() >= 4 && dnnl_node->Input(3).Exists())) {
      continue;
    }

    auto& b_zero_point = dnnl_node->Input(3);
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    if (!onnx_subgraph_viewer.GetInitializedTensor(b_zero_point.Name(), tensor_proto)) {
      continue;
    }

    if (tensor_proto == nullptr) {
      continue;
    }

    const auto& dims = tensor_proto->dims();
    auto dim_size = tensor_proto->dims_size();
    int num_elements = 1;
    for (int i = 0; i < dim_size; i++) {
      num_elements *= int(dims[i]);
    }

    // check if b_zp is all zeros, assume data is s8 since only s8 weight is supported in onednn
    bool all_zero = true;
    std::vector<int8_t> unpacked_tensor;
    unpacked_tensor.resize(num_elements, 1);
    ORT_THROW_IF_ERROR(onnxruntime::utils::UnpackTensor(*tensor_proto, tensor_proto->has_raw_data() ? tensor_proto->raw_data().data() : nullptr, tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0, reinterpret_cast<int8_t*>(unpacked_tensor.data()), num_elements));
    for (const auto& val : unpacked_tensor) {
      if (val != 0) {
        all_zero = false;
        break;
      }
    }

    if (!all_zero) {
      continue;
    }

    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "Remove weight ZP of [" << dnnl_node->Name() << "]";
    }
    // remove b_zero_point's consumer matmulint
    b_zero_point.RemoveConsumer(DnnlNodeArg(dnnl_node, 3, false));
    // detach b_zero_point from matmulint node
    dnnl_node->Inputs()[3] = nullptr;
  }
}

void DnnlGraphTransformer::MatMulIntegerBinaryEltwise(DnnlSubgraph& subgraph) {
  static int fused_index = 0;
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    std::vector<size_t> matmul_binary_eltwize_indices = {};
    auto dnnl_node = subgraph.GetDnnlNode(index);
    auto attr_node = dnnl_node;

    if (dnnl_node == nullptr || dnnl_node->OpType() != "MatMulInteger") {
      continue;
    }

    if (!IsNodeFusable(subgraph, dnnl_node)) {
      continue;
    }

    auto fused_node_inputs = dnnl_node->Inputs();
    matmul_binary_eltwize_indices.push_back(dnnl_node->Index());

    auto next_dnnl_node = dnnl_node->Output(0).GetConsumers()[0].GetNode();
    if (next_dnnl_node == nullptr) {
      continue;
    }
    if (next_dnnl_node->OpType() != "Cast") {
      continue;
    }

    if (!IsNodeFusable(subgraph, next_dnnl_node)) {
      continue;
    }

    dnnl_node = next_dnnl_node;
    matmul_binary_eltwize_indices.push_back(dnnl_node->Index());

    dnnl_node = FuseBinaryEltwisePostOps(subgraph,
                                         dnnl_node,
                                         matmul_binary_eltwize_indices,
                                         fused_node_inputs,
                                         attr_node);

    if (!(matmul_binary_eltwize_indices.size() > 1)) {
      matmul_binary_eltwize_indices.clear();
      continue;
    }

    // construct new node
    auto fused_node = std::make_unique<DnnlNode>();
    fused_node->Name() = "MatMulIntegerPostOps_fusion" + std::to_string(fused_index++);
    std::string fused_node_name = "MatMulIntegerPostOps";
    for (size_t i : matmul_binary_eltwize_indices) {
      if (subgraph.GetDnnlNode(i)->OpType() != "MatMulInteger" && subgraph.GetDnnlNode(i)->OpType() != "Cast") {
        fused_node->AppendPostOp(subgraph.GetDnnlNode(i)->OpType());
      }
    }
    fused_node->OpType() = fused_node_name;
    fused_node->Inputs() = fused_node_inputs;
    fused_node->Outputs() = {dnnl_node->Outputs()[0]};

    fused_node->Attributes().insert(attr_node->Attributes());

    if (debug_log_) {
      std::stringstream ss;
      for (size_t i : matmul_binary_eltwize_indices) {
        ss << subgraph.GetDnnlNode(i)->OpType() << "[" << subgraph.GetDnnlNode(i)->Name() << "] ";
      }
      LOGS_DEFAULT(ERROR) << fused_node->OpType() << "[" << fused_node->Name() << "] fusion of " << ss.str();
    }
    // insert new node, remove original nodes, connect new edges
    ResolveFusion(subgraph, matmul_binary_eltwize_indices, std::move(fused_node));
  }
}

DnnlNode* DnnlGraphTransformer::FuseBinaryEltwisePostOps(DnnlSubgraph& subgraph,
                                                         DnnlNode* node,
                                                         std::vector<size_t>& indices,
                                                         std::vector<DnnlTensor*>& fused_node_inputs,
                                                         DnnlNode*& attr_node) {
  std::unordered_set<std::string> binary_ops = {"Add", "Div", "Mul", "Sub"};
  std::unordered_set<std::string> elementwise_ops = {"Abs", "Elu", "Exp", "LeakyRelu", "Log", "Relu",
                                                     "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};
  // Upto 32 post-ops are supported by oneDNN framework.
  const size_t MAX_POST_OP_COUNT = 32;
  bool attribute_flag = false;
  auto dnnl_node = node;
  size_t post_op_count = 0;
  while (post_op_count < MAX_POST_OP_COUNT) {
    if (!IsNodeFusable(subgraph, dnnl_node)) {
      break;
    }

    auto next_dnnl_node = dnnl_node->Output(0).GetConsumers()[0].GetNode();
    if (next_dnnl_node == nullptr) {
      break;
    }

    auto next_type = next_dnnl_node->OpType();
    bool is_binary_op = !(binary_ops.count(next_type) == 0);
    bool is_eltwise_op = !(elementwise_ops.count(next_type) == 0);
    if (!is_binary_op && !is_eltwise_op) {
      break;
    }

    if (is_binary_op) {
      if (dnnl_node->Output(0).Name() == next_dnnl_node->Inputs()[0]->Name()) {
        fused_node_inputs.push_back(next_dnnl_node->Inputs()[1]);
      } else {
        // oneDNN can only fuse binary post op for the second input. Due to the fact
        // that division and subraction are not associative we can not fuse them if the
        // input for that node is the first input.
        if (next_dnnl_node->OpType() == "Div" || next_dnnl_node->OpType() == "Sub") {
          break;
        }
        fused_node_inputs.push_back(next_dnnl_node->Inputs()[0]);
      }
    } else if (is_eltwise_op) {
      // We only support a single node with an "alpha" attribute. If we see Elu or
      // LeakyRelu set attr_node and attribute_flag. If the attribute_flag is set break
      // out of the while loop looking for additional post ops. Since the next op cannot
      // be part of the postop fusion.
      if (next_dnnl_node->OpType() == "Elu" || next_dnnl_node->OpType() == "LeakyRelu") {
        if (attribute_flag) break;
        attr_node = next_dnnl_node;
        attribute_flag = true;
      }
    }
    dnnl_node = next_dnnl_node;
    indices.push_back(dnnl_node->Index());
    post_op_count++;
  }
  return dnnl_node;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
