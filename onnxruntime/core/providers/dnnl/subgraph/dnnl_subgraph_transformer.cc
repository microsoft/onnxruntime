// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_subgraph_transformer.h"

namespace onnxruntime {
namespace ort_dnnl {

//apply all transformation rules in order
void DnnlGraphTransformer::Apply(DnnlSubgraph& subgraph) {
  ConvRelu(subgraph);
  MatMulAdd(subgraph);
}

//resolve a fusion by replacing old_indices nodes with a new_node
//unneeded tensors will be deleted, old news' edges will be cleared
//new_node will be set with new edges and inserted to subgraph
void DnnlGraphTransformer::ResolveFusion(DnnlSubgraph& subgraph, std::vector<size_t> old_indices, std::unique_ptr<DnnlNode> new_node) {
  //the tensors to keep
  std::unordered_set<std::string> keep_tensors;

  //get keep tensors from new_node
  //all tensors related to new_node needs to be kept
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

  //find out tensors to remove, cleanup tensor consumers and producer
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

  //remove unused tensors
  for (const auto& tensor_name : tensors_to_remove) {
    auto tensor = subgraph.GetDnnlTensor(tensor_name);
    if(tensor){
      //has consumer and producer
      if(tensor->GetConsumers().size() || tensor->GetProducer().Exists()){
        continue;
      }
      else{
        subgraph.RemoveTensor(tensor_name);
      }
    }
    //subgraph.RemoveTensor(tensor_name);
  }
  //remove unused nodes
  for (auto index : old_indices) {
    subgraph.RemoveNode(index);
  }

  //reestablish producer and consumer for tensors related to new node
  //such tensors should not get deleted
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

  //new node now has correct input output tensors as well as tensor connections
  //subgraph now owns the new node
  subgraph.AddNode(std::move(new_node));
}

//helper to determine whether a tensor acts as subgraph output
bool DnnlGraphTransformer::IsGraphOutput(DnnlSubgraph& subgraph, DnnlTensor& tensor) {
  auto graph_outputs = subgraph.GetDnnlOutputs();
  if (std::find(graph_outputs.cbegin(), graph_outputs.cend(), &tensor) != graph_outputs.cend()) {
    return true;
  }
  return false;
}

//helper to determien whether
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

void DnnlGraphTransformer::ConvRelu(DnnlSubgraph& subgraph) {
  //global index of convrelu
  static int conv_relu_index = 0;

  //traverse with max index as there will be empty nodes due to fusion
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    auto dnnl_node = subgraph.GetDnnlNode(index);

    //look for conv relu pattern
    if (dnnl_node == nullptr) {
      continue;
    }

    if (dnnl_node->OpType() != "Conv") {
      continue;
    }

    if (dnnl_node->OutputCount() != 1) {
      ORT_THROW("Invalid Conv node");
    }

    if (dnnl_node->Output(0).Exists() && dnnl_node->Output(0).GetConsumers().size() != 1) {
      continue;
    }

    //check whether conv node's only output tensor is outputting the subgraph
    {
      auto graph_outputs = subgraph.GetDnnlOutputs();
      if (std::find(graph_outputs.cbegin(), graph_outputs.cend(), &dnnl_node->Output(0)) != graph_outputs.cend()) {
        continue;
      }
    }

    auto next_dnnl_node = dnnl_node->Output(0).GetConsumers()[0].GetNode();
    if (next_dnnl_node == nullptr) {
      continue;
    }
    if (next_dnnl_node->OpType() != "Relu") {
      continue;
    }

    //construct new node
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

    //insert new node, remove original nodes, connect new edges
    ResolveFusion(subgraph, {dnnl_node->Index(), next_dnnl_node->Index()}, std::move(new_node));
    LOGS_DEFAULT(INFO) << "fuse [" << dnnl_node->Name() << "] and [" << next_dnnl_node << "] into ConvRelu";
  }
}

void DnnlGraphTransformer::MatMulAdd(DnnlSubgraph& subgraph) {
  static int fused_index = 0;
  size_t max_index = subgraph.GetMaxNodeIndex();
  for (size_t index = 0; index < max_index; index++) {
    auto dnnl_node = subgraph.GetDnnlNode(index);

    //look for conv relu pattern
    if (dnnl_node == nullptr || dnnl_node->OpType() != "MatMul") {
      continue;
    }

    auto matmul_node = dnnl_node;

    if (matmul_node->OutputCount() != 1) ORT_THROW("Invalid Matmul(int) node");

    if (matmul_node->Output(0).Exists() && matmul_node->Output(0).GetConsumers().size() != 1) {
      continue;
    }

    //check whether conv node's only output tensor is outputting the subgraph
    {
      auto graph_outputs = subgraph.GetDnnlOutputs();
      if (std::find(graph_outputs.cbegin(), graph_outputs.cend(), &matmul_node->Output(0)) != graph_outputs.cend()) {
        continue;
      }
    }

    auto next_dnnl_node = matmul_node->Output(0).GetConsumers()[0].GetNode();
    if (next_dnnl_node == nullptr) {
      continue;
    }
    if (next_dnnl_node->OpType() != "Add") {
      continue;
    }

    /*
    add now has one of the input connecting to matmul's single output
    different cases:
    matmul input goes to the other add input
    matmul output goes to the other add input (adding two identical tensor)
    some other tensor goes to the other add input
    */
    auto add_node = next_dnnl_node;
    auto matmul_output_name = matmul_node->Output(0).Name();
    auto add_inputs = add_node->Inputs();
    //add is taking two inputs from the same matmul output
    //not sure if onednn would support such post ops
    if (add_inputs[0] == add_inputs[1]) {
      continue;
    }
    auto fused_node_inputs = matmul_node->Inputs();
    //the 3rd input to fused matmul
    if (matmul_output_name == add_inputs[0]->Name()) {
      fused_node_inputs.push_back(add_inputs[1]);
    } else {
      fused_node_inputs.push_back(add_inputs[0]);
    }
    auto fused_node_output = add_node->Outputs()[0];
    auto fused_node_name = matmul_node->Name() + "_" + matmul_node->OpType() + "Add_" + std::to_string(fused_index++);
    auto fused_node_type = matmul_node->OpType() + "Add";

    //construct new node
    auto fused_node = std::make_unique<DnnlNode>();
    fused_node->Name() = fused_node_name;
    fused_node->OpType() = fused_node_type;
    fused_node->Inputs() = fused_node_inputs;
    fused_node->Outputs() = {fused_node_output};
    //no attribute for matmul and add

    //insert new node, remove original nodes, connect new edges
    ResolveFusion(subgraph, {matmul_node->Index(), add_node->Index()}, std::move(fused_node));
    LOGS_DEFAULT(INFO) << "fuse [" << matmul_node->Name() << "] and [" << add_node << "] into MatMulAdd";
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
