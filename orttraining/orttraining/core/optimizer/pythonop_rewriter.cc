// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_TORCH_INTEROP

#include <string>
#include <unordered_map>
#include <vector>

#include "orttraining/core/optimizer/pythonop_rewriter.h"

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/framework/torch/torch_proxy.h"
#include "orttraining/core/framework/torch/custom_function_register.h"

namespace onnxruntime {

Status PythonOpRewriter::Apply(Graph&, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  bool modified = false;
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "PythonOp", {1}, kMSDomain) &&
      node.GetAttributes().find("tensor_reuse_map") == node.GetAttributes().end()) {
    auto func_name = static_cast<std::string>(node.GetAttributes().at("func_name").s());
    std::optional<PyObject*> input_alias_function =
        language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().TryGettingInputAliasFunction(func_name);
    if (input_alias_function.has_value()) {
      // Serialize node proto to string
      ONNX_NAMESPACE::NodeProto node_proto;
      node.ToProto(node_proto);
      std::string node_proto_str;
      node_proto.SerializeToString(&node_proto_str);

      // Call input alias function
      std::vector<int64_t> fw_all_output_to_tensor_input_reuse_map;
      std::vector<int64_t> bw_all_output_to_tensor_input_reuse_map;
      language_interop_ops::torch::TorchProxy::GetInstance().RunInputAliasFunction(
          static_cast<void*>(input_alias_function.value()),
          node_proto_str,
          fw_all_output_to_tensor_input_reuse_map,
          bw_all_output_to_tensor_input_reuse_map);

      auto input_convention = static_cast<std::string>(node.GetAttributes().at("input_convention").s());
      {
        // Handle forward input alias map.
        std::vector<int64_t> fw_tensor_output_to_tensor_input_reuse_map =
            std::vector<int64_t>((node.OutputDefs().size()), -1);

        // Map input index from `global` input index to `tensor` input index, because node.InputDefs() only contains
        // tensor inputs.
        std::unordered_map<int64_t, int64_t> position_to_tensor_index;
        int64_t tensor_index = 0;
        const size_t all_input_count = input_convention.size();
        position_to_tensor_index.reserve(all_input_count);
        for (size_t i = 0; i < all_input_count; ++i) {
          if (input_convention[i] == 'd') {
            position_to_tensor_index[i] = tensor_index;
            ++tensor_index;
          }
        }

        for (size_t i = 1; i < fw_tensor_output_to_tensor_input_reuse_map.size(); ++i) {
          if (fw_all_output_to_tensor_input_reuse_map[i - 1] != -1) {
            ORT_ENFORCE(fw_all_output_to_tensor_input_reuse_map[i - 1] < static_cast<int64_t>(all_input_count),
                        "PythonOp input alias function output index out of range. func_name: ", func_name, " ",
                        fw_all_output_to_tensor_input_reuse_map[i - 1], " >= ", all_input_count);
            fw_tensor_output_to_tensor_input_reuse_map[i] =
                position_to_tensor_index.at(fw_all_output_to_tensor_input_reuse_map[i - 1]);
          }
        }

        node.AddAttribute("tensor_reuse_map", fw_tensor_output_to_tensor_input_reuse_map);
      }

      {
        // Handle backward input alias map.
        auto& output_convention = input_convention;
        ORT_ENFORCE(bw_all_output_to_tensor_input_reuse_map.size() == output_convention.size(),
                    "PythonOpGrad input alias function output count mismatch. func_name: ", func_name, " ",
                    bw_all_output_to_tensor_input_reuse_map.size(), " != ", output_convention.size());

        std::vector<int64_t> bw_tensor_output_to_tensor_input_reuse_map =
            std::vector<int64_t>(node.InputDefs().size(), -1);
        size_t tensor_output_index = 0;
        for (size_t i = 0; i < output_convention.size(); ++i) {
          if (output_convention[i] == 'd') {
            ORT_ENFORCE(tensor_output_index < bw_tensor_output_to_tensor_input_reuse_map.size(),
                        "PythonOpGrad input alias function output count mismatch. func_name: ", func_name, " ",
                        tensor_output_index, " >= ", bw_tensor_output_to_tensor_input_reuse_map.size());
            // input index shift by 1 to skip the context
            bw_tensor_output_to_tensor_input_reuse_map[tensor_output_index] =
                bw_all_output_to_tensor_input_reuse_map[i] == -1 ? -1 : bw_all_output_to_tensor_input_reuse_map[i] + 1;
            ++tensor_output_index;
          }
        }
        node.AddAttribute("bw_tensor_reuse_map", bw_tensor_output_to_tensor_input_reuse_map);
      }

      modified = true;
    }
  }

  if (modified)
    rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;

  return Status::OK();
}

bool PythonOpRewriter::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const {
  return true;
}

}  // namespace onnxruntime

#endif
