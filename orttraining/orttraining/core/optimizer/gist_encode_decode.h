// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GistEncode
*/
class GistEncodeDecode : public RewriteRule {
 public:
  int op;
  std::string compression_type;

  static constexpr const char* GIST_PAIR_NODE_NAME_BASE = "gist";
  
  static constexpr int GIST_PACK1_FACTOR = 8;
  
  // map stores GIST signature - source operator type to destination operator type(s)
  typedef std::vector<std::string> vector_t;
  const std::unordered_map<std::string, vector_t> PATTERN_MAP = {
   {"Softmax",             {"SoftmaxGrad"}},
   {"Transpose",           {"Transpose"}},
   {"Reshape",             {"Reshape"}},
   {"Add",                 {"LayerNormalizationGrad"}},
   {"Dropout",             {"Transpose", "Reshape", "DropoutGrad"}},
   {"LayerNormalization",  {"Reshape", "Shape", "LayerNormalizationGrad"}},
   {"MatMul",              {"Shape"}},
   {"Relu",                {"ReluGrad", "Shape", "Reshape"}}
  };
  
  GistEncodeDecode() noexcept : RewriteRule("GistEncodeDecode") {}
  GistEncodeDecode(int op_flag, std::string compr_type) noexcept : RewriteRule("GistEncodeDecode") {
    op = op_flag;
    compression_type = compr_type;
  }
  
  std::vector<std::string> TargetOpTypes() const noexcept override {
     if(op == 1) {
         return{"Softmax"};
      }
      else if(op == 2) {
         return{"Transpose"};
      }
      else if(op == 3) {
         return{"Reshape"};
      }
      else if(op == 4) {
         return{"Add"};
      }
      else if(op == 5) {
         return{"Dropout"};
      }
      else if(op == 6) {
         return{"LayerNormalization"};
      }
      else if(op == 7) {
         return{"MatMul"};
      }
      else if(op == 8) {
         return{"Relu"};
      }
      else if (op == 9) {
         return {"Softmax", "Transpose", "Reshape", "Add", "Dropout", "LayerNormalization", "MatMul", "Relu"};
      }
      else {
         std::cout << "Gist op type not supported" << std::endl;
         return {};
      }
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;
  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
  bool AddEncodeDecode(Graph& graph, Node& curr_node, std::string compression_type) const;
};

}
