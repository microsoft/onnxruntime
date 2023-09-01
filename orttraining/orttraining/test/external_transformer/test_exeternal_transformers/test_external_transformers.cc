#include "core/optimizer/rewrite_rule.h"
#include "orttraining/core/optimizer/graph_transformer_registry.h"
#include "onnx/defs/schema.h"
#include <memory>
#include <iostream>

namespace onnxruntime {
namespace training {

class MyRewriteRule : public RewriteRule {
 public:
  MyRewriteRule() noexcept
      : RewriteRule("MyRewriteRule") {
  }
  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {};
  }

 private:
  bool SatisfyCondition(const Graph& /*graph*/, const Node& /*node*/, const logging::Logger& /*logger*/) const override {
    return true;
  }

  Status Apply(Graph& /*graph*/, Node& /*node*/, RewriteRuleEffect& /*rule_effect*/, const logging::Logger& /*logger*/) const override {
    std::cout << "******************Trigger Customized Graph Transformer:  MyGraphTransformer!" << std::endl;
    return Status::OK();
  }
};

void RegisterTrainingExternalTransformers() {
  ONNX_REGISTER_EXTERNAL_REWRITE_RULE(MyRewriteRule, Level1, true);
}

}  // namespace training
}  // namespace onnxruntime
