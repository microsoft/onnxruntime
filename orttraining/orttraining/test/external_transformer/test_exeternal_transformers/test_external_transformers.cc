
#include "core/optimizer/graph_transformer.h"
#include "orttraining/core/optimizer/graph_transformer_registry.h"
#include "onnx/defs/schema.h"
#include <memory>
#include <iostream>

namespace onnxruntime {
namespace training {

class MyGraphTransformer : public GraphTransformer {
 public:
  MyGraphTransformer(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MyGraphTransformer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& /*graph*/, bool& /*modified*/, int /*graph_level*/, const logging::Logger& /*logger*/) const override {
    std::cout << "******************Trigger Customized Graph Transformer:  MyGraphTransformer!" << std::endl;
    return Status::OK();
  }
};

void RegisterTrainingExternalTransformers() {
  ONNX_REGISTER_EXTERNAL_GRAPH_TRANSFORMER(MyGraphTransformer, Level1, true);
}

}
}
