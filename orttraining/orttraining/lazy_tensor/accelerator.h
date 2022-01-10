#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>

using CompiledCode = std::function<std::vector<c10::IValue>(
    at::ArrayRef<c10::IValue>&)>;

class Accelerator {
 public:
  Accelerator(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)) {}
  void run(torch::jit::Stack& stack);
  static bool supported(const torch::jit::Node* node);

 private:
  CompiledCode compile(at::ArrayRef<c10::IValue>&);
  std::shared_ptr<torch::jit::Graph> subgraph_;
  // TODO: Cache graph.
  std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledCode> cache_;
};
