#include <torch/csrc/jit/ir/ir.h>

namespace onnxruntime {
namespace lazytensor {

void OrtFuseGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const std::function<bool(torch::jit::Node*)>& fn,
    torch::jit::Symbol kind,
    size_t arg_limit = std::numeric_limits<size_t>::max());

}  // namespace lazytensor
}  // namespace onnxruntime
