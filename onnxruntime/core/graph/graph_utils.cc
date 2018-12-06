
#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace utils {
  // fusion is only done for ONNX domain ops
  bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                         const std::string& op_type,
                                         ONNX_NAMESPACE::OperatorSetVersion version,
                                         const std::string& domain) {
    if (node.OpType() != op_type ||
        node.Op()->Deprecated() || node.Op()->SinceVersion() != version ||
        (!node.Domain().empty() && node.Domain() != domain)) {
      return false;
    }
    return true;
  }
}

}  // namespace onnxruntime