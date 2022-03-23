#include <cmath>
#include <numeric>
#include <list>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>

#include "orttraining/core/graph/gradient_builder.h"

#include <cmath>
#include <numeric>
#include <list>
#include <stack>

#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/common/safeint.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "core/framework/iexecutor.h"
#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

class PInput {
};

namespace GraphParser {

// TODO: move this into pattern_graph???
Status TryReplace(Graph& graph, PatternGraph& pattern, const NodeDef& alternative,
                  std::vector<std::pair<std::string, int>> fusion_inputs,
                  std::vector<std::pair<std::string, int>> fusion_outputs);

bool HasSingleSpeciefiedConstantValue(const Graph& graph, const Node* node, double value);

int GetConstantInitializerCount(const Graph& graph, const Node* node);

Node* GetNodeOfPatternNodeName(const Graph& graph, const std::vector<PNN>& collection, const std::string& name);

NodeArg* GetNodeArgWithName(Graph& graph, const std::vector<PNN>& collection, std::string name, int idx);

template <typename T>
std::vector<T> GetConstantInitializers(const Graph& graph, const Node* node);

}  // namespace GraphParser

}  // namespace training
}  // namespace onnxruntime