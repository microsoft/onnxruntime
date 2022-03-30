#include "core/graph/graph_utils.h"
#include "orttraining/core/graph/graph_parser/pattern_graph.h"

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