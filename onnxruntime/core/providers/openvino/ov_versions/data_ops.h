#pragma once
#include <unordered_set>

namespace onnxruntime {
namespace openvino_ep {

typedef enum version_num { 
  V_2020_4,
  V_2021_1,
  V_2021_2
} version_id_e;

using ConfirmationFunction = std::function<bool(const Node*, const Provider_InitializedTensorSet&)>;

typedef struct supportedop {
  version_id_e version;
  std::vector<std::string> device_type;  
  std::string optype;
} supportedop_t;

typedef struct unsupportedopmode {
  std::vector<version_id_e> ver;
  ConfirmationFunction func; 
} unsupportedopmode_t;

class Capability{

private:
const GraphViewer& graph_viewer;
version_id_e version_id;
std::string device_id;
std::multimap<std::string, unsupportedopmode_t &> _confirmation_map;
std::vector<supportedop_t> subgraph_supported;
std::vector<supportedop_t> no_dimension_supported;
std::set<int> supported_types_vpu; 
std::set<int> supported_types_cpu;
std::set<int> supported_types_gpu;
std::set<int> supported_types_initializer;

protected:
  virtual void populate_op_mode_supported();
  virtual void populate_types_supported();
  bool check_if_op_is_supported(std::string name, std::vector<supportedop_t>& list);
  bool check_if_dimension_unsupported(const Node* node);
  bool check_if_unsupported_op_mode(const Node* node);
  bool check_if_type_is_supported(const NodeArg* node_arg, bool is_initializer);
  bool check_if_node_is_supported(const std::map<std::string, 
                                  std::set<std::string>>& op_map,
                                  const NodeIndex node_idx);
   
public:
  Capability(const GraphViewer& graph_viewer_param, version_id_e ver, std::string dev_id):
            graph_viewer(graph_viewer_param), version_id(ver), device_id(dev_id)  {
    populate_op_mode_supported();
    populate_types_supported();
  }

  virtual std::vector<NodeIndex> GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers);
  virtual bool IsOpSupportedOnlyInModel(std::string name);
  virtual bool CheckSpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers, const Node* node);
  virtual bool DoNotOmitSubGraph(const std::string& name);
  virtual bool NodePushBack(const Node* node, const std::string& name);

};

}  //namespace openvino_ep
}  //namespace onnxruntime
