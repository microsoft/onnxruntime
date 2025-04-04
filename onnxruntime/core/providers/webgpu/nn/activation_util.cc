#include "core/providers/webgpu/nn/activation_util.h"
#include "core/common/common.h"
namespace onnxruntime {
namespace webgpu {
std::string TypeSnippet(uint32_t component, std::string data_type) {
  switch (component) {
    case 1:
      return data_type;
    case 2:
      return "vec2<" + data_type + ">";
    case 3:
      return "vec3<" + data_type + ">";
    case 4:
      return "vec4<" + data_type + ">";
    default:
      ORT_THROW("Component ", component, " is not supported.");
  }
}

std::string BiasSnippet(bool has_bias) {
  return has_bias ? "value = value + getBiasByOutputCoords(coords);" : "";
}

}  // namespace webgpu
}  // namespace onnxruntime
