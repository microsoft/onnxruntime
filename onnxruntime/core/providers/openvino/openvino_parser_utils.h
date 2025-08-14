#pragma once

#include <list>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "core/framework/provider_options.h"
#include "core/providers/openvino/contexts.h"

namespace onnxruntime {
namespace openvino_ep {

class OpenVINOParserUtils {
 public:
  static std::string ParsePrecision(const ProviderOptions& provider_options,
                                    std::string& device_type,
                                    const std::string& option_name);
  static reshape_t ParseInputShape(const std::string& reshape_input_definition);
  static layout_t ParseLayout(const std::string& layout_definition);
  static std::string TrimWhitespace(const std::string& str);
  static ov::Dimension ParseDimensionRange(const std::string& range_str, const std::string& tensor_name);
  static bool Check_Valid_Layout(const std::string& layout_str, const std::string& tensor_name);
};

}  // namespace openvino_ep
}  // namespace onnxruntime
