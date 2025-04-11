#pragma once

#include <list>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "core/framework/provider_options.h"

namespace onnxruntime {
namespace openvino_ep {

class OpenVINOParserUtils {
 public:
  static std::string ParsePrecision(const ProviderOptions& provider_options,
                                    std::string& device_type,
                                    const std::string& option_name);
};

}  // namespace openvino_ep
}  // namespace onnxruntime
