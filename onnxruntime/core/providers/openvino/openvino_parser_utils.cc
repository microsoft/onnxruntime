#include <algorithm>
#include "core/providers/openvino/openvino_parser_utils.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace openvino_ep {

std::string OpenVINOParserUtils::ParsePrecision(const ProviderOptions& provider_options,
                                                std::string& device_type,
                                                const std::string& option_name) {
  using DeviceName = std::string;
  using DefaultValue = std::string;
  using ValidValues = std::list<std::string>;
  using foo = std::pair<DefaultValue, ValidValues>;
  using ParserHelper = std::map<DeviceName, foo>;

  ParserHelper helper = {
      {"GPU", {"FP16", {"FP16", "FP32", "ACCURACY"}}},
      {"NPU", {"FP16", {"FP16", "ACCURACY"}}},
      {"CPU", {"FP32", {"FP32", "ACCURACY"}}},
  };

  std::set<std::string> deprecated_device_types = {
      "CPU_FP32", "GPU_FP32", "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
      "GPU.0_FP16", "GPU.1_FP16"};

  bool is_composite = device_type.find(':') != std::string::npos;  // FOR devices AUTO:,HETERO:,MULTI:

  if (provider_options.contains(option_name)) {
    const auto& precision = provider_options.at(option_name);

    if (is_composite) {
      std::set<std::string> allowed_precisions = {"FP16", "FP32", "ACCURACY"};
      if (allowed_precisions.contains(precision)) {
        return precision;
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ",
                  precision, ".\n");
      }
    } else {
      if (helper.contains(device_type)) {
        auto const& valid_values = helper[device_type].second;

        if (precision == "ACCURACY") {
          return valid_values.back();  // Return highest supported precision
        } else {
          if (std::find(valid_values.begin(), valid_values.end(), precision) != valid_values.end()) {
            return precision;  // Return precision selected if valid
          } else {
            auto value_iter = valid_values.begin();
            std::string valid_values_joined = *value_iter;
            // Append 2nd and up, if only one then ++value_iter is same as end()
            for (++value_iter; value_iter != valid_values.end(); ++value_iter) {
              valid_values_joined += ", " + *value_iter;
            }

            ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ",
                      device_type, " only supports", valid_values_joined, ".\n");
          }
        }
      } else if (deprecated_device_types.contains(device_type)) {
        LOGS_DEFAULT(WARNING)
            << "[OpenVINO] Selected 'device_type' " + device_type + " is deprecated. \n"
            << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
            << "'GPU.1', 'NPU' or from HETERO/MULTI/AUTO options and set 'precision' separately. \n";
        auto delimit = device_type.find("_");
        device_type = device_type.substr(0, delimit);
        return device_type.substr(delimit + 1);
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported device type provided: ",
                  device_type, "\n");
      }
    }
  } else {
    if (device_type.find("NPU") != std::string::npos || device_type.find("GPU") != std::string::npos) {
      return "FP16";
    } else if (device_type.find("CPU") != std::string::npos) {
      return "FP32";
    } else {
      ORT_THROW("[ERROR] [OpenVINO] Unsupported device is selected", device_type, "\n");
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
