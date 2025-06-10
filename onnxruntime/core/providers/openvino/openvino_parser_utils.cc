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
  using ValidValues = std::vector<std::string>;
  using DefaultAndValidPair = std::pair<DefaultValue, ValidValues>;
  using ParserHelper = std::unordered_map<DeviceName, DefaultAndValidPair>;
  // {Device prefix, {Default precision, {Supported precisions}}}
  ParserHelper helper = {
      {"GPU", {"FP16", {"FP16", "FP32", "ACCURACY"}}},
      {"NPU", {"FP16", {"FP16", "ACCURACY"}}},
      {"CPU", {"FP32", {"FP32", "ACCURACY"}}},
  };

  // If we have multiple device configuration, request precision from user and check it
  if ((device_type.find("HETERO:") == 0) ||
      (device_type.find("MULTI:") == 0) ||
      (device_type.find("BATCH:") == 0) ||
      (device_type.find("AUTO:") == 0)) {
    if (!provider_options.contains(option_name)) {
      LOGS_DEFAULT(INFO) << "[OpenVINO] Precision is not set. Using default OpenVINO precision for " + device_type + ". \n";
      return "";
    } else {
      std::unordered_set<std::string> supported_precisions = {"FP16", "FP32", "ACCURACY"};
      std::string precision = provider_options.at(option_name);
      if (supported_precisions.contains(precision)) {
        return precision;
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported precision for the ", device_type, " device. Device supports only FP16, FP32, ACCURACY.\n");
      }
    }
  }

  // Deprecated device specification (CPU_FP32, GPU.0_FP32, etc.)
  if (auto delimit = device_type.find("_"); delimit != std::string::npos) {
    if (provider_options.contains(option_name)) {
      ORT_THROW("[ERROR] [OpenVINO] Precision is specified twice, please remove the _precision suffix from device name and only set the precision separately.\n");
    }
    LOGS_DEFAULT(WARNING) << "[OpenVINO] Selected 'device_type' " + device_type + " is deprecated. \n"
                          << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
                          << "'GPU.1', 'NPU' or from"
                          << " HETERO/MULTI/AUTO/BATCH options and set 'precision' separately. \n";
    std::string precision = device_type.substr(delimit + 1);
    // Device type is updated in-place
    device_type = device_type.substr(0, delimit);
    // We have to remove the index (.0, .1, etc.) to use device as key for helper
    std::string device_prefix = device_type;
    if (auto dot_delimit = device_prefix.find("."); dot_delimit != std::string::npos) {
      device_prefix = device_prefix.substr(0, dot_delimit);
    }

    if (!helper.contains(device_prefix)) {
      ORT_THROW("[ERROR] [OpenVINO] Selected 'device_type' " + device_type + " is not supported with precision suffix. \n");
    }
    const auto& valid_values = helper[device_prefix].second;
    if (std::find(std::begin(valid_values), std::end(valid_values), precision) != std::end(valid_values)) {
      return precision;
    } else {
      auto value_iter = valid_values.begin();
      std::string valid_values_joined = *value_iter;
      // Append 2nd and up, if only one then ++value_iter is same as end()
      for (++value_iter; value_iter != valid_values.end(); ++value_iter) {
        valid_values_joined += ", " + *value_iter;
      }

      ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ", device_type, " only supports ", valid_values_joined, ".\n");
    }
  }

  // Deprecated devices are already handled above
  // We have to remove the index (.0, .1, etc.) to use device as key for helper
  auto device_prefix = device_type;
  if (auto dot_delimit = device_prefix.find("."); dot_delimit != std::string::npos) {
    device_prefix = device_prefix.substr(0, dot_delimit);
  }

  if (provider_options.contains(option_name)) {
    std::string precision = provider_options.at(option_name);

    if (helper.contains(device_prefix)) {
      auto const& valid_values = helper[device_prefix].second;
      if (std::find(std::begin(valid_values), std::end(valid_values), precision) != std::end(valid_values)) {
        return precision;  // Return precision selected if valid
      } else {
        auto value_iter = valid_values.begin();
        std::string valid_values_joined = *value_iter;
        // Append 2nd and up, if only one then ++value_iter is same as end()
        for (++value_iter; value_iter != valid_values.end(); ++value_iter) {
          valid_values_joined += ", " + *value_iter;
        }

        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ", device_type, " only supports ", valid_values_joined, ".\n");
      }
    } else {
      // Not found in helper - custom device, return as is
      return precision;
    }
  } else {
    // Precision not set
    if (helper.contains(device_prefix)) {
      // If found in helper - set the default
      return helper[device_prefix].first;
    } else {
      // Not found in helper - custom device - default precision
      LOGS_DEFAULT(INFO) << "[OpenVINO] Precision is not set. Using default OpenVINO precision for " + device_type + ". \n";
      return "";
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
