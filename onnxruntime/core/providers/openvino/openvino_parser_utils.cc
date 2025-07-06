#include <algorithm>
#include <regex>
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

reshape_t OpenVINOParserUtils::ParseInputShape(const std::string& reshape_input_definition) {
  reshape_t parsed_shape_map;

  // Return empty map for empty input
  if (reshape_input_definition.empty()) {
    ORT_THROW("Empty input shape definition provided in reshape_input parameter");
  }

  // Regular expressions for parsing
  const std::regex tensor_pattern(R"(([^\[\],]+)\s*\[(.*?)\])");  // e.g. "input_1[1..5, 2, 3..4],data[1,2,3]"
  // const std::regex dimension_pattern(R"(\s*(\d+(?:\.\.\d+)?)\s*)");  // e.g. "1..5", "2", "3..4"
  const std::regex dimension_pattern(R"(\s*([^,\s]+)\s*)");
  // Find all tensor shape definitions using regex
  auto tensor_begin = std::sregex_iterator(
      reshape_input_definition.begin(),
      reshape_input_definition.end(),
      tensor_pattern);
  auto tensor_end = std::sregex_iterator();

  // If no matches found, throw error
  if (tensor_begin == tensor_end) {
    ORT_THROW("Invalid input shape definition format: " + reshape_input_definition);
  }

  // Process each tensor definition e.g. "input_1[1..5, 2, 3..4],data[1,2,3]"
  for (std::sregex_iterator i = tensor_begin; i != tensor_end; ++i) {
    std::smatch tensor_match = *i;

    // Extract tensor name and trim whitespace
    std::string tensor_name = tensor_match[1].str();  // Group 1: tensor name e.g. "input_1"
    tensor_name = TrimWhitespace(tensor_name);

    if (tensor_name.empty()) {
      ORT_THROW("Empty tensor name provided in reshape_input parameter");
    }

    // Extract dimensions string
    std::string dimensions_str = tensor_match[2].str();  // Group 2: dimensions string [e.g. "1..5, 2, 3..4"]
    std::vector<ov::Dimension> dimensions;

    // Find all dimension e.g. "1..5", "2", "3..4" using regex
    auto dim_begin = std::sregex_iterator(
        dimensions_str.begin(),
        dimensions_str.end(),
        dimension_pattern);
    auto dim_end = std::sregex_iterator();

    // Process each dimension
    for (std::sregex_iterator j = dim_begin; j != dim_end; ++j) {
      std::smatch dim_match = *j;
      std::string dim_value = dim_match[1].str();

      // Check if dimension is a range
      size_t range_separator_pos = dim_value.find("..");
      if (range_separator_pos != std::string::npos) {
        // Parse range
        dimensions.push_back(ParseDimensionRange(dim_value, tensor_name));
      } else {
        // Parse single value
        bool is_valid_integer = !dim_value.empty() &&
                                std::all_of(dim_value.begin(), dim_value.end(), [](char c) {
                                  return std::isdigit(static_cast<unsigned char>(c));
                                });

        if (!is_valid_integer) {
          ORT_THROW("Invalid dimension value: '" + dim_value + "' for tensor: " + tensor_name);
        }

        dimensions.push_back(std::stoi(dim_value));
      }
    }

    // Store parsed shape in result map
    parsed_shape_map[tensor_name] = ov::PartialShape(dimensions);
  }

  return parsed_shape_map;
}

// Helper function to trim whitespace from a string
std::string OpenVINOParserUtils::TrimWhitespace(const std::string& str) {
  const std::string whitespace = " \t\n\r\f\v";
  size_t start = str.find_first_not_of(whitespace);

  if (start == std::string::npos) {
    return "";
  }

  size_t end = str.find_last_not_of(whitespace);
  return str.substr(start, end - start + 1);
}

// Helper function to parse dimension range (e.g. "1..5")
ov::Dimension OpenVINOParserUtils::ParseDimensionRange(const std::string& range_str, const std::string& tensor_name) {
  size_t range_separator_pos = range_str.find("..");
  if (range_separator_pos == std::string::npos) {
    ORT_THROW("Invalid dimension range format: " + range_str);
  }

  std::string range_start_str = TrimWhitespace(range_str.substr(0, range_separator_pos));
  std::string range_end_str = TrimWhitespace(range_str.substr(range_separator_pos + 2));

  // Validate range values
  if (range_start_str.empty() || range_end_str.empty() ||
      !std::all_of(range_start_str.begin(), range_start_str.end(), ::isdigit) ||
      !std::all_of(range_end_str.begin(), range_end_str.end(), ::isdigit)) {
    ORT_THROW("Invalid dimension range format: '" + range_str + "' for tensor: " + tensor_name);
  }

  int range_start = std::stoi(range_start_str);
  int range_end = std::stoi(range_end_str);

  if (range_start > range_end) {
    ORT_THROW("Invalid dimension range (start > end): " + range_str + " for tensor: " + tensor_name);
  }

  return ov::Dimension(range_start, range_end);
}

}  // namespace openvino_ep
}  // namespace onnxruntime
