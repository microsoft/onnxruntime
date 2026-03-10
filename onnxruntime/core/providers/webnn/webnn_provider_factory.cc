// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/webnn_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "webnn_execution_provider.h"

#include "core/common/parse_string.h"

#include <limits>
#include <string_view>

using namespace onnxruntime;

namespace onnxruntime {

namespace {

Status ParsePositiveInt32(std::string_view input, int32_t& output) {
  int64_t value = 0;
  ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(input, value));
  ORT_RETURN_IF(value <= 0, "value must be positive.");
  ORT_RETURN_IF(value > std::numeric_limits<int32_t>::max(), "value exceeds int32 range.");
  output = static_cast<int32_t>(value);
  return Status::OK();
}

Status ParseFreeDimensionBounds(std::string_view value, webnn::FreeDimensionBounds& bounds) {
  bounds.clear();

  if (value.empty()) {
    return Status::OK();
  }

  size_t entry_begin = 0;
  while (entry_begin < value.size()) {
    const size_t entry_end = value.find(';', entry_begin);
    const std::string_view entry =
        (entry_end == std::string_view::npos) ? value.substr(entry_begin) : value.substr(entry_begin, entry_end - entry_begin);

    ORT_RETURN_IF(entry.empty(), "empty entry is not allowed.");

    const size_t colon1 = entry.find(':');
    ORT_RETURN_IF(colon1 == std::string_view::npos, "entry must be in 'name:minSize:maxSize' format.");

    const size_t colon2 = entry.find(':', colon1 + 1);
    ORT_RETURN_IF(colon2 == std::string_view::npos, "entry must be in 'name:minSize:maxSize' format.");
    ORT_RETURN_IF(entry.find(':', colon2 + 1) != std::string_view::npos,
                  "entry must contain exactly two ':' separators.");

    const std::string_view name = entry.substr(0, colon1);
    ORT_RETURN_IF(name.empty(), "dimension name must not be empty.");

    const std::string_view min_size_str = entry.substr(colon1 + 1, colon2 - colon1 - 1);
    const std::string_view max_size_str = entry.substr(colon2 + 1);
    ORT_RETURN_IF(max_size_str.empty(), "maxSize must not be empty.");

    webnn::FreeDimensionBound bound{};
    if (!min_size_str.empty()) {
      ORT_RETURN_IF_ERROR(ParsePositiveInt32(min_size_str, bound.min_size));
    }
    ORT_RETURN_IF_ERROR(ParsePositiveInt32(max_size_str, bound.max_size));
    ORT_RETURN_IF(bound.max_size < bound.min_size,
                  "maxSize must be greater than or equal to minSize for dimension: ",
                  std::string(name));

    ORT_RETURN_IF(bounds.find(std::string(name)) != bounds.end(),
                  "duplicate dimension name in FreeDimensionBounds: ", std::string(name));
    bounds.emplace(std::string(name), bound);

    if (entry_end == std::string_view::npos) {
      break;
    }
    entry_begin = entry_end + 1;
  }

  return Status::OK();
}

}  // namespace

struct WebNNProviderFactory : IExecutionProviderFactory {
  explicit WebNNProviderFactory(const std::string& webnn_device_flags,
                                const webnn::FreeDimensionBounds& free_dimension_bounds)
      : webnn_device_flags_(webnn_device_flags),
        free_dimension_bounds_(free_dimension_bounds) {}
  ~WebNNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  std::string webnn_device_flags_;
  webnn::FreeDimensionBounds free_dimension_bounds_;
};

std::unique_ptr<IExecutionProvider> WebNNProviderFactory::CreateProvider() {
  return std::make_unique<WebNNExecutionProvider>(webnn_device_flags_, free_dimension_bounds_);
}

std::shared_ptr<IExecutionProviderFactory> WebNNProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  const auto device_type_it = provider_options.find("deviceType");
  const std::string webnn_device_flags =
      (device_type_it != provider_options.end()) ? device_type_it->second : "cpu";

  webnn::FreeDimensionBounds free_dimension_bounds;
  const auto free_dimension_bounds_it = provider_options.find("FreeDimensionBounds");
  if (free_dimension_bounds_it != provider_options.end()) {
    ORT_THROW_IF_ERROR(ParseFreeDimensionBounds(free_dimension_bounds_it->second, free_dimension_bounds));
  }

  return std::make_shared<onnxruntime::WebNNProviderFactory>(webnn_device_flags, free_dimension_bounds);
}

}  // namespace onnxruntime
