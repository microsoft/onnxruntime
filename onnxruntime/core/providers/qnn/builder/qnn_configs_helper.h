// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

namespace onnxruntime {
namespace qnn {

/**
 * Helper class for building a null-terminated list of QNN configurations.
 * A QNN configuration consists of multiple objects with references to each other. This
 * class ensures that all configuration objects have the same lifetime, so that they remain valid
 * across calls to qnn_interface.xxxCreate().
 */
template <typename BaseConfigType, typename CustomConfigType>
class QnnConfigsBuilder {
 public:
  /**
   * Initializes the config build. Provide the initial/default value for each config struct type.
   * \param base_config_init The initial/default value for objects of type BaseConfigType.
   * \param custom_config_init The initial/default value for objects of type CustomConfigType.
   */
  QnnConfigsBuilder(BaseConfigType base_config_init, CustomConfigType custom_config_init)
      : base_config_init_(std::move(base_config_init)), custom_config_init_(std::move(custom_config_init)) {}

  /**
   * Returns a pointer to the beginning of a null-terminated array of QNN base configurations.
   * This result is typically passed to QNN's xxxCreate() APIs.
   *
   * \return Pointer to null-terminated BaseConfigType* array.
   */
  const BaseConfigType** GetQnnConfigs() {
    if (config_ptrs_.empty()) {
      return nullptr;
    }

    if (!IsNullTerminated()) {
      config_ptrs_.push_back(nullptr);
    }

    return config_ptrs_.data();
  }

  /**
   * Creates and returns a reference to a new custom QNN configuration object. The object is initialized to
   * the QNN recommended default value. The caller is meant to override fields in this object.
   *
   * \return A reference to a default CustomConfigType object.
   */
  gsl::not_null<CustomConfigType*> PushCustomConfig() {
    custom_configs_.push_back(std::make_unique<CustomConfigType>(custom_config_init_));
    return custom_configs_.back().get();
  }

  /**
   * Creates and returns a reference to a new QNN configuration object. The object is initialized to
   * the QNN recommended default value. The caller is meant to override fields in this object.
   *
   * \return A reference to a default BaseConfigType object.
   */
  gsl::not_null<BaseConfigType*> PushConfig() {
    configs_.push_back(std::make_unique<BaseConfigType>(base_config_init_));
    BaseConfigType* config = configs_.back().get();

    // Add pointer to this new config to the list of config pointers.
    if (IsNullTerminated()) {
      config_ptrs_.back() = config;  // Replace last nullptr entry.
    } else {
      config_ptrs_.push_back(config);
    }

    return config;
  }

 private:
  bool IsNullTerminated() const {
    return !config_ptrs_.empty() && config_ptrs_.back() == nullptr;
  }

  BaseConfigType base_config_init_;
  CustomConfigType custom_config_init_;

  // Store elements of unique_ptrs instead of by value because std::vector reallocation would change the
  // location of elements in memory. BaseConfigType objects may contain pointers to CustomConfigType objects,
  // so we need to make sure that pointers to these objects are stable in memory.
  std::vector<std::unique_ptr<CustomConfigType>> custom_configs_;
  std::vector<std::unique_ptr<BaseConfigType>> configs_;

  std::vector<const BaseConfigType*> config_ptrs_;
};

}  // namespace qnn
}  // namespace onnxruntime
