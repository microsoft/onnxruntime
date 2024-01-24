// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/inlined_containers_fwd.h>

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
  CustomConfigType& PushCustomConfig() {
    custom_configs_.push_back(custom_config_init_);
    return custom_configs_.back();
  }

  /**
   * Creates and returns a reference to a new QNN configuration object. The object is initialized to
   * the QNN recommended default value. The caller is meant to override fields in this object.
   *
   * \return A reference to a default BaseConfigType object.
   */
  BaseConfigType& PushConfig() {
    configs_.push_back(base_config_init_);
    BaseConfigType& config = configs_.back();

    // Add pointer to this new config to the list of config pointers.
    if (IsNullTerminated()) {
      config_ptrs_.back() = &config;  // Replace last nullptr entry.
    } else {
      config_ptrs_.push_back(&config);
    }

    return config;
  }

 private:
  bool IsNullTerminated() const {
    return !config_ptrs_.empty() && config_ptrs_.back() == nullptr;
  }

  BaseConfigType base_config_init_;
  CustomConfigType custom_config_init_;
  InlinedVector<CustomConfigType> custom_configs_;
  InlinedVector<BaseConfigType> configs_;
  InlinedVector<const BaseConfigType*> config_ptrs_;
};

}  // namespace qnn
}  // namespace onnxruntime
