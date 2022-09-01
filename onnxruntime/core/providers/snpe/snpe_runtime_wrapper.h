// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/RuntimeList.hpp"

namespace onnxruntime {
namespace contrib {
namespace snpe {

class SnpeRuntimeWrapper {
 public:
  SnpeRuntimeWrapper() {
    runtime_ =
#if defined(_M_X64)
        zdl::DlSystem::Runtime_t::CPU;
#else
        zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
#endif
  }

  explicit SnpeRuntimeWrapper(const std::string& runtime) : SnpeRuntimeWrapper() {
    Set(runtime);
  }

  ~SnpeRuntimeWrapper() = default;

  zdl::DlSystem::Runtime_t Get() const { return runtime_; }

  void Set(const std::string& runtime);

  void Set(zdl::DlSystem::Runtime_t runtime) { runtime_ = runtime; }

  bool IsAvailable() const {
    zdl::DlSystem::RuntimeCheckOption_t runtime_check_option = zdl::DlSystem::RuntimeCheckOption_t::DEFAULT;
    // check availability, explicitly requiring unsignedpd support
    if (runtime_ == zdl::DlSystem::Runtime_t::DSP) {
      runtime_check_option = zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK;
    }
    return zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_, runtime_check_option);
  }

  std::string ToString() const {
    return std::string(zdl::DlSystem::RuntimeList::runtimeToString(runtime_));
  }

 private:
  zdl::DlSystem::Runtime_t runtime_;
};

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
