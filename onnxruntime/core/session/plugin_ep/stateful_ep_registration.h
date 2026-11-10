// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/stateful_execution_provider.h"

namespace onnxruntime {

class RegisteredStatefulProvider {
 public:
  static Status Create(const OrtStatefulProviderDescriptor& descriptor,
                       std::shared_ptr<RegisteredStatefulProvider>& out);
  ~RegisteredStatefulProvider();
  const OrtStatefulProviderDescriptor& Descriptor() const { return descriptor_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RegisteredStatefulProvider);

 private:
  RegisteredStatefulProvider() = default;
  OrtStatefulProviderDescriptor descriptor_{};
};

// Registration is transactional within synchronous OrtEpFactory::CreateEp.
class StatefulEpCreationScope {
 public:
  StatefulEpCreationScope();
  ~StatefulEpCreationScope();
  static Status Register(OrtEp* ep, const OrtStatefulProviderDescriptor* descriptor);
  Status Take(OrtEp* ep, std::shared_ptr<RegisteredStatefulProvider>& out);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulEpCreationScope);

 private:
  static thread_local StatefulEpCreationScope* current_;
  StatefulEpCreationScope* previous_;
  InlinedHashMap<OrtEp*, std::shared_ptr<RegisteredStatefulProvider>> pending_;
};

}  // namespace onnxruntime
