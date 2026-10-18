// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/stateful_ep_registration.h"

namespace onnxruntime {

Status RegisteredStatefulProvider::Create(
    const OrtStatefulProviderDescriptor& descriptor, std::shared_ptr<RegisteredStatefulProvider>& out) {
  ORT_RETURN_IF_ERROR(StatefulProviderBinding::ValidateDescriptor(descriptor));
  auto value = std::shared_ptr<RegisteredStatefulProvider>(new RegisteredStatefulProvider());
  value->descriptor_ = descriptor;
  if (descriptor.context) {
    descriptor.Retain(descriptor.context);
  }
  out = std::move(value);
  return Status::OK();
}

RegisteredStatefulProvider::~RegisteredStatefulProvider() {
  if (descriptor_.context) {
    descriptor_.Release(descriptor_.context);
  }
}

thread_local StatefulEpCreationScope* StatefulEpCreationScope::current_ = nullptr;

StatefulEpCreationScope::StatefulEpCreationScope() : previous_(current_) {
  current_ = this;
}

StatefulEpCreationScope::~StatefulEpCreationScope() {
  current_ = previous_;
}

Status StatefulEpCreationScope::Register(OrtEp* ep, const OrtStatefulProviderDescriptor* descriptor) {
  ORT_RETURN_IF_NOT(current_ && ep && descriptor,
                    "Stateful registration requires a non-NULL EP/descriptor during synchronous CreateEp");
  ORT_RETURN_IF(current_->pending_.contains(ep), "Stateful provider already registered for this EP");
  std::shared_ptr<RegisteredStatefulProvider> registration;
  ORT_RETURN_IF_ERROR(RegisteredStatefulProvider::Create(*descriptor, registration));
  current_->pending_.emplace(ep, std::move(registration));
  return Status::OK();
}

Status StatefulEpCreationScope::Take(OrtEp* ep, std::shared_ptr<RegisteredStatefulProvider>& out) {
  if (pending_.empty()) {
    out.reset();
    return Status::OK();
  }
  const auto found = pending_.find(ep);
  ORT_RETURN_IF(pending_.size() != 1 || found == pending_.end(),
                "Stateful registration must identify the OrtEp returned by CreateEp");
  out = std::move(found->second);
  pending_.clear();
  return Status::OK();
}

}  // namespace onnxruntime
