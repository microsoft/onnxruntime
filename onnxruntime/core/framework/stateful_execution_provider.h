// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <string>

#include "core/common/status.h"
#include "core/framework/ort_value.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

class InferenceSession;

// In-tree exception boundary for C-callable CPU references; not a plugin ABI.
OrtStatus* StatefulCpuBoundary(void* context, Status (*operation)(void*)) noexcept;

struct StatefulProviderSelection {
  OrtStatefulProviderMode mode = ORT_STATEFUL_PROVIDER_CPU;
  std::string provider_name;
  OrtMemoryInfoDeviceType device_type = OrtMemoryInfoDeviceType_CPU;
  int device_id = 0;
  static Status Parse(const OrtStatefulProviderSelection* input, StatefulProviderSelection& out);
};

class StatefulProviderBinding {
 public:
  static Status Resolve(InferenceSession* session, const StatefulProviderSelection& selection,
                        StatefulProviderBinding& out);
  static Status ValidateDescriptor(const OrtStatefulProviderDescriptor& descriptor);
  static Status Query(InferenceSession& session, const char* name, OrtStatefulProviderCapabilities& out);
  const OrtStatefulProviderDescriptor& Descriptor() const;
  const StatefulProviderSelection& Selection() const { return selection_; }
  const char* Name() const;
  bool CacheCompatible(const StatefulProviderBinding& other) const;
  OrtValue RetainValue(OrtValue value, std::shared_ptr<const void> resource = {}) const;
  Status InvokeGreedy(OrtStatefulGreedyRequest request, int64_t& token) const;

 private:
  struct Lease;
  StatefulProviderSelection selection_;
  std::shared_ptr<Lease> lease_;
};

class StatefulDispatch {
 public:
  explicit StatefulDispatch(StatefulProviderBinding binding = {});
  Status Select(OrtStatefulProviderOperation operation, int element_type, bool cpu_compatible,
                bool& provider) const;
  void RecordCall(OrtStatefulProviderOperation operation, bool provider);
  void RecordDelegation(OrtStatefulProviderOperation operation);
  void RecordFailure(OrtStatefulProviderOperation operation);
  Status GetInfo(OrtStatefulProviderOperation operation, OrtStatefulDispatchInfo& out) const;
  const StatefulProviderBinding& Binding() const { return binding_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulDispatch);

 private:
  const StatefulProviderBinding binding_;
  std::array<OrtStatefulDispatchInfo, 7> records_{};
};

}  // namespace onnxruntime
