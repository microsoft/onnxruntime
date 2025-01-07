// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "QnnInterface.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"

namespace onnxruntime::qnn {

// This class manages QNN mem handles (Qnn_MemHandle_t) associated with a QNN context (Qnn_ContextHandle_t).
// In particular, it handles the registration and deregistration of mem handles.
// The associated QNN context is expected to be in scope for the lifetime of the QnnContextMemHandleManager.
class QnnContextMemHandleManager {
 public:
  QnnContextMemHandleManager(const QNN_INTERFACE_VER_TYPE& qnn_interface, Qnn_ContextHandle_t qnn_context,
                             const logging::Logger& logger);

  ~QnnContextMemHandleManager();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnContextMemHandleManager);

  // Gets an existing QNN mem handle or registers a new one.
  // `qnn_mem_handle` is set to the QNN mem handle and `did_register` is true if `qnn_mem_handle` was newly registered.
  Status GetOrRegister(void* shared_memory_address, const Qnn_Tensor_t& qnn_tensor,
                       Qnn_MemHandle_t& qnn_mem_handle, bool& did_register);

  Status Unregister(void* shared_memory_address);

  void Clear();

 private:
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
  Qnn_ContextHandle_t context_;
  const logging::Logger& logger_;

  // assume Qnn_MemHandle_t is a pointer and able to be wrapped with std::unique_ptr
  static_assert(std::is_pointer_v<Qnn_MemHandle_t>);

  using UniqueQnnMemHandle =
      std::unique_ptr<std::remove_pointer_t<Qnn_MemHandle_t>, std::function<void(Qnn_MemHandle_t)>>;

  struct MemHandleRecord {
    size_t registered_tensor_data_size;
    UniqueQnnMemHandle mem_handle;
  };

  // shared memory address -> associated mem handle record
  InlinedHashMap<const void*, MemHandleRecord> mem_handles_;
  std::mutex mem_handles_mutex_;  // synchronize access to mem_handles_
};

}  // namespace onnxruntime::qnn
