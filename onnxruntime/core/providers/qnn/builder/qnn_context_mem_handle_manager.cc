// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"

#include "HTP/QnnHtpMem.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/qnn_allocator.h"

namespace onnxruntime::qnn {

QnnContextMemHandleManager::QnnContextMemHandleManager(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                       Qnn_ContextHandle_t context,
                                                       const logging::Logger& logger)
    : qnn_interface_{qnn_interface},
      context_{context},
      logger_{logger} {
}

QnnContextMemHandleManager::~QnnContextMemHandleManager() {
  Clear();
}

Status QnnContextMemHandleManager::BatchGetOrRegister(gsl::span<const MemRegInput> inputs,
                                                      gsl::span<MemRegResult> results) {
  ORT_RETURN_IF_NOT(inputs.size() == results.size(),
                    "inputs and results must have the same size.");

  if (inputs.empty()) {
    return Status::OK();
  }

  std::scoped_lock g{mem_handles_mutex_};

  // Phase 1: Check cache and identify new registrations needed.
  InlinedVector<size_t> new_reg_indices;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];

    if (const auto it = mem_handles_.find(input.shared_memory_address); it != mem_handles_.end()) {
      const auto& record = it->second;
      const size_t qnn_tensor_data_size = utils::GetQnnTensorDataSizeInBytes(*input.qnn_tensor);
      ORT_RETURN_IF_NOT(qnn_tensor_data_size <= record.registered_tensor_data_size,
                        "Actual tensor data size (", qnn_tensor_data_size,
                        ") is larger than registered tensor data size (", record.registered_tensor_data_size, ").");

      results[i] = {record.mem_handle.get(), false};
    } else {
      new_reg_indices.push_back(i);
    }
  }

  if (new_reg_indices.empty()) {
    return Status::OK();
  }

  // Phase 2: Build descriptors for all new registrations.
  const size_t num_new = new_reg_indices.size();
  InlinedVector<Qnn_MemDescriptor_t> mem_descriptors(num_new);
  InlinedVector<QnnMemHtp_Descriptor_t> htp_mem_descriptors(num_new);
  InlinedVector<HtpSharedMemoryAllocator::SharedMemoryInfo> shared_mem_infos(num_new);

  for (size_t j = 0; j < num_new; ++j) {
    const size_t i = new_reg_indices[j];
    const auto& input = inputs[i];
    const auto& qnn_tensor = *input.qnn_tensor;

    ORT_RETURN_IF_ERROR(HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(
        input.shared_memory_address, shared_mem_infos[j]));

    auto& mem_desc = mem_descriptors[j];
    mem_desc = QNN_MEM_DESCRIPTOR_INIT;
    mem_desc.memShape.dimSize = GetQnnTensorDims(qnn_tensor);
    mem_desc.memShape.numDim = GetQnnTensorRank(qnn_tensor);
    mem_desc.memShape.shapeConfig = nullptr;
    mem_desc.dataType = GetQnnTensorDataType(qnn_tensor);
    mem_desc.memType = QNN_MEM_TYPE_CUSTOM;

    auto& htp_desc = htp_mem_descriptors[j];
    htp_desc = {};
    htp_desc.type = QNN_HTP_MEM_SHARED_BUFFER;
    htp_desc.size = shared_mem_infos[j].total_size;
    htp_desc.sharedBufferConfig.fd = shared_mem_infos[j].fd;
    htp_desc.sharedBufferConfig.offset = shared_mem_infos[j].offset;

    mem_desc.customInfo = &htp_desc;

    LOGS(logger_, VERBOSE) << "Registering QNN mem handle for context: " << context_
                           << ", shared memory (address: " << input.shared_memory_address
                           << ", offset: " << shared_mem_infos[j].offset
                           << ", fd: " << shared_mem_infos[j].fd
                           << ")";
  }

  // Phase 3: Single batched memRegister call.
  InlinedVector<Qnn_MemHandle_t> raw_mem_handles(num_new, nullptr);
  const auto register_result = qnn_interface_.memRegister(
      context_, mem_descriptors.data(), static_cast<uint32_t>(num_new), raw_mem_handles.data());
  ORT_RETURN_IF_NOT(register_result == QNN_SUCCESS,
                    "qnn_interface.memRegister() failed for batch of ", num_new, " descriptors: ",
                    utils::GetVerboseQnnErrorMessage(qnn_interface_, register_result));

  // Phase 4: Store all new handles.
  const auto unregister_mem_handle = [&qnn_interface = this->qnn_interface_](Qnn_MemHandle_t raw_mem_handle) {
    LOGS_DEFAULT(VERBOSE) << "Unregistering QNN mem handle. mem_handle: " << raw_mem_handle;

    const auto unregister_result = qnn_interface.memDeRegister(&raw_mem_handle, 1);
    if (unregister_result != QNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "qnn_interface.memDeRegister() failed: "
                          << utils::GetVerboseQnnErrorMessage(qnn_interface, unregister_result);
    }
  };

  for (size_t j = 0; j < num_new; ++j) {
    const size_t i = new_reg_indices[j];
    const auto& input = inputs[i];
    const size_t qnn_tensor_data_size = utils::GetQnnTensorDataSizeInBytes(*input.qnn_tensor);

    LOGS(logger_, VERBOSE) << "Registered QNN mem handle. mem_handle: " << raw_mem_handles[j];

    UniqueQnnMemHandle mem_handle(raw_mem_handles[j], unregister_mem_handle);
    MemHandleRecord record{qnn_tensor_data_size, std::move(mem_handle)};
    mem_handles_.emplace(input.shared_memory_address, std::move(record));

    results[i] = {raw_mem_handles[j], true};
  }

  return Status::OK();
}

Status QnnContextMemHandleManager::Unregister(void* shared_memory_address) {
  std::scoped_lock g{mem_handles_mutex_};

  auto mem_handles_it = mem_handles_.find(shared_memory_address);
  ORT_RETURN_IF_NOT(mem_handles_it != mem_handles_.end(),
                    "No mem handle found for address (", shared_memory_address, ").");

  mem_handles_.erase(mem_handles_it);

  return Status::OK();
}

void QnnContextMemHandleManager::Clear() {
  std::scoped_lock g{mem_handles_mutex_};
  mem_handles_.clear();
}

}  // namespace onnxruntime::qnn
