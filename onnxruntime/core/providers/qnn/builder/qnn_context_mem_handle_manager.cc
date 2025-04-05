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
  rpc_lib_.reset();
}

Status QnnContextMemHandleManager::GetOrRegister(void* memory_address, bool uses_shared_mem_allocator, const Qnn_Tensor_t& qnn_tensor,
                                                 Qnn_MemHandle_t& qnn_mem_handle, bool& did_register) {
  const auto qnn_tensor_rank = GetQnnTensorRank(qnn_tensor);
  auto* const qnn_tensor_dims = GetQnnTensorDims(qnn_tensor);
  const auto qnn_tensor_data_type = GetQnnTensorDataType(qnn_tensor);

  const size_t qnn_tensor_data_size =
      utils::GetQnnTensorDataSizeInBytes(gsl::span{qnn_tensor_dims, size_t{qnn_tensor_rank}}, qnn_tensor_data_type);

  {
    std::scoped_lock g{mem_handles_mutex_};

    // find existing mem handle
    if (const auto mem_handles_it = mem_handles_.find(memory_address);
        mem_handles_it != mem_handles_.end()) {
      const auto& mem_handle_record = mem_handles_it->second;

      // check that actual tensor size is less than or equal to registered tensor size
      ORT_RETURN_IF_NOT(qnn_tensor_data_size <= mem_handle_record.registered_tensor_data_size,
                        "Actual tensor data size (", qnn_tensor_data_size,
                        ") is larger than registered tensor data size (", mem_handle_record.registered_tensor_data_size,
                        ").");

      qnn_mem_handle = mem_handle_record.mem_handle.get();
      did_register = false;
      return Status::OK();
    }

    // register a new mem handle
    Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
    mem_descriptor.memShape.dimSize = qnn_tensor_dims;
    mem_descriptor.memShape.numDim = qnn_tensor_rank;
    mem_descriptor.memShape.shapeConfig = nullptr;
    mem_descriptor.dataType = qnn_tensor_data_type;
    mem_descriptor.memType = QNN_MEM_TYPE_CUSTOM;

    QnnMemHtp_Descriptor_t htp_mem_descriptor{};
    htp_mem_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;

    if (uses_shared_mem_allocator) {
      HtpSharedMemoryAllocator::SharedMemoryInfo shared_memory_info{};
      ORT_RETURN_IF_ERROR(HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(memory_address,
                                                                                  shared_memory_info));
      htp_mem_descriptor.size = shared_memory_info.total_size;
      htp_mem_descriptor.sharedBufferConfig.fd = shared_memory_info.fd;
      htp_mem_descriptor.sharedBufferConfig.offset = shared_memory_info.offset;
    } else {
      if (!rpc_lib_) {
        rpc_lib_ = std::make_unique<qnn::RpcMemLibrary>();
      }

      rpc_lib_->Api().register_buff_attr(reinterpret_cast<int*>(memory_address), static_cast<int>(qnn_tensor_data_size),
                                        0, rpcmem::FASTRPC_ATTR_IMPORT_BUFFER);
      int fd = rpc_lib_->Api().to_fd(memory_address);
      ORT_RETURN_IF(fd == rpcmem::INVALID_CLIENT_HANDLE, "Failed to register buffer to FastRPC");

      htp_mem_descriptor.size = qnn_tensor_data_size;
      htp_mem_descriptor.sharedBufferConfig.fd = fd;
      htp_mem_descriptor.sharedBufferConfig.offset = 0;
    }

    mem_descriptor.customInfo = &htp_mem_descriptor;

    LOGS(logger_, VERBOSE) << "Registering QNN mem handle for context: " << context_
                           << ", memory (address: " << memory_address
                           << ", offset: " << htp_mem_descriptor.sharedBufferConfig.offset
                           << ", fd: " << htp_mem_descriptor.sharedBufferConfig.fd
                           << ")";

    Qnn_MemHandle_t raw_mem_handle{};
    const auto register_result = qnn_interface_.memRegister(context_, &mem_descriptor, 1, &raw_mem_handle);
    ORT_RETURN_IF_NOT(register_result == QNN_SUCCESS,
                      "qnn_interface.memRegister() failed: ",
                      utils::GetVerboseQnnErrorMessage(qnn_interface_, register_result));

    LOGS(logger_, VERBOSE) << "Registered QNN mem handle. mem_handle: " << raw_mem_handle;

    // NOTE: Must use the default ORT logger inside this lambda. Don't capture this->logger_ because it may be deleted
    // by the time we need to unregister all memory handles. This happens when this->logger_ is a session logger:
    //   ~InferenceSession() -> ~Logger() -> ~QnnExecutionProvider() -> ~QnnBackendManager() ->
    //   ~QnnContextMemHandleManager() -> unregister_mem_handle() segfault
    const auto unregister_mem_handle = [uses_shared_mem_allocator, qnn_tensor_data_size, memory_address,
                                        &rpc_lib = this->rpc_lib_, &qnn_interface = this->qnn_interface_](Qnn_MemHandle_t raw_mem_handle) {
      LOGS_DEFAULT(VERBOSE) << "Unregistering QNN mem handle. mem_handle: " << raw_mem_handle;

      const auto unregister_result = qnn_interface.memDeRegister(&raw_mem_handle, 1);
      if (unregister_result != QNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "qnn_interface.memDeRegister() failed: "
                            << utils::GetVerboseQnnErrorMessage(qnn_interface, unregister_result);
      }

      if (!uses_shared_mem_allocator) {
        rpc_lib->Api().register_buff_attr(reinterpret_cast<int*>(memory_address), static_cast<int>(qnn_tensor_data_size),
                                         rpcmem::INVALID_CLIENT_HANDLE, 0);
        if (rpc_lib->Api().to_fd(memory_address) != rpcmem::INVALID_CLIENT_HANDLE) {
          LOGS_DEFAULT(ERROR) << "fastrpc buffer deregistration failed for memory address: " << memory_address;
        }
      }

    };

    UniqueQnnMemHandle mem_handle(raw_mem_handle, unregister_mem_handle);
    MemHandleRecord mem_handle_record{qnn_tensor_data_size, std::move(mem_handle)};
    mem_handles_.emplace(memory_address, std::move(mem_handle_record));

    qnn_mem_handle = raw_mem_handle;
    did_register = true;
    return Status::OK();
  }
}

Status QnnContextMemHandleManager::Unregister(void* memory_address) {
  std::scoped_lock g{mem_handles_mutex_};
  auto mem_handles_it = mem_handles_.find(memory_address);
  ORT_RETURN_IF_NOT(mem_handles_it != mem_handles_.end(),
                    "No mem handle found for address (", memory_address, ").");

  mem_handles_.erase(mem_handles_it);

  return Status::OK();
}

void QnnContextMemHandleManager::Clear() {
  std::scoped_lock g{mem_handles_mutex_};
  mem_handles_.clear();
}

}  // namespace onnxruntime::qnn
