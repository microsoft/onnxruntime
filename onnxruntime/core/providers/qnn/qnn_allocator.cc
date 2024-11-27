// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_allocator.h"

#include <algorithm>
#include <limits>

#include <QnnInterface.h>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/framework/tensor.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/shared_context.h"  // for shared mem handle access

namespace onnxruntime::qnn {

namespace {

Qnn_MemHandle_t RegisterQnnMemHandle(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                     Qnn_ContextHandle_t qnn_context_handle,
                                     int shared_memory_fd,
                                     MLDataType element_data_type, const TensorShape& shape) {
  auto qnn_shape = [shape_span = shape.GetDims()]() {
    InlinedVector<uint32_t> qnn_shape;
    std::transform(shape_span.begin(), shape_span.end(), std::back_inserter(qnn_shape),
                   [](int64_t dim) { return narrow<uint32_t>(dim); });
    return qnn_shape;
  }();

  const auto qnn_data_type = [element_data_type]() {
    Qnn_DataType_t qnn_data_type;
    ORT_ENFORCE(element_data_type->IsPrimitiveDataType());
    const auto onnx_data_type = element_data_type->AsPrimitiveDataType()->GetDataType();
    const bool is_quantized = false;  // TODO how should we set this?
    if (!utils::OnnxDataTypeToQnnDataType(onnx_data_type, qnn_data_type, is_quantized)) {
      ORT_THROW("Unable to get QNN data type from ONNX data type: ", onnx_data_type);
    }
    return qnn_data_type;
  }();

  // set up QNN memory descriptor
  Qnn_MemDescriptor_t qnn_mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
  qnn_mem_descriptor.memShape = {narrow<uint32_t>(qnn_shape.size()),
                                 qnn_shape.data(),
                                 nullptr};
  qnn_mem_descriptor.dataType = qnn_data_type;
  qnn_mem_descriptor.memType = QNN_MEM_TYPE_ION;
  qnn_mem_descriptor.ionInfo.fd = shared_memory_fd;

  Qnn_MemHandle_t qnn_mem_handle = nullptr;
  const auto register_status = qnn_interface.memRegister(qnn_context_handle, &qnn_mem_descriptor, 1,
                                                         &qnn_mem_handle);
  // TODO show error message
  ORT_ENFORCE(register_status == QNN_SUCCESS,
              "qnn_interface.memRegister() failed with error code ", register_status);

  return qnn_mem_handle;
}

void DeregisterQnnMemHandle(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                            Qnn_MemHandle_t qnn_mem_handle) {
  const auto deregister_status = qnn_interface.memDeRegister(&qnn_mem_handle, 1);
  // TODO show error message
  if (deregister_status != QNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "qnn_interface.memDeRegister() failed with error code " << deregister_status;
  }
}

using RpcMemUniquePtr = std::unique_ptr<void, void (*)(void*)>;

RpcMemUniquePtr WrapSharedMemoryWithUniquePtr(void* shared_memory_raw, const RpcMemApi& rpcmem_api) {
  return {shared_memory_raw, rpcmem_api.free};
}

}  // namespace

OrtMemoryInfo HtpSharedMemoryAllocator::MemoryInfo() {
  return OrtMemoryInfo{QNN_HTP_SHARED, OrtAllocatorType::OrtDeviceAllocator,
                       OrtDevice{OrtDevice::CPU, OrtDevice::MemType::QNN_HTP_SHARED, /* device_id */ 0},
                       /* id */ 0, OrtMemTypeDefault};
}

HtpSharedMemoryAllocator::HtpSharedMemoryAllocator(std::shared_ptr<RpcMemLibrary> rpcmem_lib,
                                                   std::shared_ptr<QnnBackendManager> qnn_backend_manager)
    : IAllocator{MemoryInfo()},
      rpcmem_lib_{std::move(rpcmem_lib)},
      qnn_backend_manager_{std::move(qnn_backend_manager)} {
  ORT_ENFORCE(rpcmem_lib_ != nullptr);
  ORT_ENFORCE(qnn_backend_manager_ != nullptr);
}

void* HtpSharedMemoryAllocator::Alloc(size_t /* size */) {
  LOGS_DEFAULT(ERROR) << "hey this ain't right";
  std::exit(1);
  ORT_THROW("HtpSharedMemoryAllocator::Alloc() is not implemented. Use HtpSharedMemoryAllocator::TensorAlloc() instead.");
}

void* HtpSharedMemoryAllocator::TensorAlloc(MLDataType element_data_type, const TensorShape& shape) {
  const auto size_in_bytes = Tensor::CalculateTensorStorageSize(element_data_type, shape);

  if (size_in_bytes == 0) {
    return nullptr;
  }

  // rpcmem_alloc() has an int size parameter. make sure we don't overflow.
  constexpr size_t max_size_in_bytes = std::numeric_limits<int>::max();
  ORT_ENFORCE(size_in_bytes <= max_size_in_bytes,
              "Allocation size (", size_in_bytes, ") is larger than maximum allowed (", max_size_in_bytes, ").");

  // allocate shared memory
  void* shared_memory_raw = rpcmem_lib_->Api().alloc(rpcmem::RPCMEM_HEAP_ID_SYSTEM, rpcmem::RPCMEM_DEFAULT_FLAGS,
                                                     static_cast<int>(size_in_bytes));

  auto shared_memory = WrapSharedMemoryWithUniquePtr(shared_memory_raw, rpcmem_lib_->Api());

  // get shared memory fd
  const auto shared_memory_fd = rpcmem_lib_->Api().to_fd(shared_memory.get());
  ORT_ENFORCE(shared_memory_fd != -1, "rpcmem_to_fd() returned invalid file descriptor.");

  // register mem handle
  // TODO synchronize calls to qnn_interface.memRegister()?
  const auto& qnn_interface = qnn_backend_manager_->GetQnnInterface();
  const auto qnn_context_handle = qnn_backend_manager_->GetQnnContext();
  const auto qnn_mem_handle = RegisterQnnMemHandle(qnn_interface, qnn_context_handle,
                                                   shared_memory_fd, element_data_type, shape);

  // save mem handle. for now, the global SharedContext will do...
  SharedContext::GetInstance().GetSharedMemHandles().Add(shared_memory.get(), qnn_mem_handle);

  return shared_memory.release();
}

void HtpSharedMemoryAllocator::Free(void* p) {
  // take ownership of shared memory and free at end of scope
  auto shared_memory = WrapSharedMemoryWithUniquePtr(p, rpcmem_lib_->Api());

  // deregister mem handle
  // TODO synchronize calls to qnn_interface.memDeRegister()?
  const auto& qnn_interface = qnn_backend_manager_->GetQnnInterface();
  const auto qnn_mem_handle = SharedContext::GetInstance().GetSharedMemHandles().GetAndRemove(p);
  DeregisterQnnMemHandle(qnn_interface, qnn_mem_handle);
}

}  // namespace onnxruntime::qnn
