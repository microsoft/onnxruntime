// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
using namespace common;

const DataTransferManager& DataTransferManager::Instance() {
  static DataTransferManager data_transfer_mgr;
  return data_transfer_mgr;
}

common::Status DataTransferManager::RegisterDataTransfer(
    OrtDevice::DeviceType src_device_type,
    OrtDevice::DeviceType dst_device_type,
    const DataTransfer& data_transfer) {
  int32_t id_key = (static_cast<int32_t>(src_device_type) << 16) | static_cast<int32_t>(dst_device_type);
  auto iter = devicetypes_datatransfer_map_.find(id_key);
  if (devicetypes_datatransfer_map_.end() != iter) {
    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "Copy tensor function has already been registered for src (",
                           src_device_type,
                           ") to dst (",
                           dst_device_type,
                           ")");
  }
  devicetypes_datatransfer_map_.insert({id_key, data_transfer});

  return Status::OK();
}

common::Status DataTransferManager::CopyTensor(const Tensor& src, Tensor& dst) const {
  return CopyTensor(src, dst, 0);
}

common::Status DataTransferManager::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  int32_t id_key = (static_cast<int32_t>(src.Location().device.Type()) << 16) | static_cast<int32_t>(dst.Location().device.Type());
  auto iter = devicetypes_datatransfer_map_.find(id_key);
  if (devicetypes_datatransfer_map_.end() == iter) {
    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "Copy tensor failed due to no copy function found for src (",
                           src.Location().device.Type(),
                           ") to dst (",
                           src.Location().device.Type(),
                           ")");
  }

  return iter->second(src, dst, exec_queue_id);
}

}  // namespace onnxruntime
