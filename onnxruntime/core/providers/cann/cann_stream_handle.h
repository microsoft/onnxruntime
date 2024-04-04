// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/stream_handles.h"
#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_call.h"

namespace onnxruntime {
void WaitCannNotificationOnDevice(Stream& stream, synchronize::Notification& notification);

struct CannStream : Stream {
  CannStream(aclrtStream stream, const OrtDevice& device, bool own_flag);

  ~CannStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  bool own_stream_{true};

  WaitNotificationFn GetWaitNotificationFn() const override { return WaitCannNotificationOnDevice; }
};

void RegisterCannStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type);

}  // namespace onnxruntime
