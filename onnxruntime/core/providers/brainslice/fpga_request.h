// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_buffer.h"
#include "FPGACoreLib.h"
namespace onnxruntime {
namespace fpga {

using FPGARequestType = uint32_t;

// A wrapper class providing write object API for fpga request buffer.
class FPGARequest : public FPGABuffer {
 public:
  FPGARequest();
  // Construct from a buffer and optional deallocator.
  explicit FPGARequest(FPGARequestType request_type,
                       void* buffer,
                       size_t size);

  // virtual destructor.
  virtual ~FPGARequest() = default;

  // Move constructor.
  FPGARequest(FPGARequest&& other) = default;

  // Move assignment
  FPGARequest& operator=(FPGARequest&& other) = default;

  // Get request type.
  FPGARequestType GetRequestType() const;

  // Get request size.
  size_t GetRequestSize() const;

  // Get stream.
  FPGABufferStream& GetStream();

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FPGARequest);
  // Request type.
  FPGARequestType request_type_;

  // Request stream.
  FPGABufferStream stream_;

  SLOT_HANDLE handle_;
};
}  // namespace fpga
}  // namespace onnxruntime
