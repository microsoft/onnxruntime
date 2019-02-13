// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_buffer.h"
#include "FPGACoreLib.h"
namespace onnxruntime {
namespace fpga {
// A wrapper class providing read object API for fpga response buffer.
class FPGAResponse : public FPGABuffer {
 public:
  // Empty constructor.
  FPGAResponse();

  // Construct.
  explicit FPGAResponse(void* p_buffer,
                        size_t p_size,
                        FPGA_STATUS p_status = FPGA_STATUS_SUCCESS);

  // Move constructor.
  FPGAResponse(FPGAResponse&& p_other);

  // Move assignment
  FPGAResponse& operator=(FPGAResponse&& p_other);

  // Virtual destructor.
  virtual ~FPGAResponse();

  // Get response status.
  FPGA_STATUS GetStatus() const;

  // Get response size;
  size_t GetResponseSize() const;

 private:
  // Response status.
  FPGA_STATUS m_status;
};
}  // namespace fpga
}  // namespace onnxruntime
