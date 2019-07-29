// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

class GatherNDBase
{
protected:
  struct Prepare {
    const uint8_t*        input_base;
    const std::string*    input_str_base;
    uint8_t*              output_base;
    std::string*          output_str_base;
    uint64_t              bytes_to_copy;
    uint64_t              element_bytes;
    uint64_t              element_to_copy;
    std::vector<uint64_t> element_offsets;

    Prepare(): input_base      (nullptr),
               input_str_base  (nullptr),
               output_base     (nullptr),
               output_str_base (nullptr),
               bytes_to_copy   (0),
               element_bytes   (0),
               element_to_copy (0),
               element_offsets (0) {}
  }; // struct Prepare

  template<typename Tind>
  Status PrepareForCompute(OpKernelContext* context, Prepare& p) const;
}; // class GatherNDBase

class GatherND final : public OpKernel, protected GatherNDBase {
public:
  explicit GatherND(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
private:
  Status GatherNumber(const Prepare& p) const;
  Status GatherString(const Prepare& p) const;
};

} // namespace contrib
} // namespace onnxruntime