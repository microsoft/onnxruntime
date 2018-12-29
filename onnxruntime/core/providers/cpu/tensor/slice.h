// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

class SliceBase {
 protected:
  SliceBase (const OpKernelInfo& info) {
    auto has_starts = info.GetAttrs("starts", attr_starts_).IsOK();
    auto has_ends   = info.GetAttrs("ends",   attr_ends_).IsOK();
    auto has_axes   = info.GetAttrs("axes",   attr_axes_).IsOK();
    ORT_ENFORCE(!has_axes || has_starts && has_ends, "Not all required attributes are specified");
  }

  Status PrepareForCompute(const std::vector<int64_t>& raw_starts,
                           const std::vector<int64_t>& raw_ends, 
                           const std::vector<int64_t>& raw_axes,
                           const size_t                dimension_count,
                           const std::vector<int64_t>& input_dimensions,
                           std::vector<int64_t>&       starts,
                           std::vector<int64_t>&       output_dims) const;

  std::vector<int64_t> attr_starts_, attr_ends_, attr_axes_;
};

template <typename T, typename Tind, bool dynamic = true>
struct Slice final : public OpKernel, public SliceBase {
  Slice(const OpKernelInfo& info) : OpKernel(info), SliceBase(info) {}
  Status Compute(OpKernelContext* context) const override;
private:
  void FillVectors(const OpKernelContext* context,
	           std::vector<int64_t>&  raw_starts,
	           std::vector<int64_t>&  raw_ends,
	           std::vector<int64_t>&  raw_axes) const;
};  // namespace onnxruntime

}  // namespace onnxruntime
