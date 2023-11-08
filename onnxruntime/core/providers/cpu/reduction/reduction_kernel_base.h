#ifndef CORE_PROVIDERS_CPU_REDUCTION_KERNEL_BASE_H
#define CORE_PROVIDERS_CPU_REDUCTION_KERNEL_BASE_H

#ifndef SHARED_PROVIDER
#include "core/common/optional.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

template <bool allow_multi_axes>
class ReduceKernelBase {
 protected:
  ReduceKernelBase(const OpKernelInfo& info, optional<int64_t> keepdims_override = {}) {
    if (allow_multi_axes) {
      axes_ = ToShapeVector(info.GetAttrsOrDefault<int64_t>("axes"));
    } else {
      auto v = info.GetAttrOrDefault<int64_t>("axis", 0);
      axes_.push_back(v);
    }
    int64_t keepdims = 1;
    if (keepdims_override.has_value()) {
      keepdims = *keepdims_override;
    } else {
      ORT_ENFORCE(info.GetAttr("keepdims", &keepdims).IsOK());
    }
    keepdims_ = (keepdims == 1);
    int64_t noop_with_empty_axes = info.GetAttrOrDefault<int64_t>("noop_with_empty_axes", 0);
    noop_with_empty_axes_ = (noop_with_empty_axes == 1);
    int64_t select_last_index = info.GetAttrOrDefault<int64_t>("select_last_index", 0);
    select_last_index_ = (select_last_index != 0);
  }

  TensorShapeVector axes_;
  bool keepdims_;
  bool noop_with_empty_axes_;
  bool select_last_index_;
};
}  // namespace onnxruntime
#endif  // !CORE_PROVIDERS_CPU_REDUCTION_KERNEL_BASE_H