// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vitisai/include/vaip/global_api.h"

namespace onnxruntime {
namespace profiling {

#if defined(USE_VITISAI)
class VitisaiProfiler final : public EpProfiler {
 public:
  VitisaiProfiler() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VitisaiProfiler);
  ~VitisaiProfiler() {}
  bool StartProfiling(TimePoint) override;
  void EndProfiling(TimePoint, Events&) override;
  void Start(uint64_t) override {}
  void Stop(uint64_t) override {}
};
#endif

}  // namespace profiling
}  // namespace onnxruntime
