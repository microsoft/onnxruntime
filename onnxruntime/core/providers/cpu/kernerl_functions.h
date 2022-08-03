#pragma once

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/ort_value.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/framework/execution_frame.h"

namespace onnxruntime {
namespace CPU_EP {
// this is just an example of kernel function
Status Mod(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs, const IExecutionProvider& provider);

}  // namespace CPU_EP

}  // namespace onnxruntime
