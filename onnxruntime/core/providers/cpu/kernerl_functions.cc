#include "core/providers/cpu/kernerl_functions.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace CPU_EP {
Status Mod(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs, const IExecutionProvider& provider) {
  const OrtValue& input_a = inputs[0];
  const OrtValue& input_b = inputs[1];
  const Tensor& a = input_a.Get<Tensor>();
  const Tensor& b = input_b.Get<Tensor>();
  InputBroadcaster input_broadcaster(a, b);

  OrtValue& output = outputs[0];
  onnxruntime::Tensor::InitOrtValue(a.DataType(), input_broadcaster.GetOutputShape(),
                                    provider.GetAllocator(0, OrtMemTypeDefault), output,
                                    {});

  if (a.IsDataType<float>()) {
    return ModImpl<float>(inputs, outputs);
  } else if (a.IsDataType<double>()) {
    return ModImpl<double>(inputs, outputs);
  } else if (a.IsDataType<int64_t>()) {
    return ModImpl<int64_t>(inputs, outputs);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Mod: unsupported data type");
  }
}
}  // namespace CPU_EP
}  // namespace onnxruntime
