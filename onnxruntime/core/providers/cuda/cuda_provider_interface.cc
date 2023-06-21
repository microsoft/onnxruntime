#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct Provider;
struct CUDA_Provider;
CUDA_Provider* GetProvider();
}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return reinterpret_cast<onnxruntime::Provider*>(onnxruntime::GetProvider());
}
}
