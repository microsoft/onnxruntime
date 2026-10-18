// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/generate/token_generation.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/stateful_execution_provider.h"

namespace onnxruntime {
namespace {
struct GreedyWork {
  const OrtValue& logits;
  std::optional<int64_t> position;
  StatefulDispatch& dispatch;
  bool called = false;
};

OrtStatus* ORT_API_CALL CpuGreedy(void* context, int64_t* out) noexcept {
  struct Call {
    void* context;
    int64_t* out;
  } call{context, out};
  return StatefulCpuBoundary(&call, [](void* argument) -> Status {
    const auto& call = *static_cast<Call*>(argument);
    ORT_RETURN_IF_NOT(call.context && call.out, "CPU greedy reference context/output is NULL");
    auto& work = *static_cast<GreedyWork*>(call.context);
    ORT_RETURN_IF(work.called, "CPU reference may be invoked only once");
    work.called = true;
    work.dispatch.RecordDelegation(ORT_STATEFUL_SAMPLE_GREEDY);
    return Sampler::Greedy(work.logits, work.position, *call.out);
  });
}
}  // namespace

Status Sampler::Sample(const OrtValue& logits, std::optional<int64_t> position,
                       StatefulDispatch& dispatch, int64_t& token) {
  ORT_RETURN_IF_NOT(logits.IsTensor(), "Greedy logits must be a tensor");
  const auto& tensor = logits.Get<Tensor>();
  const auto& shape = tensor.Shape();
  ORT_RETURN_IF((shape.NumDimensions() != 2 && shape.NumDimensions() != 3) ||
                    shape[0] != 1 || shape[shape.NumDimensions() - 1] <= 0 ||
                    (shape.NumDimensions() == 3 && shape[1] <= 0),
                "Greedy logits must have shape [1,V] or [1,S,V]");
  ORT_RETURN_IF(tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                    tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                "Greedy logits require float32 or float16");
  const int64_t rows = shape.NumDimensions() == 3 ? shape[1] : 1;
  ORT_RETURN_IF(position && (*position < 0 || *position >= rows), "Greedy position is outside logits rows");
#ifdef ENABLE_STRIDED_TENSORS
  ORT_RETURN_IF_NOT(tensor.IsContiguous(), "Greedy logits must be contiguous");
#endif
  ORT_UNUSED_PARAMETER(tensor.SizeInBytes());
  ORT_RETURN_IF_NOT(tensor.DataRaw(), "Greedy logits buffer is NULL");
  bool provider = false;
  ORT_RETURN_IF_ERROR(dispatch.Select(ORT_STATEFUL_SAMPLE_GREEDY, tensor.GetElementType(),
                                      tensor.Location().device.Type() == OrtDevice::CPU, provider));
  const auto& selection = dispatch.Binding().Selection();
  ORT_RETURN_IF(provider &&
                    (static_cast<int>(tensor.Location().device.Type()) != static_cast<int>(selection.device_type) ||
                     tensor.Location().device.Id() != selection.device_id),
                "Provider greedy logits device mismatch; no implicit transfer");
  dispatch.RecordCall(ORT_STATEFUL_SAMPLE_GREEDY, provider);
  int64_t sampled = -1;
  Status status;
  if (provider) {
    GreedyWork work{logits, position, dispatch};
    const OrtStatefulGreedyRequest request{
        sizeof(request), OrtGetApiBase()->GetApi(ORT_API_VERSION), &logits,
        position.has_value(), position.value_or(0), &work, &CpuGreedy};
    const auto& descriptor = dispatch.Binding().Descriptor();
    status = ToStatusAndRelease(descriptor.Greedy(descriptor.context, &request, &sampled));
  } else {
    status = Greedy(logits, position, sampled);
  }
  if (status.IsOK() && (sampled < 0 || sampled >= shape[shape.NumDimensions() - 1])) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider returned a token outside the logits vocabulary");
  }
  if (!status.IsOK()) {
    dispatch.RecordFailure(ORT_STATEFUL_SAMPLE_GREEDY);
    return status;
  }
  token = sampled;
  return Status::OK();
}

}  // namespace onnxruntime
