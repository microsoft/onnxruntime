// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/checkpoint_common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/status.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {
namespace training {

/**
 * @brief Create OrtValues From TensorProto objects
 *
 * @param tensor_protos vector of TensorProto
 * @param name_to_ort_value saved results.
 * @return Status
 */
Status CreateOrtValuesFromTensorProtos(
    const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    NameMLValMap& name_to_ort_value) {
  bool create_arena = true;
#if defined(USE_JEMALLOC) || defined(USE_MIMALLOC)
  // JEMalloc/mimalloc already have memory pool, so just use device allocator.
  create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64))
  // Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
  create_arena = false;
#endif
  static const AllocatorCreationInfo info{[](int) { return std::make_unique<CPUAllocator>(); }, DEFAULT_CPU_ALLOCATOR_DEVICE_ID, create_arena};
  static const AllocatorPtr cpu_allocator = CreateAllocator(info);

  for (const auto& tensor_proto : tensor_protos) {
    TensorShape tensor_shape{utils::GetTensorShapeFromTensorProto(tensor_proto)};
    const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(
                                           tensor_proto.data_type())
                                           ->GetElementType();
    auto p_tensor = std::make_unique<Tensor>(tensor_dtype, tensor_shape, cpu_allocator);
    ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), nullptr, tensor_proto, *p_tensor));

    OrtValue ort_value;
    ort_value.Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    name_to_ort_value.emplace(tensor_proto.name(), ort_value);
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
