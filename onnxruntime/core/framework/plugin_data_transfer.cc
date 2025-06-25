// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/plugin_data_transfer.h"

#include "core/framework/error_code_helper.h"

namespace onnxruntime {
namespace plugin_ep {

namespace {
static const std::function<void(void*)> no_op_deleter = [](void*) {};
static const MLDataType ml_tensor_type = DataTypeImpl::GetType<Tensor>();
}  // namespace

Status DataTransfer::CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const {
  // need to wrap the src/dst Tensor instances in OrtValue as the ORT API doesn't expose an OrtTensor.
  // Adding an OrtTensor to the API would also require adding getters for type/shape/data.
  // Those already exist for OrtValue so in order to minimize the API surface area we pay the price of a
  // const_cast to convert the `const Tensor*` src to an OrtValue.
  std::vector<OrtValue> values;
  values.resize(src_dst_pairs.size() * 2);

  for (size_t i = 0; i < src_dst_pairs.size(); ++i) {
    const auto& pair = src_dst_pairs[i];

    // we need to remove the const from the src to wrap it in an OrtValue.
    // it's passed to the impl as a const OrtValue, and the deleter is a no-op so this should be safe.
    Tensor* src_tensor = const_cast<Tensor*>(&(pair.src.get()));
    values[i * 2].Init(reinterpret_cast<void*>(src_tensor), ml_tensor_type, no_op_deleter);
    values[i * 2 + 1].Init(reinterpret_cast<void*>(&pair.dst.get()), ml_tensor_type, no_op_deleter);
  }

  std::vector<const OrtValue*> src_values;
  std::vector<OrtValue*> dst_values;
  std::vector<OrtSyncStream*> streams;
  src_values.reserve(src_dst_pairs.size());
  dst_values.reserve(src_dst_pairs.size());
  streams.reserve(src_dst_pairs.size());

  for (size_t i = 0; i < src_dst_pairs.size(); ++i) {
    src_values.push_back(&values[i * 2]);
    dst_values.push_back(&values[i * 2 + 1]);
    streams.push_back(nullptr);  // static_cast<OrtSyncStream*>(src_dst_pairs[i].src_stream));
  }

  auto* status = impl_.CopyTensors(&impl_, src_values.data(), dst_values.data(), streams.data(),
                                   src_dst_pairs.size());

  return status == nullptr ? Status::OK() : ToStatusAndRelease(status);
}

// optimized version for a single copy. see comments above in CopyTensors regarding the OrtValue usage and const_cast
Status DataTransfer::CopyTensorImpl(const Tensor& src_tensor, Tensor& dst_tensor, onnxruntime::Stream* /*stream*/) const {
  OrtValue src, dst;
  Tensor* src_tensor_ptr = const_cast<Tensor*>(&src_tensor);
  src.Init(reinterpret_cast<void*>(src_tensor_ptr), ml_tensor_type, no_op_deleter);
  dst.Init(reinterpret_cast<void*>(&dst_tensor), ml_tensor_type, no_op_deleter);
  const OrtValue* src_ptr = &src;
  OrtValue* dst_ptr = &dst;
  OrtSyncStream* stream_ptr = nullptr;  // static_cast<OrtSyncStream*>(stream);
  auto* status = impl_.CopyTensors(&impl_, &src_ptr, &dst_ptr, &stream_ptr, 1);

  return status == nullptr ? Status::OK() : ToStatusAndRelease(status);
}

}  // namespace plugin_ep
}  // namespace onnxruntime
