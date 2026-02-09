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
  std::vector<OrtValue> values(src_dst_pairs.size() * 2);
  std::vector<const OrtValue*> src_values;
  std::vector<OrtValue*> dst_values;
  std::vector<OrtSyncStream*> streams;
  std::vector<size_t> source_offsets;
  std::vector<size_t> destination_offsets;
  std::vector<size_t> sizes;
  src_values.reserve(src_dst_pairs.size());
  dst_values.reserve(src_dst_pairs.size());
  streams.reserve(src_dst_pairs.size());
  source_offsets.reserve(src_dst_pairs.size());
  destination_offsets.reserve(src_dst_pairs.size());
  sizes.reserve(src_dst_pairs.size());

  for (size_t i = 0; i < src_dst_pairs.size(); ++i) {
    const auto& pair = src_dst_pairs[i];
    Tensor* src_tensor = const_cast<Tensor*>(&(pair.src.get()));
    values[i * 2].Init(static_cast<void*>(src_tensor), ml_tensor_type, no_op_deleter);
    values[i * 2 + 1].Init(static_cast<void*>(&pair.dst.get()), ml_tensor_type, no_op_deleter);
    src_values.push_back(&values[i * 2]);
    dst_values.push_back(&values[i * 2 + 1]);
    streams.push_back(reinterpret_cast<OrtSyncStream*>(pair.src_stream));
    source_offsets.push_back(pair.source_offset);
    destination_offsets.push_back(pair.destination_offset);
    sizes.push_back(pair.size);
  }

  auto* status = impl_.CopyTensors(&impl_,
                                   src_values.data(),
                                   dst_values.data(),
                                   source_offsets.data(),
                                   destination_offsets.data(),
                                   sizes.data(),
                                   streams.data(),
                                   src_dst_pairs.size());
  return ToStatusAndRelease(status);
}

// optimized version for a single copy. see comments above in CopyTensors regarding the OrtValue usage and const_cast
Status DataTransfer::CopyTensorImpl(const Tensor& src_tensor, Tensor& dst_tensor, onnxruntime::Stream* /*stream*/) const {
  OrtValue src, dst;
  Tensor* src_tensor_ptr = const_cast<Tensor*>(&src_tensor);
  src.Init(static_cast<void*>(src_tensor_ptr), ml_tensor_type, no_op_deleter);
  dst.Init(static_cast<void*>(&dst_tensor), ml_tensor_type, no_op_deleter);
  const OrtValue* src_ptr = &src;
  OrtValue* dst_ptr = &dst;
  OrtSyncStream* stream_ptr = nullptr;  // static_cast<OrtSyncStream*>(stream);
  auto* status = impl_.CopyTensors(&impl_, &src_ptr, &dst_ptr, nullptr, nullptr, nullptr, &stream_ptr, 1);

  return ToStatusAndRelease(status);
}

}  // namespace plugin_ep
}  // namespace onnxruntime
