// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_transfer.h"
#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#endif
#include "core/framework/ortdevice.h"
#ifdef ENABLE_STRIDED_TENSORS
#include "core/framework/copy.h"
#include "core/session/environment.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/element_type_lists.h"
#endif

namespace onnxruntime {

common::Status IDataTransfer::CopyTensor(const Tensor& /*src*/, Tensor& /*dst*/) const {
  ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
}

common::Status IDataTransfer::CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const {
  for (const auto& pair : src_dst_pairs) {
    if (pair.src_stream)
      ORT_RETURN_IF_ERROR(CopyTensorAsync(pair.src, pair.dst, *pair.src_stream));
    else
      ORT_RETURN_IF_ERROR(CopyTensor(pair.src, pair.dst));
  }

  return Status::OK();
}

#if !defined(DISABLE_SPARSE_TENSORS)
common::Status IDataTransfer::CopySparseTensors(const std::vector<SparseSrcDstPair>& src_dst_pairs) const {
  for (const auto& pair : src_dst_pairs) {
    ORT_RETURN_IF_ERROR(pair.src.get().Copy(*this, pair.dst));
  }
  return Status::OK();
}
#endif

bool CPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::CPU;
}

common::Status CPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  if (src_data == dst_data) {
    // no need copying as both pointers are referring to same piece of memory.
    return Status::OK();
  }

#ifdef ENABLE_STRIDED_TENSORS
  if (!src.IsContiguous() || !dst.IsContiguous()) {
    auto dst_stride_vec = dst.Strides();
    auto src_stride_vec = src.Strides();
    onnxruntime::TensorShapeVector dst_stride{dst_stride_vec.begin(), dst_stride_vec.end()};
    onnxruntime::TensorShapeVector src_stride{src_stride_vec.begin(), src_stride_vec.end()};
    return DispatchStridedCopy<element_type_lists::All>(nullptr,
                                                        dst, 0, dst_stride,
                                                        src.Shape(),
                                                        src, 0, src_stride);
  } else {
#endif
    // Copying only happens between two same size tensors.
    ORT_ENFORCE(src.SizeInBytes() == dst.SizeInBytes());
    if (!src.IsDataTypeString()) {
      memcpy(dst_data, src_data, src.SizeInBytes());
    } else {
      const auto* src_strings = src.Data<std::string>();
      auto* dst_strings = dst.MutableData<std::string>();
      std::copy(src_strings, src_strings + src.Shape().Size(), dst_strings);
    }

    return Status::OK();
#ifdef ENABLE_STRIDED_TENSORS
  }
#endif
}

common::Status CPUDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& /*stream*/) const {
  // For being called in IDataTransfer::CopyTensors with mixture of CPU and GPU tensors.
  return this->CopyTensor(src, dst);
}

};  // namespace onnxruntime
