// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
class EncodeImage final : public OpKernel {
 public:
  EncodeImage(const OpKernelInfo& info);

  void Compute(OrtKernelContext* context);

 private:
  std::string pixel_format_;
  int quality_{70};
  std::string extension_{"jpg"};
};

/// <summary>
/// EncodeImage
///
/// Converts rank 3 BGR input with channels last ordering to the requested file type.
/// Default is 'jpg'
/// </summary>
struct CustomOpEncodeImage : Ort::CustomOpBase<CustomOpEncodeImage, KernelEncodeImage> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    Ort::CustomOpApi op_api{api};
    std::string format = op_api.KernelInfoGetAttribute<std::string>(info, "format");
    if (format != "jpg" && format != "png") {
      ORT_CXX_API_THROW("[EncodeImage] 'format' attribute value must be 'jpg' or 'png'.", ORT_RUNTIME_EXCEPTION);
    }

    return new KernelEncodeImage(api, format);
  }

  const char* GetName() const {
    return "EncodeImage";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        ORT_CXX_API_THROW(MakeString("Invalid input index ", index), ORT_INVALID_ARGUMENT);
    }
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        ORT_CXX_API_THROW(MakeString("Invalid output index ", index), ORT_INVALID_ARGUMENT);
    }
  }
};
}  // namespace ort_extensions
