// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "winml_adapter_c_api.h"
#include <memory>
#include "core/graph/onnx_protobuf.h"

class ModelInfo;
struct OrtModelImpl;

// opaque so we don't clash with the ORT OrtModel
struct OrtModel {
  // OrtModel is always OrtModelImpl. Add some helpers here so the reinterpret_cast is in one place.
  OrtModelImpl* ToInternal();
  const OrtModelImpl* ToInternal() const;

 protected:
  OrtModel() = default;  // can only be created as OrtModelImpl
};

struct OrtModelImpl : public OrtModel {
 public:
  static OrtStatus* CreateEmptyModel(int64_t opset, ::OrtModel** model);
  static OrtStatus* CreateOrtModelFromPath(const char* path, size_t len, OrtModel** model);
  static OrtStatus* CreateOrtModelFromData(void* data, size_t len, OrtModel** model);
  static OrtStatus* CreateOrtModelFromProto(std::unique_ptr<onnx::ModelProto>&& model_proto, OrtModel** model);
  const ModelInfo* UseModelInfo() const;

  onnx::ModelProto* UseModelProto() const;
  std::unique_ptr<onnx::ModelProto> DetachModelProto();

  void RefreshModelInfo();

 private:
  OrtModelImpl(std::unique_ptr<onnx::ModelProto> model_proto);
  OrtModelImpl(const OrtModel& other) = delete;
  OrtModelImpl& operator=(const OrtModel& other) = delete;

 private:
  std::unique_ptr<onnx::ModelProto> model_proto_;
  std::unique_ptr<ModelInfo> model_info_;
};

inline OrtModelImpl* OrtModel::ToInternal() {
  return static_cast<OrtModelImpl*>(this);
}

inline const OrtModelImpl* OrtModel::ToInternal() const {
  return static_cast<const OrtModelImpl*>(this);
}
