// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "winml_adapter_c_api.h"
#include <memory>
#include "core/graph/onnx_protobuf.h"

class ModelInfo;

struct OrtModel {
 public:
  static OrtStatus* CreateEmptyModel(int64_t opset, OrtModel** model);
  static OrtStatus* CreateOrtModelFromPath(const char* path, size_t len, OrtModel** model);
  static OrtStatus* CreateOrtModelFromData(void* data, size_t len, OrtModel** model);
  static OrtStatus* CreateOrtModelFromProto(std::unique_ptr<onnx::ModelProto>&& model_proto, OrtModel** model);
  const ModelInfo* UseModelInfo() const;

  onnx::ModelProto* UseModelProto() const;
  std::unique_ptr<onnx::ModelProto> DetachModelProto();

 private:
  OrtModel(std::unique_ptr<onnx::ModelProto> model_proto);
  OrtModel(const OrtModel& other) = delete;
  OrtModel& operator=(const OrtModel& other) = delete;

 private:
  std::unique_ptr<onnx::ModelProto> model_proto_;
  std::unique_ptr<ModelInfo> model_info_;
};
