// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/env.h"
#define LOAD_PYOP_SYM(n, v, m) ORT_ENFORCE(Env::Default().GetSymbolFromLibrary(handle_, n, reinterpret_cast<void**>(&v)) == Status::OK(), m)

#include "core/session/onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#ifdef _WIN32
#include <Windows.h>
#else
#define HMODULE void*
#endif

namespace ONNX_NAMESPACE {
class NodeProto;
}

namespace onnxruntime {

using OnnxTypes = std::vector<ONNXTensorElementDataType>;
using OnnxAttrs = std::unordered_map<std::string, std::string>;
using PyOpLogFunc = std::function<void(const char*)>;

class PyOpLibProxy {
 public:
  static PyOpLibProxy& GetInstance();
  void ReleaseInstance(void*);
  bool InvokePythonFunc(void*,
                        const char*,
                        const std::vector<const void*>&,
                        const std::vector<int32_t>&,
                        const std::vector<std::vector<int64_t>>&,
                        std::vector<std::unique_ptr<char[]>>&,
                        std::vector<int32_t>&,
                        std::vector<std::vector<int64_t>>&,
                        std::function<void(const char*)>);
  const char* GetLastErrorMessage(std::string&);
  void* NewInstance(const char*, const char*, const OnnxAttrs&);
  bool Initialized() const { return initialized_; }
  int32_t GetGil() const;
  void PutGil(int32_t) const;

 private:
  PyOpLibProxy();
  ~PyOpLibProxy();
  bool initialized_ = false;
};

struct PyCustomKernel {
  PyCustomKernel(const OnnxAttrs& attrs,
                 const std::string& module,
                 const std::string& class_name,
                 const std::string& compute,
                 PyOpLogFunc logging_func);
  ~PyCustomKernel();
  void GetOutputShape(OrtKernelContext*, size_t, OrtTensorTypeAndShapeInfo*);
  void Compute(OrtKernelContext* context);
  int32_t GetNumpyType(int32_t elem_type) const;

 private:
  OnnxAttrs attrs_;
  std::string module_;
  std::string class_name_;
  std::string compute_;
  void* instance_ = nullptr;
  PyOpLogFunc logging_func_;
};

struct PyCustomOp : Ort::CustomOpBase<PyCustomOp, PyCustomKernel> {
  PyCustomOp(
      const OnnxAttrs& attrs,
      const OnnxTypes& inputs_type,
      const OnnxTypes& outputs_type,
      const std::string& module,
      const std::string& class_name,
      const std::string& compute = "compute",
      PyOpLogFunc logging_func = [](const char*) {});
  void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;

 private:
  OnnxAttrs attrs_;
  OnnxTypes inputs_type_;
  OnnxTypes outputs_type_;
  std::string module_;
  std::string class_name_;
  std::string compute_;
  PyOpLogFunc logging_func_;
};  // struct PyCustomOp

PyCustomOp* LoadPyOp(const ONNX_NAMESPACE::NodeProto& node_proto, PyOpLogFunc log_func);
}  // namespace onnxruntime
