// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pyop.h"
#ifdef _WIN32
#define LIB_PYOP "onnxruntime_pywrapper.dll"
#define LOAD_PYOP_LIB(n, v, m) ORT_ENFORCE((v = LoadLibraryA(n)) != nullptr, m)
#else
#ifdef __APPLE__
#define LIB_PYOP "./libonnxruntime_pywrapper.dylib"
#else
#define LIB_PYOP "./libonnxruntime_pywrapper.so"
#endif
#define LOAD_PYOP_LIB(n, v, m) ORT_ENFORCE((v = dlopen(n, RTLD_NOW | RTLD_GLOBAL)) != nullptr, m)
#include "dlfcn.h"
#endif
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
// #define LOAD_PYOP_SYM(n, v, m) ORT_ENFORCE(Env::Default().GetSymbolFromLibrary(handle_, n, reinterpret_cast<void**>(&v)) == Status::OK(), m)

namespace onnxruntime {

const PyOpLibProxy& PyOpLibProxy::GetInstance() {
  static PyOpLibProxy proxy;
  return proxy;
}

PyOpLibProxy::PyOpLibProxy() {
  std::string err;
  LOAD_PYOP_LIB(LIB_PYOP, handle_, "Failed to load pyop library");
  LOAD_PYOP_SYM("Initialize", initialize_, "Failed to import function: Initialize");
  LOAD_PYOP_SYM("NewInstance", new_instance_, "Failed to import function: NewInstance");
  LOAD_PYOP_SYM("InvokePythonFunc", invoke_python_func_, "Failed to import function: InvokePythonFunc");
  LOAD_PYOP_SYM("ReleaseInstance", release_instance_, "Failed to import function: ReleaseInstance");
  LOAD_PYOP_SYM("GetLastErrorMessage", get_last_error_message_, "Failed to import function: GetLastErrorMessage");
  ORT_ENFORCE(initialize_(), get_last_error_message_(err));
}

PyOpLibProxy::~PyOpLibProxy() {
  Env::Default().UnloadDynamicLibrary(handle_);
}

PyCustomKernel::PyCustomKernel(Ort::CustomOpApi ort,
                               const OnnxAttrs& attrs,
                               const std::string& module,
                               const std::string& class_name,
                               const std::string& compute,
                               PyOpLogFunc logging_func) : ort_(ort), attrs_(attrs), module_(module), class_name_(class_name), compute_(compute), logging_func_(logging_func) {
  std::string err;
  instance_ = PyOpLibProxy::GetInstance().new_instance_(module.c_str(), class_name_.c_str(), attrs_);
  ORT_ENFORCE(nullptr != instance_, PyOpLibProxy::GetInstance().get_last_error_message_(err));
}

PyCustomKernel::~PyCustomKernel() {
  if (nullptr != instance_) {
    PyOpLibProxy::GetInstance().release_instance_(instance_);
    instance_ = nullptr;
  }
}

// Do nothing since Custom Op does not trigger shape inference
void PyCustomKernel::GetOutputShape(OrtKernelContext*, size_t, OrtTensorTypeAndShapeInfo*) {}

void PyCustomKernel::Compute(OrtKernelContext* context) {
  ORT_ENFORCE(nullptr != context);
  auto inputs_count = (size_t) reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
  std::vector<const void*> inputs;
  std::vector<std::unique_ptr<char[]>> outputs;
  std::vector<int32_t> inputs_type, outputs_elem_size;
  std::vector<std::vector<int64_t>> inputs_dim, outputs_dim;

  for (size_t i = 0; i < inputs_count; ++i) {
    auto ort_value = ort_.KernelContext_GetInput(context, i);
    inputs.push_back(const_cast<MLValue*>(ort_value)->Get<Tensor>().DataRaw());
    inputs_type.push_back(GetType(ort_value));
    inputs_dim.push_back(const_cast<MLValue*>(ort_value)->Get<Tensor>().Shape().GetDims());
  }

  std::string err;
  ORT_ENFORCE(PyOpLibProxy::GetInstance().invoke_python_func_(instance_, compute_.c_str(), inputs, inputs_type,
                                                              inputs_dim, outputs, outputs_elem_size,
                                                              outputs_dim, logging_func_),
              PyOpLibProxy::GetInstance().get_last_error_message_(err));  //ORT_ENFORCE

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto ort_output = ort_.KernelContext_GetOutput(context, i, outputs_dim[i].data(), outputs_dim[i].size());
    auto output_mem_addr = ort_.GetTensorMutableData<char>(ort_output);
    auto output_len = std::accumulate(begin(outputs_dim[i]), end(outputs_dim[i]), static_cast<int64_t>(outputs_elem_size[i]), std::multiplies<int64_t>());
    memcpy(output_mem_addr, outputs[i].get(), output_len);
  }
}

int32_t PyCustomKernel::GetType(const OrtValue* input) const {
  int32_t numpy_type;
  ORT_ENFORCE(nullptr != input);
  ORT_ENFORCE(input->IsTensor(), "input must be a tensor");
  auto elem_type = input->Get<Tensor>().GetElementType();

  namespace on = ONNX_NAMESPACE;
  switch (elem_type) {
    case on::TensorProto_DataType_BOOL:
      numpy_type = 0;
      break;
    case on::TensorProto_DataType_INT8:
      numpy_type = 1;
      break;
    case on::TensorProto_DataType_UINT8:
      numpy_type = 2;
      break;
    case on::TensorProto_DataType_INT16:
      numpy_type = 3;
      break;
    case on::TensorProto_DataType_UINT16:
      numpy_type = 4;
      break;
    case on::TensorProto_DataType_INT32:
      numpy_type = 5;
      break;
    case on::TensorProto_DataType_UINT32:
      numpy_type = 6;
      break;
    case on::TensorProto_DataType_INT64:
      numpy_type = 9;
      break;
    case on::TensorProto_DataType_UINT64:
      numpy_type = 10;
      break;
    case on::TensorProto_DataType_FLOAT:
      numpy_type = 11;
      break;
    case on::TensorProto_DataType_DOUBLE:
      numpy_type = 12;
      break;
    default:
      ORT_THROW("Input primitive type not supported: ", DataTypeImpl::ToString(input->Get<Tensor>().DataType()));
  }
  return numpy_type;
}

PyCustomOp::PyCustomOp(const OnnxAttrs& attrs,
                       const OnnxTypes& inputs_type,
                       const OnnxTypes& outputs_type,
                       const std::string& module,
                       const std::string& class_name,
                       const std::string& compute,
                       PyOpLogFunc logging_func) : attrs_(attrs), inputs_type_(inputs_type), outputs_type_(outputs_type), module_(module), class_name_(class_name), compute_(compute), logging_func_(logging_func) { OrtCustomOp::version = ORT_API_VERSION; }

void* PyCustomOp::CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo*) {
  return new PyCustomKernel(api, attrs_, module_, class_name_, compute_, logging_func_);
}

const char* PyCustomOp::GetName() const { return "PyOp"; }

size_t PyCustomOp::GetInputTypeCount() const { return inputs_type_.size(); }
ONNXTensorElementDataType PyCustomOp::GetInputType(size_t index) const { return inputs_type_[index]; }

size_t PyCustomOp::GetOutputTypeCount() const { return outputs_type_.size(); }
ONNXTensorElementDataType PyCustomOp::GetOutputType(size_t index) const { return outputs_type_[index]; }

PyCustomOp* LoadPyOp(const ONNX_NAMESPACE::NodeProto& node_proto, PyOpLogFunc log_func) {
  OnnxAttrs onnx_attrs;
  OnnxTypes input_types, output_types;
  std::string module, class_name, compute = "compute";
  for (int j = 0; j < node_proto.attribute_size(); ++j) {
    const auto& attr = node_proto.attribute(j);
    if (utils::HasString(attr)) {
      if (attr.name() == "module")
        module = attr.s();
      else if (attr.name() == "class_name")
        class_name = attr.s();
      else if (attr.name() == "compute")
        compute = attr.s();
      else
        onnx_attrs[attr.name()] = attr.s();
    } else if (attr.ints_size() > 0) {
      if (attr.name() == "input_types") {
        for (int k = 0; k < attr.ints_size(); ++k) {
          input_types.push_back(static_cast<ONNXTensorElementDataType>(attr.ints(k)));
        }
      } else if (attr.name() == "output_types") {
        for (int k = 0; k < attr.ints_size(); ++k) {
          output_types.push_back(static_cast<ONNXTensorElementDataType>(attr.ints(k)));
        }
      }
    }
  }  //for
  ORT_ENFORCE(module != "", "PyOp module not specified");
  ORT_ENFORCE(class_name != "", "PyOp class name not specified");
  ORT_ENFORCE(!input_types.empty(), "PyOp node inputs not specified");
  ORT_ENFORCE(!output_types.empty(), "PyOp node outputs not specified");
  return new PyCustomOp(onnx_attrs, input_types, output_types, module, class_name, compute, log_func);
}
}  // namespace onnxruntime
