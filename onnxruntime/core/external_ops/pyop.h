#pragma once
#ifdef _WIN32
#include <Windows.h>
#define LIB_PYOP           "onnxruntime_pyop.dll"
#define LOAD_PYOP_LIB(n)   LoadLibraryA(n)
#define LOAD_PYOP_SYM(h,n) GetProcAddress(h,n)
#define UNLD_PYOP_LIB(h)   FreeLibrary(h)
#else
#ifdef __APPLE__
#define LIB_PYOP           "./libonnxruntime_pyop.dylib"
#else
#define LIB_PYOP           "./libonnxruntime_pyop.so"
#endif
#define LOAD_PYOP_LIB(n)   dlopen(n,RTLD_NOW|RTLD_GLOBAL)
#define LOAD_PYOP_SYM(h,n) dlsym(h,n)
#define UNLD_PYOP_LIB(h)   dlclose(h)
#define HMODULE            void*
#include "dlfcn.h"
#endif
#include "core/framework/ml_value.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/op_kernel_context_internal.h"
#include <iostream>
#include <vector>
#include <unordered_map>

using ONNX_TYPES = std::vector<ONNXTensorElementDataType>;
using ONNX_ATTRS = std::unordered_map<std::string, std::string>;
using ORT_SHAPE  = OrtTensorTypeAndShapeInfo;
using LOG_FUNC   = std::function<void(const char*)>;
using ORT_API    = onnxruntime::CustomOpApi;

typedef bool INIT();
typedef bool PYFUNC(const char*,
                    const char*,
                    const std::vector<const void*>&,
                    const std::vector<int32_t>&,
                    const std::vector<std::vector<int64_t>>&,
                    std::vector<const void*>&,
                    std::vector<int32_t>&,
                    std::vector<std::vector<int64_t>>&);
typedef void* NEWINST(const char*, const char*, const ONNX_ATTRS&);
typedef bool INVOKE(void*,
                    const char*,
                    const std::vector<const void*>&,
                    const std::vector<int32_t>&,
                    const std::vector<std::vector<int64_t>>&,
                    std::vector<const void*>&,
                    std::vector<int32_t>&,
                    std::vector<std::vector<int64_t>>&,
                    std::function<void(const char*)>);
typedef void RELEASE(void*);
typedef const char* LASTERR(std::string&);
typedef void SETPATH(const wchar_t*);

namespace onnxruntime {

class PythonWrapper {

public:
    static const PythonWrapper& GetInstance() {
        static PythonWrapper wrapper;
        return wrapper;
    }

    HMODULE     handle  = nullptr;
    INIT*       init    = nullptr;
    NEWINST*    newInst = nullptr;
    INVOKE*     invoke  = nullptr;
    RELEASE*    release = nullptr;
    LASTERR*    lastErr = nullptr;

private:
    PythonWrapper() {

        handle = LOAD_PYOP_LIB(LIB_PYOP);
        ORT_ENFORCE(nullptr != handle, "Failed to load pyop library");

        init = (INIT*)LOAD_PYOP_SYM(handle, "Initialize");
        ORT_ENFORCE(nullptr != init, "Failed to import function: Initialize");

        newInst = (NEWINST*)LOAD_PYOP_SYM(handle, "NewInstance");
        ORT_ENFORCE(nullptr != newInst, "Failed to import function: NewInstance");

        invoke = (INVOKE*)LOAD_PYOP_SYM(handle, "InvokePythonFunc");
        ORT_ENFORCE(nullptr != invoke, "Failed to import function: InvokePythonFunc");

        release = (RELEASE*)LOAD_PYOP_SYM(handle, "ReleaseInstance");
        ORT_ENFORCE(nullptr != release, "Failed to import function: ReleaseInstance");

        lastErr = (LASTERR*)LOAD_PYOP_SYM(handle, "GetLastErrorMessage"); 
        ORT_ENFORCE(nullptr != lastErr, "Failed to import function: GetLastErrorMessage");

        std::string err;
        ORT_ENFORCE(init(), lastErr(err));
    }

    ~PythonWrapper() {
        UNLD_PYOP_LIB(handle);
    }
};

struct PyCustomKernel {

    PyCustomKernel(ORT_API             ort,
                   const ONNX_ATTRS&   attrs,
                   const std::string&  module,
                   const std::string&  class_name,
                   const std::string&  compute,
                   LOG_FUNC            logging_func):
                   ort_(ort), attrs_(attrs), module_(module), class_name_(class_name),
                   compute_(compute), logging_func_(logging_func) {
        std::string err;
        instance_ = PythonWrapper::GetInstance().newInst(module.c_str(), class_name_.c_str(), attrs_);
        ORT_ENFORCE(nullptr != instance_, PythonWrapper::GetInstance().lastErr(err));
    }

    ~PyCustomKernel() {
        if (nullptr != instance_) {
            PythonWrapper::GetInstance().release(instance_);
            instance_ = nullptr;
        }
    }

    // Do nothing since Custom Op does not trigger share inference
    void GetOutputShape(OrtKernelContext*, size_t, OrtTensorTypeAndShapeInfo*) {}

    void Compute(OrtKernelContext* context) {

        ORT_ENFORCE (nullptr != context);
        auto inputs_count = (size_t)reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
        std::vector<const void*>            inputs,      outputs;
        std::vector<int32_t>                inputs_type, outputs_elem_size;
        std::vector<std::vector<int64_t>>   inputs_dim,  outputs_dim;

        for (size_t i = 0; i < inputs_count; ++i) {
            auto ort_value = ort_.KernelContext_GetInput(context, i);
            inputs.push_back(((MLValue*)ort_value)->Get<Tensor>().DataRaw());
            inputs_type.push_back(GetType(ort_value));
            inputs_dim.push_back(((MLValue*)ort_value)->Get<Tensor>().Shape().GetDims());
        }

        std::string err;
        ORT_ENFORCE (PythonWrapper::GetInstance().invoke(instance_, compute_.c_str(), inputs, inputs_type, inputs_dim, outputs, outputs_elem_size, outputs_dim, logging_func_), PythonWrapper::GetInstance().lastErr(err));
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto ort_output  = ort_.KernelContext_GetOutput(context, i, outputs_dim[i].data(), outputs_dim[i].size());
            auto output_mem_addr = ort_.GetTensorMutableData<char>(ort_output);
            auto output_len = std::accumulate(begin(outputs_dim[i]), end(outputs_dim[i]), static_cast<int64_t>(outputs_elem_size[i]), std::multiplies<int64_t>());
            memcpy(output_mem_addr, outputs[i], output_len);
            free(const_cast<void*>(outputs[i]));
        }
    }

    int32_t GetType(OrtValue* input) const
    {
        int32_t numpy_type;
        ORT_ENFORCE (nullptr != input);
        ORT_ENFORCE(((MLValue*)input)->IsTensor(), "input must be a tensor");
        auto data_type = ((MLValue*)input)->Get<Tensor>().DataType();
        if (data_type == DataTypeImpl::GetType<bool>()) {
            numpy_type = 0;
        } else if (data_type == DataTypeImpl::GetType<int8_t>()) {
            numpy_type = 1;
        } else if (data_type == DataTypeImpl::GetType<uint8_t>()) {
            numpy_type = 2;
        } else if (data_type == DataTypeImpl::GetType<int16_t>()) {
            numpy_type = 3;
        } else if (data_type == DataTypeImpl::GetType<uint16_t>()) {
            numpy_type = 4;
        } else if (data_type == DataTypeImpl::GetType<int32_t>()) {
            numpy_type = 5;
        } else if (data_type == DataTypeImpl::GetType<uint32_t>()) {
            numpy_type = 6;
        } else if (data_type == DataTypeImpl::GetType<int64_t>()) {
            numpy_type = 9;
        } else if (data_type == DataTypeImpl::GetType<uint64_t>()) {
            numpy_type = 10;
        } else if (data_type == DataTypeImpl::GetType<float>()) {
            numpy_type = 11;
        } else if (data_type == DataTypeImpl::GetType<double>()) {
            numpy_type = 12;
        } else if (data_type == DataTypeImpl::GetType<std::string>()) {
            numpy_type = 18;
        } else if (data_type == DataTypeImpl::GetType<MLFloat16>() ||
                   data_type == DataTypeImpl::GetType<BFloat16>()) {
            numpy_type = 23;
        } else ORT_ENFORCE(false, "Input type not supported");
        return numpy_type;
    }

private:
    ORT_API        ort_;
    ONNX_ATTRS     attrs_;
    std::string    module_;
    std::string    class_name_;
    std::string    compute_;
    void*          instance_ = nullptr;
    LOG_FUNC       logging_func_;
};

struct PyCustomOp: onnxruntime::CustomOpBase<PyCustomOp, PyCustomKernel> {

    PyCustomOp(const ONNX_ATTRS&    attrs,
               const ONNX_TYPES&    inputs_type,
               const ONNX_TYPES&    outputs_type,
               const std::string&   module,
               const std::string&   class_name,
               const std::string&   compute      = "compute",
               LOG_FUNC             logging_func = [](const char*){}):
               attrs_(attrs), inputs_type_(inputs_type),
               outputs_type_(outputs_type), module_(module),
               class_name_(class_name), compute_(compute),
               logging_func_(logging_func) {
               OrtCustomOp::version = ORT_API_VERSION; }
 
    void* CreateKernel (onnxruntime::CustomOpApi api, const OrtKernelInfo*) {
        return new PyCustomKernel(api, attrs_, module_, class_name_, compute_, logging_func_);
    }

    const char* GetName() const { return "PyOp"; }

    size_t GetInputTypeCount() const { return inputs_type_.size(); }
    ONNXTensorElementDataType GetInputType(size_t index) const { return inputs_type_[index]; }

    size_t GetOutputTypeCount() const { return outputs_type_.size(); }
    ONNXTensorElementDataType GetOutputType(size_t index) const { return outputs_type_[index]; }

private:

    ONNX_ATTRS     attrs_;
    ONNX_TYPES     inputs_type_;
    ONNX_TYPES     outputs_type_;
    std::string    module_;
    std::string    class_name_;
    std::string    compute_;
    LOG_FUNC       logging_func_;
};//PyCusomOp

}
