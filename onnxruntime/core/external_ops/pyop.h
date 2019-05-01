#pragma once
#ifdef _WIN32
#include <Windows.h>
#define PY_LIB_NAME        "onnxruntime_pyop.dll"
#define PY_LOAD_LIB(n)     LoadLibraryA(n)
#define PY_LOAD_SYM(h,n)   GetProcAddress(h,n)
#define PY_UNLOAD_LIB(h)   FreeLibrary(h)
#define PYDLE              HMODULE
#else
#define PY_LIB_NAME        "./libonnxruntime_pyop.so"
#define PY_LOAD_LIB(n)     dlopen(n,RTLD_NOW|RTLD_GLOBAL)
#define PY_LOAD_SYM(h,n)   dlsym(h,n)
#define PY_UNLOAD_LIB(h)   dlclose(h)
#define PYDLE              void*
#include "dlfcn.h"
#endif
#include "core/framework/ml_value.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/op_kernel_context_internal.h"
#include <iostream>
#include <vector>
#include <unordered_map>

#define PYOP(op) (static_cast<PyCustomOp*>(op))
#define PYKL(kl) (static_cast<PyCustomKernel*>(kl))

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

struct PyCustomKernel;

class PythonWrapper {

    friend struct PyCustomKernel;

    PythonWrapper() {

        handle = PY_LOAD_LIB(PY_LIB_NAME);
        ORT_ENFORCE(nullptr != handle, "Failed to load pyop library");

        init = (INIT*)PY_LOAD_SYM(handle, "Initialize");
        ORT_ENFORCE(nullptr != init, "Failed to import function: Initialize");

        newInst = (NEWINST*)PY_LOAD_SYM(handle, "NewInstance");
        ORT_ENFORCE(nullptr != newInst, "Failed to import function: NewInstance");

        invoke = (INVOKE*)PY_LOAD_SYM(handle, "InvokePythonFunc");
        ORT_ENFORCE(nullptr != invoke, "Failed to import function: InvokePythonFunc");

        release = (RELEASE*)PY_LOAD_SYM(handle, "ReleaseInstance");
        ORT_ENFORCE(nullptr != release, "Failed to import function: ReleaseInstance");

        lastErr = (LASTERR*)PY_LOAD_SYM(handle, "GetLastErrorMessage"); 
        ORT_ENFORCE(nullptr != lastErr, "Failed to import function: GetLastErrorMessage");

        std::string err;
        ORT_ENFORCE(init(), lastErr(err));
    }

    ~PythonWrapper() {
        PY_UNLOAD_LIB(handle);
    }

    PYDLE       handle  = nullptr;
    INIT*       init    = nullptr;
    NEWINST*    newInst = nullptr;
    INVOKE*     invoke  = nullptr;
    RELEASE*    release = nullptr;
    LASTERR*    lastErr = nullptr;
};

struct PyCustomKernel {

    PyCustomKernel (ORT_API               ort,
                    const ONNX_ATTRS&     attrs,
                    const std::string&    module,
                    const std::string&    class_name,
                    const std::string&    compute,
                    const std::string&    shape_infer,
                    LOG_FUNC              logging_func):
                    ort_(ort), attrs_(attrs), module_(module), class_name_(class_name),
                    compute_(compute), shape_infer_(shape_infer), logging_func_(logging_func) {

        std::string err;
        instance_ = GetPyWrapper().newInst(module.c_str(), class_name_.c_str(), attrs_);
        ORT_ENFORCE(nullptr != instance_, GetPyWrapper().lastErr(err));
    }

    ~PyCustomKernel() {
        if (nullptr != instance_) {
            GetPyWrapper().release(instance_);
            instance_ = nullptr;
        }
    }

    void GetOutputShape (OrtKernelContext* context, size_t index, OrtTensorTypeAndShapeInfo* info) {

        if ("" == shape_infer_) {
            return; // do nothing if shape inference function not specified
        }

        ORT_ENFORCE (nullptr != context);
        ORT_ENFORCE (nullptr != info);

        auto inputs_count = (size_t)reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
        ORT_ENFORCE(inputs_count > 0);
 
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
        ORT_ENFORCE (GetPyWrapper().invoke(instance_, shape_infer_.c_str(), inputs, inputs_type, inputs_dim, outputs, outputs_elem_size, outputs_dim, logging_func_), GetPyWrapper().lastErr(err));
        ORT_ENFORCE (outputs.size() > index, "output count is less then ort output index");
        ort_.SetDimensions(info, (const int64_t*)outputs[index], outputs_dim[index][0]);
        for (auto mem: outputs) {
            free(const_cast<void*>(mem));
        }
    }

    void Compute (OrtKernelContext* context) {

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
        ORT_ENFORCE (GetPyWrapper().invoke(instance_, compute_.c_str(), inputs, inputs_type, inputs_dim, outputs, outputs_elem_size, outputs_dim, logging_func_), GetPyWrapper().lastErr(err));
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

    PythonWrapper& GetPyWrapper()
    {
        static PythonWrapper pyWrapper;
        return pyWrapper;
    }

    ORT_API        ort_;
    ONNX_ATTRS     attrs_;
    std::string    module_;
    std::string    class_name_;
    std::string    compute_;
    std::string    shape_infer_;
    void*          instance_ = nullptr;
    LOG_FUNC       logging_func_;
};

struct PyCustomOp: onnxruntime::CustomOpBase<PyCustomOp, PyCustomKernel> {

    PyCustomOp(const ONNX_ATTRS&    attrs,
               const ONNX_TYPES&    input_types,
               const ONNX_TYPES&    output_types,
               const std::string&   module,
               const std::string&   class_name,
               const std::string&   compute      = "compute",
               const std::string&   shape_infer  = "", // shape inference is optional
               LOG_FUNC             logging_func = [](const char*){}):
               attrs_(attrs), input_types_(input_types),
               output_types_(output_types), module_(module),
               class_name_(class_name), compute_(compute),
               shape_infer_(shape_infer), logging_func_(logging_func) {
               OrtCustomOp::version = ORT_API_VERSION; }
 
    void* CreateKernel (onnxruntime::CustomOpApi api, const OrtKernelInfo*) {
        return new PyCustomKernel(api, attrs_, module_, class_name_, compute_, shape_infer_, logging_func_);
    }

    const char* GetName() const { return "PyOp"; }

    size_t GetInputTypeCount() const { return input_types_.size(); }
    ONNXTensorElementDataType GetInputType(size_t index) const { return input_types_[index]; }

    size_t GetOutputTypeCount() const { return output_types_.size(); }
    ONNXTensorElementDataType GetOutputType(size_t index) const { return output_types_[index]; }

private:

    ONNX_ATTRS     attrs_;
    ONNX_TYPES     input_types_;
    ONNX_TYPES     output_types_;
    std::string    module_;
    std::string    class_name_;
    std::string    compute_;
    std::string    shape_infer_;
    LOG_FUNC       logging_func_;
};//PyCusomOp

}
