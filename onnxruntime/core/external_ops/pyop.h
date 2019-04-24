#pragma once
#include "dlfcn.h"
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
typedef const char* LASTERR();
typedef void SETPATH(const wchar_t*);

namespace onnxruntime {

struct PythonWrapper {

    PythonWrapper() {

        handle = dlopen("./libonnxruntime_pyop.so", RTLD_NOW | RTLD_GLOBAL);
        ORT_ENFORCE(nullptr != handle, dlerror());

        init = (INIT*)dlsym(handle, "Initialize");
        ORT_ENFORCE(nullptr != init, dlerror());

        pyFunc = (PYFUNC*)dlsym(handle, "CallPythonFunction");
        ORT_ENFORCE(nullptr != pyFunc, dlerror());

        newInst = (NEWINST*)dlsym(handle, "NewInstance");
        ORT_ENFORCE(nullptr != newInst, dlerror());

        invoke = (INVOKE*)dlsym(handle, "InvokePythonFunc");
        ORT_ENFORCE(nullptr != invoke, dlerror());

        release = (RELEASE*)dlsym(handle, "ReleaseInstance");
        ORT_ENFORCE(nullptr != release, dlerror());

        lastErr = (LASTERR*)dlsym(handle, "GetLastErrorMessage"); 
        ORT_ENFORCE(nullptr != lastErr, dlerror());

        setPath = (SETPATH*)dlsym(handle, "SetSysPath");
        ORT_ENFORCE(nullptr != setPath, dlerror());

        ORT_ENFORCE(init(), lastErr());
        setPath(L".");      
    }

    ~PythonWrapper() {
        dlclose(handle);
    }

    void*       handle  = nullptr;
    INIT*       init    = nullptr;
    PYFUNC*     pyFunc  = nullptr;
    NEWINST*    newInst = nullptr;
    INVOKE*     invoke  = nullptr;
    RELEASE*    release = nullptr;
    LASTERR*    lastErr = nullptr;
    SETPATH*    setPath = nullptr;
};

struct PyCustomKernel {

    PyCustomKernel (onnxruntime::CustomOpApi ort,
                    const ONNX_ATTRS&        attrs,
                    const std::string&       module,
                    const std::string&       class_name,
                    const std::string&       compute,
                    const std::string&       shape_infer): 
                    ort_(ort), attrs_(attrs), module_(module),
                    class_name_(class_name), compute_(compute), shape_infer_(shape_infer) {
        instance_ = GetPyWrapper().newInst(module.c_str(), class_name_.c_str(), attrs_);
        ORT_ENFORCE(nullptr != instance_, GetPyWrapper().lastErr());
    }

    ~PyCustomKernel() {
        if (nullptr != instance_) {
            GetPyWrapper().release(instance_);
            instance_ = nullptr;
        }
    }

    void GetOutputShape (OrtKernelContext* context, size_t index, OrtTensorTypeAndShapeInfo* info) {
        auto input_count = (size_t)reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
        ORT_ENFORCE(input_count > 0);
        std::vector<const void*>            input,      output;
        std::vector<int32_t>                input_type, output_size;
        std::vector<std::vector<int64_t>>   input_dim,  output_dim;

        for (size_t i = 0; i < input_count; ++i) {
            auto ort_value = ort_.KernelContext_GetInput(context, i);
            input.push_back(((MLValue*)ort_value)->Get<Tensor>().DataRaw());
            input_type.push_back(GetType(ort_value));
            input_dim.push_back(((MLValue*)ort_value)->Get<Tensor>().Shape().GetDims());
        }

//       ORT_ENFORCE (GetPyWrapper().pyFunc(module_.c_str(), shape_infer_.c_str(), input, input_type, input_dim, output, output_size, output_dim), GetPyWrapper().lastErr());
        ORT_ENFORCE (GetPyWrapper().invoke(instance_, shape_infer_.c_str(), input, input_type, input_dim, output, output_size, output_dim, [](const char*){}), GetPyWrapper().lastErr());
        ORT_ENFORCE (output.size() > index, "output count is less then ort output index");
        ort_.SetDimensions(info, (const int64_t*)output[index], output_dim[index][0]);
        for (auto mem: output) {
            free(const_cast<void*>(mem));
        }
    }

    void Compute (OrtKernelContext* context) {
        auto input_count = (size_t)reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->InputCount();
        std::vector<const void*>            input,      output;
        std::vector<int32_t>                input_type, output_size;
        std::vector<std::vector<int64_t>>   input_dim,  output_dim;

        for (size_t i = 0; i < input_count; ++i) {
            auto ort_value = ort_.KernelContext_GetInput(context, i);
            input.push_back(((MLValue*)ort_value)->Get<Tensor>().DataRaw());
            input_type.push_back(GetType(ort_value));
            input_dim.push_back(((MLValue*)ort_value)->Get<Tensor>().Shape().GetDims());
        }

//      ORT_ENFORCE (GetPyWrapper().pyFunc(module_.c_str(), compute_.c_str(), input, input_type, input_dim, output, output_size, output_dim), GetPyWrapper().lastErr());
        ORT_ENFORCE (GetPyWrapper().invoke(instance_, compute_.c_str(), input, input_type, input_dim, output, output_size, output_dim, [](const char*){}), GetPyWrapper().lastErr());
        for (size_t i = 0; i < output.size(); ++i) {
            OrtValue* ort_output  = ort_.KernelContext_GetOutput(context, i, output_dim[i].data(), output_dim[i].size());
            char* output_mem_addr = ort_.GetTensorMutableData<char>(ort_output);
            auto output_len = std::accumulate(begin(output_dim[i]), end(output_dim[i]), output_size[i], std::multiplies<int64_t>());
            memcpy(output_mem_addr, output[i], output_len);
            free(const_cast<void*>(output[i]));
        }
    }

    int32_t GetType(OrtValue* input) const
    {
        int32_t numpy_type;
        ORT_ENFORCE(((MLValue*)input)->IsTensor(), "input is not tensor");
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
        } else ORT_ENFORCE(false, "Input type not supported");

        return numpy_type;
    }

private:

    PythonWrapper& GetPyWrapper()
    {
        static PythonWrapper pyWrapper;
        return pyWrapper;
    }

    onnxruntime::CustomOpApi ort_;
    ONNX_ATTRS  attrs_;
    std::string module_;
    std::string class_name_;
    std::string compute_;
    std::string shape_infer_;
    void* instance_ = nullptr;
};

struct PyCustomOp: onnxruntime::CustomOpBase<PyCustomOp, PyCustomKernel> {

    PyCustomOp(const ONNX_ATTRS&    attrs,
               const ONNX_TYPES&    input_types,
               const ONNX_TYPES&    output_types,
               const std::string&   module,
               const std::string&   class_name,
               const std::string&   compute     = "compute",
               const std::string&   shape_infer = "shape_infer"):
               attrs_(attrs),
               input_types_(input_types),
               output_types_(output_types),
               module_(module), class_name_(class_name), compute_(compute),
               shape_infer_(shape_infer) { OrtCustomOp::version = ORT_API_VERSION; } 
 
    void* CreateKernel (onnxruntime::CustomOpApi api, const OrtKernelInfo*) {
        return new PyCustomKernel(api, attrs_, module_, class_name_, compute_, shape_infer_);
    }

    const char* GetName() const { return class_name_.c_str(); }

    size_t GetInputTypeCount() const { return input_types_.size(); }
    ONNXTensorElementDataType GetInputType(size_t index) const { return input_types_[index]; }

    size_t GetOutputTypeCount() const { return output_types_.size(); }
    ONNXTensorElementDataType GetOutputType(size_t index) const { return output_types_[index]; }

private:
    ONNX_ATTRS  attrs_;
    ONNX_TYPES  input_types_;
    ONNX_TYPES  output_types_;
    std::string module_;
    std::string class_name_;
    std::string compute_;
    std::string shape_infer_;
};//PyCusomOp

}
