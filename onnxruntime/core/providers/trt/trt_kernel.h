// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "NvInfer.h"

namespace onnxruntime
{
static const int kBatchSize = 1;
static const int max_batch_size = 1;
static const int max_workspace_size = 1 << 30;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

class TRTKernel final : public OpKernel
{
public:
    explicit TRTKernel(const OpKernelInfo& info);
    common::Status Compute(OpKernelContext* context) const override;

private:
    nvinfer1::IExecutionContext *tensorrt_context_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::vector<int> graph_input_index_;
    std::vector<std::string> graph_input_name_;
    std::vector<int> input_binding_index_;
    std::vector<int> input_dim_size_;
    std::vector<std::vector<int>> input_dimension_;
    std::vector<int> output_binding_index_;
    std::vector<int> output_dim_size_;
    std::vector<std::vector<int>> output_dimension_;
};

struct InferDeleter
{
    template<typename T>
    void operator()(T* obj) const
    {
        if( obj )
        {
            obj->destroy();
        }
    }
};

template<typename T>
inline std::shared_ptr<T> InferObject(T* obj)
{
    if( !obj )
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

class TRTLogger : public nvinfer1::ILogger
{
    nvinfer1::ILogger::Severity _verbosity;
    std::ostream* _ostream;
public:
    TRTLogger(Severity verbosity=Severity::kWARNING,
               std::ostream& ostream=std::cout)
        : _verbosity(verbosity), _ostream(&ostream) {}
    void log(Severity severity, const char* msg) override
    {
        if( severity <= _verbosity )
        {
            time_t rawtime = std::time(0);
            char buf[256];
            strftime(&buf[0], 256,
                     "%Y-%m-%d %H:%M:%S",
                     std::gmtime(&rawtime));
            const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" :
                                  severity == Severity::kERROR          ? "  ERROR" :
                                  severity == Severity::kWARNING        ? "WARNING" :
                                  severity == Severity::kINFO           ? "   INFO" :
                                  "UNKNOWN");
            (*_ostream) << "[" << buf << " " << sevstr << "] "
                        << msg
                        << std::endl;
        }
    }
};
}  // namespace Lotus

