// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "NvInfer.h"

namespace onnxruntime{
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

class TRTKernel final : public OpKernel{
public:
    explicit TRTKernel(const OpKernelInfo& info);
    common::Status Compute(OpKernelContext* context) const override;

private:
    nvinfer1::IExecutionContext *tensorrt_context_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::vector<int> graph_input_indexes_;
    std::vector<std::string> graph_input_names_;
    std::vector<int> input_binding_indexes_;
    std::vector<int> input_dim_sizes_;
    std::vector<std::vector<int>> input_shapes_;
    std::vector<int> output_binding_indexes_;
    std::vector<int> output_dim_sizes_;
    std::vector<std::vector<int>> output_shapes_;
    int num_inputs_, num_outputs_;

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TRTKernel);
};

struct InferDeleter{
    template<typename T>
    void operator()(T* obj) const{
        if( obj ){
            obj->destroy();
        }
    }
};

template<typename T>
inline std::shared_ptr<T> InferObject(T* obj){
    if( !obj ){
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

class TRTLogger : public nvinfer1::ILogger{
    nvinfer1::ILogger::Severity verbosity_;
    std::ostream* ostream_;
public:
    TRTLogger(Severity verbosity=Severity::kWARNING,
               std::ostream& ostream=std::cout)
        : verbosity_(verbosity), ostream_(&ostream) {}
    void log(Severity severity, const char* msg) override{
        if( severity <= verbosity_ ){
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
            (*ostream_) << "[" << buf << " " << sevstr << "] "
                        << msg
                        << std::endl;
        }
    }
};
}  // namespace Lotus


