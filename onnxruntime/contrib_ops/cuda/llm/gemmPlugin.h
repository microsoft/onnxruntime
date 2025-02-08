/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include  "contrib_ops/cuda/llm/common/cublasMMWrapper.h"
#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <string>
#include <vector>

namespace onnxruntime::llm
{

using CublasGemmWrapper = onnxruntime::llm::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;

class CublasLtGemmPluginProfiler
    : public GemmPluginProfiler<cublasLtMatmulHeuristicResult_t, CublasGemmWrapperPtr, GemmIdCublas, GemmIdCublasHash>
{
public:
    using Config = cublasLtMatmulHeuristicResult_t;

    void setTranspose(bool transposeA, bool transposeB)
    {
        mTransA = transposeA;
        mTransB = transposeB;
    }

    void setPadLd(int padLda, int padLdb, int padLdc)
    {
        mPadLda = padLda;
        mPadLdb = padLdb;
        mPadLdc = padLdc;
    }

    void setOutputType(nvinfer1::DataType type)
    {
        mOutputType = type;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    bool checkTactic(int m, int n, int k, Config const& tactic) const override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    bool mTransA;
    bool mTransB;
    int mPadLda;
    int mPadLdb;
    int mPadLdc;
    nvinfer1::DataType mOutputType;

    static constexpr size_t ALIGNMENT = 256;
};

class GemmPlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<CublasLtGemmPluginProfiler>;

    GemmPlugin() = delete;

    GemmPlugin(int transA, int transB, int padLda, int padLdb, int padLdc, nvinfer1::DataType type, bool useFp8,
        float alpha, PluginProfilerPtr const& profiler);

    GemmPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~GemmPlugin() override = default;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    void init();
    void configGemm();
    void setGemmConfig();

private:
    const std::string mLayerName;

    int mTransA;
    int mTransB;
    int mPadLda;
    int mPadLdb;
    int mPadLdc;
    nvinfer1::DataType mType;
    nvinfer1::DataType mOutputType;

    static thread_local CublasGemmWrapperPtr mCublasWrapper;

    GemmDims mDims{};
    GemmIdCublas mGemmId{};
    bool mUseFp8{false};
    float mAlpha{1.f};

    PluginProfilerPtr mPluginProfiler;
};

class GemmPluginCreator : public BaseCreator
{
public:
    GemmPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<CublasLtGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}  // namespace onnxruntime::llm


using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

class Gemm final : public CudaKernel {
 public:
 Gemm(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct KernelLaunchDispatcher {
    void operator()(cudaStream_t stream, int64_t input_size, int64_t bias_size, const Tensor& X, const Tensor& B,
                    Tensor& Y) const;
  };
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
