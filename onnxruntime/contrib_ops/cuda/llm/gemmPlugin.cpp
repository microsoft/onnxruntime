/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "contrib_ops/cuda/llm/gemmPlugin.h"

#include "contrib_ops/cuda/llm/gemm_profiler.h"
// #include "contrib_ops/cuda/llm/common/plugin.h"
#include "contrib_ops/cuda/llm/common/pluginUtils.h"
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/cudaCoreGemm.h"
#include "contrib_ops/cuda/llm/common/debugUtils.h"

#include <cassert>

using namespace nvinfer1;
using namespace onnxruntime::llm::common;
using onnxruntime::llm::GemmDims;
using onnxruntime::llm::GemmPluginCreator;
using onnxruntime::llm::GemmPlugin;
using onnxruntime::llm::CublasLtGemmPluginProfiler;
using onnxruntime::llm::CublasGemmWrapperPtr;
using onnxruntime::llm::read;
using onnxruntime::llm::write;

static char const* GEMM_PLUGIN_VERSION{"1"};
static char const* GEMM_PLUGIN_NAME{"Gemm"};
PluginFieldCollection GemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GemmPluginCreator::mPluginAttributes;

void getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda, int& ldb, int& ldc,
                      bool transA, bool transB, int M, int N, int K, int padLda, int padLdb, int padLdc)
{
    transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    m = N;
    n = M;
    k = K;
    lda = transB ? K + padLdb : N + padLdb;
    ldb = transA ? M + padLda : K + padLda;
    ldc = N + padLdc;
}

void runGemm(int const M, int const N, int const K, bool const transA, bool const transB, int const padLda,
    int const padLdb, int const padLdc, nvinfer1::DataType const type, CublasGemmWrapperPtr const& cublasWrapperPtr,
    void const* act, void const* weight, float const alpha, void* output,
    std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic, void* workspace, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0)
        return;

    cublasWrapperPtr->setStream(stream);
    cublasWrapperPtr->setWorkspace(workspace);

    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K, padLda, padLdb, padLdc);

    cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, alpha, 0.0f, heuristic);
    cublasWrapperPtr->destroyDescriptors();
}

void CublasLtGemmPluginProfiler::runTactic(
    int m, int n, int k, CublasLtGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    size_t dataSize = sizeof(half);
    if (mType == nvinfer1::DataType::kFLOAT)
    {
        dataSize = sizeof(float);
    }

    void* actPtr = reinterpret_cast<void*>(workspace);
    void* weightPtr = reinterpret_cast<void*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(actPtr), m * k * dataSize, ALIGNMENT));
    void* outputPtr = reinterpret_cast<void*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(weightPtr), n * k * dataSize, ALIGNMENT));
    char* workspacePtr = reinterpret_cast<char*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(outputPtr), m * (n + mPadLdc) * dataSize, ALIGNMENT));
    runGemm(m, n, k, mTransA, mTransB, mPadLda, mPadLdb, mPadLdc, mType, mRunner, actPtr, weightPtr, 1.0f, outputPtr,
        {tactic}, workspacePtr, stream);
}

bool CublasLtGemmPluginProfiler::checkTactic(int m, int n, int k, Config const& tactic) const
{
    cublasOperation_t transa, transb;
    int M = m, N = n, K = k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K, mPadLda, mPadLdb, mPadLdc);

    mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

    auto const checkResult = mRunner->checkTactic(transa, transb, m, n, k, lda, ldb, ldc, tactic.algo);

    mRunner->destroyDescriptors();

    return checkResult;
}

void CublasLtGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    size_t dataSize = getDTypeSize(mType);
    size_t outputDataSize = getDTypeSize(mOutputType);

    std::vector<size_t> workspaces = {
        maxM * k * dataSize,                   // A
        n * k * dataSize,                      // B
        maxM * (n + mPadLdc) * outputDataSize, // C
        CUBLAS_WORKSPACE_SIZE                  // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size(), ALIGNMENT);
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<CublasLtGemmPluginProfiler::Config> CublasLtGemmPluginProfiler::getTactics(int M, int N, int K) const
{
    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K, mPadLda, mPadLdb, mPadLdc);

    mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    auto const heruistics = mRunner->getTactics(transa, transb, m, n, k, lda, ldb, ldc);
    mRunner->destroyDescriptors();

    return heruistics;
}

GemmPlugin::GemmPlugin(int transA, int transB, int padLda, int padLdb, int padLdc, nvinfer1::DataType type, bool useFp8,
    float alpha, GemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mTransA(transA)
    , mTransB(transB)
    , mPadLda(padLda)
    , mPadLdb(padLdb)
    , mPadLdc(padLdc)
    , mType(type)
    , mOutputType(type)
    , mUseFp8(useFp8)
    , mAlpha(alpha)
    , mPluginProfiler(pluginProfiler)
{
    init();
}

// Parameterized constructor
GemmPlugin::GemmPlugin(void const* data, size_t length, GemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mTransA);
    read(d, mTransB);
    read(d, mPadLda);
    read(d, mPadLdb);
    read(d, mPadLdc);
    read(d, mType);
    read(d, mUseFp8);
    read(d, mAlpha);
    read(d, mDims);
    read(d, mOutputType);

    init();

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

thread_local CublasGemmWrapperPtr GemmPlugin::mCublasWrapper = nullptr;

void GemmPlugin::init()
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

    mPluginProfiler->setTranspose(mTransA, mTransB);
    mPluginProfiler->setOutputType(mOutputType);
    mPluginProfiler->setPadLd(mPadLda, mPadLdb, mPadLdc);

    mGemmId = GemmIdCublas(mDims.n, mDims.k, mType, mTransA, mTransB, mOutputType);
}

void GemmPlugin::setGemmConfig()
{
    if (mType == nvinfer1::DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig(trtToCublasDtype(mOutputType));
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
        mCublasWrapper->setBF16GemmConfig(trtToCublasDtype(mOutputType));
    }
#endif

#ifdef ENABLE_FP8
    if (mUseFp8)
    {
        mCublasWrapper->setFP8GemmConfig(trtToCublasDtype(mOutputType));
    }
#endif
}

void GemmPlugin::configGemm()
{
    if (!mDims.isInitialized())
    {
        return;
    }

    setGemmConfig();

    mPluginProfiler->profileTactics(mCublasWrapper, mType, mDims, mGemmId);
}


nvinfer1::DimsExprs GemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[1].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsA + nbDimsB - 2;

        if (mTransA)
        {
            for (int i = 1; i < nbDimsA; ++i)
            {
                ret.d[i - 1] = inputs[0].d[i];
            }
        }
        else
        {
            for (int i = 0; i < nbDimsA - 1; ++i)
            {
                ret.d[i] = inputs[0].d[i];
            }
        }
        if (mTransB)
        {
            for (int i = 0; i < nbDimsB - 1; ++i)
            {
                ret.d[nbDimsA - 1 + i] = exprBuilder.constant(inputs[1].d[i]->getConstantValue() + mPadLdc);
            }
        }
        else
        {
            for (int i = 1; i < nbDimsB; ++i)
            {
                ret.d[nbDimsA - 2 + i] = exprBuilder.constant(inputs[1].d[i]->getConstantValue() + mPadLdc);
            }
        }
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    auto const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (pos < nbInputs)
    {
        // If use FP8, act/weight dtype should be kFP8
        if (mUseFp8)
        {
            return desc.type == nvinfer1::DataType::kFP8;
        }
        else
        {
            return desc.type == mType;
        }
    }

    return desc.type == mType || desc.type == nvinfer1::DataType::kFLOAT;
}

void GemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const nbDimsA = in[0].max.nbDims;

    auto const minM = utils::computeMDimension(mTransA, in[0].min);
    auto const maxM = utils::computeMDimension(mTransA, in[0].max);
    auto const N = utils::computeNDimension(mTransB, in[1].max);
    auto const K = static_cast<utils::DimType64>(mTransA ? in[0].max.d[0] : in[0].max.d[nbDimsA - 1]);

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId.n = N;
    mGemmId.k = K;

    mOutputType = out[0].desc.type;
}

size_t GemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return CUBLAS_WORKSPACE_SIZE;
}

int GemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     mat1 [M, K] (mTransA = False)
    //     mat2 [K, N] (mTransB = False)
    // outputs
    //     mat [M, N]
    if (mCublasWrapper == nullptr)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        mCublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }
    setGemmConfig();

    int const nbDimsA = inputDesc[0].dims.nbDims;
    int const padM = mTransA ? mPadLda : 0;
    int const padN = mTransB ? 0 : mPadLdb;
    int const padK = mTransA ? 0 : mPadLda;
    auto const M = utils::computeMDimension(mTransA, inputDesc[0].dims) - padM;
    auto const N = utils::computeNDimension(mTransB, inputDesc[1].dims) - padN;
    int const K = static_cast<utils::DimType64>(
        mTransA ? inputDesc[0].dims.d[0] - padK : inputDesc[0].dims.d[nbDimsA - 1] - padK);

    bool noPadDim = padM == 0 && padN == 0 && padK == 0 && mPadLdc == 0;
    bool cudaKernelSupportType = mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kFLOAT
        || mType == nvinfer1::DataType::kBF16;

    // skip computation for a TRT empty tensor
    if (M == 0)
    {
        return 0;
    }

    std::string mnkStr = "MNK={" + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + "}";
    {
        std::string const activationStr = "GEMM layer's activation before GEMM with " + mnkStr;
        TLLM_CHECK_DEBUG_WITH_INFO(
            tensorrt_llm::runtime::utils::tensorHasInvalid(M, K, mType, inputs[0], stream, activationStr) == false,
            "Found invalid number (NaN or Inf) in " + activationStr);
    }

    bool cudaKernelFinished = false;
    // TODO: sub tensor matmul is not supported in fp8 gemm cuda kernel
    if (M <= 4 && N <= 128000 && mUseFp8 && noPadDim && cudaKernelSupportType)
    {
        tensorrt_llm::common::QuantMode quantMode = tensorrt_llm::common::QuantMode::fromQuantAlgo("FP8");
        tensorrt_llm::kernels::cuda_core_gemm::Params params(reinterpret_cast<void const*>(inputs[0]),
            reinterpret_cast<void const*>(inputs[1]), mAlpha, reinterpret_cast<void*>(outputs[0]), M, N, K, quantMode,
            nvinfer1::DataType::kFP8, mOutputType);
        cudaKernelFinished = tensorrt_llm::kernels::cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
    }
    else if (M <= 6 && N <= 128000 && !mUseFp8 && noPadDim && cudaKernelSupportType)
    {
        tensorrt_llm::common::QuantMode quantMode;
        tensorrt_llm::kernels::cuda_core_gemm::Params params(reinterpret_cast<void const*>(inputs[0]),
            reinterpret_cast<void const*>(inputs[1]), mAlpha, reinterpret_cast<void*>(outputs[0]), M, N, K, quantMode,
            mType, mOutputType);
        cudaKernelFinished = tensorrt_llm::kernels::cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
    }

    if (!cudaKernelFinished)
    {
        auto bestTactic = mPluginProfiler->getBestConfig(M, mGemmId);
        runGemm(M, N, K, mTransA, mTransB, mPadLda, mPadLdb, mPadLdc, mType, mCublasWrapper, inputs[0], inputs[1],
            mAlpha, outputs[0], bestTactic, workspace, stream);
    }

    {
        std::string const outputStr = "GEMM layer's output after GEMM with " + mnkStr;
        TLLM_CHECK_DEBUG_WITH_INFO(
            tensorrt_llm::runtime::utils::tensorHasInvalid(M, N + mPadLdc, mType, outputs[0], stream, outputStr)
                == false,
            "Found invalid number (NaN or Inf) in " + outputStr);
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* GemmPlugin::getPluginType() const noexcept
{
    return GEMM_PLUGIN_NAME;
}

char const* GemmPlugin::getPluginVersion() const noexcept
{
    return GEMM_PLUGIN_VERSION;
}

int GemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void GemmPlugin::destroy() noexcept
{
    delete this;
}

size_t GemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mTransA) + sizeof(mTransB) + sizeof(mPadLda) + sizeof(mPadLdb) + sizeof(mPadLdc) + sizeof(mType)
        + sizeof(mDims) + sizeof(mUseFp8) + sizeof(mAlpha) + mPluginProfiler->getSerializationSize(mGemmId)
        + sizeof(mOutputType); // selected tactics container size
}

void GemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mTransA);
    write(d, mTransB);
    write(d, mPadLda);
    write(d, mPadLdb);
    write(d, mPadLdc);
    write(d, mType);
    write(d, mUseFp8);
    write(d, mAlpha);
    write(d, mDims);
    write(d, mOutputType);
    mPluginProfiler->serialize(d, mGemmId);

    TLLM_CHECK(d == a + getSerializationSize());
}

void GemmPlugin::terminate() noexcept {}

///////////////

GemmPluginCreator::GemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("transA", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("transB", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("padLda", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("padLdb", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("padLdc", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_fp8", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GemmPluginCreator::getPluginName() const noexcept
{
    return GEMM_PLUGIN_NAME;
}

char const* GemmPluginCreator::getPluginVersion() const noexcept
{
    return GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* GemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int transA{};
    int transB{};
    int padLda{};
    int padLdb{};
    int padLdc{};
    nvinfer1::DataType type{};
    int useFp8{};
    float alpha = 1.F;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "transa"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transA = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "transb"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transB = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "pad_lda"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            padLda = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "pad_ldb"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            padLdb = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "pad_ldc"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            padLdc = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_fp8"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            useFp8 = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "alpha"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            alpha = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
    }
    try
    {
        // GemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false, /* skip */ true);
        auto* obj = new GemmPlugin(transA, transB, padLda, padLdb, padLdc, type, useFp8, alpha, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GemmPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GemmPlugin::destroy()
    try
    {
        // GemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true, /* skip */ true);
        auto* obj = new GemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}


using namespace onnxruntime::common;
namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BiasGelu, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    BiasGelu);

template <typename T>
void BiasGelu::KernelLaunchDispatcher<T>::operator()(cudaStream_t stream, int64_t input_size, int64_t bias_size,
                                                     const Tensor& X, const Tensor& B, Tensor& Y) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  LaunchBiasGeluKernel<CudaT>(stream, input_size, bias_size, reinterpret_cast<const CudaT*>(X.template Data<T>()),
                              reinterpret_cast<const CudaT*>(B.template Data<T>()),
                              reinterpret_cast<CudaT*>(Y.template MutableData<T>()));
}

Status BiasGelu::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X);
  const auto* B = context->Input<Tensor>(1);
  ORT_ENFORCE(B);

  const auto& input_shape = X->Shape();
  const auto& bias_shape = B->Shape();
  ORT_ENFORCE(input_shape.NumDimensions() >= 1 && bias_shape.NumDimensions() == 1 &&
                  input_shape.GetDims().back() == bias_shape.GetDims().back(),
              "B must be 1-dimensional and match the last dimension of X.");

  auto* Y = context->Output(0, input_shape);
  ORT_ENFORCE(Y);

  const auto input_size = input_shape.Size();
  const auto bias_size = bias_shape.Size();
  utils::MLTypeCallDispatcher<MLFloat16, float, double, BFloat16> dispatcher{X->GetElementType()};
  dispatcher.Invoke<KernelLaunchDispatcher>(Stream(context), input_size, bias_size, *X, *B, *Y);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
