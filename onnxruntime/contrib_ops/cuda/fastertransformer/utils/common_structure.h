/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

template<typename T>
struct DenseWeight{
    const T* kernel = nullptr;
    const T* bias = nullptr;
};

template<typename T>
struct LayerNormWeight{
    const T* gamma = nullptr;
    const T* beta = nullptr;
};

template<typename T>
struct AttentionWeight{
    DenseWeight<T> query_weight;
    DenseWeight<T> key_weight;
    DenseWeight<T> value_weight;
    DenseWeight<T> attention_output_weight;
};

template<typename T>
struct FFNWeight{
    DenseWeight<T> intermediate_weight;
    DenseWeight<T> output_weight;
};

namespace fastertransformer{

enum class ActivationType{RELU, GELU};

template<OperationType OpType_>
class TransformerTraits;

template<>
class TransformerTraits<OperationType::FP32>
{
  public:
    typedef float DataType;
    static const OperationType OpType = OperationType::FP32;
    static cudaDataType_t const computeType = CUDA_R_32F;
    static cudaDataType_t const scaleType = CUDA_R_32F;
    static cudaDataType_t const AType = CUDA_R_32F;
    static cudaDataType_t const BType = CUDA_R_32F;
    static cudaDataType_t const CType = CUDA_R_32F;
};

template<>
class TransformerTraits<OperationType::FP16>
{
  public:
    typedef half DataType;
    static const OperationType OpType = OperationType::FP16;
    static cudaDataType_t const computeType = CUDA_R_16F;
    static cudaDataType_t const scaleType = CUDA_R_16F;
    static cudaDataType_t const AType = CUDA_R_16F;
    static cudaDataType_t const BType = CUDA_R_16F;
    static cudaDataType_t const CType = CUDA_R_16F;
};

} // end of fastertransformer 