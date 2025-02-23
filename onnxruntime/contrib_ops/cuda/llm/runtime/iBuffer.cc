/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/runtime/iBuffer.h"
#include "contrib_ops/cuda/llm/runtime/iTensor.h"
#include "contrib_ops/cuda/llm/runtime/tllmBuffers.h"

#include "contrib_ops/cuda/llm/common/assert.h"
#include "contrib_ops/cuda/llm/common/cudaUtils.h"
#include "contrib_ops/cuda/llm/runtime/bufferView.h"

#include <cuda_runtime_api.h>

#include <memory>

using namespace onnxruntime::llm::runtime;

MemoryType IBuffer::memoryType(void const* data)
{
    cudaPointerAttributes attributes{};
    TLLM_CUDA_CHECK(::cudaPointerGetAttributes(&attributes, data));
    switch (attributes.type)
    {
    case cudaMemoryTypeHost: return MemoryType::kPINNEDPOOL;
    case cudaMemoryTypeDevice: return MemoryType::kGPU;
    case cudaMemoryTypeManaged: return MemoryType::kUVM;
    case cudaMemoryTypeUnregistered: return MemoryType::kCPU;
    }

    TLLM_THROW("Unsupported memory type");
}

IBuffer::UniquePtr IBuffer::slice(IBuffer::SharedPtr buffer, std::size_t offset, std::size_t size)
{
    return std::make_unique<BufferView>(std::move(buffer), offset, size);
}

IBuffer::UniquePtr IBuffer::wrap(void* data, nvinfer1::DataType type, std::size_t size, std::size_t capacity)
{
    TLLM_CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    IBuffer::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericBuffer<PinnedBorrowingAllocator>( // NOLINT(modernize-make-unique)
            capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kPINNEDPOOL:
        result.reset(new GenericBuffer<PinnedPoolBorrowingAllocator>( // NOLINT(modernize-make-unique)
            capacity, type, PinnedPoolBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericBuffer<CpuBorrowingAllocator>(capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericBuffer<GpuBorrowingAllocator>(capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    default: TLLM_THROW("Unknown memory type");
    }
    result->resize(size);
    return result;
}

std::ostream& onnxruntime::llm::runtime::operator<<(std::ostream& output, IBuffer const& buffer)
{
    auto data = const_cast<IBuffer&>(buffer).data();
    auto tensor = ITensor::wrap(data, buffer.getDataType(),
        ITensor::makeShape({static_cast<SizeType32>(buffer.getSize())}), buffer.getCapacity());
    return output << *tensor;
}

char const* IBuffer::getDataTypeName(DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kINT64: return DataTypeTraits<nvinfer1::DataType::kINT64>::name;
    case nvinfer1::DataType::kINT32: return DataTypeTraits<nvinfer1::DataType::kINT32>::name;
    case nvinfer1::DataType::kFLOAT: return DataTypeTraits<nvinfer1::DataType::kFLOAT>::name;
#ifdef ENABLE_BF16    
    case nvinfer1::DataType::kBF16: return DataTypeTraits<nvinfer1::DataType::kBF16>::name;
#endif    
    case nvinfer1::DataType::kHALF: return DataTypeTraits<nvinfer1::DataType::kHALF>::name;
    case nvinfer1::DataType::kBOOL: return DataTypeTraits<nvinfer1::DataType::kBOOL>::name;
    case nvinfer1::DataType::kUINT8: return DataTypeTraits<nvinfer1::DataType::kUINT8>::name;
    case nvinfer1::DataType::kINT8: return DataTypeTraits<nvinfer1::DataType::kINT8>::name;
#ifdef ENABLE_BF8    
    case nvinfer1::DataType::kFP8: return DataTypeTraits<nvinfer1::DataType::kFP8>::name;
#endif
    case nvinfer1::DataType::kINT4: [[fallthrough]] /* do nothing */;
    case nvinfer1::DataType::kFP4: /* do nothing */;
    }
    TLLM_THROW("Unknown data type");
}

char const* IBuffer::getDataTypeName() const
{
    return getDataTypeName(getDataType());
}

char const* IBuffer::getMemoryTypeName() const
{
    switch (getMemoryType())
    {
    case MemoryType::kPINNED: return MemoryTypeString<MemoryType::kPINNED>::value;
    case MemoryType::kPINNEDPOOL: return MemoryTypeString<MemoryType::kPINNEDPOOL>::value;
    case MemoryType::kCPU: return MemoryTypeString<MemoryType::kCPU>::value;
    case MemoryType::kGPU: return MemoryTypeString<MemoryType::kGPU>::value;
    case MemoryType::kUVM: return MemoryTypeString<MemoryType::kUVM>::value;
    }
    TLLM_THROW("Unknown memory type");
}
