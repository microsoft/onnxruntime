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

#include "contrib_ops/cuda/llm/runtime/iTensor.h"

#include "contrib_ops/cuda/llm/common/memoryUtils.h"
#include "contrib_ops/cuda/llm/common/stringUtils.h"
#include "contrib_ops/cuda/llm/runtime/bufferManager.h"
#include "contrib_ops/cuda/llm/runtime/tensorView.h"
#include "contrib_ops/cuda/llm/runtime/tllmBuffers.h"

#include <initializer_list>
#include <memory>

using namespace onnxruntime::llm::runtime;

namespace tc = onnxruntime::llm::common;

ITensor::UniquePtr ITensor::slice(SharedPtr tensor, std::size_t offset, std::size_t size)
{
    TLLM_CHECK(tensor);
    return std::make_unique<TensorView>(std::move(tensor), offset, size);
}

ITensor::UniquePtr ITensor::slice(SharedPtr tensor, ITensor::Shape const& offsetDims, ITensor::DimType64 size)
{
    auto shape = tensor->getShape();
    TLLM_CHECK(offsetDims.nbDims >= 0);
    TLLM_CHECK(shape.nbDims >= offsetDims.nbDims);
    TLLM_CHECK(size >= 0);

    ITensor::Shape strides = ITensor::strides(shape);
    DimType64 offset{0};
    for (SizeType32 di = 0; di < offsetDims.nbDims - 1; di++)
    {
        TLLM_CHECK(0 <= offsetDims.d[di] && offsetDims.d[di] < shape.d[di]);
        offset += offsetDims.d[di] * strides.d[di];
    }

    if (TLLM_LIKELY(offsetDims.nbDims > 0))
    {
        TLLM_CHECK(offsetDims.d[offsetDims.nbDims - 1] + size <= shape.d[offsetDims.nbDims - 1]);
        offset += offsetDims.d[offsetDims.nbDims - 1] * strides.d[offsetDims.nbDims - 1];
    }
    else
    {
        TLLM_CHECK(size >= 0 && size <= 1);
        TLLM_CHECK(shape.nbDims == 0 ? size == 0 : true);
    }

    ITensor::Shape dims;
    dims.nbDims = shape.nbDims - offsetDims.nbDims + 1;
    dims.d[0] = size;
    for (SizeType32 di = 1; di < dims.nbDims; di++)
    {
        dims.d[di] = shape.d[di - 1 + offsetDims.nbDims];
    }

    return std::make_unique<TensorView>(std::move(tensor), offset, volume(dims), dims);
}

ITensor::UniquePtr ITensor::view(IBuffer::SharedPtr buffer, ITensor::Shape const& dims)
{
    auto const size = buffer->getSize();
    return std::make_unique<TensorView>(std::move(buffer), 0, size, dims);
}

ITensor::Shape ITensor::makeShape(std::initializer_list<int64_t> const& dims)
{
    TLLM_CHECK_WITH_INFO(dims.size() <= ITensor::Shape::MAX_DIMS, "Number of dimensions is too large");
    ITensor::Shape shape{};
    shape.nbDims = static_cast<decltype(ITensor::Shape::nbDims)>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
    return shape;
}

std::string ITensor::toString(ITensor::Shape const& dims)
{
    if (dims.nbDims < 0)
    {
        return "invalid";
    }
    else if (dims.nbDims == 0)
    {
        return "()";
    }
    else
    {
        return onnxruntime::llm::common::arr2str(dims.d, dims.nbDims);
    }
}

ITensor::UniquePtr ITensor::wrap(void* data, nvinfer1::DataType type, ITensor::Shape const& shape, std::size_t capacity)
{
    auto const size = volumeNonNegative(shape);
    TLLM_CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    ITensor::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSizeInBits() / 8;
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericTensor<PinnedBorrowingAllocator>( // NOLINT(modernize-make-unique)
            shape, capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kPINNEDPOOL:
        result.reset(new GenericTensor<PinnedPoolBorrowingAllocator>( // NOLINT(modernize-make-unique)
            shape, capacity, type, PinnedPoolBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<CpuBorrowingAllocator>(
                shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<GpuBorrowingAllocator>(
                shape, capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kUVM:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<ManagedBorrowingAllocator>(
                shape, capacity, type, ManagedBorrowingAllocator(data, capacityInBytes)));
        break;
    default: TLLM_THROW("Invalid memory type."); break;
    }
    return result;
}

ITensor::Shape ITensor::squeeze(ITensor::Shape const& shape, SizeType32 dim)
{
    TLLM_CHECK_WITH_INFO(0 < shape.nbDims, "Cannot squeeze 1-dimensional tensor");
    TLLM_CHECK_WITH_INFO(
        dim < shape.nbDims, tc::fmtstr("Invalid index %d, tensor has %d dimensions", dim, shape.nbDims));
    TLLM_CHECK_WITH_INFO(shape.d[dim] == 1, "Can only squeeze dimension of size 1");

    ITensor::Shape newDims;
    newDims.nbDims = shape.nbDims - 1;
    std::copy(shape.d, shape.d + dim, newDims.d);
    std::copy(shape.d + dim + 1, shape.d + shape.nbDims, newDims.d + dim);
    return newDims;
}

ITensor::Shape ITensor::unsqueeze(ITensor::Shape const& shape, SizeType32 dim)
{
    TLLM_CHECK_WITH_INFO(shape.nbDims < Shape::MAX_DIMS, "Too many dimensions to unsqueeze");
    TLLM_CHECK_WITH_INFO(
        0 <= dim && dim <= shape.nbDims, tc::fmtstr("Invalid dim %d, tensor has %d dimensions", dim, shape.nbDims));

    ITensor::Shape newDims;
    newDims.nbDims = shape.nbDims + 1;
    std::copy(shape.d, shape.d + dim, newDims.d);
    newDims.d[dim] = 1;
    std::copy(shape.d + dim, shape.d + shape.nbDims, newDims.d + dim + 1);
    return newDims;
}

namespace
{
template <typename T>
void printTensor(ITensor const& tensor, std::ostream& out)
{
    TLLM_CHECK_WITH_INFO(tensor.getDataType() == TRTDataType<typename std::remove_cv<T>::type>::value,
        tc::fmtstr("Data type mismatch: %d vs %d", static_cast<std::int32_t>(tensor.getDataType()),
            static_cast<std::int32_t>(TRTDataType<typename std::remove_cv<T>::type>::value)));
    auto const& shape = tensor.getShape();
    out << "shape: " << shape << std::endl;
    out << "vals: " << std::endl;

    BufferManager::ITensorPtr host{};
    T const* hostData;
    if (tensor.getMemoryType() == MemoryType::kGPU)
    {
        auto streamPtr = std::make_shared<CudaStream>();
        BufferManager manager{streamPtr};
        host = manager.copyFrom(tensor, MemoryType::kCPU);
        streamPtr->synchronize();
        hostData = bufferCast<T>(*host);
    }
    else
    {
        hostData = bufferCast<T>(tensor);
    }

    using TOutput
        = std::conditional_t<std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>, std::int32_t, T>;
    if (shape.nbDims > 3)
    {
        out << "Not printing elements for more than 3 dims\n";
    }
    else if (shape.nbDims == 3 && shape.d[2] > 1)
    {
        for (int i = 0; i < shape.d[0]; ++i)
        {
            for (int j = 0; j < shape.d[1]; ++j)
            {
                out << "i=" << i << " j=" << j << ": ";
                tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, j, 0), shape.d[2]) << "\n";
            }
        }
    }
    else if (shape.nbDims >= 2 && shape.d[1] > 1)
    {
        for (int i = 0; i < shape.d[0]; ++i)
        {
            out << "i=" << i << ": ";
            tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, 0), shape.d[1]) << "\n";
        }
    }
    else
    {
        tc::arr2outCasted<TOutput>(out, hostData, shape.d[0]) << "\n";
    }
    out << std::flush;
}

} // namespace

std::ostream& onnxruntime::llm::runtime::operator<<(std::ostream& out, ITensor const& tensor)
{
    switch (tensor.getDataType())
    {
    case nvinfer1::DataType::kFLOAT: printTensor<float>(tensor, out); break;
    case nvinfer1::DataType::kHALF: printTensor<half>(tensor, out); break;
    case nvinfer1::DataType::kBOOL: printTensor<bool>(tensor, out); break;
    case nvinfer1::DataType::kINT8: printTensor<std::int8_t>(tensor, out); break;
    case nvinfer1::DataType::kINT32: printTensor<std::int32_t>(tensor, out); break;
    case nvinfer1::DataType::kINT64: printTensor<std::int64_t>(tensor, out); break;
    case nvinfer1::DataType::kUINT8: printTensor<std::uint8_t>(tensor, out); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: printTensor<__nv_bfloat16>(tensor, out); break;
#endif
    default: TLLM_THROW("Unsupported data type");
    }

    return out;
}
