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

#pragma once

#include "contrib_ops/cuda/llm/runtime/bufferView.h"
#include "contrib_ops/cuda/llm/runtime/iTensor.h"

#include <stdexcept>

namespace onnxruntime::llm::runtime
{
class TensorView : virtual public ITensor, public BufferView
{
public:
    using Base = BufferView;

    TensorView(ITensor::SharedPtr const& buffer, size_t offset, size_t size)
        : BufferView{buffer, offset * sizeDim0(*buffer), size * sizeDim0(*buffer)}
        , mDims{buffer->getShape()}
    {
        auto const dim0 = static_cast<size_t>((mDims.nbDims >= 0 && mDims.d[0] >= 0) ? mDims.d[0] : 0);
        if (offset > dim0)
        {
            throw std::out_of_range("offset exceeds dimension 0");
        }

        if (offset + size > dim0)
        {
            throw std::out_of_range("slice exceeds dimension 0");
        }
        mDims.d[0] = size;
    }

    TensorView(IBuffer::SharedPtr const& buffer, size_t offset, size_t size, nvinfer1::Dims const& dims)
        : BufferView{buffer, offset, size}
        , mDims{dims}
    {
        Base::resize(ITensor::volumeNonNegative(dims));
    }

    [[nodiscard]] nvinfer1::Dims const& getShape() const override
    {
        return mDims;
    }

    void reshape(nvinfer1::Dims const& dims) override
    {
        Base::resize(ITensor::volumeNonNegative(dims));
        mDims = dims;
    }

    void resize(std::size_t newSize) override
    {
        ITensor::resize(newSize);
    }

    void release() override
    {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    static std::size_t sizeDim0(ITensor const& tensor)
    {
        auto& shape = tensor.getShape();
        return shape.nbDims > 0 && shape.d[0] > 0 ? ITensor::volume(shape) / shape.d[0] : 0;
    }

    nvinfer1::Dims mDims{};
};
} // namespace onnxruntime::llm::runtime
