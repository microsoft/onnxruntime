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

#include "contrib_ops/cuda/llm/runtime/tllmBuffers.h"

namespace onnxruntime::llm::runtime
{
template <typename TAllocator>
typename PoolAllocator<TAllocator>::PoolType& PoolAllocator<TAllocator>::getPool()
{
    static PoolType pool;
    return pool;
}

// IpcNvlsTensorView::IpcNvlsTensorView(std::weak_ptr<IpcNvlsTensor> const& tensor, bool unicastView)
//     : mTensor(tensor)
//     , mUnicastView(unicastView)
//     , mDims(mTensor.lock()->getShape())
// {
// }

// IpcNvlsTensorView::IpcNvlsTensorView(IpcNvlsTensorView&& other) noexcept
//     : mTensor(std::move(other.mTensor))
//     , mUnicastView(other.mUnicastView)
//     , mDims(mTensor.lock()->getShape())
// {
// }

// IpcNvlsTensorView& IpcNvlsTensorView::operator=(IpcNvlsTensorView&& other) noexcept
// {
//     if (this != &other)
//     {
//         // Reset tensor.
//         mTensor.reset();
//         mTensor.swap(other.mTensor);
//         mUnicastView = other.mUnicastView;
//         mDims = mTensor.lock()->getShape();
//     }
//     return *this;
// }

// std::shared_ptr<IpcNvlsBuffer> IpcNvlsTensorView::lock() const
// {
//     auto sp = mTensor.lock();
//     TLLM_CHECK(sp != nullptr);
//     return sp;
// }

// ///////////////////////////////////////
// // IpcNvlsTensorView ITensor methods
// ///////////////////////////////////////
// nvinfer1::Dims const& IpcNvlsTensorView::getShape() const
// {
//     return mDims;
// }

// void IpcNvlsTensorView::reshape(nvinfer1::Dims const& dims)
// {
//     auto new_size = nonNegative(volume(dims));
//     if (new_size > getCapacity())
//     {
//         TLLM_THROW("IpcNvlsTensorView::reshape() cannot be larger than origin tensor.");
//     }
//     mDims = dims;
// }

// ///////////////////////////////////////
// // IpcNvlsTensorView IBuffer methods
// ///////////////////////////////////////
// void* IpcNvlsTensorView::_data() const
// {
//     if (mUnicastView)
//     {
//         return lock()->data();
//     }
//     else
//     {
//         return lock()->dataMC();
//     }
// }

// std::size_t IpcNvlsTensorView::getSize() const
// {
//     return lock()->getSize();
// }

// std::size_t IpcNvlsTensorView::getCapacity() const
// {
//     return lock()->getCapacity();
// }

// nvinfer1::DataType IpcNvlsTensorView::getDataType() const
// {
//     return lock()->getDataType();
// }

// MemoryType IpcNvlsTensorView::getMemoryType() const
// {
//     return lock()->getMemoryType();
// }

// explicit instantiations
template class PoolAllocator<PinnedAllocator>;
} // namespace onnxruntime::llm::runtime
