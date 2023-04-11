// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <optional>
#include <array>

#include "core/framework/ortmemoryinfo.h"
#include "core/framework/tensor_shape.h"
#include "core/common/common.h"
#include "core/framework/shard_utils.h"

namespace onnxruntime {

using DeleterFnPtr = std::function<void(void*)>;
using ShapeType = TensorShape;
struct Buffer {
    Buffer() = default;
    Buffer(std::size_t sizeInBytes, MemoryLocation location, std::size_t offset, void *ptr, DeleterFnPtr deleter)
        : sizeInBytes_(sizeInBytes),
          memoryLocation_(location),
          byte_offset_(offset),
          p_data_(ptr),
          deleter_(deleter) {}

    ~Buffer() {
        if (deleter_)
            deleter_(p_data_);
    }
    void* Data() const { return p_data_; }
    MemoryLocation Location() const { return memoryLocation_; }
    size_t Offset() const { return byte_offset_; }
    void SetOffset(ptrdiff_t byte_offset) { byte_offset_ = byte_offset; }
    size_t Size() const { return sizeInBytes_; }
    bool OwnBuffer() const { return deleter_ != nullptr; }
protected:
    size_t sizeInBytes_{0};
    MemoryLocation memoryLocation_{MemoryLocation::Uniform};
    size_t byte_offset_{0};
    void *p_data_{nullptr};

    /**
        if deleter_ is null, it means tensor does not own the buffer.
        otherwise tensor will use the deleter to release the buffer when
        tensor is released.
    */
    DeleterFnPtr deleter_;
};

struct ShardAccessor {
    ShardAccessor(Buffer& buffer, ShardDim& shardDims, std::uint32_t index) : buffer_(buffer), shardDims_(shardDims), index_(index) {}

    void* Data() const { return buffer_.Data(); }
    MemoryLocation Location() const { return buffer_.Location(); }
    size_t Offset() const { return buffer_.Offset(); }
    void SetOffset(ptrdiff_t byte_offset) { buffer_.SetOffset(byte_offset); }
    size_t Size() const { return buffer_.Size(); }
    bool OwnBuffer() const { return buffer_.OwnBuffer(); }
    ShardDim const& GetShardDim() const { return shardDims_; }
    std::uint32_t Index() const { return index_; }

private:
    Buffer& buffer_;
    ShardDim& shardDims_;
    std::uint32_t index_;
};

struct Storage {
    class ShardRange
    {
        public:
        struct Iterator
        {
            friend class ShardRange;

            using difference_type   = std::ptrdiff_t;
            using value_type        = ShardAccessor;
            using pointer           = value_type*;
            using reference         = value_type&;
            using iterator_category = std::forward_iterator_tag;

            Iterator(Storage& storage, ShapeType& shape, std::uint32_t pos)
                : storage_(storage), shape_(shape), pos_(pos) {}

            value_type operator*() const { return storage_.Shard(pos_); }
            Iterator&  operator++() { pos_++; return *this; }
            Iterator  operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

            bool       operator==(Iterator const& rhs) const { return IsEqual_(rhs); }
            bool       operator!=(Iterator const& rhs) const { return !IsEqual_(rhs); }
            private:
            constexpr bool IsEqual_(Iterator const& rhs) const
            { return (&storage_ == &rhs.storage_) && (shape_ == rhs.shape_) && (pos_ == rhs.pos_); }

            Storage& storage_;
            ShapeType& shape_;
            std::uint32_t pos_;

        };

        ShardRange(Iterator pBegin, Iterator pEnd) : begin_(pBegin), end_(pEnd)
        {
        }

        explicit ShardRange(Storage& storage, ShapeType& shape)
            : begin_(storage, shape, 0), end_(storage, shape, ShardUtils::NumShards(storage.ShardDims().value())) {}

        Iterator begin() const { return begin_; }
        Iterator end() const { return end_; }

        private:
        Iterator begin_, end_;
    };  // ShardRange ends

    ORT_DISALLOW_COPY_AND_ASSIGNMENT(Storage);
    Storage(std::vector<Buffer>&& buffers, std::optional<ShardDim> shardDims)
        : buffers_(std::move(buffers)), shardDims_(shardDims.value_or(ShardDim()))
    {
    }
    Storage(Buffer&& buffer) : buffers_(CreateBuffer(std::move(buffer)))
    {
    }
    std::vector<Buffer> CreateBuffer(Buffer&& buffer)
    {
        std::vector<Buffer> buffers;
        buffers.emplace_back(std::move(buffer));
        return buffers;
    }
    void* Data(uint32_t index=0) const  { return buffers_.at(index).Data(); }
    MemoryLocation Location(uint32_t index=0) const  { return buffers_.at(index).Location(); }
    size_t Offset(uint32_t index=0) const  { return buffers_.at(index).Offset(); }
    void SetOffset(ptrdiff_t byte_offset, uint32_t index=0) { buffers_.at(index).SetOffset(byte_offset); }
    size_t Size(uint32_t index=0) const  { return buffers_.at(index).Size(); }
    ShardAccessor Shard(uint32_t index=0) { return ShardAccessor{ buffers_.at(index), shardDims_, index }; }
    void Apply(std::function<void(Buffer&)> fn) { std::for_each(buffers_.begin(), buffers_.end(), fn); }
    std::optional<ShardDim> ShardDims() const { return std::make_optional(shardDims_); }
    bool OwnsBuffer() const { return buffers_[0].OwnBuffer(); }
    ShardRange Shards(ShapeType& shape) { return ShardRange{*this, shape}; }
protected:
    std::vector<Buffer> buffers_;
    ShardDim shardDims_;
};

} // namespace onnxruntime ends
