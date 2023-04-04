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

using DeleterFnPtr = void (*)(void*);
//using ShardDim = std::array<std::int32_t, 5>;

//typedef int32_t ShardDim[16];
//using MemoryLocations = std::array<MemoryLocation, 16>;
//using ShardDim = TensorShapeVector;
// ShardUtils::Index(offset, shardDims_)
// struct ShardDim : TensorShapeVector
// {
//     using TensorShapeVector::TensorShapeVector;
//     constexpr operator uint64_t() const {
//         return ShardUtils::Index(*this, *this);
//     }
// };

struct Buffer {
    Buffer() = default;
    explicit Buffer(std::size_t sizeInBytes, MemoryLocation location, std::size_t offset, void *ptr, DeleterFnPtr deleter)
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


struct OneShard : Buffer {
    std::optional<ShardDim> shardDims_;
    std::optional<ShardDim> shardOffset_;
};

#if 0
class ShardRange
{
    public:
    struct Iterator
    {
        using difference_type   = std::ptrdiff_t;
        using value_type        = OneShard;
        using pointer           = value_type*;
        using reference         = value_type&;
        using iterator_category = std::forward_iterator_tag;

        Iterator() = delete;
        Iterator(Storage& storage, ShardDim& shape, std::int64_t pos)
            : storage_(storage), shape_(shape), pos_(pos) {}

        value_type operator*() const { return storage_.OneShard(pos_); }
        Iterator&  operator++() { Advance(shape_, pos_); return *this; }
        bool       operator==(Iterator const& rhs) const { return (&storage_ == &rhs.storage_) && (shape_ == rhs.shape_) && (pos_ == rhs.pos_); }
        private:
        Storage& storage_;
        ShardDim& shape_;
        ShardDim pos_;
    };

    ShardRange(Iterator pBegin, Iterator pEnd) : begin_(pBegin), end_(pEnd)
    {
        assert(begin_.shape_ != end_.shape_);
        assert(begin_.shardDims_ == end_.shardDims_);
    }

    explicit ShardRange(Storage& storage, ShardDim& shape)
        : begin_(storage, shape, 0), end_(storage, storage, ShardUtils::NumShards(storage_.ShardDims().value())) {}

    Iterator begin() const { return begin_; }
    Iterator end() const { return end_; }

    private:
    Iterator begin_, end_;
};
#endif

#if 1
struct Storage {
    //ORT_DISALLOW_COPY_AND_ASSIGNMENT(Storage);
    Storage(std::vector<Buffer>&& buffers, std::optional<ShardDim> shardDims)
        : buffers_(std::move(buffers)), shardDims_(shardDims.value_or(ShardDim()))
    {
    }
    Storage(Buffer&& buffers)
    {
        buffers_.emplace_back(std::move(buffers));
    }
    void* Data(uint32_t index=0) const  { return buffers_.at(index).Data(); }
    MemoryLocation Location(uint32_t index=0) const  { return buffers_.at(index).Location(); }
    size_t Offset(uint32_t index=0) const  { return buffers_.at(index).Offset(); }
    void SetOffset(ptrdiff_t byte_offset, uint32_t index=0) { buffers_.at(index).SetOffset(byte_offset); }
    size_t Size(uint32_t index=0) const  { return buffers_.at(index).Size(); }
    OneShard Shard(uint32_t index=0) const { return {buffers_.at(index), shardDims_, std::nullopt}; }
    OneShard Shard(ShardDim const& offset) const { return { buffers_.at(ShardUtils::Index(offset, shardDims_)), shardDims_, offset }; }
    void Apply(std::function<void(Buffer&)> fn) { std::for_each(buffers_.begin(), buffers_.end(), fn); }
    std::optional<ShardDim> ShardDims() const { return std::make_optional(shardDims_); }
    bool OwnsBuffer() const { return buffers_[0].OwnBuffer(); }
 private:
    std::vector<Buffer> buffers_;
    ShardDim shardDims_;
};
#else
struct IStorage {
    virtual void* Data(uint32_t index=0) const=0;
    virtual MemoryLocation Location(uint32_t index=0) const=0;
    virtual size_t Offset(uint32_t index=0) const=0;
    virtual size_t SetOffset(ptrdiff_t byte_offset, uint32_t index=0) const=0;
    virtual size_t Size(uint32_t index=0) const=0;
    virtual OneShard Shard(uint32_t index=0) const=0;
    virtual void Apply(std::function<void(Buffer&)> fn)=0;
    virtual std::optional<ShardDim> ShardDims() const { return std::nullopt; }
};

struct SingleShard : IStorage {
    //ORT_DISALLOW_COPY_AND_ASSIGNMENT(SingleShard);
    SingleShard(Buffer&& buffer) : buffer_(std::move(buffer)) {}
    void* Data(uint32_t /*index*/) const  { return buffer_.Data(); }
    MemoryLocation Location(uint32_t /*index*/) const  { return buffer_.Location(); }
    size_t Offset(uint32_t /*index*/) const  { return buffer_.Offset(); }
    size_t Size(uint32_t /*index*/) const  { return buffer_.Size(); }
    OneShard const& Shard(ShardDim const& /*offset*/) const { return {buffer_}; }
    void Apply(std::function<void(Buffer&)> fn) { fn(buffer_); }
 private:
    Buffer buffer_;
};

struct Storage : IStorage {
    //ORT_DISALLOW_COPY_AND_ASSIGNMENT(Storage);
    Storage(std::vector<Buffer>&& buffers, std::optional<ShardDim> shardDims)
        : buffers_(std::move(buffers)), shardDims_(shardDims.value_or(ShardDim()))
    {
    }
    void* Data(uint32_t index=0) const  { return buffers_.at(index).Data(); }
    MemoryLocation Location(uint32_t index=0) const  { return buffers_.at(index).Location(); }
    size_t Offset(uint32_t index=0) const  { return buffers_.at(index).Offset(); }
    size_t Size(uint32_t index=0) const  { return buffers_.at(index).Size(); }
    OneShard Shard(uint32_t index=0) const { return { buffers_.at(index), offset, shardDims_ }; }
    OneShard Shard(ShardDim const& offset) const { return { buffers_.at(ShardUtils::Index(offset, shardDims_)), offset, shardDims_ }; }
    void Apply(std::function<void(Buffer&)> fn) { std::for_each(buffers_.begin(), buffers_.end(), fn); }
    std::optional<ShardDim> ShardDims() const { return std::make_optional(shardDims_); }
 private:
    std::vector<Buffer> buffers_;
    ShardDim shardDims_;
};
#endif

} // namespace onnxruntime ends
