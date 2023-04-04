#pragma once

#include <cstdint>
#include <cstddef>
#include <optional>

#include "core/framework/tensor_shape.h"
#include "core/framework/ortmemoryinfo.h"

namespace onnxruntime {

struct ShardUtils {
    // Given shape and shardDims, returns vector of offsets in shardDims
    // For example of [4,8,12] and [2,1,2], returns [2,8,6].
    // Currently assumes evenly sharding tensors.
    static TensorShape GetShardSize(TensorShape const& shape, ShardDim const& shardDims)
    {
        TensorShape shardShape;
        assert(shape.NumDimensions() == shardDims.size());
        for (auto i = 0u; i < shape.NumDimensions(); ++i)
        {
            assert(shape[i] % shardDims[i] == 0);
            shardShape[i] = shape[i] / shardDims[i];
        }
        return shardShape;
    }

    // Given shape and shardSize, returns vector of offsets in shardDims
    // For example of [4, 8, 12] and [2, 1, 2],
    // (0, 0, 0), (0, 8, 6), (2, 8, 0), (2, 8, 6)
    static ShardDim GetShardOffset(size_t rank, ShardDim const& stride, TensorShape const& shardSize)
    {
        const auto lastShardDim = shardSize.NumDimensions() - 1;
        ShardDim offset;

        for (auto idx=0u; idx<=lastShardDim; idx++)
        {
            offset[idx] = (rank / stride[idx]) * shardSize[idx];
            rank = rank % stride[idx];
        }
        return offset;
    }

    static ShardDim GetShardStride(ShardDim const& shardDims)
    {
        ShardDim shardStride;
        auto accumulator = 1u;
        for (int64_t i = shardDims.size() - 1; i >= 0; --i)
        {
            shardStride[i] = accumulator;
            accumulator *= shardDims[i];
        }
        return shardStride;
    }

    static std::vector<ShardDim> GetShardOffsets(TensorShape const& shape, ShardDim const& shardDims)
    {
        assert(shape.NumDimensions() == shardDims.size());
        const auto numShards = NumShards(shardDims);
        std::vector<ShardDim> shardOffsets;
        shardOffsets.reserve(numShards);

        ShardDim stride = GetShardStride(shardDims);
        TensorShape shardSize = GetShardSize(shape, shardDims);

        for (auto i=0u; i < numShards; i++)
        {
            shardOffsets.emplace_back(GetShardOffset(i, stride, shardSize));
        }
        return shardOffsets;
    }

    static size_t NumShards(ShardDim const& shardDims)
    {
        auto numDims = shardDims.size();
        auto accumulator = 1ul;
      for (auto i=0u; i<numDims; i++)
        {
            accumulator *= shardDims[i];
        }
        return accumulator;
    }

    static uint64_t Index(ShardDim const& offsets, ShardDim const& shardDims)
    {
        ShardDim shardStride;
        auto accumulator = 1u;
        auto offset = 0u;
        for (int64_t i = shardDims.size() - 1; i >= 0; --i)
        {
            offset += offsets[i] * accumulator;
            accumulator *= shardDims[i];
        }
        return offset;
    }

    static size_t GetIdxInShardDim(ShardDim const& shardDims, ShardDim const& shardIdx)
    {
        assert(shardDims.size() == shardIdx.size());
        const auto lastShardDim = shardDims.size() - 1;
        auto accumulator = 1ul;
        size_t offset = 0;
        for (int64_t i = lastShardDim; i >= 0; --i)
        {
            //stride[i] = accumulator;
            offset += shardIdx[i] * accumulator;
            accumulator *= shardDims[i];
        }
        return offset;
    }
};




struct ShardSpec : TensorShape {
    ShardSpec(TensorShape const& shape)
        : TensorShape(shape), shardLocations_(shape.shardInfo().value_or(ShardInfo{}))
    {
    }

    ShardSpec(TensorShape const& shape, ShardDim const& shardDim, MemoryLocations const& locations)
        : TensorShape(shape), shardLocations_(ShardInfo{shardDim, locations})
    {
    }

    std::optional<ShardInfo> shardInfo() const override { return std::make_optional(shardLocations_); }

private:
    ShardInfo shardLocations_;
};

}
