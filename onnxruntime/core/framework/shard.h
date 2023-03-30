#pragma once

#include <cstdint>
#include <cstddef>
#include <optinal>

#include "core/framework/tensor_shape.h"
#include "core/framework/ortmemoryinfo.h"

namespace onnxruntime {

struct ShardUtils {
    // given shape, returns vector of offsets in shardDims
    // For example of [4, 8, 12] and [2, 1, 2],
    // (0, 0, 0), (0, 8, 6), (2, 8, 0), (2, 8, 6)
    static ShardDim GetShardSize(TensorShape const& shape, ShardDim const& shardDims)
    {
        ShardDim shardSize;
        assert(shape.NumDimensions() == shardDims.size());
        for (auto i = 0u; i < shape.NumDimensions(); ++i)
        {
            assert(shape[i] % shardDims[i] == 0);
            shardSize[i] = shape[i] / shardDims[i];
        }
        return shardSize;
    }

    static ShardDim GetShardStride(ShardDim const& shardDims)
    {
        ShardDim shardStride;
        auto accumulator = 1u;
        for (auto i = shardDims.size() - 1; i >= 0; --i)
        {
            shardStride[i] = accumulator;
            accumulator *= shardDims[i];
        }
        return shardStride;
    }

    static ShardDim GetShardOffset(size_t rank, ShardDim const& stride, ShardDim const& shardSize)
    {
        const auto lastShardDim = shardSize.size() - 1;
        ShardDim offset;

        for (auto idx=0u; idx<=lastShardDim; idx++)
        {
            offset[idx] = (rank / stride[idx]) * shardSize[idx];
            rank = rank % stride[idx];
        }
        return offset;
    }

    static std::vector<ShardDim> GetShardOffsets(TensorShape const& shape, ShardDim const& shardDims)
    {
        assert(shape.NumDimensions() == shardDims.size());
        const auto numShards = NumShards(shardDims);
        std::vector<ShardDim> shardOffsets;
        shardOffsets.reserve(numShards);

        ShardDim stride = GetShardStride(shardDims);
        ShardDim shardSize = GetShardSize(shape, shardDims);

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

    static size_t GetIdxInShardDim(ShardDim const& shardDims, ShardDim const& shardIdx)
    {
        assert(shardDims.size() == shardIdx.size());
        const auto lastShardDim = shardDims.size() - 1;
        auto accumulator = 1ul;
        size_t offset = 0;
        for (auto i = lastShardDim; i >= 0; --i)
        {
            //stride[i] = accumulator;
            offset += shardIdx[i] * accumulator;
            accumulator *= shardDims[i];
        }
        return offset;
    }
};

using MemoryLocations = std::array<MemoryLocation, onnxruntime::kTensorShapeSmallBufferElementsSize>;

struct ShardSpec : TensorShape {
    ShardSpec(TensorShape const& shape)
        : TensorShape(shape), shardDims_(shape.shardDims())
    {
    }

    ShardSpec(TensorShape const& shape, TensorShape shardDim, MemoryLocations const& locations)
        : TensorShape(shape), shardDims_(shardDim), locations_(locations)
    {
    }

    std::optional<ShardDim> shardDims() const override { return shardDims_; }

private:
    ShardDim shardDims_;
    MemoryLocations locations_;
};

}
