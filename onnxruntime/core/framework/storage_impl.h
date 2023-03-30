#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/tensor_shape.h"
#include "core/common/common.h"

namespace onnxruntime {

using DeleterFnPtr = void (*)(void*);


struct Buffer final {
    Buffer(std::size_t sizeInBytes, MemoryLocation location, std::size_t offset, void *ptr, DeleterFnPtr deleter)
        : sizeInBytes_(sizeInBytes),
          memoryLocation_(location),
          offset_(offset),
          ptr_(ptr),
          deleter_(deleter) {}
    ~Buffer() {
        if (deleter_)
            deleter_(ptr_);
    }
private:
    MemoryLocation memoryLocation_;
    size_t sizeInBytes_;
    size_t offset_;
    void *ptr_;

    /**
        if deleter_ is null, it means tensor does not own the buffer.
        otherwise tensor will use the deleter to release the buffer when
        tensor is released.
    */
    DeleterFnPtr deleter_;
};

struct StorageImpl {
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(StorageImpl);
    StorageImpl(std::vector<Buffer>&& buffers)
        : buffers_(std::move(buffers))
    {
    }
    StorageImpl(std::size_t sizeInBytes, MemoryLocation location, std::size_t offset, void *ptr, DeleterFnPtr deleter)
    {
        buffers_.emplace_back(sizeInBytes, location, offset, ptr, deleter);
    }
 private:
    std::vector<Buffer> buffers_;
    // ShardDims shardDims
};



}
