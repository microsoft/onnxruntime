#pragma once

#include <vector>
#include <memory>

#include "core/framework/storage_impl.h"

struct Buffer;
struct StorageImpl;
struct Storage final {
    Storage() = default;
    Storage(std::vector<Buffer>&& buffers)
        : _impl(std::make_shared<StorageImpl>(std::move(buffers)))
    {}
    Storage(std::size_t sizeInBytes, MemoryLocation location, std::size_t offset, void *ptr, DeleterFnPtr deleter)
        : _impl(std::make_shared<StorageImpl>(sizeInBytes, location, offset, deleter))
    {}

 private:
    std::shared_ptr<StorageImpl> _impl;
};
