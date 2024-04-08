// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GpuEvent.h"

namespace Dml
{
    struct ReusableCopyKey
    {
        ID3D12Resource* dstResource;
        uint64_t dstOffset;
        uint64_t srcSizeInBytes;
        bool operator==(const ReusableCopyKey& other) const
        {
            return dstResource == other.dstResource && dstOffset == other.dstOffset && srcSizeInBytes == other.srcSizeInBytes;
        }
    };
}

namespace std
{
    template <>
    struct hash<Dml::ReusableCopyKey> {
        size_t operator()(const Dml::ReusableCopyKey& reusableCopyKey) const
        {
            auto h = std::hash<uint64_t>()(reusableCopyKey.dstOffset);
            onnxruntime::HashCombine(reusableCopyKey.srcSizeInBytes, h);
            onnxruntime::HashCombine(reusableCopyKey.dstResource, h);
            return h;
        }
    };
}

namespace Dml
{
    class ExecutionContext;

    // Implements a non-blocking, ring-buffer style upload heap for copying CPU data to GPU resources.
    class PooledUploadHeap
    {
    public:
        PooledUploadHeap(ID3D12Device* device, std::shared_ptr<ExecutionContext> executionContext);

        // Makes a copy of the source data and begins copying it into the destination resource, and returns a GpuEvent
        // which will become signaled when the copy is complete. The destination resource must be a default or readback
        // buffer.
        GpuEvent BeginUploadToGpu(
            ID3D12Resource* dst,
            uint64_t dstOffset,
            D3D12_RESOURCE_STATES dstState,
            gsl::span<const std::byte> src);

        // Same as BeginUploadToGpu, but reuses a command list that already has a copy recorded into it instead of
        // creating a new one.
        GpuEvent BeginReusableUploadToGpu(
            ID3D12Resource* dst,
            uint64_t dstOffset,
            D3D12_RESOURCE_STATES dstState,
            gsl::span<const std::byte> src);

        // Releases unused capacity.
        void Trim();

        size_t Capacity() const { return m_totalCapacity; }

    private:
        static constexpr size_t c_minChunkSize = 1024 * 1024; // 1MB
        static constexpr size_t c_allocationAlignment = 512; // In bytes; as per D3D12 requirement for buffers

        // A suballoction from a chunk
        struct Allocation
        {
            size_t sizeInBytes;

            // The offset, in bytes, from the beginning of the chunk to the beginning of this allocation
            size_t offsetInChunk;

            // The event that will be signaled to when the GPU is done executing work that uses this allocation
            GpuEvent doneEvent;

            // Whether the allocation is locked because it is being used by a reusable command list
            bool locked;
        };

        // Represents a single contiguous upload heap from which we carve out suballocations. Ranges are suballocated
        // from the upload heap in a ring-buffer fashion.
        struct Chunk
        {
            size_t capacityInBytes; // The total size of the upload heap, in bytes
            ComPtr<ID3D12Resource> resource;

            // Allocations are sorted by ascending fence value - that is, least to most recently allocated
            std::list<Allocation> allocations;
        };

        // Calls AssertInvariants on construction and again on destruction
        class InvariantChecker
        {
        public:
            InvariantChecker(PooledUploadHeap* parent)
                : m_parent(parent)
            {
                m_parent->AssertInvariants();
            }

            ~InvariantChecker()
            {
                m_parent->AssertInvariants();
            }

        private:
            PooledUploadHeap* m_parent;
        };

        struct ReusableCopyCommandListState
        {
            ComPtr<ID3D12GraphicsCommandList> commandList;
            ComPtr<ID3D12CommandAllocator> commandAllocator;
            ID3D12Resource* chunkResource;
            Allocation* allocation;
        };

        // Attempts to find enough unused space in the supplied chunk to accommodate the given allocation size.
        // Returns the offset of that memory if successful, null if there wasn't enough space.
        static std::optional<size_t> FindOffsetForAllocation(const Chunk& chunk, size_t sizeInBytes);

        static Chunk CreateChunk(ID3D12Device* device, size_t sizeInBytes);

        // Finds or creates a chunk with enough space to accommodate an allocation of the given size, and returns a
        // pointer to the chunk and allocation offset.
        std::pair<Chunk*, size_t> Reserve(size_t sizeInBytes);

        void ReclaimAllocations(); // Frees all allocations which are no longer being used by the GPU.
        void AssertInvariants();

        ComPtr<ID3D12Device> m_device;
        std::shared_ptr<ExecutionContext> m_executionContext;

        // We keep an LRU cache of reusable command lists. A command list can be reused for a copy if the following conditions are met:
        //   1. The destination resource is the same
        //   2. The destination offset is the same
        //   3. The size of the source data is the same
        // The least recently used command list is located at the beginning, and the most recently used one is located at the end.
        constexpr static int maxReusableCommandLists = 3;
        std::list<ReusableCopyCommandListState> m_reusableCommandLists;
        std::unordered_map<ReusableCopyKey, std::list<ReusableCopyCommandListState>::iterator> m_reusableCommandListsCache;

        std::vector<Chunk> m_chunks; // sorted ascending by capacity (upload heap size)
        size_t m_totalCapacity = 0; // Total size of all chunks, in bytes
    };

} // namespace Dml
