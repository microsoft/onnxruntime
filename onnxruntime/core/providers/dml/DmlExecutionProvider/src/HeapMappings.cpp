#include "HeapMappings.h"
#include "precomp.h"

namespace
{
    struct Hasher
    {
        size_t Value = 0;

        template <typename T> void Add(const T& value)
        {
            auto hash = std::hash<T>{}(value);
            Value = (Value + 0x9e3779b9 + (hash << 6) + (hash >> 2)) ^ hash;
        }
    };
} // namespace

namespace Dml
{
    bool HeapMapping::operator==(const HeapMapping& value) const
    {
        return Heap == value.Heap && ResourceSegment == value.ResourceSegment && HeapSegments == value.HeapSegments;
    }

    bool HeapMappingsComparer::operator()(const HeapMappings& a, const HeapMappings& b) const
    {
        return a == b;
    }

    size_t HeapMappingsHasher::operator()(const HeapMappings& value) const
    {
        Hasher hash;

        for (auto& mapping : value)
        {
            hash.Add(mapping.Heap);
            hash.Add(mapping.ResourceSegment.Start);
            hash.Add(mapping.ResourceSegment.Size);

            for (auto& segment : mapping.HeapSegments)
            {
                hash.Add(segment.Start);
                hash.Add(segment.Size);
            }
        }

        return hash.Value;
    }

    bool ResourceComparer::operator()(const ComPtr<ID3D12Resource>& a, const ComPtr<ID3D12Resource>& b) const
    {
        return a.Get() == b.Get();
    }

    size_t ResourceHasher::operator()(const ComPtr<ID3D12Resource>& value) const
    {
        return size_t(value.Get());
    }
} // namespace Dml