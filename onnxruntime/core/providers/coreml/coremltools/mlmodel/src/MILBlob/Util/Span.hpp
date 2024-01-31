// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Util/Verify.hpp"
#include <array>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace MILBlob {
namespace Util {

constexpr std::size_t DynamicExtent = std::numeric_limits<std::size_t>::max();

namespace span_helpers {

//----------------------------------------------------------------------
// helper traits
//----------------------------------------------------------------------

template <size_t Extent>
struct IsDynamicExtent {
    static constexpr bool value = false;
};

template <>
struct IsDynamicExtent<DynamicExtent> {
    static constexpr bool value = true;
};

template <size_t Index, size_t Extent>
struct IsIndexValid {
    static constexpr bool value = (Index < Extent);
};

template <size_t Index>
struct IsIndexValid<Index, DynamicExtent> {
    static constexpr bool value = false;
};

//----------------------------------------------------------------------
// helper storage size
//----------------------------------------------------------------------

template <size_t Extent>
class SpanSize final {
public:
    SpanSize() = default;
    ~SpanSize() = default;
    SpanSize(const SpanSize&) = default;
    SpanSize(SpanSize&&) noexcept = default;
    SpanSize& operator=(const SpanSize&) = default;
    SpanSize& operator=(SpanSize&&) noexcept = default;

    constexpr size_t Size() const
    {
        return m_size;
    }

private:
    static constexpr size_t m_size = Extent;
};

template <>
class SpanSize<DynamicExtent> final {
public:
    SpanSize() = delete;
    ~SpanSize() = default;
    SpanSize(const SpanSize&) = default;
    SpanSize(SpanSize&&) noexcept = default;
    SpanSize& operator=(const SpanSize&) = default;
    SpanSize& operator=(SpanSize&&) noexcept = default;

    explicit SpanSize(size_t size) : m_size(size) {}

    size_t Size() const
    {
        return m_size;
    }

private:
    size_t m_size;
};

}  // namespace span_helpers

//----------------------------------------------------------------------
// Span<T, Extent> is a custom implementation of an array view, similar
// to std::span introduced in C++20.
//
// If Extent is specified, Span supports compile-time bounds checking
// when the Get<> method is used.
//
// This version of Span also supports iterating slices and dimensions
// of multi-dimensional contiguous memory blocks.
//----------------------------------------------------------------------

template <typename T, size_t Extent = DynamicExtent>
class Span final {
public:
    using value_type = T;
    using pointer = typename std::add_pointer<value_type>::type;
    using reference = typename std::add_lvalue_reference<value_type>::type;
    using iterator = pointer;

    using const_value_type = typename std::add_const<value_type>::type;
    using const_pointer = typename std::add_pointer<const_value_type>::type;
    using const_iterator = const_pointer;

    template <size_t Extent_>
    using SpanSize = span_helpers::SpanSize<Extent_>;

    template <size_t Extent_>
    using IsDynamicExtent = span_helpers::IsDynamicExtent<Extent_>;

    template <size_t Index, size_t Extent_>
    using IsIndexValid = span_helpers::IsIndexValid<Index, Extent_>;

    class SliceIterator final {
    public:
        SliceIterator(pointer p, size_t stride) : m_ptr(p), m_stride(stride) {}

        bool operator==(const SliceIterator& other) const
        {
            return m_ptr == other.m_ptr && m_stride == other.m_stride;
        }

        bool operator!=(const SliceIterator& other) const
        {
            return !(*this == other);
        }

        SliceIterator& operator++()
        {
            m_ptr += m_stride;
            return *this;
        }

        // NOLINTNEXTLINE(cert-dcl21-cpp)
        SliceIterator operator++(int) const
        {
            return SliceIterator(m_ptr + m_stride, m_stride);
        }

        Span<T> operator*() const
        {
            return Span<T>(m_ptr, m_stride);
        }

    private:
        pointer m_ptr;
        size_t m_stride;
    };

    template <size_t Stride>
    class StaticSliceIterator final {
    public:
        explicit StaticSliceIterator(pointer p) : m_ptr(p) {}

        bool operator==(const StaticSliceIterator<Stride>& other) const
        {
            return m_ptr == other.m_ptr;
        }

        bool operator!=(const StaticSliceIterator<Stride>& other) const
        {
            return !(*this == other);
        }

        StaticSliceIterator& operator++()
        {
            m_ptr += Stride;
            return *this;
        }

        // NOLINTNEXTLINE(cert-dcl21-cpp)
        StaticSliceIterator operator++(int) const
        {
            return StaticSliceIterator<Stride>(m_ptr + Stride);
        }

        Span<T, Stride> operator*() const
        {
            return Span<T, Stride>(m_ptr);
        }

    private:
        pointer m_ptr;
    };

    template <typename Iterator>
    class IteratorProvider final {
    public:
        IteratorProvider(Iterator begin, Iterator end) : m_begin(begin), m_end(end) {}

        Iterator begin() const
        {
            return m_begin;
        }

        Iterator end() const
        {
            return m_end;
        }

    private:
        Iterator m_begin;
        Iterator m_end;
    };

    ~Span() = default;

    Span(const Span<T, Extent>&) = default;
    Span(Span<T, Extent>&&) noexcept = default;

    Span<T, Extent>& operator=(const Span<T, Extent>&) = default;
    Span<T, Extent>& operator=(Span<T, Extent>&&) noexcept = default;

    /** Implicit copy constructor for converting a mutable span to a const span. Extent and type must be the same. */
    template <typename NonConstT,
              typename std::enable_if<!std::is_same<T, NonConstT>::value &&
                                          std::is_same<T, typename std::add_const<NonConstT>::type>::value,
                                      int>::type = 0>
    Span(const Span<NonConstT, Extent>& other) : m_ptr(other.Data())
                                               , m_size(other.Size())
    {}

    /** Implicit move constructor for converting a mutable span to a const span. Extent and type must be the same. */
    template <typename NonConstT,
              typename std::enable_if<!std::is_same<T, NonConstT>::value &&
                                          std::is_same<T, typename std::add_const<NonConstT>::type>::value,
                                      int>::type = 0>
    Span(Span<NonConstT, Extent>&& other) : m_ptr(other.Data())
                                          , m_size(other.Size())
    {}

    template <size_t Extent__ = Extent, typename std::enable_if<IsDynamicExtent<Extent__>::value, int>::type = 0>
    Span() : m_ptr(nullptr)
           , m_size(0)
    {}

    template <size_t Extent__ = Extent, typename std::enable_if<!IsDynamicExtent<Extent__>::value, int>::type = 0>
    explicit Span(pointer p) : m_ptr(p)
    {}

    template <size_t Extent__ = Extent, typename std::enable_if<IsDynamicExtent<Extent__>::value, int>::type = 0>
    Span(pointer p, size_t size) : m_ptr(size == 0 ? nullptr : p)
                                 , m_size(size)
    {}

    //
    // properties
    //

    pointer Data() const
    {
        return m_ptr;
    }

    size_t Size() const
    {
        return m_size.Size();
    }

    constexpr bool IsEmpty() const
    {
        return Size() == 0;
    }

    //
    // random access
    //

    reference operator[](size_t index) const
    {
        MILDebugVerifyIsTrue(index < Size(), std::range_error, "index out of bounds");
        return m_ptr[index];
    }

    reference At(size_t index) const
    {
        MILVerifyIsTrue(index < Size(), std::range_error, "index out of bounds");
        return m_ptr[index];
    }

    // Get<N>() returns a reference to the value at index N.
    // This method only exists for fixed-sized Span instantiations.
    // The bounds of N are compile-time checked.
    template <
        size_t Index,
        typename std::enable_if<!IsDynamicExtent<Extent>::value && IsIndexValid<Index, Extent>::value, int>::type = 0>
    reference Get() const
    {
        return (*this)[Index];
    }

    //
    // slicing
    //

    /** Gets a sub-span starting at index */
    Span<T> Slice(size_t index) const
    {
        MILVerifyIsTrue(index < Size(), std::range_error, "index out of bounds");
        return Span<T>(Data() + index, Size() - index);
    }

    /** Gets a sub-span starting at index with length size */
    Span<T> Slice(size_t index, size_t size) const
    {
        MILVerifyIsTrue(size > 0 && index < Size() && index + size <= Size(), std::range_error, "index out of bounds");
        return Span<T>(Data() + index, size);
    }

    /** Slices into num_slices dimensions, and returns the span corresponding to slice_index */
    Span<T> SliceByDimension(size_t num_slices, size_t slice_index) const
    {
        MILVerifyIsTrue(Size() % num_slices == 0, std::range_error, "index out of bounds");
        size_t stride = Size() / num_slices;
        return Slice(slice_index * stride, stride);
    }

    //
    // reinterpreting data
    //

    template <size_t NewExtent>
    Span<T, NewExtent> StaticResize() const
    {
        MILVerifyIsTrue(NewExtent <= Size(), std::range_error, "index out of bounds");
        return Span<T, NewExtent>(Data());
    }

    //
    // basic C++ iterators
    //

    iterator begin() const
    {
        return Data();
    }

    iterator end() const
    {
        return Data() + Size();
    }

    const_iterator cbegin() const
    {
        return Data();
    }

    const_iterator cend() const
    {
        return Data() + Size();
    }

    std::reverse_iterator<iterator> rbegin() const
    {
        return std::reverse_iterator<iterator>(Data() + Size());
    }

    std::reverse_iterator<iterator> rend() const
    {
        return std::reverse_iterator<iterator>(Data());
    }

    std::reverse_iterator<const_iterator> crbegin() const
    {
        return std::reverse_iterator<const_iterator>(Data() + Size());
    }

    std::reverse_iterator<const_iterator> crend() const
    {
        return std::reverse_iterator<const_iterator>(Data());
    }

    //
    // complex C++ iterators
    //

    /** Iterates based on slices. This iterator will produce Size() % sliceSice slices. */
    IteratorProvider<SliceIterator> IterateSlices(size_t sliceSize) const
    {
        MILVerifyIsTrue(Size() % sliceSize == 0, std::range_error, "index out of bounds");

        return IteratorProvider<SliceIterator>(SliceIterator(Data(), sliceSize),
                                               SliceIterator(Data() + Size(), sliceSize));
    }

    template <size_t SliceSize>
    IteratorProvider<StaticSliceIterator<SliceSize>> IterateSlices() const
    {
        MILVerifyIsTrue(Size() % SliceSize == 0, std::range_error, "index out of bounds");

        return IteratorProvider<StaticSliceIterator<SliceSize>>(StaticSliceIterator<SliceSize>(Data()),
                                                                StaticSliceIterator<SliceSize>(Data() + Size()));
    }

    /**
     Iterates based on dimensions. Similar to IterateBySlices, but based on the number of slices (dimensions) rather
     than the size of the slice.
    */
    IteratorProvider<SliceIterator> IterateByDimension(size_t dim) const
    {
        return IterateSlices(Size() / dim);
    }

private:
    pointer m_ptr;
    SpanSize<Extent> m_size;
};

// MakeSpan for std::vector<T> yields Span<T, DynamicExtent>
// Examples:
// (1) create a mutable span
//     std::vector<int> v = { 1, 2, 3 };
//     auto span = MakeSpan(v); // span is Span<int>
// (2) create an immutable span
//     const std::vector<int> v = { 1, 2, 3 };
//     auto span = MakeSpan(v); // span is Span<const int>
// (3) create an immutable span from a mutable vector
//     std::vector<int> v = { 1, 2, 3 };
//     auto span = MakeSpan<const int>(v); // span is Span<const int>

template <typename T, template <typename, typename...> class C, typename... Args>
Span<T> MakeSpan(C<T, Args...>& c)
{
    return Span<T>(c.data(), c.size());
}

template <typename T, template <typename, typename...> class C, typename... Args>
Span<const T> MakeSpan(const C<T, Args...>& c)
{
    return Span<const T>(c.data(), c.size());
}

template <typename TargetT,
          typename T,
          template <typename, typename...>
          class C,
          typename... Args,
          std::enable_if_t<std::is_const<TargetT>::value, bool> = true>
Span<TargetT> MakeSpan(const C<T, Args...>& c)
{
    return Span<TargetT>(c.data(), c.size());
}

// MakeSpan for std::array<T, N> yields Span<T, N>.
// Examples:
// (1) create a mutable span
//     std::array<int, 3> v = { 1, 2, 3 };
//     auto span = MakeSpan(v); // span is Span<int, 3>
// (2) create an immutable span from a mutable vector
//     std::array<int, 3> v = { 1, 2, 3 };
//     auto span = MakeSpan<const int>(v); // span is Span<const int, 3>
// (3) create an immutable span
//     const std::array<int, 3> v = { 1, 2, 3 };
//     auto span = MakeSpan(v); // span is Span<const int, 3>

template <typename T, size_t N>
Span<T, N> MakeSpan(std::array<T, N>& v)
{
    return Span<T, N>(v.data());
}

template <typename T, size_t N, typename MutableT = typename std::remove_const<T>::type>
Span<T, N> MakeSpan(const std::array<MutableT, N>& v)
{
    return Span<T, N>(v.data());
}

template <typename T, size_t N, typename ConstT = typename std::add_const<T>::type>
Span<ConstT, N> MakeSpan(const std::array<T, N>& v)
{
    return Span<ConstT, N>(v.data());
}

}  // namespace Util
}  // namespace MILBlob
