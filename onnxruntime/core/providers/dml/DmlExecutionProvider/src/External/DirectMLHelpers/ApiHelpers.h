// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

union ActivationOperatorDescUnion
{
    DML_ACTIVATION_IDENTITY_OPERATOR_DESC identity;
    DML_ACTIVATION_ELU_OPERATOR_DESC elu;
    DML_ACTIVATION_HARDMAX_OPERATOR_DESC hardmax;
    DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC hardSigmoid;
    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leakyRelu;
    DML_ACTIVATION_LINEAR_OPERATOR_DESC linear;
    DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC logSoftmax;
    DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC parameterizedRelu;
    DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC parametricSoftplus;
    DML_ACTIVATION_RELU_OPERATOR_DESC relu;
    DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC scaledTanh;
    DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC scaledElu;
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid;
    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax;
    DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC softplus;
    DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC softsign;
    DML_ACTIVATION_TANH_OPERATOR_DESC tanh;
    DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC thresholdedRelu;
    DML_ACTIVATION_SHRINK_OPERATOR_DESC shrink;
};

struct ActivationOperatorDesc
{
    ActivationOperatorDescUnion params;
    DML_OPERATOR_TYPE activationType;

    DML_OPERATOR_DESC GetDmlDesc() const
    {
        switch (activationType)
        {
        case DML_OPERATOR_ACTIVATION_ELU: return { activationType, &params.elu };
        case DML_OPERATOR_ACTIVATION_HARDMAX: return { activationType, &params.hardmax };
        case DML_OPERATOR_ACTIVATION_HARD_SIGMOID: return { activationType, &params.sigmoid };
        case DML_OPERATOR_ACTIVATION_IDENTITY: return { activationType, &params.identity };
        case DML_OPERATOR_ACTIVATION_LEAKY_RELU: return { activationType, &params.leakyRelu };
        case DML_OPERATOR_ACTIVATION_LINEAR: return { activationType, &params.linear };
        case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX: return { activationType, &params.logSoftmax };
        case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU: return { activationType, &params.parameterizedRelu };
        case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS: return { activationType, &params.parametricSoftplus };
        case DML_OPERATOR_ACTIVATION_RELU: return { activationType, &params.relu };
        case DML_OPERATOR_ACTIVATION_SCALED_ELU: return { activationType, &params.scaledElu };
        case DML_OPERATOR_ACTIVATION_SCALED_TANH: return { activationType, &params.scaledTanh };
        case DML_OPERATOR_ACTIVATION_SIGMOID: return { activationType, &params.sigmoid };
        case DML_OPERATOR_ACTIVATION_SOFTMAX: return { activationType, &params.softmax };
        case DML_OPERATOR_ACTIVATION_SOFTPLUS: return { activationType, &params.softplus };
        case DML_OPERATOR_ACTIVATION_SOFTSIGN: return { activationType, &params.softsign };
        case DML_OPERATOR_ACTIVATION_TANH: return { activationType, &params.tanh };
        case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU: return { activationType, &params.thresholdedRelu };
        case DML_OPERATOR_ACTIVATION_SHRINK: return { activationType, &params.shrink };
        default: THROW_HR(E_INVALIDARG);
        }
    }
};

// DML_BUFFER_TENSOR_DESC (DML_TENSOR_TYPE_BUFFER)
struct DmlBufferTensorDesc
{
    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
    std::vector<uint32_t> sizes;
    std::optional<std::vector<uint32_t>> strides;
    uint64_t totalTensorSizeInBytes = 0;
    uint32_t guaranteedBaseOffsetAlignment = 0;

    DmlBufferTensorDesc() = default;

    /*implicit*/ DmlBufferTensorDesc(const DML_BUFFER_TENSOR_DESC& desc)
        : dataType(desc.DataType)
        , flags(desc.Flags)
        , sizes(desc.Sizes, desc.Sizes + desc.DimensionCount)
        , totalTensorSizeInBytes(desc.TotalTensorSizeInBytes)
        , guaranteedBaseOffsetAlignment(desc.GuaranteedBaseOffsetAlignment)
    {
        if (desc.Strides)
        {
            strides.emplace(desc.Strides, desc.Strides + desc.DimensionCount);
        }
    }

    // Constructs a DmlBufferTensorDesc from a generic DML_TENSOR_DESC. The type must be DML_TENSOR_TYPE_BUFFER.
    /*implicit*/ DmlBufferTensorDesc(const DML_TENSOR_DESC& desc)
        : DmlBufferTensorDesc(*static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc))
    {
        assert(desc.Type == DML_TENSOR_TYPE_BUFFER);
    }
};

template <size_t Size>
class StackAllocator
{
public:
    StackAllocator() = default;

    // Non-copiable, non-movable
    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    StackAllocator(StackAllocator&&) = delete;
    StackAllocator& operator=(StackAllocator&&) = delete;

    template <typename T>
    T* Allocate(size_t count = 1)
    {
        static_assert(std::is_trivial_v<T>,
            "This class may only be used to allocate trivial types, as it does not invoke constructors.");

        // Allocate from the fixed bucket before falling back to dynamic
        Bucket* lastBucket = m_dynamic.empty() ? static_cast<Bucket*>(&m_fixed) : static_cast<Bucket*>(&m_dynamic.back());

        size_t sizeInBytes = sizeof(T) * count;
        void* memory = lastBucket->TryAllocate(sizeInBytes, alignof(T));

        if (!memory)
        {
            // Not enough capacity remains; allocate a new dynamic bucket
            size_t minimumSize = sizeInBytes;
            m_dynamic.emplace_back(minimumSize);

            memory = m_dynamic.back().TryAllocate(sizeInBytes, alignof(T));
        }

        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }

    void Reset()
    {
        m_fixed.allocatedSize = 0;
        m_dynamic.clear();
    }

private:
    struct Bucket
    {
        void* data;
        size_t allocatedSize;
        size_t capacity;

        Bucket() = default;

        // Non-copiable, non-movable
        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;
        Bucket(Bucket&&) = delete;
        Bucket& operator=(Bucket&&) = delete;

        template <typename T>
        static T RoundUpToMultiple(T value, T multiple)
        {
            static_assert(std::is_integral_v<T>);

            T remainder = value % multiple;
            if (remainder != 0)
            {
                value += multiple - remainder;
            }

            return value;
        }

        void* TryAllocate(size_t sizeInBytes, size_t alignment)
        {
            size_t alignedOffset = RoundUpToMultiple(allocatedSize, alignment);
            size_t newAllocatedSize = alignedOffset + sizeInBytes;

            if (newAllocatedSize > capacity)
            {
                return nullptr; // Not enough capacity
            }

            allocatedSize = newAllocatedSize;
            return static_cast<byte*>(data) + alignedOffset;
        }
    };

    struct FixedBucket : Bucket
    {
        std::array<byte, Size> stack;

        FixedBucket()
        {
            this->data = stack.data();
            this->allocatedSize = 0;
            this->capacity = stack.size();
        }
    };

    struct DynamicBucket : Bucket
    {
        explicit DynamicBucket(size_t minimumSize)
        {
            this->allocatedSize = 0;
            this->capacity = RoundUpToMultiple<size_t>(minimumSize, 4096); // Round up to nearest page granularity

            this->data = VirtualAlloc(nullptr, this->capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            THROW_LAST_ERROR_IF_NULL(this->data);
        }

        ~DynamicBucket()
        {
            if (data)
            {
                (void)VirtualFree(data, 0, MEM_RELEASE);
            }
        }
    };

    // This allocator first retrieves memory from a fixed-size stack-allocated array before falling back to dynamically
    // allocated memory if the fixed stack array is exhausted.
    FixedBucket m_fixed;
    std::deque<DynamicBucket> m_dynamic;
};