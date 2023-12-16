//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
// clang-format off

#pragma once
// TODO (pavignol): Revert
// #include <DirectML.h>
#include "core/providers/dml/DirectML2.h"

#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#include <deque>
#include <memory>
#include <utility>
#include <type_traits>
#include <functional>

#include <wrl/client.h> // For Microsoft::WRL::ComPtr

#if DMLX_USE_ABSEIL
    #if __cpp_lib_span
        #include <span>
    #endif
#elif __cplusplus >= 201703L && __has_include(<optional>)
    // stl optional is only available in cpp17 and above.
    #include <optional>
#elif __has_include("dml_optional_extensions.h")
    #include "dml_optional_extensions.h"
    #define DMLX_OPTIONAL_EXTENDED
#endif

/** Calculates the minimum number of bytes required to store a buffer tensor with the specified type, sizes, and
    strides. The formula can be expressed as the following:

    IndexOfLastElement = dot(Sizes - 1, Strides);
    MinimumImpliedSizeInBytes = roundup((IndexOfLastElement + 1) * ElementSizeInBytes, 4)

    In other words, the minimum size of a tensor is the index of the one-past-the-end element, multiplied by the
    element size (e.g. 2 bytes for a FLOAT16 tensor). Additionally DirectML requires that all buffers bound must have
    a total size which is DWORD-aligned, and hence the minimum implied size in bytes must be rounded up to the nearest
    4-byte boundary.
    */

inline UINT64 DMLCalcBufferTensorSize(
    DML_TENSOR_DATA_TYPE dataType,
    UINT dimensionCount,
    _In_reads_(dimensionCount) const UINT* sizes,
    _In_reads_opt_(dimensionCount) const UINT* strides)
{
    UINT elementSizeInBytes = 0;
    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
        elementSizeInBytes = 4;
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
        elementSizeInBytes = 2;
        break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
        elementSizeInBytes = 1;
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_INT64:
        elementSizeInBytes = 8;
        break;

    default:
        return 0; // Invalid data type
    }

    UINT64 minimumImpliedSizeInBytes = 0;
    if (!strides)
    {
        minimumImpliedSizeInBytes = 1;
        for (UINT i = 0; i < dimensionCount; ++i)
        {
            minimumImpliedSizeInBytes *= sizes[i];
        }
        minimumImpliedSizeInBytes *= elementSizeInBytes;
    }
    else
    {
        UINT64 indexOfLastElement = 0;
        for (UINT i = 0; i < dimensionCount; ++i)
        {
            indexOfLastElement += (sizes[i] - 1) * strides[i];
        }

        minimumImpliedSizeInBytes = (indexOfLastElement + 1) * elementSizeInBytes;
    }

    // Round up to the nearest 4 bytes.
    minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ull;

    return minimumImpliedSizeInBytes;
}

namespace dml
{
    namespace detail
    {
        // Provide non-member size() and data(). Defaults to standard library implementation (if available)
#if __cpp_lib_nonmember_container_access
        template <typename C>
        constexpr auto size(const C& c) -> decltype(c.size())
        {
            return std::size(c);
        }

        template <typename T, std::size_t N>
        constexpr std::size_t size(const T(&array)[N]) noexcept
        {
            return std::size(array);
        }

        template <typename C>
        constexpr auto data(C& c) -> decltype(c.data())
        {
            return std::data(c);
        }

        template <typename T, std::size_t N>
        constexpr T* data(T(&array)[N]) noexcept
        {
            return std::data(array);
        }
#else
        template <typename C>
        constexpr auto size(const C& c) -> decltype(c.size())
        {
            return c.size();
        }

        template <typename T, std::size_t N>
        constexpr std::size_t size(const T(&array)[N]) noexcept
        {
            return N;
        }

        template <typename C>
        constexpr auto data(C& c) -> decltype(c.data())
        {
            return c.data();
        }

        template <typename T, std::size_t N>
        constexpr T* data(T(&array)[N]) noexcept
        {
            return array;
        }
#endif

        template <typename T>
        class span
        {
        public:
            span() = default;

            constexpr span(std::initializer_list<T> i) : m_begin(i.begin()), m_end(i.end()) {}
            constexpr span(T* begin, T* end) : m_begin(begin), m_end(end) {}
            constexpr span(T* begin, size_t elementCount) : m_begin(begin), m_end(begin + elementCount) {}

            template <typename ContiguousContainer>
            constexpr span(ContiguousContainer&& container)
                : m_begin(dml::detail::data(container)), m_end(m_begin + dml::detail::size(container)) {}

            template <size_t N>
            constexpr span(T(&a)[N]) noexcept : span(a, N) {}

            T* data() noexcept { return m_begin; }
            T* begin() noexcept { return m_begin; }
            T* end() noexcept { return m_end; }
            T const* data() const noexcept { return m_begin; }
            T const* begin() const noexcept { return m_begin; }
            T const* end() const noexcept { return m_end; }
            bool empty() const noexcept { return m_end == m_begin; }
            size_t size() const noexcept { return m_end - m_begin; }
            size_t size_bytes() const noexcept { return sizeof(T) * size(); }
            T& operator[](size_t index) const noexcept { return m_begin[index]; }
            span<T> subspan(size_t index, size_t count) { return span<T>(m_begin + index, m_begin + index + count); }

        protected:
            T* m_begin = nullptr;
            T* m_end = nullptr;
        };
    }

#if DMLX_USE_ABSEIL
    template <typename T>
    using Optional = absl::optional<T>;

    constexpr absl::nullopt_t NullOpt = absl::nullopt;

    template <typename T, size_t N>
    using SmallVector = absl::InlinedVector<T, N>;

    template <typename T>
    using Span = absl::Span<T>;

    using absl::make_unique;
#else
    #ifndef DMLX_OPTIONAL_EXTENDED
        template <typename T>
            using Optional = std::optional<T>;
            constexpr std::nullopt_t NullOpt = std::nullopt;
    #endif

    template <typename T, size_t N>
    using SmallVector = std::vector<T>;

    #if __cpp_lib_span
        template <typename T>
        using Span = std::span<T>;
    #elif DMLX_USE_GSL
        template <typename T>
        using Span = gsl::span<T>;
    #else
        template <typename T>
        using Span = dml::detail::span<T>;
    #endif

    using std::make_unique;
#endif

#if __cpp_exceptions
    #if DMLX_USE_WIL
        #define DMLX_THROW_IF_FAILED(_hr) THROW_IF_FAILED(_hr)
        #define DMLX_THROW(_hr) THROW_HR(_hr)
    #else
        #define DMLX_THROW_IF_FAILED(_hr) if (FAILED(_hr)) { throw std::runtime_error(#_hr); }
        #define DMLX_THROW(_hr) throw std::runtime_error(#_hr);
    #endif
#else
    #define DMLX_THROW_IF_FAILED(_hr) if (FAILED(_hr)) { std::abort(); }
    #define DMLX_THROW(_hr) { std::abort(); }
#endif

    class Graph;
    class Expression;

    using TensorDimensions = SmallVector<uint32_t, 4>;
    using TensorStrides = SmallVector<uint32_t, 4>;

    // The custom properties returned by a TensorPolicy.
    struct TensorProperties
    {
        Optional<TensorStrides> strides;
        uint64_t totalTensorSizeInBytes;
        uint32_t guaranteedBaseOffsetAlignment;
    };

    // Provides a way to customize the properties that DMLX automatically sets on tensors. Callers may provide their
    // own TensorPolicy implementation to provide custom strides, total tensor sizes, and alignment. TensorPolicy
    // objects can be set using Graph::SetTensorPolicy().
    class TensorPolicy
    {
    public:
        // A function type that returns a TensorProperties object given a tensor data type, flags, and sizes.
        using Func = std::function<
            TensorProperties (DML_TENSOR_DATA_TYPE dataType, DML_TENSOR_FLAGS flags, Span<const uint32_t> sizes)
            >;

        TensorPolicy() = default;
        /*implicit*/ TensorPolicy(Func impl)
            : m_impl(impl)
        {}

        TensorProperties Get(
            DML_TENSOR_DATA_TYPE dataType,
            DML_TENSOR_FLAGS flags,
            Span<const uint32_t> sizes) const
        {
            // Empty/uninitialized policy falls back to default.
            if (!m_impl)
            {
                return ComputeDefault(dataType, flags, sizes);
            }

            return m_impl(dataType, flags, sizes);
        }

        // Returns the default tensor policy, which doesn't produce any changes to tensor layout, has no guaranteed
        // alignment, and which uses DMLCalcBufferTensorSize to compute the total tensor size.
        static TensorPolicy Default()
        {
            return TensorPolicy();
        }

        // A tensor policy that returns strides which produce tensors with a layout transposed to dimension order
        // (0, 2, ..., n, 1). This is often referred to as "NHWC" or "interleaved channel" layout. This is useful,
        // for example, when applied to 2D Convolution to produce outputs in an NHWC layout (as opposed to NCHW, which
        // is the DirectML default for 2D Convolution).
        //
        // Examples of the transposes produced by this policy:
        //   NCW -> NWC
        //   NCHW -> NHWC
        //   NCDHW -> NDHWC
        static TensorPolicy InterleavedChannel()
        {
            return TensorPolicy(&ComputeInterleavedChannel);
        }

    private:
        static TensorProperties ComputeDefault(
            DML_TENSOR_DATA_TYPE dataType,
            DML_TENSOR_FLAGS /*flags*/,
            Span<const uint32_t> sizes)
        {
            uint32_t dimensionCount = static_cast<uint32_t>(sizes.size());
            TensorProperties props;
            props.strides = NullOpt; // no strides
            props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimensionCount, sizes.data(), nullptr);
            props.guaranteedBaseOffsetAlignment = 0;
            return props;
        }

        static TensorProperties ComputeInterleavedChannel(
            DML_TENSOR_DATA_TYPE dataType,
            DML_TENSOR_FLAGS /*flags*/,
            Span<const uint32_t> sizes)
        {
            uint32_t dimensionCount = static_cast<uint32_t>(sizes.size());
            TensorStrides strides(dimensionCount);

            enum Axes { N, C, /* spatial dimensions ... */ };

            // N dimension strides
            if (dimensionCount >= 1)
            {
                strides[N] = 1;
                for (uint32_t i = 1; i < dimensionCount; ++i)
                {
                    strides[N] *= sizes[i];
                }
            }

            // C dimension strides
            if (dimensionCount >= 2)
            {
                strides[C] = 1;
            }

            // Spatial dimension strides
            if (dimensionCount >= 3)
            {
                uint32_t stride = sizes[C];
                for (uint32_t i = dimensionCount - 1; i >= 2; --i)
                {
                    strides[i] = stride;
                    stride *= sizes[i];
                }
            }

            TensorProperties props;
            props.strides = std::move(strides);
            props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimensionCount, sizes.data(), props.strides->data());
            props.guaranteedBaseOffsetAlignment = 0;
            return props;
        }

        Func m_impl;
    };

    struct TensorDesc
    {
    public:
        using Dimensions = TensorDimensions;
        using Strides = TensorStrides;

        DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
        DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
        Dimensions sizes;
        Optional<Strides> strides;
        uint64_t totalTensorSizeInBytes = 0;
        uint32_t guaranteedBaseOffsetAlignment = 0;

        TensorDesc() = default;

        TensorDesc(DML_TENSOR_DATA_TYPE dataType, Dimensions sizes, const TensorPolicy& policy = {})
            : TensorDesc(dataType, DML_TENSOR_FLAG_NONE, sizes, policy)
        {}

        TensorDesc(DML_TENSOR_DATA_TYPE dataType, DML_TENSOR_FLAGS flags, Dimensions sizes, const TensorPolicy& policy = {})
        {
            TensorProperties props = policy.Get(dataType, flags, sizes);
            Initialize(
                dataType,
                flags,
                std::move(sizes),
                std::move(props.strides),
                props.totalTensorSizeInBytes,
                props.guaranteedBaseOffsetAlignment);
        }

        TensorDesc(
            DML_TENSOR_DATA_TYPE dataType,
            DML_TENSOR_FLAGS flags,
            Dimensions sizes,
            Optional<Dimensions> strides,
            uint64_t totalTensorSizeInBytes,
            uint32_t guaranteedBaseOffsetAlignment)
        {
            Initialize(dataType, flags, std::move(sizes), std::move(strides), totalTensorSizeInBytes, guaranteedBaseOffsetAlignment);
        }

        /* implicit */ TensorDesc(const DML_TENSOR_DESC& desc)
            : TensorDesc(*static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc))
        {
            assert(desc.Type == DML_TENSOR_TYPE_BUFFER);
            assert(desc.Desc != nullptr);
        }

        /* implicit */ TensorDesc(const DML_BUFFER_TENSOR_DESC& desc)
        {
            this->dataType = desc.DataType;
            this->flags = desc.Flags;
            this->sizes.assign(desc.Sizes, desc.Sizes + desc.DimensionCount);
            if (desc.Strides)
            {
                this->strides.emplace();
                this->strides->assign(desc.Strides, desc.Strides + desc.DimensionCount);
            }
            this->totalTensorSizeInBytes = desc.TotalTensorSizeInBytes;
            this->guaranteedBaseOffsetAlignment = desc.GuaranteedBaseOffsetAlignment;
        }

        // Returns an equivalent DML_TENSOR_DESC or DML_BUFFER_TENSOR_DESC. The returned object contains pointers
        // into the TensorDesc, so it is only valid as long as the TensorDesc itself is alive.
        template <typename T>
        T* AsPtr()
        {
            // "sizeof(T) == -1" is always false; this is just to make the static_assert dependent on the template
            // parameter and therefore not evaluated until template instantiation
            static_assert(sizeof(T) == -1, "Invalid type");
        }

        template <>
        DML_BUFFER_TENSOR_DESC* AsPtr<DML_BUFFER_TENSOR_DESC>()
        {
            assert(!strides || sizes.size() == strides->size());

            m_bufferDesc.DataType = this->dataType;
            m_bufferDesc.Flags = this->flags;
            m_bufferDesc.DimensionCount = static_cast<UINT>(sizes.size());
            m_bufferDesc.Sizes = this->sizes.data();
            m_bufferDesc.Strides = this->strides ? this->strides->data() : nullptr;
            m_bufferDesc.TotalTensorSizeInBytes = this->totalTensorSizeInBytes;
            m_bufferDesc.GuaranteedBaseOffsetAlignment = this->guaranteedBaseOffsetAlignment;
            return &m_bufferDesc;
        }

        template <>
        DML_TENSOR_DESC* AsPtr<DML_TENSOR_DESC>()
        {
            m_tensorDesc = DML_TENSOR_DESC{ DML_TENSOR_TYPE_BUFFER, AsPtr<DML_BUFFER_TENSOR_DESC>() };
            return &m_tensorDesc;
        }

    private:
        DML_BUFFER_TENSOR_DESC m_bufferDesc;
        DML_TENSOR_DESC m_tensorDesc;

        void Initialize(
            DML_TENSOR_DATA_TYPE tensorDataType,
            DML_TENSOR_FLAGS tensorFlags,
            Dimensions tensorSizes,
            Optional<Dimensions> tensorStrides,
            uint64_t totalTensorSizeInBytesVal,
            uint32_t guaranteedBaseOffsetAlignmentVal)
        {
            assert(!tensorStrides || tensorStrides->size() == static_cast<uint32_t>(tensorSizes.size()));

            this->dataType = tensorDataType;
            this->flags = tensorFlags;
            this->sizes = std::move(tensorSizes);
            this->strides = std::move(tensorStrides);
            this->totalTensorSizeInBytes = totalTensorSizeInBytesVal;
            this->guaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignmentVal;
        }
    };

    namespace detail
    {
        class GraphBuilder;
        class NodeOutput;

        // A node in the graph which represents a graph input.
        struct InputNode
        {
            uint32_t inputIndex;
        };

        // A node in the graph which represents a DML operator.
        struct OperatorNode
        {
            Microsoft::WRL::ComPtr<IDMLOperator> op;

            // The inputs to this node
            std::vector<NodeOutput*> inputs;
        };

        // Used for representing reshapes and type punning
        struct ReinterpretNode
        {
            NodeOutput* input;
        };

        enum class NodeType
        {
            Invalid,
            Input,
            Operator,
            Reinterpret,
        };

        // Identifies a node in the graph.
        struct NodeID
        {
            NodeType type;
            uint32_t index; // The index of this node in the GraphBuilder
        };

        // Represents one of the outputs of a node.
        class NodeOutput
        {
        public:
            NodeOutput(GraphBuilder* owner, NodeID node, uint32_t outputIndex, TensorDesc tensorDesc)
                : m_owner(owner)
                , m_node(node)
                , m_outputIndex(outputIndex)
                , m_tensorDesc(std::move(tensorDesc))
            {}

            // Retrieves the GraphBuilder that owns this object.
            GraphBuilder* GetGraphBuilder() const { return m_owner; }

            NodeID GetNode() const { return m_node; }
            uint32_t GetOutputIndex() const { return m_outputIndex; }
            const TensorDesc& GetOutputDesc() const { return m_tensorDesc; }

        private:
            GraphBuilder* m_owner;
            NodeID m_node;

            // An operator can have multiple outputs; this index identifies which one of the operator's  outputs this
            // NodeOutput represents.
            uint32_t m_outputIndex;

            TensorDesc m_tensorDesc;
        };

        struct GraphDesc
        {
            uint32_t inputCount;
            uint32_t outputCount;
            std::vector<DML_OPERATOR_GRAPH_NODE_DESC> nodes;
            std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
            std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        };

        class GraphBuilder
        {
        public:
            GraphBuilder(IDMLDevice* device, TensorPolicy tensorPolicy = {})
                : m_device(device)
                , m_tensorPolicy(tensorPolicy)
            {}

            IDMLDevice* GetDevice() const
            {
                return m_device.Get();
            }

            void SetTensorPolicy(TensorPolicy policy) { m_tensorPolicy = std::move(policy); }
            const TensorPolicy& GetTensorPolicy() const { return m_tensorPolicy; }
            TensorPolicy& GetTensorPolicy() { return m_tensorPolicy; }

            // Creates a DML operator node owned by this graph builder and returns a NodeInfo identifier. The
            // inputs to this node must be supplied in the correct order matching the DML operator.
            NodeID CreateOperatorNode(DML_OPERATOR_TYPE type, const void* desc, Span<NodeOutput* const> inputs);
            NodeID CreateInputNode(uint32_t inputIndex);
            NodeID CreateReinterpretNode(NodeOutput* input);
            NodeOutput* CreateNodeOutput(NodeID node, uint32_t outputIndex, TensorDesc tensorDesc);
            GraphDesc GetGraphDesc(Span<const Expression> outputs) const;

        private:
            Microsoft::WRL::ComPtr<IDMLDevice> m_device;
            TensorPolicy m_tensorPolicy;
            std::vector<InputNode> m_inputNodes;
            std::vector<OperatorNode> m_operatorNodes;
            std::vector<ReinterpretNode> m_reinterpretNodes;
            std::deque<NodeOutput> m_nodeOutputs; // deque doesn't invalidate references to elements when it resizes
        };

    } // namespace detail

    class Expression
    {
    public:
        /*implicit*/ Expression(detail::NodeOutput* nodeOutput = nullptr)
            : m_nodeOutput(nodeOutput)
        {}

        // Returns a struct containing the required properties of the tensor to hold the output of this expression,
        // once evaluated.
        const TensorDesc& GetOutputDesc() const { return Impl()->GetOutputDesc(); }

        // For internal use only
        detail::NodeOutput* Impl() const { return m_nodeOutput; }

        explicit operator bool() const
        {
            return m_nodeOutput != nullptr;
        }

    private:
        detail::NodeOutput* m_nodeOutput; // weak; this is owned by the GraphBuilder
    };

    class Graph
    {
    public:
        explicit Graph(IDMLDevice* device, TensorPolicy tensorPolicy = {})
            : m_graphBuilder(make_unique<detail::GraphBuilder>(device, tensorPolicy))
        {}

        // For internal use only
        detail::GraphBuilder* Impl() { return m_graphBuilder.get(); }

        // Sets/gets the tensor policy. If not set, defaults to TensorPolicy::Default(). Tensor policies can be used
        // to control properties (such as strides) on output tensors produced by this Graph.
        void SetTensorPolicy(TensorPolicy policy) { m_graphBuilder->SetTensorPolicy(std::move(policy)); }
        const TensorPolicy& GetTensorPolicy() const { return m_graphBuilder->GetTensorPolicy(); }
        TensorPolicy& GetTensorPolicy() { return m_graphBuilder->GetTensorPolicy(); }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> Compile(
            DML_EXECUTION_FLAGS flags,
            Span<const Expression> outputs,
            uint32_t inputCount = 0) const
        {
            detail::GraphDesc graph = m_graphBuilder->GetGraphDesc(outputs);

            // If supplied, the requested number of inputs to the compiled operator can be larger than the actual
            // number of input nodes on the graph (e.g. in the case of unused empty inputs), but never smaller.
            assert(inputCount == 0 || inputCount >= graph.inputCount);

            std::vector<DML_GRAPH_NODE_DESC> graphNodes(graph.nodes.size());
            for (size_t i = 0; i < graphNodes.size(); ++i)
            {
                graphNodes[i] = { DML_GRAPH_NODE_TYPE_OPERATOR, &graph.nodes[i] };
            }

            std::vector<DML_GRAPH_EDGE_DESC> inputEdges(graph.inputEdges.size());
            for (size_t i = 0; i < inputEdges.size(); ++i)
            {
                inputEdges[i] = { DML_GRAPH_EDGE_TYPE_INPUT, &graph.inputEdges[i] };
            }

            std::vector<DML_GRAPH_EDGE_DESC> outputEdges(graph.outputEdges.size());
            for (size_t i = 0; i < outputEdges.size(); ++i)
            {
                outputEdges[i] = { DML_GRAPH_EDGE_TYPE_OUTPUT, &graph.outputEdges[i] };
            }

            std::vector<DML_GRAPH_EDGE_DESC> intermediateEdges(graph.intermediateEdges.size());
            for (size_t i = 0; i < intermediateEdges.size(); ++i)
            {
                intermediateEdges[i] = { DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &graph.intermediateEdges[i] };
            }

            DML_GRAPH_DESC graphDesc = {};
            graphDesc.InputCount = inputCount ? inputCount : graph.inputCount;
            graphDesc.OutputCount = graph.outputCount;
            graphDesc.NodeCount = static_cast<UINT>(graphNodes.size());
            graphDesc.Nodes = graphNodes.data();
            graphDesc.InputEdgeCount = static_cast<UINT>(inputEdges.size());
            graphDesc.InputEdges = inputEdges.data();
            graphDesc.OutputEdgeCount = static_cast<UINT>(outputEdges.size());
            graphDesc.OutputEdges = outputEdges.data();
            graphDesc.IntermediateEdgeCount = static_cast<UINT>(intermediateEdges.size());
            graphDesc.IntermediateEdges = intermediateEdges.data();

            Microsoft::WRL::ComPtr<IDMLDevice1> device1;
            DMLX_THROW_IF_FAILED(m_graphBuilder->GetDevice()->QueryInterface(IID_PPV_ARGS(&device1)));

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledGraph;
            DMLX_THROW_IF_FAILED(device1->CompileGraph(&graphDesc, flags, IID_PPV_ARGS(&compiledGraph)));

            return compiledGraph;
        }

    private:
        std::unique_ptr<detail::GraphBuilder> m_graphBuilder;
    };

    // Represents an activation to be fused with an existing operator. The meaning of param1 and param2 depend on the
    // activation to be fused.
    //
    // For HARD_SIGMOID, LINEAR, PARAMETRIC_SOFTPLUS, and SCALED_TANH: param1 = Alpha and param2 = Beta
    // For ELU, LEAKY_RELU, THRESHOLDED_RELU, and CELU: param1 = Alpha. param2 is unused.
    // For SCALED_ELU, param1 = Alpha and param2 = Gamma.
    // For SHRINK, param1 = Bias and param2 = Threshold
    // For SOFTPLUS, param1 = Steepness.
    // For all other activations, both param1 and param2 are unused.
    struct FusedActivation
    {
        DML_OPERATOR_TYPE activation = DML_OPERATOR_INVALID;
        float param1 = 0.0f;
        float param2 = 0.0f;

        FusedActivation() = default;

        explicit FusedActivation(DML_OPERATOR_TYPE activation, float param1 = 0.0f, float param2 = 0.0f)
            : activation(activation), param1(param1), param2(param2)
        {}

        static FusedActivation None()
        {
            return FusedActivation();
        }

        static FusedActivation Elu(float alpha = 1.0f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_ELU, alpha);
        }

        static FusedActivation HardSigmoid(float alpha = 0.2f, float beta = 0.5f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_HARD_SIGMOID, alpha, beta);
        }

        static FusedActivation Identity()
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_IDENTITY);
        }

        static FusedActivation LeakyRelu(float alpha = 0.01f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_LEAKY_RELU, alpha);
        }

        static FusedActivation Linear(float alpha, float beta)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_LINEAR, alpha, beta);
        }

        static FusedActivation ParametricSoftplus(float alpha, float beta)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS, alpha, beta);
        }

        static FusedActivation Relu()
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_RELU);
        }

        static FusedActivation ScaledElu(float alpha = 1.67326319217681884765625f, float gamma = 1.05070102214813232421875f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_SCALED_ELU, alpha, gamma);
        }

        static FusedActivation ScaledTanh(float alpha = 1.0f, float beta = 0.5f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_SCALED_TANH, alpha, beta);
        }

        static FusedActivation Sigmoid()
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_SIGMOID);
        }

        static FusedActivation Softplus(float steepness = 1.0f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_SOFTPLUS, steepness);
        }

        static FusedActivation Softsign()
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_SOFTSIGN);
        }

        static FusedActivation Tanh()
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_TANH);
        }

        static FusedActivation ThresholdedRelu(float alpha = 1.0f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU, alpha);
        }

        static FusedActivation Shrink(float bias = 0.0f, float threshold = 0.5f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_SHRINK, bias, threshold);
        }

        static FusedActivation Celu(float alpha = 1.0f)
        {
            return FusedActivation(DML_OPERATOR_ACTIVATION_CELU, alpha);
        }
    };

    // Implementation detail helper for determining if a list of expressions share the same GraphBuilder.
    namespace detail
    {
        inline bool HasSameOwner(Span<const Expression> exprs)
        {
            if (exprs.size() == 0)
            {
                return true;
            }

            detail::GraphBuilder* owner = exprs.begin()->Impl()->GetGraphBuilder();
            for (Expression expr : exprs)
            {
                if (expr.Impl()->GetGraphBuilder() != owner)
                {
                    return false;
                }
            }

            return true;
        }

        inline bool HasSameOwner(std::initializer_list<Expression> exprs)
        {
            Span<const Expression> span(exprs.begin(), exprs.size());
            return HasSameOwner(span);
        }

        inline bool HasSameDataType(Span<const Expression> exprs)
        {
            if (exprs.size() == 0)
            {
                return true;
            }

            DML_TENSOR_DATA_TYPE dataType = exprs.begin()->Impl()->GetOutputDesc().dataType;
            for (Expression expr : exprs)
            {
                if (expr.Impl()->GetOutputDesc().dataType != dataType)
                {
                    return false;
                }
            }

            return true;
        }

        inline bool HasSameDataType(std::initializer_list<Expression> exprs)
        {
            Span<const Expression> span(exprs.begin(), exprs.size());
            return HasSameDataType(span);
        }
    } // namespace detail

    // Expression implementation helpers
    namespace detail
    {
        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseUnary(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias)
        {
            detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

            TensorDesc inputTensor = input.Impl()->GetOutputDesc();
            TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

            TDesc desc = {};
            desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;

            detail::NodeOutput* const inputs[] = { input.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseUnary(Expression input, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UNKNOWN)
        {
            detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

            TensorDesc inputTensor = input.Impl()->GetOutputDesc();

            if (outputDataType == DML_TENSOR_DATA_TYPE_UNKNOWN)
            {
                outputDataType = inputTensor.dataType;
            }
            TensorDesc outputTensor(outputDataType, inputTensor.sizes, builder->GetTensorPolicy());

            TDesc desc = {};
            desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

            detail::NodeOutput* const inputs[] = { input.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseBinary(Expression a, Expression b)
        {
            assert(detail::HasSameOwner({ a, b }));
            detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

            TensorDesc aTensor = a.Impl()->GetOutputDesc();
            TensorDesc bTensor = b.Impl()->GetOutputDesc();
            TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetTensorPolicy()); // Same as input

            TDesc desc = {};
            desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
            desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

            detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseComparison(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
        {
            assert(detail::HasSameOwner({ a, b }));
            detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

            TensorDesc aTensor = a.Impl()->GetOutputDesc();
            TensorDesc bTensor = b.Impl()->GetOutputDesc();
            TensorDesc outputTensor(outputDataType, aTensor.sizes, builder->GetTensorPolicy());

            TDesc desc = {};
            desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
            desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

            detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        // Used to reserve some space on the stack for setting up fused activation operator descs.
        struct FusedActivationStorage
        {
            DML_OPERATOR_DESC opDesc;

            // All fuseable activation descs have a common layout: two tensor desc pointers and up to 2 optional
            // float parameters, so just use LINEAR as an archetype
            DML_ACTIVATION_LINEAR_OPERATOR_DESC activationDesc;
        };

        // Returns the correct value for filling out fused activation fields in the DML API, e.g.
        // DML_CONVOLUTION_OPERATOR_DESC::FusedActivation. The descs themselves are stored in the `storage` outptr.
        inline const DML_OPERATOR_DESC* GetFusedActivationPtr(
            FusedActivation fusedActivation,
            _Out_ FusedActivationStorage* storage)
        {
            if (fusedActivation.activation == DML_OPERATOR_INVALID)
            {
                // No fused activation
                return nullptr;
            }

            storage->activationDesc.InputTensor = nullptr;
            storage->activationDesc.OutputTensor = nullptr;
            storage->activationDesc.Alpha = fusedActivation.param1;
            storage->activationDesc.Beta = fusedActivation.param2;

            storage->opDesc.Type = fusedActivation.activation;
            storage->opDesc.Desc = &storage->activationDesc;

            return &storage->opDesc;
        }

    } // namespace detail

    inline Expression InputTensor(Graph& graph, uint32_t inputIndex, TensorDesc desc)
    {
        detail::GraphBuilder* builder = graph.Impl();

        detail::NodeID node = builder->CreateInputNode(inputIndex);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(desc));
        return output;
    }

    inline Expression Identity(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_IDENTITY, DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Abs(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ABS, DML_ELEMENT_WISE_ABS_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ACos(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ACOS, DML_ELEMENT_WISE_ACOS_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Add(Expression a, Expression b)
    {
        assert(detail::HasSameOwner({ a, b }));
        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_ADD_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_ADD, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Add(Expression a, Expression b, FusedActivation fusedActivation)
    {
        assert(detail::HasSameOwner({ a, b }));
        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetTensorPolicy()); // Same as input
        detail::FusedActivationStorage storage;

        DML_ELEMENT_WISE_ADD1_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_ADD1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ASin(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ASIN, DML_ELEMENT_WISE_ASIN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ATan(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ATAN, DML_ELEMENT_WISE_ATAN_OPERATOR_DESC>(input, scaleBias);
    }

#if DML_TARGET_VERSION >= 0x3100

    inline Expression ATanYX(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_ATAN_YX, DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC>(a, b);
    }

#endif // DML_TARGET_VERSION >= 0x3100

    inline Expression Ceil(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_CEIL, DML_ELEMENT_WISE_CEIL_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Clip(Expression input, float min, float max, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_CLIP_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;
        desc.Min = min;
        desc.Max = max;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_CLIP, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

#if DML_TARGET_VERSION >= 0x3100

    inline Expression ClipGrad(Expression input, Expression inputGradient, float min, float max)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc inputGradientTensor = inputGradient.Impl()->GetOutputDesc();
        TensorDesc outputGradientTensor(inputGradientTensor.dataType, inputGradientTensor.sizes, builder->GetTensorPolicy());

        DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InputGradientTensor = inputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputGradientTensor = outputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Min = min;
        desc.Max = max;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputGradientTensor));

        return output;
    }

#endif // DML_TARGET_VERSION >= 0x3100

    inline Expression Cos(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_COS, DML_ELEMENT_WISE_COS_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Divide(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_DIVIDE, DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>(a, b);
    }

    inline Expression Exp(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_EXP, DML_ELEMENT_WISE_EXP_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Floor(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_FLOOR, DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Log(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_LOG, DML_ELEMENT_WISE_LOG_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression LogicalAnd(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND, DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC>(a, b);
    }

    inline Expression Equals(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseComparison<
            DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS,
            DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC>(a, b, outputDataType);
    }

    inline Expression GreaterThan(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseComparison<
            DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN,
            DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC>(a, b, outputDataType);
    }

    inline Expression GreaterThanOrEqual(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseComparison<
            DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL,
            DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC>(a, b, outputDataType);
    }

    inline Expression LessThan(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseComparison<
            DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN,
            DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC>(a, b, outputDataType);
    }

    inline Expression LessThanOrEqual(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseComparison<
            DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL,
            DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC>(a, b, outputDataType);
    }

    inline Expression LogicalNot(Expression input)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression LogicalOr(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR, DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC>(a, b);
    }

    inline Expression LogicalXor(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR, DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC>(a, b);
    }

    inline Expression Max(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MAX, DML_ELEMENT_WISE_MAX_OPERATOR_DESC>(a, b);
    }

    inline Expression Mean(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MEAN, DML_ELEMENT_WISE_MEAN_OPERATOR_DESC>(a, b);
    }

    inline Expression Min(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MIN, DML_ELEMENT_WISE_MIN_OPERATOR_DESC>(a, b);
    }

    inline Expression Multiply(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MULTIPLY, DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>(a, b);
    }

    inline Expression Pow(Expression input, Expression exponent, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        assert(detail::HasSameOwner({ input, exponent }));
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc exponentTensor = exponent.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_POW_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ExponentTensor = exponentTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;

        detail::NodeOutput* const inputs[] = { input.Impl(), exponent.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_POW, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Pow(Expression input, float exponent, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;
        desc.Exponent = exponent;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Recip(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_RECIP, DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Sin(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SIN, DML_ELEMENT_WISE_SIN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Sqrt(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SQRT, DML_ELEMENT_WISE_SQRT_OPERATOR_DESC>(input, scaleBias);
    }

#if DML_TARGET_VERSION >= 0x3100

    inline Expression DifferenceSquare(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE, DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC>(a, b);
    }

#endif // DML_TARGET_VERSION >= 0x3100

    inline Expression Subtract(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_SUBTRACT, DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>(a, b);
    }

    inline Expression Tan(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_TAN, DML_ELEMENT_WISE_TAN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Threshold(Expression input, float min, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;
        desc.Min = min;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_THRESHOLD, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression QuantizeLinear(Expression input, Expression scale, Expression zeroPoint, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        assert(detail::HasSameOwner({ input, scale, zeroPoint }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc zeroPointTensor = zeroPoint.Impl()->GetOutputDesc();
        TensorDesc outputTensor(outputDataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ZeroPointTensor = zeroPointTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl(), scale.Impl(), zeroPoint.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression DequantizeLinear(Expression input, Expression scale, Expression zeroPoint)
    {
        assert(detail::HasSameOwner({ input, scale, zeroPoint }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc zeroPointTensor = zeroPoint.Impl()->GetOutputDesc();
        TensorDesc outputTensor(DML_TENSOR_DATA_TYPE_FLOAT32, inputTensor.sizes, builder->GetTensorPolicy());

        DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ZeroPointTensor = zeroPointTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl(), scale.Impl(), zeroPoint.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Sign(Expression a)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SIGN, DML_ELEMENT_WISE_SIGN_OPERATOR_DESC>(a);
    }

    inline Expression IsNaN(Expression input, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_IS_NAN, DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC>(input, outputDataType);
    }

    inline Expression Erf(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ERF, DML_ELEMENT_WISE_ERF_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Sinh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SINH, DML_ELEMENT_WISE_SINH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Cosh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_COSH, DML_ELEMENT_WISE_COSH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Tanh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_TANH, DML_ELEMENT_WISE_TANH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ASinh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ASINH, DML_ELEMENT_WISE_ASINH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ACosh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ACOSH, DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ATanh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ATANH, DML_ELEMENT_WISE_ATANH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression If(Expression condition, Expression a, Expression b)
    {
        assert(detail::HasSameOwner({ condition, a, b }));
        assert(detail::HasSameDataType({ a, b }));

        detail::GraphBuilder* builder = condition.Impl()->GetGraphBuilder();

        TensorDesc conditionTensor = condition.Impl()->GetOutputDesc();
        assert(conditionTensor.dataType == DML_TENSOR_DATA_TYPE_UINT8);

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetTensorPolicy());

        DML_ELEMENT_WISE_IF_OPERATOR_DESC desc = {};
        desc.ConditionTensor = conditionTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { condition.Impl(), a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_IF, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression BitShiftLeft(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT, DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC>(a, b);
    }

    inline Expression BitShiftRight(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT, DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC>(a, b);
    }

    inline Expression BitAnd(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_AND, DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC>(a, b);
    }

    inline Expression BitOr(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_OR, DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>(a, b);
    }

    inline Expression BitXor(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_XOR, DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>(a, b);
    }

    inline Expression BitNot(Expression a)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_BIT_NOT, DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC>(a);
    }

    inline Expression BitCount(Expression a, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_BIT_COUNT, DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC>(a, outputDataType);
    }

    inline Expression Round(Expression input, DML_ROUNDING_MODE roundingMode = DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); // Same as input

        DML_ELEMENT_WISE_ROUND_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.RoundingMode = roundingMode;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_ROUND, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression IsInfinity(
        Expression input,
        DML_IS_INFINITY_MODE infinityMode = DML_IS_INFINITY_MODE_EITHER,
        DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(outputDataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InfinityMode = infinityMode;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_IS_INFINITY, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ModulusTruncate(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE, DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC>(a, b);
    }

    inline Expression ModulusFloor(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR, DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC>(a, b);
    }

#pragma region detail
#define DMLX_ACTIVATION_IMPL(_name) \
    do { \
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder(); \
        \
        TensorDesc inputTensor = input.Impl()->GetOutputDesc(); \
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); \
        \
        DML_##_name##_OPERATOR_DESC desc = {}; \
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>(); \
        \
        detail::NodeOutput* const inputs[] = { input.Impl() }; \
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_##_name, &desc, inputs); \
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor)); \
        \
        return output; \
    } while(0)

#define DMLX_ACTIVATION_IMPL_1(_name, _param1Name, _param1) \
    do { \
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder(); \
        \
        TensorDesc inputTensor = input.Impl()->GetOutputDesc(); \
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); \
        \
        DML_##_name##_OPERATOR_DESC desc = {}; \
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc._param1Name = _param1; \
        \
        detail::NodeOutput* const inputs[] = { input.Impl() }; \
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_##_name, &desc, inputs); \
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor)); \
        \
        return output; \
    } while(0)

#define DMLX_ACTIVATION_IMPL_2(_name, _param1Name, _param1, _param2Name, _param2) \
    do { \
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder(); \
        \
        TensorDesc inputTensor = input.Impl()->GetOutputDesc(); \
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy()); \
        \
        DML_##_name##_OPERATOR_DESC desc = {}; \
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc._param1Name = _param1; \
        desc._param2Name = _param2; \
        \
        detail::NodeOutput* const inputs[] = { input.Impl() }; \
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_##_name, &desc, inputs); \
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor)); \
        \
        return output; \
    } while(0)
#pragma endregion

    inline Expression ActivationElu(Expression input, float alpha = 1.0f)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_ELU, Alpha, alpha);
    }

    inline Expression ActivationHardmax(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_HARDMAX);
    }

    inline Expression ActivationHardSigmoid(Expression input, float alpha = 0.2f, float beta = 0.5f)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_HARD_SIGMOID, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationIdentity(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_IDENTITY);
    }

    inline Expression ActivationLeakyRelu(Expression input, float alpha = 0.01f)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_LEAKY_RELU, Alpha, alpha);
    }

    inline Expression ActivationLinear(Expression input, float alpha, float beta)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_LINEAR, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationLogSoftmax(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_LOG_SOFTMAX);
    }

    inline Expression ActivationParameterizedRelu(Expression input, Expression slope)
    {
        assert(detail::HasSameOwner({ input, slope }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc slopeTensor = slope.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.SlopeTensor = slopeTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl(), slope.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ActivationParametricSoftplus(Expression input, float alpha, float beta)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_PARAMETRIC_SOFTPLUS, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationRelu(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_RELU);
    }

    inline Expression ActivationScaledElu(Expression input, float alpha = 1.67326319217681884765625f, float gamma = 1.05070102214813232421875f)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_SCALED_ELU, Alpha, alpha, Gamma, gamma);
    }

    inline Expression ActivationScaledTanh(Expression input, float alpha = 1.0f, float beta = 0.5f)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_SCALED_TANH, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationSigmoid(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_SIGMOID);
    }

    inline Expression ActivationSoftmax(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_SOFTMAX);
    }

    inline Expression ActivationSoftplus(Expression input, float steepness = 1.0f)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_SOFTPLUS, Steepness, steepness);
    }

    inline Expression ActivationSoftsign(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_SOFTSIGN);
    }

    inline Expression ActivationTanh(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_TANH);
    }

    inline Expression ActivationThresholdedRelu(Expression input, float alpha = 1.0f)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_THRESHOLDED_RELU, Alpha, alpha);
    }

    inline Expression ActivationShrink(Expression input, float bias = 0.0f, float threshold = 0.5f)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_SHRINK, Bias, bias, Threshold, threshold);
    }

    inline Expression ActivationCelu(Expression input, float alpha = 1.0f)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_CELU, Alpha, alpha);
    }

#undef DMLX_ACTIVATION_IMPL
#undef DMLX_ACTIVATION_IMPL_1
#undef DMLX_ACTIVATION_IMPL_2

    // ---------------------------------------------------------------------------------------------------------------

    // If not specified, parameters are defaulted to the following values:
    //   Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION
    //   Direction = DML_CONVOLUTION_DIRECTION_FORWARD
    //   Strides = { 1, 1 } for 2D convolution, { 1, 1, 1 } for 3D convolution
    //   Dilations = { 1, 1 } for 2D convolution, { 1, 1, 1 } for 3D convolution
    //   StartPadding = { 0, 0 } for 2D convolution, { 0, 0, 0 } for 3D convolution
    //   EndPadding = { 0, 0 } for 2D convolution, { 0, 0, 0 } for 3D convolution
    //   OutputPadding = { 0, 0 } for 2D convolution, { 0, 0, 0 } for 3D convolution
    //   GroupCount = 1
    //   FusedActivation = nullptr
    //   OutputSizes = computed from other parameters
    inline Expression Convolution(
        Expression input,
        Expression filter,
        Optional<Expression> bias = NullOpt,
        DML_CONVOLUTION_MODE mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        DML_CONVOLUTION_DIRECTION direction = DML_CONVOLUTION_DIRECTION_FORWARD,
        Span<const uint32_t> strides = {},
        Span<const uint32_t> dilations = {},
        Span<const uint32_t> startPadding = {},
        Span<const uint32_t> endPadding = {},
        Span<const uint32_t> outputPadding = {},
        uint32_t groupCount = 1,
        FusedActivation fusedActivation = FusedActivation::None(),
        TensorDimensions outputSizes = {})
    {
        assert(detail::HasSameOwner({ input, filter }));
        assert(!bias || detail::HasSameOwner({ input, *bias }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc filterTensor = filter.Impl()->GetOutputDesc();
        TensorDesc biasTensor;
        if (bias)
        {
            biasTensor = bias->Impl()->GetOutputDesc();
        }

        uint32_t dimensionCount = static_cast<uint32_t>(inputTensor.sizes.size());

        assert(dimensionCount == 4 || dimensionCount == 5);
        uint32_t spatialDimensionCount = dimensionCount - 2;

        // If the spatial dimension count is 2, we'll just use the first two elements by setting
        // DimensionCount = 2 in the desc
        const uint32_t defaultStridesAndDilations[3] = { 1, 1, 1 };
        const uint32_t defaultPadding[3] = { 0, 0, 0 };

        assert(strides.empty() || strides.size() == spatialDimensionCount);
        assert(dilations.empty() || dilations.size() == spatialDimensionCount);
        assert(startPadding.empty() || startPadding.size() == spatialDimensionCount);
        assert(endPadding.empty() || endPadding.size() == spatialDimensionCount);
        assert(outputPadding.empty() || outputPadding.size() == spatialDimensionCount);
        assert(outputSizes.empty() || outputSizes.size() == inputTensor.sizes.size());

        strides = strides.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : strides;
        dilations = dilations.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : dilations;
        startPadding = startPadding.empty() ? Span<const uint32_t>{ defaultPadding } : startPadding;
        endPadding = endPadding.empty() ? Span<const uint32_t>{ defaultPadding } : endPadding;
        outputPadding = outputPadding.empty() ? Span<const uint32_t>{ defaultPadding } : outputPadding;

        // Compute the output shapes

        if (outputSizes.empty())
        {
            if (direction == DML_CONVOLUTION_DIRECTION_FORWARD)
            {
                outputSizes.push_back(inputTensor.sizes[0]); // output[N] = input[N]
                outputSizes.push_back(filterTensor.sizes[0]); // output[C] = filter[N]

                for (uint32_t dim = 0; dim < spatialDimensionCount; ++dim)
                {
                    uint32_t inputSize = inputTensor.sizes[dim + 2];
                    uint32_t paddedSize = inputSize + startPadding[dim] + endPadding[dim];

                    uint32_t windowSize = filterTensor.sizes[dim + 2];
                    uint32_t kernelSize = 1 + (windowSize - 1) * dilations[dim];

                    assert(kernelSize <= paddedSize);
                    assert(strides[dim] != 0);

                    outputSizes.push_back(1 + (paddedSize - kernelSize) / strides[dim]);
                }
            }
            else if (direction == DML_CONVOLUTION_DIRECTION_BACKWARD)
            {
                // TODO: implement me
                assert(false);
            }
            else
            {
                assert(false);
                DMLX_THROW(E_UNEXPECTED);
            }
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());
        detail::FusedActivationStorage storage;

        DML_CONVOLUTION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.FilterTensor = filterTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BiasTensor = bias ? biasTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Mode = mode;
        desc.Direction = direction;
        desc.DimensionCount = spatialDimensionCount;
        desc.Strides = strides.data();
        desc.Dilations = dilations.data();
        desc.StartPadding = startPadding.data();
        desc.EndPadding = endPadding.data();
        desc.OutputPadding = outputPadding.data();
        desc.GroupCount = groupCount;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] = { input.Impl(), filter.Impl(), bias ? bias->Impl() : nullptr };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_CONVOLUTION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    // Helper for setting parameters for the Convolution operator. Sample usage:
    //
    //   auto conv = dml::ConvolutionBuilder(...)
    //        .StartPadding(...)
    //        .EndPadding(...)
    //        .Strides(...)
    //        .Build();
    //
    // Parameters left unspecified will be defaulted with the same values as dml::Convolution().
    class ConvolutionBuilder
    {
    public:
        ConvolutionBuilder(Expression input, Expression filter, Optional<Expression> bias = NullOpt)
            : m_input(input), m_filter(filter), m_bias(bias)
        {}

        ConvolutionBuilder& Mode(DML_CONVOLUTION_MODE mode) { m_mode = mode; return *this; }
        ConvolutionBuilder& Direction(DML_CONVOLUTION_DIRECTION direction) { m_direction = direction; return *this; }
        ConvolutionBuilder& Strides(Span<const uint32_t> strides) { m_strides.assign(strides.begin(), strides.end()); return *this; }
        ConvolutionBuilder& Dilations(Span<const uint32_t> dilations) { m_dilations.assign(dilations.begin(), dilations.end()); return *this; }
        ConvolutionBuilder& StartPadding(Span<const uint32_t> startPadding) { m_startPadding.assign(startPadding.begin(), startPadding.end()); return *this; }
        ConvolutionBuilder& EndPadding(Span<const uint32_t> endPadding) { m_endPadding.assign(endPadding.begin(), endPadding.end()); return *this; }
        ConvolutionBuilder& OutputPadding(Span<const uint32_t> outputPadding) { m_outputPadding.assign(outputPadding.begin(), outputPadding.end()); return *this; }
        ConvolutionBuilder& GroupCount(uint32_t groupCount) { m_groupCount = groupCount; return *this; }
        ConvolutionBuilder& FusedActivation(FusedActivation fusedActivation) { m_fusedActivation = fusedActivation; return *this; }
        ConvolutionBuilder& OutputSizes(TensorDimensions outputSizes) { m_outputSizes = std::move(outputSizes); return *this; }

        Expression Build() const
        {
            return Convolution(
                m_input,
                m_filter,
                m_bias,
                m_mode,
                m_direction,
                m_strides,
                m_dilations,
                m_startPadding,
                m_endPadding,
                m_outputPadding,
                m_groupCount,
                m_fusedActivation,
                m_outputSizes);
        }

    private:
        Expression m_input;
        Expression m_filter;
        Optional<Expression> m_bias;
        DML_CONVOLUTION_MODE m_mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        DML_CONVOLUTION_DIRECTION m_direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        SmallVector<uint32_t, 3> m_strides = {};
        SmallVector<uint32_t, 3> m_dilations = {};
        SmallVector<uint32_t, 3> m_startPadding = {};
        SmallVector<uint32_t, 3> m_endPadding = {};
        SmallVector<uint32_t, 3> m_outputPadding = {};
        uint32_t m_groupCount = 1;
        dml::FusedActivation m_fusedActivation;
        TensorDimensions m_outputSizes = {};
    };

    // ---------------------------------------------------------------------------------------------------------------

    inline Expression Gemm(
        Expression a,
        Expression b,
        Optional<Expression> c = NullOpt,
        DML_MATRIX_TRANSFORM transA = DML_MATRIX_TRANSFORM_NONE,
        DML_MATRIX_TRANSFORM transB = DML_MATRIX_TRANSFORM_NONE,
        float alpha = 1.0f,
        float beta = 1.0f,
        FusedActivation fusedActivation = FusedActivation::None())
    {
        assert(detail::HasSameOwner({ a, b }));
        assert(!c || detail::HasSameOwner({ a, *c }));

        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc cTensor;
        if (c)
        {
            cTensor = c->Impl()->GetOutputDesc();
        }

        TensorDimensions outputSizes;
        outputSizes.push_back(aTensor.sizes[0]); // output[N] = input[N]
        outputSizes.push_back(aTensor.sizes[1]); // output[C] = input[C]
        outputSizes.push_back(transA == DML_MATRIX_TRANSFORM_NONE ? aTensor.sizes[2] : aTensor.sizes[3]);
        outputSizes.push_back(transB == DML_MATRIX_TRANSFORM_NONE ? bTensor.sizes[3] : bTensor.sizes[2]);

        TensorDesc outputTensor(aTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());
        detail::FusedActivationStorage storage;

        DML_GEMM_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.CTensor = c ? cTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.TransA = transA;
        desc.TransB = transB;
        desc.Alpha = alpha;
        desc.Beta = beta;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl(), c ? c->Impl() : nullptr };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GEMM, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    // Helper for setting parameters for the Gemm operator. Parameters left unspecified will be defaulted with the
    // same values as dml::Gemm().
    class GemmBuilder
    {
    public:
        GemmBuilder(Expression a, Expression b, Optional<Expression> c = NullOpt)
            : m_a(a), m_b(b), m_c(c)
        {}

        GemmBuilder& TransA(DML_MATRIX_TRANSFORM transA) { m_transA = transA; return *this; }
        GemmBuilder& TransB(DML_MATRIX_TRANSFORM transB) { m_transB = transB; return *this; }
        GemmBuilder& Alpha(float alpha) { m_alpha = alpha; return *this; }
        GemmBuilder& Beta(float beta) { m_beta = beta; return *this; }
        GemmBuilder& FusedActivation(FusedActivation fusedActivation) { m_fusedActivation = fusedActivation; return *this; }

        Expression Build() const
        {
            return Gemm(m_a, m_b, m_c, m_transA, m_transB, m_alpha, m_beta, m_fusedActivation);
        }

    private:
        Expression m_a;
        Expression m_b;
        Optional<Expression> m_c;
        DML_MATRIX_TRANSFORM m_transA = DML_MATRIX_TRANSFORM_NONE;
        DML_MATRIX_TRANSFORM m_transB = DML_MATRIX_TRANSFORM_NONE;
        float m_alpha = 1.0f;
        float m_beta = 1.0f;
        dml::FusedActivation m_fusedActivation;
    };

    // ---------------------------------------------------------------------------------------------------------------

    // If `axes` is not specified, by default this reduces the entire tensor to single element.
    inline Expression Reduce(Expression input, DML_REDUCE_FUNCTION function, Span<const uint32_t> axes = {})
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        uint32_t dimensionCount = static_cast<uint32_t>(inputTensor.sizes.size());

        SmallVector<uint32_t, 4> defaultAxes;
        if (axes.empty())
        {
            for (uint32_t i = 0; i < dimensionCount; ++i)
            {
                defaultAxes.push_back(i);
            }
            axes = defaultAxes;
        }

        // Compute the output tensor dimensions
        TensorDimensions outputSizes;
        for (uint32_t i = 0; i < dimensionCount; ++i)
        {
            // If the dimension is to be reduced, this dimension in the output tensor has a size of 1, otherwise
            // it matches the input tensor.
            const bool dimensionIsReduced = std::find(axes.begin(), axes.end(), i) != axes.end();
            if (dimensionIsReduced)
            {
                outputSizes.push_back(1);
            }
            else
            {
                outputSizes.push_back(inputTensor.sizes[i]);
            }
        }

        // ARGMIN and ARGMAX reduction produce a UINT32 output; all other reductions produce an output with the same
        // type as the input.
        DML_TENSOR_DATA_TYPE outputDataType;
        if (function == DML_REDUCE_FUNCTION_ARGMIN || function == DML_REDUCE_FUNCTION_ARGMAX)
        {
            outputDataType = DML_TENSOR_DATA_TYPE_UINT32;
        }
        else
        {
            outputDataType = inputTensor.dataType;
        }

        TensorDesc outputTensor(outputDataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_REDUCE_OPERATOR_DESC desc = {};
        desc.Function = function;
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.AxisCount = static_cast<uint32_t>(axes.size());
        desc.Axes = axes.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_REDUCE, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression AveragePooling(
        Expression input,
        Span<const uint32_t> strides,
        Span<const uint32_t> windowSizes,
        Span<const uint32_t> startPadding,
        Span<const uint32_t> endPadding,
        bool includePadding,
        TensorDimensions outputSizes = {})
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        assert(strides.size() == windowSizes.size());
        assert(strides.size() == startPadding.size());
        assert(strides.size() == endPadding.size());

        // Calculate output size, if not explicitly provided
        if (outputSizes.empty())
        {
            outputSizes.push_back(inputTensor.sizes[0]); // N
            outputSizes.push_back(inputTensor.sizes[1]); // C
            for (size_t i = 0; i < windowSizes.size(); ++i)
            {
                uint32_t paddedInputSize = inputTensor.sizes[2 + i] + startPadding[i] + endPadding[i];
                uint32_t outputSize = (paddedInputSize - windowSizes[i]) / strides[i] + 1;
                outputSizes.push_back(outputSize);
            }
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_AVERAGE_POOLING_OPERATOR_DESC averagePoolDesc = {};
        averagePoolDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        averagePoolDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        averagePoolDesc.DimensionCount = static_cast<uint32_t>(windowSizes.size());
        averagePoolDesc.Strides = strides.data();
        averagePoolDesc.WindowSize = windowSizes.data();
        averagePoolDesc.StartPadding = startPadding.data();
        averagePoolDesc.EndPadding = endPadding.data();
        averagePoolDesc.IncludePadding = includePadding;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_AVERAGE_POOLING, &averagePoolDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    //
    // TODO: LpPooling
    //

    // ---------------------------------------------------------------------------------------------------------------

    struct MaxPoolingOutputs
    {
        Expression values;
        Expression indices; // Only valid if outputIndices = true is supplied to MaxPooling()
    };

    // If not specified, parameters are defaulted to the following values:
    //   Strides = 1 for each spatial dimension
    //   StartPadding = 0 for each spatial dimension
    //   EndPadding = 0 for each spatial dimension
    //   Dilations = 1 for each spatial dimension
    //   OutputIndices = false
    inline MaxPoolingOutputs MaxPooling(
        Expression input,
        Span<const uint32_t> windowSize,
        Span<const uint32_t> strides = {},
        Span<const uint32_t> startPadding = {},
        Span<const uint32_t> endPadding = {},
        Span<const uint32_t> dilations = {},
        bool outputIndices = false,
        TensorDimensions outputSizes = {})
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        // If the spatial dimension count is 2, we'll just use the first two elements by setting
        // DimensionCount = 2 in the desc
        const uint32_t defaultStridesAndDilations[3] = { 1, 1, 1 };
        const uint32_t defaultPadding[3] = { 0, 0, 0 };

        assert(windowSize.size() == 2 || windowSize.size() == 3);
        assert(strides.empty() || strides.size() == windowSize.size());
        assert(dilations.empty() || dilations.size() == windowSize.size());
        assert(startPadding.empty() || startPadding.size() == windowSize.size());
        assert(endPadding.empty() || endPadding.size() == windowSize.size());

        strides = strides.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : strides;
        dilations = dilations.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : dilations;
        startPadding = startPadding.empty() ? Span<const uint32_t>{ defaultPadding } : startPadding;
        endPadding = endPadding.empty() ? Span<const uint32_t>{ defaultPadding } : endPadding;

        // Calculate output size, if not explicitly provided
        if (outputSizes.empty())
        {
            outputSizes.push_back(inputTensor.sizes[0]); // N
            outputSizes.push_back(inputTensor.sizes[1]); // C
            for (size_t i = 0; i < windowSize.size(); i++)
            {
                uint32_t paddedInputSize = inputTensor.sizes[2 + i] + startPadding[i] + endPadding[i];
                uint32_t dilatedWindowSize = 1 + (windowSize[i] - 1) * dilations[i];
                uint32_t outputSize = (dilatedWindowSize >= paddedInputSize) ? 1 : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
                outputSizes.push_back(outputSize);
            }
        }

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetTensorPolicy());
        TensorDesc outputIndicesTensor(DML_TENSOR_DATA_TYPE_UINT32, std::move(outputSizes), builder->GetTensorPolicy());

        DML_MAX_POOLING2_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputIndicesTensor = outputIndices ? outputIndicesTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.DimensionCount = static_cast<uint32_t>(windowSize.size());
        desc.Strides = strides.data();
        desc.WindowSize = windowSize.data();
        desc.StartPadding = startPadding.data();
        desc.EndPadding = endPadding.data();
        desc.Dilations = dilations.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_MAX_POOLING2, &desc, inputs);

        detail::NodeOutput* outputExpr = builder->CreateNodeOutput(node, 0, std::move(outputTensor));
        if (outputIndices)
        {
            detail::NodeOutput* outputIndicesExpr = builder->CreateNodeOutput(node, 1, std::move(outputIndicesTensor));
            return { outputExpr, outputIndicesExpr };
        }
        return { outputExpr, Expression() };
    }

    // Helper for setting parameters for the MaxPooling operator. Sample usage:
    //
    //   auto [out, outIndices] = dml::MaxPoolingBuilder(...)
    //        .StartPadding(...)
    //        .EndPadding(...)
    //        .OutputIndices(...)
    //        .Build();
    //
    // Parameters left unspecified will be defaulted with the same values as dml::MaxPooling().
    class MaxPoolingBuilder
    {
    public:
        MaxPoolingBuilder(Expression input, Span<const uint32_t> windowSize)
            : m_input(input), m_windowSize(windowSize.begin(), windowSize.end())
        {}

        MaxPoolingBuilder& Strides(Span<const uint32_t> strides) { m_strides.assign(strides.begin(), strides.end()); return *this; }
        MaxPoolingBuilder& StartPadding(Span<const uint32_t> startPadding) { m_startPadding.assign(startPadding.begin(), startPadding.end()); return *this; }
        MaxPoolingBuilder& EndPadding(Span<const uint32_t> endPadding) { m_endPadding.assign(endPadding.begin(), endPadding.end()); return *this; }
        MaxPoolingBuilder& Dilations(Span<const uint32_t> dilations) { m_dilations.assign(dilations.begin(), dilations.end()); return *this; }
        MaxPoolingBuilder& OutputIndices(bool outputIndices) { m_outputIndices = outputIndices; return *this; }
        MaxPoolingBuilder& OutputSizes(TensorDimensions outputSizes) { m_outputSizes = std::move(outputSizes); return *this; }

        MaxPoolingOutputs Build() const
        {
            return MaxPooling(
                m_input,
                m_windowSize,
                m_strides,
                m_startPadding,
                m_endPadding,
                m_dilations,
                m_outputIndices,
                m_outputSizes);
        }

    private:
        Expression m_input;
        SmallVector<uint32_t, 3> m_windowSize;
        SmallVector<uint32_t, 3> m_strides = {};
        SmallVector<uint32_t, 3> m_startPadding = {};
        SmallVector<uint32_t, 3> m_endPadding = {};
        SmallVector<uint32_t, 3> m_dilations = {};
        bool m_outputIndices = false;
        TensorDimensions m_outputSizes = {};
    };

    // ---------------------------------------------------------------------------------------------------------------

    //
    // TODO: MaxUnpooling
    //

    //
    // TODO: ROIPooling
    //

    inline Expression Slice(
        Expression input,
        Span<const uint32_t> inputWindowOffsets,
        Span<const uint32_t> inputWindowSizes,
        Span<const int32_t> inputWindowStrides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDimensions outputSizes(inputTensor.sizes);

        assert(inputWindowOffsets.size() == outputSizes.size());
        assert(inputWindowOffsets.size() == inputWindowStrides.size());
        assert(inputWindowOffsets.size() == inputWindowSizes.size());

        for (size_t i = 0; i < outputSizes.size(); i++)
        {
            uint32_t minimumInputSize = (inputWindowSizes[i] - 1) / abs(inputWindowStrides[i]) + 1;
            outputSizes[i] = minimumInputSize;
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_SLICE1_OPERATOR_DESC sliceDesc = {};
        sliceDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        sliceDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        sliceDesc.DimensionCount = static_cast<uint32_t>(inputWindowOffsets.size());
        sliceDesc.InputWindowOffsets = inputWindowOffsets.data();
        sliceDesc.InputWindowSizes = inputWindowSizes.data();
        sliceDesc.InputWindowStrides = inputWindowStrides.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SLICE1, &sliceDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Cast(Expression input, DML_TENSOR_DATA_TYPE targetDataType)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(targetDataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_CAST_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_CAST, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline std::vector<Expression> Split(
        Expression input,
        uint32_t axis,
        Span<const uint32_t> outputAxisSizes)
    {
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        uint32_t axisSizeSum = 0;

        std::vector<TensorDesc> outputTensors;
        outputTensors.reserve(outputAxisSizes.size());

        std::vector<DML_TENSOR_DESC> outputDescs;
        outputDescs.reserve(outputAxisSizes.size());

        for (uint32_t outputAxisSize : outputAxisSizes)
        {
            TensorDimensions outputSizes = inputTensor.sizes;
            outputSizes[axis] = outputAxisSize;

            TensorDesc tensorDesc(inputTensor.dataType, outputSizes, builder->GetTensorPolicy());
            outputTensors.push_back(std::move(tensorDesc));
            outputDescs.push_back(*outputTensors.back().AsPtr<DML_TENSOR_DESC>());

            axisSizeSum += outputAxisSize;
        }

        assert(axisSizeSum == inputTensor.sizes[axis]);

        DML_SPLIT_OPERATOR_DESC desc = {};
        desc.Axis = axis;
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensors = outputDescs.data();
        desc.OutputCount = static_cast<uint32_t>(outputAxisSizes.size());

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SPLIT, &desc, inputs);

        std::vector<Expression> outputs;
        outputs.reserve(outputAxisSizes.size());

        for (uint32_t i = 0; i < outputAxisSizes.size(); ++i)
        {
            outputs.push_back(builder->CreateNodeOutput(node, i, std::move(outputTensors[i])));
        }

        return outputs;
    }

    inline Expression Join(
        Span<const Expression> inputs,
        uint32_t axis)
    {
        assert(!inputs.empty());

        detail::GraphBuilder* builder = inputs[0].Impl()->GetGraphBuilder();
        DML_TENSOR_DATA_TYPE dataType = inputs[0].Impl()->GetOutputDesc().dataType;

        TensorDimensions outputSizes = inputs[0].Impl()->GetOutputDesc().sizes;
        outputSizes[axis] = 0;

        std::vector<TensorDesc> inputTensors;
        inputTensors.reserve(inputs.size());

        std::vector<DML_TENSOR_DESC> inputDescs;
        inputDescs.reserve(inputs.size());

        std::vector<detail::NodeOutput*> inputNodes;
        inputNodes.reserve(inputs.size());

        for (Expression input : inputs)
        {
            inputTensors.push_back(input.Impl()->GetOutputDesc());
            TensorDesc& inputTensor = inputTensors.back();
            outputSizes[axis] += inputTensor.sizes[axis];
            inputDescs.push_back(*inputTensor.AsPtr<DML_TENSOR_DESC>());
            inputNodes.push_back(input.Impl());
        }

        TensorDesc outputTensor(dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_JOIN_OPERATOR_DESC desc = {};
        desc.Axis = axis;
        desc.InputCount = static_cast<uint32_t>(inputDescs.size());
        desc.InputTensors = inputDescs.data();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_JOIN, &desc, inputNodes);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Padding(
        Expression input,
        DML_PADDING_MODE paddingMode,
        float paddingValue,
        Span<const uint32_t> startPadding,
        Span<const uint32_t> endPadding)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDimensions outputSizes = inputTensor.sizes;

        assert(outputSizes.size() == startPadding.size());
        assert(outputSizes.size() == endPadding.size());

        for (size_t i = 0; i < outputSizes.size(); i++)
        {
            outputSizes[i] += startPadding[i] + endPadding[i];
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_PADDING_OPERATOR_DESC paddingDesc = {};
        paddingDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        paddingDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        paddingDesc.PaddingMode = paddingMode;
        paddingDesc.PaddingValue = paddingValue;
        paddingDesc.DimensionCount = static_cast<uint32_t>(startPadding.size());
        paddingDesc.StartPadding = startPadding.data();
        paddingDesc.EndPadding = endPadding.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_PADDING, &paddingDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ValueScale2D(
        Expression input,
        float scale,
        Span<const float> bias)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_VALUE_SCALE_2D_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Scale = scale;
        desc.ChannelCount = static_cast<uint32_t>(bias.size());
        desc.Bias = bias.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_VALUE_SCALE_2D, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Upsample2D(Expression input, DML_SIZE_2D scaleSize, DML_INTERPOLATION_MODE interpolationMode)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        assert(inputTensor.sizes.size() == 4 || inputTensor.sizes.size() == 5);

        uint32_t i = 0;
        TensorDimensions outputSizes;
        outputSizes.push_back(inputTensor.sizes[i++]);                    // output[N] = input[N]
        outputSizes.push_back(inputTensor.sizes[i++]);                    // output[C] = input[C]
        if (inputTensor.sizes.size() == 5)
        {
            outputSizes.push_back(inputTensor.sizes[i++]);                // output[D] = input[D]
        }
        outputSizes.push_back(inputTensor.sizes[i++] * scaleSize.Height); // output[H] = input[H] * scaleH
        outputSizes.push_back(inputTensor.sizes[i++] * scaleSize.Width);  // output[W] = input[W] * scaleW
        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_UPSAMPLE_2D_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleSize = scaleSize;
        desc.InterpolationMode = interpolationMode;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_UPSAMPLE_2D, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Gather(
        Expression input,
        Expression indices,
        uint32_t axis,
        uint32_t indexDimensions)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();

        uint32_t dimensionCount = static_cast<uint32_t>(inputTensor.sizes.size());
        assert(indicesTensor.sizes.size() == dimensionCount);
        assert(axis < dimensionCount);
        assert(indexDimensions <= dimensionCount);

        TensorDimensions outputSizes(dimensionCount, 1);

        // All dimensions after the axis should be the same as the input
        int outputDim = static_cast<int>(dimensionCount) - 1;
        for (; static_cast<uint32_t>(outputDim) > axis; --outputDim)
        {
            outputSizes[outputDim] = inputTensor.sizes[outputDim];
        }

        // All dimensions within the range [axis - indexDimensions, axis] should be the same as the indices
        int indexDim = static_cast<int>(dimensionCount) - 1;
        for (; outputDim > static_cast<int>(axis) - static_cast<int>(indexDimensions); --outputDim, --indexDim)
        {
            outputSizes[outputDim] = indicesTensor.sizes[indexDim];
        }

        // All dimensions before (axis - indexDimensions) should be the same as the input
        int inputDim = axis - 1;
        for (; outputDim >= 0 && inputDim >= 0; --outputDim, --inputDim)
        {
            outputSizes[outputDim] = inputTensor.sizes[inputDim];
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_GATHER_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;
        desc.IndexDimensions = indexDimensions;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GATHER, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression GatherElements(
        Expression input,
        Expression indices,
        uint32_t axis)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();

        TensorDesc outputTensor(inputTensor.dataType, indicesTensor.sizes, builder->GetTensorPolicy());

        DML_GATHER_ELEMENTS_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GATHER_ELEMENTS, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression GatherND(
        Expression input,
        Expression indices,
        uint32_t inputDimensionCount,
        uint32_t indicesDimensionCount,
        uint32_t batchDimensionCount)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();

        assert(inputDimensionCount >= 1u && inputDimensionCount <= inputTensor.sizes.size());
        assert(indicesDimensionCount >= 1u && indicesDimensionCount <= indicesTensor.sizes.size());
        assert(batchDimensionCount < inputDimensionCount);
        assert(batchDimensionCount < indicesDimensionCount);

        uint32_t numberOfCoordinatesPerIndex = indicesTensor.sizes.back();
        assert(numberOfCoordinatesPerIndex >= 1u && numberOfCoordinatesPerIndex <= inputDimensionCount - batchDimensionCount);

        uint32_t numberOfOutputDimensionsFromInput = inputDimensionCount - batchDimensionCount - numberOfCoordinatesPerIndex;
        uint32_t outputPaddingAmount = static_cast<uint32_t>(inputTensor.sizes.size()) - (indicesDimensionCount + numberOfOutputDimensionsFromInput - 1);

        TensorDimensions outputSizes(outputPaddingAmount, 1);
        outputSizes.insert(outputSizes.end(), indicesTensor.sizes.end() - indicesDimensionCount, indicesTensor.sizes.end() - 1);
        outputSizes.insert(outputSizes.end(), inputTensor.sizes.end() - numberOfOutputDimensionsFromInput, inputTensor.sizes.end());

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_GATHER_ND1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InputDimensionCount = inputDimensionCount;
        desc.IndicesDimensionCount = indicesDimensionCount;
        desc.BatchDimensionCount = batchDimensionCount;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GATHER_ND1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ScatterElements(
        Expression input,
        Expression indices,
        Expression updates,
        uint32_t axis)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();
        TensorDesc updatesTensor = updates.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_SCATTER_ELEMENTS_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.UpdatesTensor = updatesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl(), updates.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SCATTER_ELEMENTS, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ScatterND(
        Expression input,
        Expression indices,
        Expression updates,
        uint32_t inputDimensionCount,
        uint32_t indicesDimensionCount)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();
        TensorDesc updatesTensor = updates.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_SCATTER_ND_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.UpdatesTensor = updatesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InputDimensionCount = inputDimensionCount;
        desc.IndicesDimensionCount = indicesDimensionCount;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl(), updates.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SCATTER_ND, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression SpaceToDepth(
        Expression input,
        uint32_t blockSize,
        DML_DEPTH_SPACE_ORDER order = DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        assert(inputTensor.sizes.size() == 4);

        dml::TensorDesc::Dimensions outputSizes = {
            inputTensor.sizes[0],
            inputTensor.sizes[1] * blockSize * blockSize,
            inputTensor.sizes[2] / blockSize,
            inputTensor.sizes[3] / blockSize
        };

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetTensorPolicy());

        DML_SPACE_TO_DEPTH1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BlockSize = blockSize;
        desc.Order = order;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SPACE_TO_DEPTH1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression DepthToSpace(
        Expression input,
        uint32_t blockSize,
        DML_DEPTH_SPACE_ORDER order = DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        assert(inputTensor.sizes.size() == 4);

        dml::TensorDesc::Dimensions outputSizes = {
            inputTensor.sizes[0],
            inputTensor.sizes[1] / (blockSize * blockSize),
            inputTensor.sizes[2] * blockSize,
            inputTensor.sizes[3] * blockSize
        };

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetTensorPolicy());

        DML_DEPTH_TO_SPACE1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BlockSize = blockSize;
        desc.Order = order;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_DEPTH_TO_SPACE1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Tile(Expression input, Span<const uint32_t> repeats)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDimensions outputSizes = input.GetOutputDesc().sizes;

        assert(repeats.size() == outputSizes.size());

        for (size_t i = 0; i < repeats.size(); ++i)
        {
            outputSizes[i] *= repeats[i];
        }

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_TILE_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.RepeatsCount = static_cast<uint32_t>(repeats.size());
        desc.Repeats = repeats.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_TILE, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    struct TopKOutputs
    {
        Expression value;
        Expression index;
    };

    inline TopKOutputs TopK(Expression input, uint32_t axis, uint32_t k, DML_AXIS_DIRECTION axisDirection)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        TensorDimensions outputSizes = inputTensor.sizes;
        outputSizes.back() = k;

        TensorDesc outputValueTensor(inputTensor.dataType, outputSizes, builder->GetTensorPolicy());
        TensorDesc outputIndexTensor(DML_TENSOR_DATA_TYPE_UINT32, outputSizes, builder->GetTensorPolicy());

        DML_TOP_K1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputValueTensor = outputValueTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputIndexTensor = outputIndexTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;
        desc.K = k;
        desc.AxisDirection = axisDirection;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_TOP_K1, &desc, inputs);
        detail::NodeOutput* outputValue = builder->CreateNodeOutput(node, 0, std::move(outputValueTensor));
        detail::NodeOutput* outputIndex = builder->CreateNodeOutput(node, 1, std::move(outputIndexTensor));

        return { outputValue, outputIndex };
    }

    inline Expression BatchNormalization(
        Expression input,
        Expression mean,
        Expression variance,
        Expression scale,
        Expression bias,
        bool spatial,
        float epsilon,
        FusedActivation fusedActivation = FusedActivation::None())
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc meanTensor = mean.Impl()->GetOutputDesc();
        TensorDesc varianceTensor = variance.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc biasTensor = bias.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        detail::FusedActivationStorage storage;

        DML_BATCH_NORMALIZATION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.MeanTensor = meanTensor.AsPtr<DML_TENSOR_DESC>();
        desc.VarianceTensor = varianceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BiasTensor = biasTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Spatial = spatial;
        desc.Epsilon = epsilon;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] = { input.Impl(), mean.Impl(), variance.Impl(), scale.Impl(), bias.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_BATCH_NORMALIZATION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression BatchNormalization(
        Expression input,
        Expression mean,
        Expression variance,
        Expression scale,
        Expression bias,
        bool spatial,
        float epsilon,
        const DML_OPERATOR_DESC* fusedActivation)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc meanTensor = mean.Impl()->GetOutputDesc();
        TensorDesc varianceTensor = variance.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc biasTensor = bias.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_BATCH_NORMALIZATION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.MeanTensor = meanTensor.AsPtr<DML_TENSOR_DESC>();
        desc.VarianceTensor = varianceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BiasTensor = biasTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Spatial = spatial;
        desc.Epsilon = epsilon;
        desc.FusedActivation = fusedActivation;

        detail::NodeOutput* const inputs[] = { input.Impl(), mean.Impl(), variance.Impl(), scale.Impl(), bias.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_BATCH_NORMALIZATION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

#if DML_TARGET_VERSION >= 0x3100

    struct BatchNormalizationGradOutputs
    {
        Expression gradient;
        Expression scaleGradient;
        Expression biasGradient;
    };

    inline BatchNormalizationGradOutputs BatchNormalizationGrad(
        Expression input,
        Expression inputGradient,
        Expression mean,
        Expression variance,
        Expression scale,
        float epsilon)
    {
        dml::detail::GraphBuilder* builder = mean.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc inputGradientTensor = inputGradient.Impl()->GetOutputDesc();
        TensorDesc meanTensor = mean.Impl()->GetOutputDesc();
        TensorDesc varianceTensor = variance.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc outputGradientTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());
        TensorDesc outputScaleGradientTensor(meanTensor.dataType, meanTensor.sizes, builder->GetTensorPolicy());
        TensorDesc outputBiasGradientTensor(meanTensor.dataType, meanTensor.sizes, builder->GetTensorPolicy());

        DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InputGradientTensor = inputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.MeanTensor = meanTensor.AsPtr<DML_TENSOR_DESC>();
        desc.VarianceTensor = varianceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Epsilon = epsilon;

        desc.OutputGradientTensor = outputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputScaleGradientTensor = outputScaleGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputBiasGradientTensor = outputBiasGradientTensor.AsPtr<DML_TENSOR_DESC>();

        dml::detail::NodeOutput* const inputs[] = { input.Impl(), inputGradient.Impl(), mean.Impl(), variance.Impl(), scale.Impl() };
        dml::detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_BATCH_NORMALIZATION_GRAD, &desc, inputs);

        BatchNormalizationGradOutputs outputValues;
        outputValues.gradient = builder->CreateNodeOutput(node, 0, *desc.OutputGradientTensor);
        outputValues.scaleGradient = builder->CreateNodeOutput(node, 1, *desc.OutputScaleGradientTensor);
        outputValues.biasGradient = builder->CreateNodeOutput(node, 2, *desc.OutputBiasGradientTensor);

        return outputValues;
    }

#endif // DML_TARGET_VERSION >= 0x3100

#if DML_TARGET_VERSION >= 0x4100
    struct BatchNormalizationTrainingOutputs
    {
        Expression output;
        Expression mean;
        Expression variance;
    };

    inline BatchNormalizationTrainingOutputs BatchNormalizationTraining(
        Expression input,
        Expression scale,
        Expression bias,
        Optional<Expression> fusedAdd,
        float epsilon,
        FusedActivation fusedActivation = FusedActivation::None())
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc biasTensor = bias.Impl()->GetOutputDesc();

        TensorDesc fusedAddTensor;
        if (fusedAdd)
        {
            fusedAddTensor = fusedAdd->Impl()->GetOutputDesc();
        }

        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());
        TensorDesc outputMeanTensor(inputTensor.dataType, scaleTensor.sizes, builder->GetTensorPolicy());
        TensorDesc outputVarianceTensor(inputTensor.dataType, scaleTensor.sizes, builder->GetTensorPolicy());

        detail::FusedActivationStorage storage;

        DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BiasTensor = biasTensor.AsPtr<DML_TENSOR_DESC>();
        desc.FusedAddTensor = fusedAdd.has_value() ? fusedAddTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputMeanTensor = outputMeanTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputVarianceTensor = outputVarianceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Epsilon = epsilon;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] = { input.Impl(), scale.Impl(), bias.Impl(), fusedAdd ? fusedAdd->Impl() : nullptr };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_BATCH_NORMALIZATION_TRAINING, &desc, inputs);
        detail::NodeOutput* output         = builder->CreateNodeOutput(node, 0, std::move(outputTensor));
        detail::NodeOutput* outputMean     = builder->CreateNodeOutput(node, 1, std::move(outputMeanTensor));
        detail::NodeOutput* outputVariance = builder->CreateNodeOutput(node, 2, std::move(outputVarianceTensor));

        return {output, outputMean, outputVariance};
    }

    inline BatchNormalizationGradOutputs BatchNormalizationTrainingGrad(
        Expression input,
        Expression inputGradient,
        Expression mean,
        Expression variance,
        Expression scale,
        float epsilon)
    {
        dml::detail::GraphBuilder* builder = mean.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc inputGradientTensor = inputGradient.Impl()->GetOutputDesc();
        TensorDesc meanTensor = mean.Impl()->GetOutputDesc();
        TensorDesc varianceTensor = variance.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc outputGradientTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());
        TensorDesc outputScaleGradientTensor(meanTensor.dataType, meanTensor.sizes, builder->GetTensorPolicy());
        TensorDesc outputBiasGradientTensor(meanTensor.dataType, meanTensor.sizes, builder->GetTensorPolicy());

        DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InputGradientTensor = inputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.MeanTensor = meanTensor.AsPtr<DML_TENSOR_DESC>();
        desc.VarianceTensor = varianceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Epsilon = epsilon;

        desc.OutputGradientTensor = outputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputScaleGradientTensor = outputScaleGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputBiasGradientTensor = outputBiasGradientTensor.AsPtr<DML_TENSOR_DESC>();

        dml::detail::NodeOutput* const inputs[] = { input.Impl(), inputGradient.Impl(), mean.Impl(), variance.Impl(), scale.Impl() };
        dml::detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD, &desc, inputs);

        BatchNormalizationGradOutputs outputValues;
        outputValues.gradient = builder->CreateNodeOutput(node, 0, *desc.OutputGradientTensor);
        outputValues.scaleGradient = builder->CreateNodeOutput(node, 1, *desc.OutputScaleGradientTensor);
        outputValues.biasGradient = builder->CreateNodeOutput(node, 2, *desc.OutputBiasGradientTensor);

        return outputValues;
    }
#endif // DML_TARGET_VERSION >= 0x4100

    inline Expression MeanVarianceNormalization(
        Expression input,
        Optional<Expression> scale,
        Optional<Expression> bias,
        Span<const uint32_t> axes,
        bool normalizeVariance,
        float epsilon,
        FusedActivation fusedActivation = FusedActivation::None())
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());
        TensorDesc scaleTensor;
        TensorDesc biasTensor;

        if (scale)
        {
            scaleTensor = scale->Impl()->GetOutputDesc();
        }
        if (bias)
        {
            biasTensor = bias->Impl()->GetOutputDesc();
        }

        detail::FusedActivationStorage storage;

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scale ? scaleTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.BiasTensor = bias ? biasTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.AxisCount = static_cast<UINT>(axes.size());
        desc.Axes = axes.data();
        desc.NormalizeVariance = normalizeVariance;
        desc.Epsilon = epsilon;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] =
        {
            input.Impl(),
            scale ? scale->Impl() : nullptr,
            bias ? bias->Impl() : nullptr
        };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression LocalResponseNormalization(
        Expression input,
        bool crossChannel,
        uint32_t localSize,
        float alpha,
        float beta,
        float bias)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.CrossChannel = crossChannel;
        desc.LocalSize = localSize;
        desc.Alpha = alpha;
        desc.Beta = beta;
        desc.Bias = bias;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    //
    // TODO: LpNormalization
    //

    //
    // TODO: RNN
    //

    //
    // TODO: LSTM
    //

    enum class GRUOutputOptions
    {
        Both,
        Sequence,
        Single,
    };

    struct GRUOutputs
    {
        Expression sequence;
        Expression single;
    };

    inline GRUOutputs GRU(
        Expression input,
        Expression weight,
        Expression recurrence,
        Optional<Expression> bias,
        Optional<Expression> hiddenInit,
        Optional<Expression> sequenceLengths,
        Span<const FusedActivation> activationDescs,
        DML_RECURRENT_NETWORK_DIRECTION direction,
        bool linearBeforeReset,
        GRUOutputOptions outputOptions)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc weightTensor = weight.Impl()->GetOutputDesc();
        TensorDesc recurrenceTensor = recurrence.Impl()->GetOutputDesc();
        TensorDesc biasTensor;
        TensorDesc hiddenInitTensor;
        TensorDesc sequenceLengthsTensor;
        TensorDesc outputSequenceTensor;
        TensorDesc outputSingleTensor;
        if (bias)
        {
            biasTensor = bias->Impl()->GetOutputDesc();
        }
        if (hiddenInit)
        {
            hiddenInitTensor = hiddenInit->Impl()->GetOutputDesc();
        }
        if (sequenceLengths)
        {
            sequenceLengthsTensor = sequenceLengths->Impl()->GetOutputDesc();
        }

        TensorDesc::Dimensions outputSequenceSizes(4);
        TensorDesc::Dimensions outputSingleSizes(4);
        uint32_t directionCount = (direction == DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL) ? 2 : 1;
        if (outputOptions == GRUOutputOptions::Sequence || outputOptions == GRUOutputOptions::Both)
        {
            outputSequenceSizes[0] = inputTensor.sizes[1]; // SequenceLength
            outputSequenceSizes[1] = directionCount;
            outputSequenceSizes[2] = inputTensor.sizes[2]; // BatchSize
            outputSequenceSizes[3] = recurrenceTensor.sizes[3]; // HiddenSize
            outputSequenceTensor = TensorDesc(inputTensor.dataType, outputSequenceSizes, builder->GetTensorPolicy());
        }
        if (outputOptions == GRUOutputOptions::Single || outputOptions == GRUOutputOptions::Both)
        {
            outputSingleSizes[0] = 1;
            outputSingleSizes[1] = directionCount;
            outputSingleSizes[2] = inputTensor.sizes[2]; // BatchSize
            outputSingleSizes[3] = recurrenceTensor.sizes[3]; // HiddenSize
            outputSingleTensor = TensorDesc(inputTensor.dataType, outputSingleSizes, builder->GetTensorPolicy());
        }

        uint32_t activationCount = static_cast<uint32_t>(activationDescs.size());
        if (activationCount > 4)
        {
            DMLX_THROW(E_INVALIDARG);
        }

        detail::FusedActivationStorage storage[4];
        DML_OPERATOR_DESC activationDescArray[4];
        for (uint32_t i = 0; i < activationCount; ++i)
        {
            activationDescArray[i] = *detail::GetFusedActivationPtr(activationDescs[i], &storage[i]);
        }

        DML_GRU_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.WeightTensor = weightTensor.AsPtr<DML_TENSOR_DESC>();
        desc.RecurrenceTensor = recurrenceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BiasTensor = bias ? biasTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.HiddenInitTensor = hiddenInit ? hiddenInitTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.SequenceLengthsTensor = sequenceLengths ? sequenceLengthsTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputSequenceTensor = outputSequenceTensor.sizes.empty() ? nullptr : outputSequenceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputSingleTensor = outputSingleTensor.sizes.empty() ? nullptr : outputSingleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ActivationDescCount = activationCount;
        desc.ActivationDescs = activationDescArray;
        desc.Direction = direction;
        desc.LinearBeforeReset = linearBeforeReset;

        detail::NodeOutput* const inputs[] =
        {
            input.Impl(),
            weight.Impl(),
            recurrence.Impl(),
            bias ? bias->Impl() : nullptr,
            hiddenInit ? hiddenInit->Impl() : nullptr,
            sequenceLengths ? sequenceLengths->Impl() : nullptr
        };

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GRU, &desc, inputs);

        detail::NodeOutput* outputSequenceExpr = nullptr;
        detail::NodeOutput* outputSingleExpr = nullptr;
        if (outputOptions == GRUOutputOptions::Sequence || outputOptions == GRUOutputOptions::Both)
        {
             outputSequenceExpr = builder->CreateNodeOutput(node, 0, std::move(outputSequenceTensor));
        }
        if (outputOptions == GRUOutputOptions::Single || outputOptions == GRUOutputOptions::Both)
        {
             outputSingleExpr = builder->CreateNodeOutput(node, 1, std::move(outputSingleTensor));
        }
        return { outputSequenceExpr, outputSingleExpr };
    }

    //
    // TODO: DiagonalMatrix
    //

    inline Expression OneHot(
        Expression indices,
        Expression values,
        uint32_t outputLength,
        uint32_t axis)
    {
        detail::GraphBuilder* builder = indices.Impl()->GetGraphBuilder();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();
        TensorDesc valuesTensor = values.Impl()->GetOutputDesc();

        assert(axis < static_cast<uint32_t>(indicesTensor.sizes.size()));

        // The output and indices sizes must all match except for the active axis, which is supplied as outputLength.
        TensorDimensions outputSizes = indicesTensor.sizes;
        outputSizes[axis] = outputLength;

        TensorDesc outputTensor(valuesTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_ONE_HOT_OPERATOR_DESC desc = {};
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ValuesTensor = valuesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;

        detail::NodeOutput* const inputs[] = { indices.Impl(), values.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ONE_HOT, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    // If not specified, parameters are defaulted to the following values:
    //   Scales = computed by dividing the output sizes by the input sizes
    //   InputPixelOffsets = 0.5f for each dimension
    //   OutputPixelOffsets = -0.5f for each dimension
    inline Expression Resample(
        Expression input,
        TensorDimensions outputSizes,
        DML_INTERPOLATION_MODE mode,
        Span<const float> scales = {},
        Span<const float> inputPixelOffsets = {},
        Span<const float> outputPixelOffsets = {})
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        uint32_t dimensionCount = static_cast<uint32_t>(inputTensor.sizes.size());
        assert(outputSizes.size() == dimensionCount);

        SmallVector<float, 4> defaultScales;
        if (scales.empty())
        {
            for (uint32_t i = 0; i < dimensionCount; ++i)
            {
                defaultScales.push_back(static_cast<float>(outputSizes[i]) / static_cast<float>(inputTensor.sizes[i]));
            }
            scales = defaultScales;
        }

        SmallVector<float, 4> defaultInputPixelOffsets;
        if (inputPixelOffsets.empty())
        {
            defaultInputPixelOffsets.assign(dimensionCount, 0.5f);
            inputPixelOffsets = defaultInputPixelOffsets;
        }

        SmallVector<float, 4> defaultOutputPixelOffsets;
        if (outputPixelOffsets.empty())
        {
            defaultOutputPixelOffsets.assign(dimensionCount, -0.5f);
            outputPixelOffsets = defaultOutputPixelOffsets;
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_RESAMPLE1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InterpolationMode = mode;
        desc.DimensionCount = static_cast<UINT>(scales.size());
        desc.Scales = scales.data();
        desc.InputPixelOffsets = inputPixelOffsets.data();
        desc.OutputPixelOffsets = outputPixelOffsets.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_RESAMPLE1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression FillValueConstant(
        Graph& graph,
        TensorDimensions outputSizes,
        DML_TENSOR_DATA_TYPE valueDataType,
        DML_SCALAR_UNION value)
    {
        detail::GraphBuilder* builder = graph.Impl();
        TensorDesc outputTensor(valueDataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC desc = {};
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ValueDataType = valueDataType;
        desc.Value = value;

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_FILL_VALUE_CONSTANT, &desc, {});
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression FillValueSequence(
        Graph& graph,
        TensorDimensions outputSizes,
        DML_TENSOR_DATA_TYPE valueDataType,
        DML_SCALAR_UNION valueStart,
        DML_SCALAR_UNION valueDelta)
    {
        detail::GraphBuilder* builder = graph.Impl();
        TensorDesc outputTensor(valueDataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC desc = {};
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ValueDataType = valueDataType;
        desc.ValueStart = valueStart;
        desc.ValueDelta = valueDelta;

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_FILL_VALUE_SEQUENCE, &desc, {});
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression CumulativeSummation(
        Expression input,
        uint32_t axis,
        DML_AXIS_DIRECTION axisDirection,
        bool hasExclusiveSum)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_CUMULATIVE_SUMMATION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;
        desc.AxisDirection = axisDirection;
        desc.HasExclusiveSum = hasExclusiveSum;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_CUMULATIVE_SUMMATION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

#if DML_TARGET_VERSION >= 0x3100

    inline Expression CumulativeProduct(
        Expression input,
        uint32_t axis,
        DML_AXIS_DIRECTION axisDirection,
        bool hasExclusiveProduct)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_CUMULATIVE_PRODUCT_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;
        desc.AxisDirection = axisDirection;
        desc.HasExclusiveProduct = hasExclusiveProduct;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_CUMULATIVE_PRODUCT, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

#endif // DML_TARGET_VERSION >= 0x3100

    inline Expression ReverseSubsequences(
        Expression input,
        Expression sequenceLengths,
        uint32_t axis)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc sequenceLengthsTensor = sequenceLengths.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC reverseDesc = {};
        reverseDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        reverseDesc.SequenceLengthsTensor = sequenceLengthsTensor.AsPtr<DML_TENSOR_DESC>();
        reverseDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        reverseDesc.Axis = axis;

        detail::NodeOutput* const inputs[] = { input.Impl(), sequenceLengths.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_REVERSE_SUBSEQUENCES, &reverseDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    //
    // TODO: MatrixMultiplyInteger
    //

    //
    // TODO: QuantizedLinearMatrixMultiply
    //

    //
    // TODO: ConvolutionInteger
    //

    //
    // TODO: QuantizedLinearConvolution
    //

    //
    // TODO: ReluGrad
    //

    //
    // TODO: AveragePoolingGrad
    //

    //
    // TODO: MaxPoolingGrad
    //

    struct RandomGeneratorOutputs
    {
        Expression values;
        Expression state; // Only valid if outputState = true is supplied to RandomGenerator
    };

    inline RandomGeneratorOutputs RandomGenerator(
        Expression inputState,
        TensorDimensions outputSizes,
        bool outputState = true,
        DML_RANDOM_GENERATOR_TYPE type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10)
    {
        detail::GraphBuilder* builder = inputState.Impl()->GetGraphBuilder();

        TensorDesc inputStateTensor = inputState.Impl()->GetOutputDesc();
        TensorDesc outputTensor(DML_TENSOR_DATA_TYPE_UINT32, std::move(outputSizes), builder->GetTensorPolicy());

        DML_RANDOM_GENERATOR_OPERATOR_DESC desc = {};
        desc.Type = type;
        desc.InputStateTensor = inputStateTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        if (outputState)
        {
            // Input and output state have the same TensorDesc.
            desc.OutputStateTensor = inputStateTensor.AsPtr<DML_TENSOR_DESC>();
        }

        RandomGeneratorOutputs out;

        detail::NodeOutput* const inputs[] = { inputState.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_RANDOM_GENERATOR, &desc, inputs);
        out.values = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        if (outputState)
        {
            TensorDesc outputStateTensor = inputStateTensor;
            out.state = builder->CreateNodeOutput(node, 1, std::move(outputStateTensor));
        }

        return out;
    }

    struct NonZeroCoordinatesOutputs
    {
        Expression count;
        Expression coordinates;
    };
    inline NonZeroCoordinatesOutputs NonZeroCoordinates(Expression input)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        const auto& inputTensorSizes = inputTensor.sizes;
        uint32_t dimensionCount = static_cast<uint32_t>(inputTensorSizes.size());

        TensorDimensions outputCountSizes = {1};
        uint32_t totalElements = 1;
        for (uint32_t i = 0; i < dimensionCount; ++i)
        {
            totalElements *= inputTensorSizes[i];
        }
        TensorDesc outputCountTensor(DML_TENSOR_DATA_TYPE_UINT32, outputCountSizes, builder->GetTensorPolicy());
        TensorDesc outputCoordinatesTensor(DML_TENSOR_DATA_TYPE_UINT32, {totalElements, dimensionCount}, builder->GetTensorPolicy());

        DML_NONZERO_COORDINATES_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputCountTensor = outputCountTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputCoordinatesTensor = outputCoordinatesTensor.AsPtr<DML_TENSOR_DESC>();

        NonZeroCoordinatesOutputs output;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_NONZERO_COORDINATES, &desc, inputs);
        output.count = builder->CreateNodeOutput(node, 0, std::move(outputCountTensor));
        output.coordinates = builder->CreateNodeOutput(node, 1, std::move(outputCoordinatesTensor));
        return output;
    }

    // If not specified, parameters are defaulted to the following values:
    //   Scales = computed by dividing the input sizes by the output sizes
    //   InputPixelOffsets = 0.5f for each dimension
    //   OutputPixelOffsets = -0.5f for each dimension
    inline Expression ResampleGrad(
        Expression input,
        TensorDimensions outputSizes,
        DML_INTERPOLATION_MODE mode,
        Span<const float> scales = {},
        Span<const float> inputPixelOffsets = {},
        Span<const float> outputPixelOffsets = {})
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        uint32_t dimensionCount = static_cast<uint32_t>(inputTensor.sizes.size());
        assert(outputSizes.size() == dimensionCount);

        SmallVector<float, 4> defaultScales;
        if (scales.empty())
        {
            for (uint32_t i = 0; i < dimensionCount; ++i)
            {
                defaultScales.push_back(static_cast<float>(inputTensor.sizes[i]) / static_cast<float>(outputSizes[i]));
            }
            scales = defaultScales;
        }

        SmallVector<float, 4> defaultInputPixelOffsets;
        if (inputPixelOffsets.empty())
        {
            defaultInputPixelOffsets.assign(dimensionCount, 0.5f);
            inputPixelOffsets = defaultInputPixelOffsets;
        }

        SmallVector<float, 4> defaultOutputPixelOffsets;
        if (outputPixelOffsets.empty())
        {
            defaultOutputPixelOffsets.assign(dimensionCount, -0.5f);
            outputPixelOffsets = defaultOutputPixelOffsets;
        }

        TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetTensorPolicy());

        DML_RESAMPLE_GRAD_OPERATOR_DESC desc = {};
        desc.InputGradientTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputGradientTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InterpolationMode = mode;
        desc.DimensionCount = static_cast<UINT>(scales.size());
        desc.Scales = scales.data();
        desc.InputPixelOffsets = inputPixelOffsets.data();
        desc.OutputPixelOffsets = outputPixelOffsets.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_RESAMPLE_GRAD, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression SliceGrad(
        Expression inputGradient,
        TensorDimensions outputGradientSizes,
        Span<const uint32_t> inputWindowOffsets,
        Span<const uint32_t> inputWindowSizes,
        Span<const int32_t> inputWindowStrides)
    {
        detail::GraphBuilder* builder = inputGradient.Impl()->GetGraphBuilder();

        TensorDesc inputGradientTensor = inputGradient.Impl()->GetOutputDesc();

        assert(inputWindowOffsets.size() == inputGradientTensor.sizes.size());
        assert(inputWindowOffsets.size() == outputGradientSizes.size());
        assert(inputWindowOffsets.size() == inputWindowStrides.size());
        assert(inputWindowOffsets.size() == inputWindowSizes.size());

        TensorDesc outputGradientTensor(inputGradientTensor.dataType, std::move(outputGradientSizes), builder->GetTensorPolicy());

        DML_SLICE_GRAD_OPERATOR_DESC sliceGradDesc = {};
        sliceGradDesc.InputGradientTensor = inputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        sliceGradDesc.OutputGradientTensor = outputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        sliceGradDesc.DimensionCount = static_cast<uint32_t>(inputWindowOffsets.size());
        sliceGradDesc.InputWindowOffsets = inputWindowOffsets.data();
        sliceGradDesc.InputWindowSizes = inputWindowSizes.data();
        sliceGradDesc.InputWindowStrides = inputWindowStrides.data();

        detail::NodeOutput* const inputs[] = { inputGradient.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SLICE_GRAD, &sliceGradDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputGradientTensor));

        return output;
    }

    //
    // TODO: AdamOptimizer
    //

    //
    // TODO: Argmin
    //

    //
    // TODO: Argmax
    //

#if DML_TARGET_VERSION >= 0x4000

    inline Expression RoiAlign(
        Expression input,
        Expression roi,
        Expression batchIndices,
        DML_REDUCE_FUNCTION reductionFunction,
        DML_INTERPOLATION_MODE interpolationMode,
        float spatialScaleX,
        float spatialScaleY,
        float inputPixelOffset,
        float outputPixelOffset,
        float outOfBoundsInputValue,
        uint32_t minimumSamplesPerOutput,
        uint32_t maximumSamplesPerOutput,
        bool alignRegionsToCorners,
        uint32_t outputHeight,
        uint32_t outputWidth)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc roiTensor = roi.Impl()->GetOutputDesc();
        TensorDesc batchIndicesTensor = batchIndices.Impl()->GetOutputDesc();

        uint32_t channelCount = inputTensor.sizes[1];
        uint32_t roiCount = roiTensor.sizes.size() < 2 ? 1u : roiTensor.sizes[roiTensor.sizes.size() - 2];

        TensorDesc::Dimensions outputSizes({
            roiCount,
            channelCount,
            outputHeight,
            outputWidth,
        });

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetTensorPolicy());

        DML_ROI_ALIGN1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ROITensor = roiTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BatchIndicesTensor = batchIndicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ReductionFunction = reductionFunction;
        desc.InterpolationMode = interpolationMode;
        desc.SpatialScaleX = spatialScaleX;
        desc.SpatialScaleY = spatialScaleY;
        desc.InputPixelOffset = inputPixelOffset;
        desc.OutputPixelOffset = outputPixelOffset;
        desc.OutOfBoundsInputValue = outOfBoundsInputValue;
        desc.MinimumSamplesPerOutput = minimumSamplesPerOutput;
        desc.MaximumSamplesPerOutput = maximumSamplesPerOutput;
        desc.AlignRegionsToCorners = alignRegionsToCorners;

        detail::NodeOutput* const inputs[] = { input.Impl(), roi.Impl(), batchIndices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ROI_ALIGN1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

#endif // DML_TARGET_VERSION >= 0x4000

#if DML_TARGET_VERSION >= 0x4100
    struct RoiAlignGradOutputs
    {
        Expression outputGradient;
        Expression outputROIGradient;
    };

    inline RoiAlignGradOutputs RoiAlignGrad(
        Optional<Expression> input,
        Expression inputGradient,
        Expression roi,
        Expression batchIndices,
        DML_REDUCE_FUNCTION reductionFunction,
        DML_INTERPOLATION_MODE interpolationMode,
        float spatialScaleX,
        float spatialScaleY,
        float inputPixelOffset,
        float outputPixelOffset,
        uint32_t minimumSamplesPerOutput,
        uint32_t maximumSamplesPerOutput,
        bool alignRegionsToCorners,
        uint32_t batchSize,
        uint32_t imageHeight,
        uint32_t imageWidth,
        bool computeOutputGradient,
        bool computeOutputROIGradient)
    {
        detail::GraphBuilder* builder = inputGradient.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.has_value() ? input->Impl()->GetOutputDesc() : TensorDesc();
        TensorDesc inputGradientTensor = inputGradient.Impl()->GetOutputDesc();
        TensorDesc roiTensor = roi.Impl()->GetOutputDesc();
        TensorDesc batchIndicesTensor = batchIndices.Impl()->GetOutputDesc();

        assert(computeOutputGradient || computeOutputROIGradient);
        assert(inputGradientTensor.sizes.size() > 1);

        TensorDesc outputGradientTensor;
        if (computeOutputGradient)
        {
            TensorDesc::Dimensions outputGradientSizes({
                batchSize,
                inputGradientTensor.sizes[1],
                imageHeight,
                imageWidth,
            });

            outputGradientTensor = TensorDesc(inputGradientTensor.dataType, outputGradientSizes, builder->GetTensorPolicy());
        }

        TensorDesc outputROIGradientTensor = computeOutputROIGradient ? TensorDesc(roiTensor.dataType, roiTensor.sizes, builder->GetTensorPolicy()) : TensorDesc();
        assert(!computeOutputROIGradient || outputROIGradientTensor.sizes == roiTensor.sizes);

        DML_ROI_ALIGN_GRAD_OPERATOR_DESC desc = {};
        desc.InputTensor = input ? inputTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.InputGradientTensor = inputGradientTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ROITensor = roiTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BatchIndicesTensor = batchIndicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputGradientTensor = computeOutputGradient ? outputGradientTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputROIGradientTensor = computeOutputROIGradient ? outputROIGradientTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.ReductionFunction = reductionFunction;
        desc.InterpolationMode = interpolationMode;
        desc.SpatialScaleX = spatialScaleX;
        desc.SpatialScaleY = spatialScaleY;
        desc.InputPixelOffset = inputPixelOffset;
        desc.OutputPixelOffset = outputPixelOffset;
        desc.MinimumSamplesPerOutput = minimumSamplesPerOutput;
        desc.MaximumSamplesPerOutput = maximumSamplesPerOutput;
        desc.AlignRegionsToCorners = alignRegionsToCorners;

        detail::NodeOutput* const inputs[] = { input ? input->Impl() : nullptr, inputGradient.Impl(), roi.Impl(), batchIndices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(static_cast<DML_OPERATOR_TYPE>(DML_OPERATOR_ROI_ALIGN_GRAD), &desc, inputs);

        RoiAlignGradOutputs outputs {};

        if (computeOutputGradient)
        {
            outputs.outputGradient = builder->CreateNodeOutput(node, 0, std::move(outputGradientTensor));
        }

        if (computeOutputROIGradient)
        {
            outputs.outputROIGradient = builder->CreateNodeOutput(node, 1, std::move(outputROIGradientTensor));
        }

        return outputs;
    }
#endif

    // Reinterprets the memory of a tensor with a different type and dimensions (analogously to using
    // reinterpret_cast to access raw bits). Note that this is different to the DML Cast operator, which performs
    // a type cast on the contents of a tensor (analogously to static_cast). The total tensor size of the output
    // (which depends on the supplied type/sizes/strides) must match the input.
    inline Expression Reinterpret(
        Expression input,
        DML_TENSOR_DATA_TYPE newType,
        TensorDimensions newSizes,
        Optional<TensorStrides> newStrides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc newTensor(
            newType,
            inputTensor.flags,
            std::move(newSizes),
            std::move(newStrides),
            inputTensor.totalTensorSizeInBytes,
            inputTensor.guaranteedBaseOffsetAlignment);

        detail::NodeID node = builder->CreateReinterpretNode(input.Impl());
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(newTensor));

        return output;
    }

    // Same as Reinterpret above, but only adjusts tensor dimensions without affecting type.
    inline Expression Reinterpret(
        Expression input,
        TensorDimensions newSizes,
        Optional<TensorStrides> newStrides)
    {
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        return Reinterpret(input, inputTensor.dataType, std::move(newSizes), std::move(newStrides));
    }

    // Same as Reinterpret above, but only adjusts tensor type without affecting sizes or strides.
    inline Expression Reinterpret(Expression input, DML_TENSOR_DATA_TYPE newType)
    {
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        return Reinterpret(input, newType, inputTensor.sizes, inputTensor.strides);
    }

    // Operator overloads for convenience, which merely map to one of the functions above
    inline Expression operator+(Expression a, Expression b) { return dml::Add(a, b); }
    inline Expression operator-(Expression a, Expression b) { return dml::Subtract(a, b); }
    inline Expression operator*(Expression a, Expression b) { return dml::Multiply(a, b); }
    inline Expression operator/(Expression a, Expression b) { return dml::Divide(a, b); }
    inline Expression operator%(Expression a, Expression b) { return dml::ModulusTruncate(a, b); }
    inline Expression operator&(Expression a, Expression b) { return dml::BitAnd(a, b); }
    inline Expression operator|(Expression a, Expression b) { return dml::BitOr(a, b); }
    inline Expression operator^(Expression a, Expression b) { return dml::BitXor(a, b); }
    inline Expression operator<<(Expression a, Expression b) { return dml::BitShiftLeft(a, b); }
    inline Expression operator>>(Expression a, Expression b) { return dml::BitShiftRight(a, b); }
    inline Expression& operator+=(Expression& a, Expression b) { a = a + b; return a; }
    inline Expression& operator-=(Expression& a, Expression b) { a = a - b; return a; }
    inline Expression& operator*=(Expression& a, Expression b) { a = a * b; return a; }
    inline Expression& operator/=(Expression& a, Expression b) { a = a / b; return a; }
    inline Expression& operator%=(Expression& a, Expression b) { a = a % b; return a; }
    inline Expression& operator&=(Expression& a, Expression b) { a = a & b; return a; }
    inline Expression& operator|=(Expression& a, Expression b) { a = a | b; return a; }
    inline Expression& operator^=(Expression& a, Expression b) { a = a ^ b; return a; }
    inline Expression& operator<<=(Expression& a, Expression b) { a = a << b; return a; }
    inline Expression& operator>>=(Expression& a, Expression b) { a = a >> b; return a; }

    // Operations involving scalars can be reduced to elementwise identity
    inline Expression operator+(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ 1.0f, b }); }
    inline Expression operator-(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ 1.0f, -b }); }
    inline Expression operator*(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ b, 0.0f }); }
    inline Expression operator/(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ 1.0f / b, 0.0f }); }
    inline Expression operator+(float a, Expression b) { return dml::Identity(b, DML_SCALE_BIAS{ 1.0f, a }); }
    inline Expression operator-(float a, Expression b) { return dml::Identity(b, DML_SCALE_BIAS{ -1.0f, a }); }
    inline Expression operator*(float a, Expression b) { return dml::Identity(b, DML_SCALE_BIAS{ a, 0.0f }); }
    inline Expression operator/(float a, Expression b) { return dml::Recip(b, DML_SCALE_BIAS{ a, 0.0f }); }
    inline Expression& operator+=(Expression& a, float b) { a = a + b; return a; }
    inline Expression& operator-=(Expression& a, float b) { a = a - b; return a; }
    inline Expression& operator*=(Expression& a, float b) { a = a * b; return a; }
    inline Expression& operator/=(Expression& a, float b) { a = a / b; return a; }

    // Unary
    inline Expression operator~(Expression input) { return dml::BitNot(input); }
    inline Expression operator+(Expression input) { return dml::Identity(input); }
    inline Expression operator-(Expression input) { return dml::Identity(input, DML_SCALE_BIAS{ -1.0f, 0.0f }); }

    // Logical
    inline Expression operator!(Expression a) { return dml::LogicalNot(a); }
    inline Expression operator&&(Expression a, Expression b) { return dml::LogicalAnd(a, b); }
    inline Expression operator||(Expression a, Expression b) { return dml::LogicalOr(a, b); }
    inline Expression operator>(Expression a, Expression b) { return dml::GreaterThan(a, b); }
    inline Expression operator<(Expression a, Expression b) { return dml::LessThan(a, b); }
    inline Expression operator==(Expression a, Expression b) { return dml::Equals(a, b); }
    inline Expression operator!=(Expression a, Expression b) { return !(a == b); }
    inline Expression operator>=(Expression a, Expression b) { return dml::GreaterThanOrEqual(a, b); }
    inline Expression operator<=(Expression a, Expression b) { return dml::LessThanOrEqual(a, b); }

    // GraphBuilder implementation details
    namespace detail
    {
        inline NodeID GraphBuilder::CreateOperatorNode(
            DML_OPERATOR_TYPE type,
            const void* desc,
            Span<NodeOutput* const> inputs)
        {
            DML_OPERATOR_DESC opDesc = { type, desc };

            Microsoft::WRL::ComPtr<IDMLOperator> op;
            DMLX_THROW_IF_FAILED(m_device->CreateOperator(&opDesc, IID_PPV_ARGS(&op)));

            OperatorNode node = {};
            node.op = std::move(op);
            node.inputs.assign(inputs.begin(), inputs.end());

            uint32_t index = static_cast<uint32_t>(m_operatorNodes.size());
            m_operatorNodes.push_back(std::move(node));

            return { NodeType::Operator, index };
        }

        inline NodeID GraphBuilder::CreateInputNode(uint32_t inputIndex)
        {
            uint32_t index = static_cast<uint32_t>(m_inputNodes.size());
            m_inputNodes.push_back(InputNode{ inputIndex });
            return { NodeType::Input, index };
        }

        inline NodeID GraphBuilder::CreateReinterpretNode(NodeOutput* input)
        {
            uint32_t index = static_cast<uint32_t>(m_reinterpretNodes.size());
            m_reinterpretNodes.push_back(ReinterpretNode{ input });
            return { NodeType::Reinterpret, index };
        }

        inline NodeOutput* GraphBuilder::CreateNodeOutput(NodeID node, uint32_t outputIndex, TensorDesc tensorDesc)
        {
            // Construct the object in the deque, which doesn't invalidate references to elements as it grows
            m_nodeOutputs.emplace_back(this, node, outputIndex, std::move(tensorDesc));

            return &m_nodeOutputs.back();
        }

        inline GraphDesc GraphBuilder::GetGraphDesc(Span<const Expression> outputs) const
        {
            GraphDesc desc = {};
            desc.inputCount = static_cast<uint32_t>(m_inputNodes.size());
            desc.outputCount = static_cast<uint32_t>(outputs.size());

            for (const OperatorNode& node : m_operatorNodes)
            {
                uint32_t nodeIndex = static_cast<uint32_t>(desc.nodes.size());
                desc.nodes.push_back(DML_OPERATOR_GRAPH_NODE_DESC{ node.op.Get() });

                // Walk through each of this node's inputs and add it as an edge
                const uint32_t inputCount = static_cast<uint32_t>(node.inputs.size());
                for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
                {
                    NodeOutput* input = node.inputs[inputIndex];
                    if (input == nullptr)
                    {
                        continue;
                    }
                    NodeID inputNode = input->GetNode();

                    // Reinterpret nodes aren't "real" nodes, they're just used to modify TensorDescs across
                    // edges. So we follow this node backwards until it hits a real node.
                    while (inputNode.type == NodeType::Reinterpret)
                    {
                        input = m_reinterpretNodes[inputNode.index].input;
                        inputNode = input->GetNode();
                    }

                    if (inputNode.type == NodeType::Input)
                    {
                        DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
                        inputEdge.GraphInputIndex = m_inputNodes[inputNode.index].inputIndex;
                        inputEdge.ToNodeIndex = nodeIndex;
                        inputEdge.ToNodeInputIndex = inputIndex;

                        desc.inputEdges.push_back(inputEdge);
                    }
                    else if (inputNode.type == NodeType::Operator)
                    {
                        DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
                        intermediateEdge.FromNodeIndex = inputNode.index;
                        intermediateEdge.FromNodeOutputIndex = input->GetOutputIndex();
                        intermediateEdge.ToNodeIndex = nodeIndex;
                        intermediateEdge.ToNodeInputIndex = inputIndex;

                        desc.intermediateEdges.push_back(intermediateEdge);
                    }
                    else
                    {
                        assert(false); // Invalid node type
                        DMLX_THROW(E_UNEXPECTED);
                    }
                }
            }

            // Add output edges
            for (uint32_t outputIndex = 0; outputIndex < desc.outputCount; ++outputIndex)
            {
                NodeOutput* output = outputs[outputIndex].Impl();
                if (output == nullptr)
                {
                    continue;
                }
                NodeID outputNode = output->GetNode();

                // Reinterpret nodes are meaningless on outputs (they're no-ops), so just follow them back until we
                // get to a real operator node.
                while (outputNode.type == NodeType::Reinterpret)
                {
                    output = m_reinterpretNodes[outputNode.index].input;
                    outputNode = output->GetNode();
                }

                if (outputNode.type == NodeType::Input)
                {
                    // It's not valid to connect an output of the graph directly to an input without an intervening
                    // node. If this behavior is desired, it should instead be accomplished with a copy e.g. using
                    // the elementwise identity operator.
                    DMLX_THROW(E_INVALIDARG);
                }

                assert(outputNode.type == NodeType::Operator);

                DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
                outputEdge.FromNodeIndex = output->GetNode().index;
                outputEdge.FromNodeOutputIndex = output->GetOutputIndex();
                outputEdge.GraphOutputIndex = outputIndex;

                desc.outputEdges.push_back(outputEdge);
            }

            // Sanity
            assert(desc.nodes.size() == m_operatorNodes.size());
            assert(desc.outputEdges.size() == desc.outputCount);
            assert(desc.outputCount == outputs.size());

            return desc;
        }
    } // namespace detail

} // namespace dml
