// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#pragma warning(push)
#pragma warning(disable : 6387)

#include "LearningModelBinding.h"
#include "LearningModelDevice.h"
#include "LearningModelSession.h"
#include "TensorKindFrom.h"
#include "TensorMemoryBufferReference.h"
#include "TensorBaseHelpers.h"

#include "core/session/onnxruntime_c_api.h"

namespace Windows::AI::MachineLearning {
// TensorBase
//
// This is the base class for all data based Tensor types. It exposes array and IVectorView
// based getter and setters.
//
// Supported derived classes:
//    Float, Int8, UInt8, UInt16, Int16, Int32, Int64, Boolean, Double, UInt32, UInt64
//
// Unsupported types
//    Float16 and String have different access patterns and Int8, Complex64, Complex128 are unsupported
//
template <typename T, typename ViewT, typename TDerived, typename TInterface, typename TBase>
struct TensorBase : TBase {
  template <typename ElementType = T, typename ElementViewType = ViewT>
  static void ASSERT_TEMPLATE_PARAMETERS() {
    // This adds compile time checks that ensure that the API can only be called when:
    //   1) the first template parameter matches the internal type (T),
    //      since the api attempts copy the tensor memory of type T into a vector of type ElementType.
    //   2) the second template parameter matches the return type
    static_assert(
        std::is_same<T, ElementType>::value,
        "This API can only be called with template parameters that match its internal data type T.");
    static_assert(
        std::is_same<ViewT, ElementViewType>::value,
        "This API can only be called with template parameters that match its internal data type T.");
  }

  template <typename ElementType = T, typename ElementViewType = ViewT>
  static void ASSERT_TEMPLATE_PARAMETERS_EXACT() {
    // This adds compile time checks that ensure that the API can only be called when:
    //   1) the conditions of ASSERT_TEMPLATE_PARAMETERS() are met.
    //   2) the ABI type (ViewT) matches the internal type (t).
    ASSERT_TEMPLATE_PARAMETERS<ElementType, ElementViewType>();

    static_assert(
        std::is_same<T, ViewT>::value,
        "This API can only be called with matching T and ViewT. Explicit specialization is required.");
  }

  /// On creation, tensors can either:
  ///  1) act as a placeholder without any backing memory (output tensors, chained values). In this case we
  ///     create the backing memory when the buffer is accessed. The buffer is allocated one of there scenarios:
  ///         GPUTensorize during binding (used to create DML resources for chaining)
  ///         UpdateSourceResourceData after eval (used for output placeholder tensors or unbound outputs)
  ///         GetBuffer when accessed by users
  ///    a) TensorBase()
  ///  2) allocate backing cpu memory (when a shape is provided)
  ///    a) TensorBase(std::vector<int64_t> const& shape)
  ///    b) TensorBase(winrt::Windows::Foundation::Collections::IIterable<int64_t> const& shape)
  ///  3) use provided backing gpu memory
  ///    a) TensorBase(std::vector<int64_t> const& shape, ID3D12Resource* pResource)
  TensorBase() : m_resources(std::make_shared<TensorResources<T>>()) {
  }

  TensorBase(winrt::Windows::Foundation::Collections::IIterable<int64_t> const& shape) : m_shape(begin(shape), end(shape)),
                                                                                         m_resources(std::make_shared<TensorResources<T>>()) {
    GetCpuResource() = std::make_shared<WinML::Tensor<T>>(m_shape);
  }

  TensorBase(std::vector<int64_t> const& shape) : m_shape(shape),
                                                  m_resources(std::make_shared<TensorResources<T>>()) {
    GetCpuResource() = std::make_shared<WinML::Tensor<T>>(m_shape);
  }

  TensorBase(std::vector<int64_t> const& shape, ID3D12Resource* pResource) : m_shape(shape),
                                                                             m_resources(std::make_shared<TensorResources<T>>()) {
    // This Api is not supported for TensorString
    WINML_THROW_HR_IF_TRUE_MSG(
        E_ILLEGAL_METHOD_CALL,
        (std::is_same<T, std::string>::value),
        "TensorString objects cannot be created from a ID3D12Resource!");

    GetGpuResource() = std::make_shared<DMLResource>(pResource);
  }

  OrtValue CreateGPUMLValue(std::shared_ptr<DMLResource>& resource, BindingContext& context) {
    auto shape = onnxruntime::TensorShape(m_shape);
    auto type = onnxruntime::DataTypeImpl::GetType<T>();
    return TensorBaseHelpers::CreateGPUMLValue(resource, context, shape, type);
  }

  OrtValue CPUTensorize(WinML::BindingContext& context) {
    if (GetCpuResource() != nullptr) {
      return GetCpuResource()->MLValue();
    }

    // If there is no matching cpu resource, then fallback to a gpu resource
    if (GetGpuResource() != nullptr) {
      return CreateGPUMLValue(GetGpuResource(), context);
    }

    WINML_THROW_HR(WINML_ERR_INVALID_BINDING);
  }

  OrtValue GPUTensorize(WinML::BindingContext& context) {
    if (GetGpuResource() != nullptr) {
      return CreateGPUMLValue(GetGpuResource(), context);
    }

    // If there is no matching gpu resource, then fallback to a cpu resource
    if (GetCpuResource() != nullptr) {
      return GetCpuResource()->MLValue();
    }

    if (TensorKind() == winrt::Windows::AI::MachineLearning::TensorKind::String) {
      // Lazily allocate the cpu TensorString resource
      // TensorStrings are CPU only, and so a gpu resource cannot be allocated for them.
      GetCpuResource() = std::make_shared<WinML::Tensor<T>>(m_shape);
      return GetCpuResource()->MLValue();
    } else {
      // Try to allocate the backing memory for the caller
      auto bufferSize = std::accumulate(std::begin(m_shape), std::end(m_shape), static_cast<int64_t>(1), std::multiplies<int64_t>());
      auto bufferByteSize = sizeof(T) * bufferSize;

      // DML needs the resources' sizes to be a multiple of 4 bytes
      if (bufferByteSize % 4 != 0) {
        bufferByteSize += 4 - (bufferByteSize % 4);
      }

      D3D12_HEAP_PROPERTIES heapProperties = {
          D3D12_HEAP_TYPE_DEFAULT,
          D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
          D3D12_MEMORY_POOL_UNKNOWN,
          0,
          0};
      D3D12_RESOURCE_DESC resourceDesc = {
          D3D12_RESOURCE_DIMENSION_BUFFER,
          0,
          static_cast<uint64_t>(bufferByteSize),
          1,
          1,
          1,
          DXGI_FORMAT_UNKNOWN,
          {1, 0},
          D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

      auto spSession = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
      auto spDevice = spSession->Device().as<winrt::Windows::AI::MachineLearning::implementation::LearningModelDevice>();

      winrt::com_ptr<ID3D12Resource> pGPUResource = nullptr;
      spDevice->GetD3DDevice()->CreateCommittedResource(
          &heapProperties,
          D3D12_HEAP_FLAG_NONE,
          &resourceDesc,
          D3D12_RESOURCE_STATE_COMMON,
          nullptr,
          __uuidof(ID3D12Resource),
          pGPUResource.put_void());

      GetGpuResource() = std::make_shared<DMLResource>(pGPUResource.get());
      return CreateGPUMLValue(GetGpuResource(), context);
    }
  }

  void EnsureBufferNotInUse() {
    auto isBufferInUse =
        std::any_of(
            m_outstandingReferences.begin(),
            m_outstandingReferences.end(),
            [](auto weakRef) { return weakRef.get() != nullptr; });

    WINML_THROW_HR_IF_TRUE_MSG(WINML_ERR_INVALID_BINDING, isBufferInUse, "The tensor has outstanding memory buffer references that must be closed prior to evaluation!");
  }

  // ILotusValueProviderPrivate::GetOrtValue
  STDMETHOD(GetOrtValue)
  (WinML::BindingContext& context, OrtValue* mlValue) {
    RETURN_HR_IF_NULL_MSG(
        WINML_ERR_INVALID_BINDING,
        m_resources,
        "The tensor has been closed and its resources have been detached!");

    EnsureBufferNotInUse();

    auto spSession = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto spDevice = spSession->Device().as<winrt::Windows::AI::MachineLearning::implementation::LearningModelDevice>();
    if (spDevice->IsCpuDevice()) {
      *mlValue = CPUTensorize(context);
    } else {
      *mlValue = GPUTensorize(context);
    }

    return S_OK;
  }

  // ILotusValueProviderPrivate::UpdateSourceResourceData
  STDMETHOD(UpdateSourceResourceData)
  (BindingContext& context, OrtValue& mlValue) {
    RETURN_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources have been detached during evaluation!");

    auto spSession = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto& tensor = mlValue.Get<onnxruntime::Tensor>();
    auto pResource = const_cast<void*>(tensor.DataRaw());

    m_shape = tensor.Shape().GetDims();

    if (GetCpuResource() == nullptr) {
      GetCpuResource() = std::make_shared<WinML::Tensor<T>>(m_shape);
    }

    if (!strcmp(tensor.Location().name, onnxruntime::CPU) ||
        tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput ||
        tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput) {
      // Get the data pointer and size
      T* pData;
      uint32_t pSize;
      std::tie(pSize, pData) = GetCpuResource()->buffer();

      if (pResource != reinterpret_cast<void*>(pData)) {
        // Only copy the data if the source and destination are not the same!
        // The engine provided buffer will not match the tensor buffer when
        // the tensor is created as a placeholder output, or as an unbound output.
        GetCpuResource()->set(static_cast<uint32_t>(tensor.Shape().Size()), reinterpret_cast<T*>(pResource));
      }
    } else {
      // If we got a gpu resource, we should move the data to the cpu so accessors can retrieve the data.
      // We don't need to copy the engine provided dx resource into a local copy since we always preallocate gpu
      // resources for tensors. Therefore we are certain that the returned dxresource is the same as the one we passed in
      // and was updated in place.
      auto cpuValue = GetCpuResource()->MLValue();
      auto& cpuLotusTensor = const_cast<onnxruntime::Tensor&>(cpuValue.Get<onnxruntime::Tensor>());
      RETURN_HR_IF(E_FAIL, !Dml::CopyTensor(spSession->GetExecutionProvider(), tensor, cpuLotusTensor).IsOK());
    }

    return S_OK;
  }

  ///
  /// Tensor Creation Patterns
  ///

  // ITensor<T>::Create
  static typename TBase::class_type Create() try {
    return winrt::make<TDerived>();
  }
  WINML_CATCH_ALL

  // ITensor<T>::Create
  static typename TBase::class_type Create(
      winrt::Windows::Foundation::Collections::IIterable<int64_t> const& shape) try {
    typename TBase::class_type tensorValue = winrt::make<TDerived>();
    auto tensorValueImpl = tensorValue.as<TDerived>();
    tensorValueImpl->m_shape = std::vector<int64_t>(begin(shape), end(shape));
    return tensorValue;
  }
  WINML_CATCH_ALL

  // ITensor<T>::CreateFromIterable
  static typename TBase::class_type CreateFromIterable(
      winrt::Windows::Foundation::Collections::IIterable<int64_t> shape,
      winrt::Windows::Foundation::Collections::IIterable<ViewT> const& data) try {
    std::vector<int64_t> vecShape(begin(shape), end(shape));
    if (HasFreeDimensions(vecShape)) {
      // If the tensor is being created with a free dimension, the data needs to
      // provide its actual size so that the free dimension can be computed.
      // In the case of IIterable<T>, there is no Size accessor, and so we require that
      // in this case the underlying object also implement IVectorView, so that we may
      // efficiently query the size of the data.
      if (auto vectorView = data.try_as<winrt::Windows::Foundation::Collections::IVectorView<ViewT>>()) {
        vecShape = GetAdjustedShape(vecShape, vectorView.Size());
      }
    }

    typename TBase::class_type tensorValue = winrt::make<TDerived>(vecShape);
    auto tensorValueImpl = tensorValue.as<TDerived>();
    tensorValueImpl->SetBufferFromIterable(data);
    return tensorValue;
  }
  WINML_CATCH_ALL

  // ITensor<T>::CreateFromArray
  static typename TBase::class_type CreateFromArray(
      winrt::Windows::Foundation::Collections::IIterable<int64_t> shape,
      winrt::array_view<ViewT const> data) try {
    std::vector<int64_t> vecShape(begin(shape), end(shape));
    return CreateFromArrayInternal(vecShape, data);
  }
  WINML_CATCH_ALL

  // ITensor<T>::CreateFromShapeArrayAndDataArray
  static typename TBase::class_type CreateFromShapeArrayAndDataArray(
      winrt::array_view<int64_t const> shape,
      winrt::array_view<ViewT const> data) try {
    std::vector<int64_t> vecShape(shape.begin(), shape.end());
    return CreateFromArrayInternal(vecShape, data);
  }
  WINML_CATCH_ALL

  static typename TBase::class_type CreateFromArrayInternal(
      std::vector<int64_t> shape,
      winrt::array_view<ViewT const> data) {
    if (HasFreeDimensions(shape)) {
      shape = GetAdjustedShape(shape, data.size());
    }

    typename TBase::class_type tensorValue = winrt::make<TDerived>(shape);
    auto tensorValueImpl = tensorValue.as<TDerived>();
    tensorValueImpl->SetBufferFromArray(data);
    return tensorValue;
  }

  // ITensor<T>::CreateFromBuffer
  static typename TBase::class_type CreateFromBuffer(
      winrt::array_view<int64_t const> shape,
      winrt::Windows::Storage::Streams::IBuffer const& buffer) try {
    std::vector<int64_t> vecShape(shape.begin(), shape.end());
    typename TBase::class_type tensorValue = winrt::make<TDerived>();
    auto tensorValueImpl = tensorValue.as<TDerived>();
    tensorValueImpl->m_shape = vecShape;
    tensorValueImpl->GetCpuResource() = std::make_shared<WinML::Tensor<T>>(vecShape, buffer);
    return tensorValue;
  }
  WINML_CATCH_ALL

  // ITensorNative::CreateFromD3D12Resource
  static HRESULT CreateFromD3D12Resource(
      ID3D12Resource* value,
      __int64* shape,
      int shapeCount,
      IUnknown** result) {
    try {
      // make sure they gave us a valid shape
      THROW_HR_IF(E_INVALIDARG, shape == nullptr);
      THROW_HR_IF(E_INVALIDARG, shapeCount == 0);

      // turn the shape into a vector<>
      std::vector<int64_t> shapeVector(shape, shape + shapeCount);

      // for each dim
      UINT64 width = 1;
      for (int i = 0; i < shapeCount; i++) {
        // find out it's total width
        width *= shapeVector[i];
        // make sure there are no invalid dimensions (-1 or any invalid shape)
        THROW_HR_IF(E_INVALIDARG, shapeVector[i] <= 0);
      }
      width *= sizeof(T);

      // make sure they gave us a valid value
      THROW_HR_IF(E_INVALIDARG, value == nullptr);

      // make sure it's a d3d12 buffer (!texture)
      auto desc = value->GetDesc();
      THROW_HR_IF(E_INVALIDARG, desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER);

      // make sure it's big enough
      THROW_HR_IF(E_INVALIDARG, desc.Width < width);

      // make the underlying winrt object
      typename TBase::class_type tensorValue = winrt::make<TDerived>(shapeVector, value);

      // return it (the caller owns the ref)
      *result = tensorValue.as<IUnknown>().detach();
      return S_OK;
    }
    WINML_CATCH_ALL_COM
  }

  static std::vector<int64_t> GetAdjustedShape(
      std::vector<int64_t> shape,
      uint64_t actualSize) {
    auto shapeSize = std::accumulate(std::begin(shape), std::end(shape), static_cast<int64_t>(1),
                                     [](const auto& accumulatedValue, const auto& next) {
                                       if (next == -1) {
                                         return accumulatedValue;
                                       } else {
                                         return accumulatedValue * next;
                                       }
                                     });

    THROW_HR_IF(E_INVALIDARG, actualSize % shapeSize != 0);

    auto foundIt = std::find_if(std::begin(shape), std::end(shape), [](auto dim) { return dim == -1; });
    auto iFreeDimension = std::distance(std::begin(shape), foundIt);

    shape[iFreeDimension] = static_cast<int64_t>(actualSize / shapeSize);
    return shape;
  }

  static bool HasFreeDimensions(std::vector<int64_t> const& shape) {
    // Ensure that all dimension values are either -1, or positive
    auto unsupportedIt =
        std::find_if(begin(shape), end(shape),
                     [](const auto& dim) {
                       return dim < -1;
                     });
    THROW_HR_IF(E_INVALIDARG, unsupportedIt != end(shape));

    auto nFreeDimensions = std::count(begin(shape), end(shape), -1);
    if (nFreeDimensions == 0) {
      return false;
    } else if (nFreeDimensions == 1) {
      return true;
    } else {
      throw winrt::hresult_invalid_argument();
    }
  }

  ///
  /// Tensor Data Buffer Accessor APIs
  ///

  // IMemoryBuffer::CreateReference
  winrt::Windows::Foundation::IMemoryBufferReference CreateReference() try {
    // Create a TensorMemoryBufferReference<T>

    // Per IMemoryBuffer.CreateReference (https://docs.microsoft.com/en-us/uwp/api/windows.foundation.imemorybuffer.createreference)
    // "This method always successfully returns a new IMemoryBufferReference object even after the IMemoryBuffer
    // "has been closed. In that case, the returned IMemoryBufferReference is already closed."
    // Creating a TensorMemoryBufferReference<T> with a null pointer is equivalent to creating it as closed.

    auto memoryBufferReference = winrt::make<TensorMemoryBufferReference<T>>(m_shape, m_resources);

    // Create and cache a weak reference to the TensorMemoryBufferReference<T>
    winrt::weak_ref<TensorMemoryBufferReference<T>> weak(memoryBufferReference.as<TensorMemoryBufferReference<T>>());
    m_outstandingReferences.push_back(weak);

    // Return the strong ref to the caller
    return memoryBufferReference;
  }
  WINML_CATCH_ALL

  // IMemoryBuffer::Close
  void Close() try {
    // Let go of the lifetime of the resources, this is will indicate that the memorybuffer is closed
    m_resources = nullptr;
  }
  WINML_CATCH_ALL

  // ITensorNative::GetBuffer
  STDMETHOD(GetBuffer)
  (BYTE** value, UINT32* capacity) {
    // This Api is not supported for TensorString
    RETURN_HR_IF_MSG(
        ERROR_INVALID_FUNCTION,
        (std::is_same<T, std::string>::value),
        "TensorString objects cannot return byte buffers!");

    RETURN_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources have been detached!");

    return m_resources->GetBuffer(m_shape, value, capacity);
  }

  // ITensorNative::GetD3D12Resource
  STDMETHOD(GetD3D12Resource)
  (ID3D12Resource** ppResource) {
    try {
      // This Api is not supported for TensorString
      RETURN_HR_IF(ERROR_INVALID_FUNCTION, (std::is_same<T, std::string>::value));
      RETURN_HR_IF_NULL_MSG(
          E_ILLEGAL_METHOD_CALL,
          m_resources,
          "The tensor has been closed and its resources have been detached!");

      GetGpuResource()->DXResource.copy_to(ppResource);
      return S_OK;
    }
    WINML_CATCH_ALL_COM
  }

  // ITensor<T>::GetAsVectorView
  template <typename ElementType = T, typename ElementViewType = ViewT>
  winrt::Windows::Foundation::Collections::IVectorView<ElementViewType> GetAsVectorView() try {
    // This adds compile time checks that ensure that the API can only be called when:
    //   1) the conditions of ASSERT_TEMPLATE_PARAMETERS_EXACT() are met.
    //   2) the signature of the method conforms to the ABI signature and the return value matches the ABI Return Type (ViewT).
    ASSERT_TEMPLATE_PARAMETERS_EXACT<ElementType, ElementViewType>();

    // This method returns the raw tensor data as an IVectorView.
    // This is a slow API that performs a buffer copy into a caller
    // owned IVectorView object.

    // Get the raw buffer pointer from the native tensor implementation.
    uint32_t size;
    ElementType* pData;
    std::tie(size, pData) = GetCpuResource()->buffer();

    // Copy data that will be passed back to caller.
    auto copy = std::vector<ElementType>(pData, pData + size);

    // Create IVectorView from copied data.
    return winrt::single_threaded_vector<ElementViewType>(std::move(copy)).GetView();
  }
  WINML_CATCH_ALL

  // Specialized version to convert float16 to float
  template <>
  winrt::Windows::Foundation::Collections::IVectorView<float> GetAsVectorView<onnxruntime::MLFloat16, float>() try {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<onnxruntime::MLFloat16, float>();

    uint32_t size;
    onnxruntime::MLFloat16* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    // Copy the HALFs to floats
    std::vector<float> floatValue(size);
    DirectX::PackedVector::XMConvertHalfToFloatStream(
        floatValue.data(),
        sizeof(float) /* output stride */,
        reinterpret_cast<DirectX::PackedVector::HALF*>(pBuffer),
        sizeof(DirectX::PackedVector::HALF) /* input stride */,
        size);

    // Create IVectorView from copied data.
    return winrt::single_threaded_vector<float>(std::move(floatValue)).GetView();
  }
  WINML_CATCH_ALL

  // Specialized version to convert string to hstring
  template <>
  winrt::Windows::Foundation::Collections::IVectorView<winrt::hstring> GetAsVectorView<std::string, winrt::hstring>() try {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<std::string, winrt::hstring>();

    uint32_t size;
    std::string* pData;
    std::tie(size, pData) = GetCpuResource()->buffer();

    auto copy = std::vector<winrt::hstring>(size, L"");
    std::generate(
        copy.begin(),
        copy.end(),
        [n = 0, &pData]() mutable {
          return WinML::Strings::HStringFromUTF8(pData[n++]);
        });

    return winrt::single_threaded_vector<winrt::hstring>(std::move(copy)).GetView();
  }
  WINML_CATCH_ALL

  // Specialized version to convert int8_t to uint8_t
  template <>
  winrt::Windows::Foundation::Collections::IVectorView<uint8_t> GetAsVectorView<int8_t, uint8_t>() try {
    ASSERT_TEMPLATE_PARAMETERS<int8_t, uint8_t>();

    uint32_t size;
    int8_t* pData;
    std::tie(size, pData) = GetCpuResource()->buffer();

    // Copy data that will be passed back to caller.

    gsl::span<uint8_t> span(reinterpret_cast<uint8_t*>(pData), size);
    std::vector<uint8_t> copy(span.begin(), span.begin() + size);

    // Create IVectorView from copied data.
    return winrt::single_threaded_vector<uint8_t>(std::move(copy)).GetView();
  }
  WINML_CATCH_ALL

  ///
  /// Tensor Property Accessors
  ///

  // ILearningModelFeatureValue implementation
  winrt::Windows::AI::MachineLearning::LearningModelFeatureKind Kind() try {
    return winrt::Windows::AI::MachineLearning::LearningModelFeatureKind::Tensor;
  }
  WINML_CATCH_ALL

  // ITensor::TensorKind
  winrt::Windows::AI::MachineLearning::TensorKind TensorKind() try {
    return TensorKindFrom<TInterface>::Type;
  }
  WINML_CATCH_ALL

  // ITensor::Shape
  winrt::Windows::Foundation::Collections::IVectorView<int64_t> Shape() try {
    std::vector<int64_t> copy(m_shape.cbegin(), m_shape.cend());
    return winrt::single_threaded_vector(std::move(copy)).GetView();
  }
  WINML_CATCH_ALL

  // ILotusValueProviderPrivate::AbiRepresentation
  STDMETHOD(AbiRepresentation)
  (winrt::Windows::Foundation::IInspectable& abiRepresentation) {
    using ABIType = typename TBase::class_type;
    ABIType to = nullptr;
    RETURN_IF_FAILED(this->QueryInterface(
        winrt::guid_of<ABIType>(),
        reinterpret_cast<void**>(winrt::put_abi(to))));

    to.as(abiRepresentation);

    return S_OK;
  }

  // ILotusValueProviderPrivate::IsPlaceholder
  STDMETHOD(IsPlaceholder)
  (bool* pIsPlaceHolder) {
    RETURN_HR_IF_NULL(E_POINTER, pIsPlaceHolder);
    RETURN_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources have been detached!");

    *pIsPlaceHolder = GetCpuResource() == nullptr && GetGpuResource() == nullptr;
    return S_OK;
  }

 private:
  ///
  /// SetBufferFromArray and parameterized specializations for MLFloat16, int8_t, and std::string
  ///
  template <typename ElementType = T, typename ElementViewType = ViewT>
  void SetBufferFromArray(winrt::array_view<ElementViewType const> data) {
    // This adds compile time checks that ensure that the API can only be called when
    // the conditions of ASSERT_TEMPLATE_PARAMETERS_EXACT() are met.
    ASSERT_TEMPLATE_PARAMETERS_EXACT<ElementType, ElementViewType>();

    // This method accepts data as an array, T[], from the caller.
    // This is a non-destructive API, so the caller data is
    // left untouched, and the data is copied into internal buffers.
    GetCpuResource()->set(data.size(), data.data());
  }

  // Specialized version to convert floats to float16
  template <>
  void SetBufferFromArray<onnxruntime::MLFloat16, float>(winrt::array_view<float const> data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<onnxruntime::MLFloat16, float>();

    uint32_t size;
    onnxruntime::MLFloat16* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    THROW_HR_IF(E_UNEXPECTED, data.size() != size);
    DirectX::PackedVector::XMConvertFloatToHalfStream(
        reinterpret_cast<DirectX::PackedVector::HALF*>(pBuffer),
        sizeof(DirectX::PackedVector::HALF) /* output stride */,
        data.data(),
        sizeof(float) /* input stride */,
        data.size());
  }

  // Specialized version to convert uint8_t to int8_t
  template <>
  void SetBufferFromArray<int8_t, uint8_t>(winrt::array_view<uint8_t const> data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<int8_t, uint8_t>();

    auto size = data.size();
    auto pData = data.data();

    GetCpuResource()->set(size, reinterpret_cast<int8_t*>(const_cast<uint8_t*>(pData)));
  }

  // Specialized version to convert hstring to string
  template <>
  void SetBufferFromArray<std::string, winrt::hstring>(winrt::array_view<winrt::hstring const> data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<std::string, winrt::hstring>();

    uint32_t size;
    std::string* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();
    THROW_HR_IF(E_UNEXPECTED, data.size() > size);

    // Convert and copy into the underlying buffer
    std::transform(
        data.begin(), data.end(), pBuffer,
        [](auto& element) mutable {
          return WinML::Strings::UTF8FromHString(element);
        });
  }

  ///
  /// SetBufferFromIterable and parameterized specializations for MLFloat16, int8_t, and std::string
  ///
  template <typename ElementType = T, typename ElementViewType = ViewT>
  void SetBufferFromIterable(
      winrt::Windows::Foundation::Collections::IIterable<ElementViewType> const& data) {
    // This adds compile time checks that ensure that the API can only be called when
    // the conditions of ASSERT_TEMPLATE_PARAMETERS_EXACT() are met.
    ASSERT_TEMPLATE_PARAMETERS_EXACT<ElementType, ElementViewType>();

    uint32_t size;
    ElementType* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    // This method accepts data as an IVectorView<T>.
    // This is a non-destructive API, so the caller data is
    // left untouched, and the data is copied into internal buffers.
    std::copy(begin(data), end(data), pBuffer);
  }

  // Specialized version to convert floats to float16
  template <>
  void SetBufferFromIterable<onnxruntime::MLFloat16, float>(
      winrt::Windows::Foundation::Collections::IIterable<float> const& data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<onnxruntime::MLFloat16, float>();

    uint32_t size;
    onnxruntime::MLFloat16* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    // Now that we take in IIterables and not vector views
    // how do we validate size???
    // THROW_HR_IF(E_UNEXPECTED, data.Size() != size);

    std::transform(
        begin(data),
        end(data),
        reinterpret_cast<DirectX::PackedVector::HALF*>(pBuffer),
        DirectX::PackedVector::XMConvertFloatToHalf);
  }

  // Specialized version to convert uint8_t to int8_t
  template <>
  void SetBufferFromIterable<int8_t, uint8_t>(
      winrt::Windows::Foundation::Collections::IIterable<uint8_t> const& data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<int8_t, uint8_t>();

    uint32_t size;
    int8_t* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    std::transform(begin(data), end(data), pBuffer, [](auto element) { return static_cast<int8_t>(element); });
  }

  // Specialized version to convert hstring to string
  template <>
  void SetBufferFromIterable<std::string, winrt::hstring>(
      winrt::Windows::Foundation::Collections::IIterable<winrt::hstring> const& data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<std::string, winrt::hstring>();

    uint32_t size;
    std::string* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    // Convert and copy into the underlying buffer
    std::transform(begin(data), end(data), pBuffer, [](const auto& element) {
      return WinML::Strings::UTF8FromHString(element);
    });
  }

  std::shared_ptr<WinML::Tensor<T>>& GetCpuResource() {
    WINML_THROW_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources are detached!");

    return m_resources->CpuResource;
  }

  std::shared_ptr<DMLResource>& GetGpuResource() {
    WINML_THROW_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources are detached!");

    return m_resources->GpuResource;
  }

 private:
  std::vector<int64_t> m_shape;
  std::shared_ptr<TensorResources<T>> m_resources;
  std::vector<winrt::weak_ref<TensorMemoryBufferReference<T>>> m_outstandingReferences;
  bool m_isClosed = false;
};

}  // namespace Windows::AI::MachineLearning

#pragma warning(pop)
