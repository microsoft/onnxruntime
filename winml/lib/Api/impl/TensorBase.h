// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#pragma warning(push)
#pragma warning(disable : 6387)

#include "LearningModelBinding.h"
#include "LearningModelDevice.h"
#include "LearningModelSession.h"
#include "TensorKindFrom.h"
#include "TensorMemoryBufferReference.h"

#include "core/session/onnxruntime_c_api.h"

namespace _winml {

// TensorBase
//
// This is the base class for all data based Tensor types. It exposes array and IVectorView
// based getter and setters.
//
// Look in FeatureValue.h to see where all of them actually get created with CREATE_TENSOR()
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

  TensorBase(wfc::IIterable<int64_t> const& shape) : shape_(begin(shape), end(shape)),
                                                     m_resources(std::make_shared<TensorResources<T>>()) {
    GetCpuResource() = std::make_shared<_winml::Tensor<T>>(shape_);
  }

  TensorBase(std::vector<int64_t> const& shape) : shape_(shape),
                                                  m_resources(std::make_shared<TensorResources<T>>()) {
    GetCpuResource() = std::make_shared<_winml::Tensor<T>>(shape_);
  }

  TensorBase(std::vector<int64_t> const& shape, ID3D12Resource* resource) : shape_(shape),
                                                                            m_resources(std::make_shared<TensorResources<T>>()) {
    // This Api is not supported for TensorString
    WINML_THROW_HR_IF_TRUE_MSG(
        E_ILLEGAL_METHOD_CALL,
        (std::is_same<T, std::string>::value),
        "TensorString objects cannot be created from a ID3D12Resource!");

    GetGpuResource().copy_from(resource);
  }

  HRESULT CreateGPUMLValue(ID3D12Resource* resource, BindingContext& context, IValue** out) {
    THROW_HR_IF_NULL(E_INVALIDARG, resource);

    auto session = context.session.as<winmlp::LearningModelSession>();
    auto device = session->Device().as<winmlp::LearningModelDevice>();
    WINML_THROW_HR_IF_TRUE_MSG(WINML_ERR_INVALID_BINDING,
                               device->IsCpuDevice(),
                               "Cannot create GPU tensor on CPU device");

    auto engine = session->GetEngine();
    RETURN_IF_FAILED(engine->CreateTensorValueFromExternalD3DResource(resource, shape_.data(), shape_.size(), TensorKind(), out));
    return S_OK;
  }

  HRESULT CPUTensorize(_winml::BindingContext& context, IValue** out) {
    auto session = context.session.as<winmlp::LearningModelSession>();
    auto engine = session->GetEngine();

    if (GetCpuResource() != nullptr) {
      return CreateTensorValueFromExternalBuffer(engine, out);
    }

    // If there is no matching cpu resource, then fallback to a gpu resource
    if (GetGpuResource() != nullptr) {
      return CreateGPUMLValue(GetGpuResource().get(), context, out);
    }

    WINML_THROW_HR(WINML_ERR_INVALID_BINDING);
  }

  HRESULT GPUTensorize(_winml::BindingContext& context, IValue** out) {
    if (GetGpuResource() != nullptr) {
      return CreateGPUMLValue(GetGpuResource().get(), context, out);
    }

    // Get engine
    auto session = context.session.as<winmlp::LearningModelSession>();
    auto engine = session->GetEngine();

    // If there is no matching gpu resource, then fallback to a cpu resource
    if (GetCpuResource() != nullptr) {
      return CreateTensorValueFromExternalBuffer(engine, out);
    }

    if (TensorKind() == winml::TensorKind::String) {
      // Lazily allocate the cpu TensorString resource
      // TensorStrings are CPU only, and so a gpu resource cannot be allocated for them.
      GetCpuResource() = std::make_shared<_winml::Tensor<T>>(shape_);
      return CreateTensorValueFromExternalBuffer(engine, out);
    } else {
      // Try to allocate the backing memory for the caller
      auto bufferSize = std::accumulate(std::begin(shape_), std::end(shape_), static_cast<int64_t>(1), std::multiplies<int64_t>());
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

      auto device = session->Device().as<winmlp::LearningModelDevice>();

      winrt::com_ptr<ID3D12Resource> gpu_resource = nullptr;
      device->GetD3DDevice()->CreateCommittedResource(
          &heapProperties,
          D3D12_HEAP_FLAG_NONE,
          &resourceDesc,
          D3D12_RESOURCE_STATE_COMMON,
          nullptr,
          __uuidof(ID3D12Resource),
          gpu_resource.put_void());

      GetGpuResource() = gpu_resource;

      return CreateGPUMLValue(GetGpuResource().get(), context, out);
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
  STDMETHOD(GetValue)
  (_winml::BindingContext& context, IValue** out) {
    RETURN_HR_IF_NULL_MSG(
        WINML_ERR_INVALID_BINDING,
        m_resources,
        "The tensor has been closed and its resources have been detached!");

    EnsureBufferNotInUse();

    auto spSession = context.session.as<winmlp::LearningModelSession>();
    auto spDevice = spSession->Device().as<winmlp::LearningModelDevice>();

    if (spDevice->IsCpuDevice()) {
      RETURN_IF_FAILED(CPUTensorize(context, out));
    } else {
      RETURN_IF_FAILED(GPUTensorize(context, out));
    }

    return S_OK;
  }

  static int64_t ShapeSize(std::vector<int64_t> shape) {
    // for each dim
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      // find out it's total size
      size *= shape[i];
      // make sure there are no invalid dimensions (-1 or any invalid shape)
      THROW_HR_IF(E_INVALIDARG, shape[i] <= 0);
    }
    return size;
  }

  template <typename ElementType = T, typename ElementViewType = ViewT>
  void SetBufferFromValueResourceBuffer(uint32_t size, void* data) {
    // This adds compile time checks that ensure that the API can only be called when
    // the conditions of ASSERT_TEMPLATE_PARAMETERS_EXACT() are met.
    ASSERT_TEMPLATE_PARAMETERS<ElementType, ElementViewType>();

    GetCpuResource()->set(size, reinterpret_cast<ElementType*>(data));
  }

  template <>
  void SetBufferFromValueResourceBuffer<std::string, winrt::hstring>(uint32_t size, void* data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<std::string, winrt::hstring>();

    GetCpuResource()->get_tensor_buffer()->Set(size, reinterpret_cast<std::string_view*>(data));
  }

  template <typename ElementType = T, typename ElementViewType = ViewT>
  HRESULT CreateTensorValueFromExternalBuffer(_winml::IEngine* engine, IValue** value) {
    // This adds compile time checks that ensure that the API can only be called when
    // the conditions of ASSERT_TEMPLATE_PARAMETERS_EXACT() are met.
    ASSERT_TEMPLATE_PARAMETERS<ElementType, ElementViewType>();

    RETURN_IF_FAILED_MSG(engine->CreateTensorValueFromExternalBuffer(
                             GetCpuResource()->buffer().second, GetCpuResource()->size_in_bytes(), GetCpuResource()->shape().data(),
                             GetCpuResource()->shape().size(), TensorKind(), value),
                         "Failed to prepare buffer for copy back from device resource.");
    return S_OK;
  }

  template <>
  HRESULT CreateTensorValueFromExternalBuffer<std::string, winrt::hstring>(_winml::IEngine* engine, IValue** value) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<std::string, winrt::hstring>();

    std::vector<const char*> raw_values;
    auto string_array = GetCpuResource()->buffer().second;
    std::transform(
        string_array,
        string_array + GetCpuResource()->size_in_bytes(),
        std::back_inserter(raw_values),
        [&](auto& str) { return str.c_str(); });

    RETURN_IF_FAILED_MSG(engine->CreateStringTensorValueFromDataWithCopy(
                             raw_values.data(), raw_values.size(), GetCpuResource()->shape().data(),
                             GetCpuResource()->shape().size(), value),
                         "Failed to prepare buffer for copy back from device resource.");
    return S_OK;
  }

  // ILotusValueProviderPrivate::UpdateSourceResourceData
  STDMETHOD(UpdateSourceResourceData)
  (BindingContext& context, IValue* value) {
    RETURN_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources have been detached during evaluation!");

    _winml::Resource updated_resource;
    RETURN_IF_FAILED(value->GetResource(updated_resource));

    // get the shape
    RETURN_IF_FAILED_MSG(value->GetTensorShape(shape_), "Failed to get the tensor shape from resource!");

    // make sure we always have a CPU resource
    if (GetCpuResource() == nullptr) {
      GetCpuResource() = std::make_shared<_winml::Tensor<T>>(shape_);
    }

    bool is_cpu;
    if (SUCCEEDED(value->IsCpu(&is_cpu)) && is_cpu) {
      // Get the data pointer and size
      T* data;
      uint32_t size;
      std::tie(size, data) = GetCpuResource()->buffer();

      if (updated_resource.get() != reinterpret_cast<void*>(data)) {
        // Only copy the data if the source and destination are not the same!
        // The engine provided buffer will not match the tensor buffer when
        // the tensor is created as a placeholder output, or as an unbound output.
        auto shape_size = static_cast<uint32_t>(ShapeSize(shape_));
        SetBufferFromValueResourceBuffer(shape_size, updated_resource.get());
      }
    } else {
      // If we got a gpu resource, we should move the data to the cpu so accessors can retrieve the data.
      // We don't need to copy the engine provided dx resource into a local copy since we always preallocate gpu
      // resources for tensors. Therefore we are certain that the returned dxresource is the same as the one we passed in
      // and was updated in place.
      auto spSession = context.session.as<winmlp::LearningModelSession>();
      auto engine = spSession->GetEngine();

      winrt::com_ptr<IValue> dest;
      RETURN_IF_FAILED_MSG(CreateTensorValueFromExternalBuffer(engine, dest.put()),
                           "Failed to prepare buffer for copy back from device resource.");
      RETURN_IF_FAILED(engine->CopyValueAcrossDevices(value, dest.get()));
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
      wfc::IIterable<int64_t> const& shape) try {
    typename TBase::class_type tensorValue = winrt::make<TDerived>();
    auto tensorValueImpl = tensorValue.as<TDerived>();
    tensorValueImpl->shape_ = std::vector<int64_t>(begin(shape), end(shape));
    return tensorValue;
  }
  WINML_CATCH_ALL

  // ITensor<T>::CreateFromIterable
  static typename TBase::class_type CreateFromIterable(
      wfc::IIterable<int64_t> shape,
      wfc::IIterable<ViewT> const& data) try {
    std::vector<int64_t> vecShape(begin(shape), end(shape));
    if (HasFreeDimensions(vecShape)) {
      // If the tensor is being created with a free dimension, the data needs to
      // provide its actual size so that the free dimension can be computed.
      // In the case of IIterable<T>, there is no Size accessor, and so we require that
      // in this case the underlying object also implement IVectorView, so that we may
      // efficiently query the size of the data.
      if (auto vectorView = data.try_as<wfc::IVectorView<ViewT>>()) {
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
      wfc::IIterable<int64_t> shape,
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
      wss::IBuffer const& buffer) try {
    std::vector<int64_t> vecShape(shape.begin(), shape.end());
    typename TBase::class_type tensorValue = winrt::make<TDerived>();
    auto tensorValueImpl = tensorValue.as<TDerived>();
    tensorValueImpl->shape_ = vecShape;
    tensorValueImpl->GetCpuResource() = std::make_shared<_winml::Tensor<T>>(vecShape, buffer);
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
      UINT64 width = ShapeSize(shapeVector) * sizeof(T);

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
  wf::IMemoryBufferReference CreateReference() try {
    // Create a TensorMemoryBufferReference<T>

    // Per IMemoryBuffer.CreateReference (https://docs.microsoft.com/en-us/uwp/api/windows.foundation.imemorybuffer.createreference)
    // "This method always successfully returns a new IMemoryBufferReference object even after the IMemoryBuffer
    // "has been closed. In that case, the returned IMemoryBufferReference is already closed."
    // Creating a TensorMemoryBufferReference<T> with a null pointer is equivalent to creating it as closed.

    auto memoryBufferReference = winrt::make<TensorMemoryBufferReference<T>>(shape_, m_resources);

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
        (std::is_same_v<T, std::string>),
        "TensorString objects cannot return byte buffers!");

    RETURN_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources have been detached!");

    return m_resources->GetBuffer(shape_, value, capacity);
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

      GetGpuResource().copy_to(ppResource);
      return S_OK;
    }
    WINML_CATCH_ALL_COM
  }

  // ITensor<T>::GetAsVectorView
  template <typename ElementType = T, typename ElementViewType = ViewT>
  wfc::IVectorView<ElementViewType> GetAsVectorView() try {
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
  wfc::IVectorView<float> GetAsVectorView<_winml::Half, float>() try {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<_winml::Half, float>();

    uint32_t size;
    _winml::Half* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    // Copy the HALFs to floats
    std::vector<float> floatValue(size);
    DirectX::PackedVector::XMConvertHalfToFloatStream(
        floatValue.data(),
        sizeof(float) /* output stride */,
        reinterpret_cast<DirectX::PackedVector::HALF*>(pBuffer),
        sizeof(_winml::Half) /* input stride */,
        size);

    // Create IVectorView from copied data.
    return winrt::single_threaded_vector<float>(std::move(floatValue)).GetView();
  }
  WINML_CATCH_ALL

  // Specialized version to convert string to hstring
  template <>
  wfc::IVectorView<winrt::hstring> GetAsVectorView<std::string, winrt::hstring>() try {
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
          return _winml::Strings::HStringFromUTF8(pData[n++]);
        });

    return winrt::single_threaded_vector<winrt::hstring>(std::move(copy)).GetView();
  }
  WINML_CATCH_ALL

  // Specialized version to convert int8_t to uint8_t
  template <>
  wfc::IVectorView<uint8_t> GetAsVectorView<int8_t, uint8_t>() try {
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
  winml::LearningModelFeatureKind Kind() try {
    return winml::LearningModelFeatureKind::Tensor;
  }
  WINML_CATCH_ALL

  // ITensor::TensorKind
  winml::TensorKind TensorKind() try {
    return TensorKindFrom<TInterface>::Type;
  }
  WINML_CATCH_ALL

  // ITensor::Shape
  wfc::IVectorView<int64_t> Shape() try {
    std::vector<int64_t> copy(shape_.cbegin(), shape_.cend());
    return winrt::single_threaded_vector(std::move(copy)).GetView();
  }
  WINML_CATCH_ALL

  // ILotusValueProviderPrivate::AbiRepresentation
  STDMETHOD(AbiRepresentation)
  (wf::IInspectable& abiRepresentation) {
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
  void SetBufferFromArray<_winml::Half, float>(winrt::array_view<float const> data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<_winml::Half, float>();

    uint32_t size;
    _winml::Half* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    THROW_HR_IF(E_UNEXPECTED, data.size() != size);
    DirectX::PackedVector::XMConvertFloatToHalfStream(
        reinterpret_cast<DirectX::PackedVector::HALF*>(pBuffer),
        sizeof(_winml::Half) /* output stride */,
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
          return _winml::Strings::UTF8FromHString(element);
        });
  }

  ///
  /// SetBufferFromIterable and parameterized specializations for MLFloat16, int8_t, and std::string
  ///
  template <typename ElementType = T, typename ElementViewType = ViewT>
  void SetBufferFromIterable(
      wfc::IIterable<ElementViewType> const& data) {
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
  void SetBufferFromIterable<_winml::Half, float>(
      wfc::IIterable<float> const& data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<_winml::Half, float>();

    uint32_t size;
    _winml::Half* pBuffer;

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
      wfc::IIterable<uint8_t> const& data) {
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
      wfc::IIterable<winrt::hstring> const& data) {
    // Ensure that this call is being called with the correct template parameters
    ASSERT_TEMPLATE_PARAMETERS<std::string, winrt::hstring>();

    uint32_t size;
    std::string* pBuffer;

    // Get the data pointer and size
    std::tie(size, pBuffer) = GetCpuResource()->buffer();

    // Convert and copy into the underlying buffer
    std::transform(begin(data), end(data), pBuffer, [](const auto& element) {
      return _winml::Strings::UTF8FromHString(element);
    });
  }

  std::shared_ptr<_winml::Tensor<T>>& GetCpuResource() {
    WINML_THROW_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources are detached!");

    return m_resources->CpuResource;
  }

  winrt::com_ptr<ID3D12Resource>& GetGpuResource() {
    WINML_THROW_HR_IF_NULL_MSG(
        E_ILLEGAL_METHOD_CALL,
        m_resources,
        "The tensor has been closed and its resources are detached!");

    return m_resources->GpuResource;
  }

 private:
  std::vector<int64_t> shape_;
  std::shared_ptr<TensorResources<T>> m_resources;
  std::vector<winrt::weak_ref<TensorMemoryBufferReference<T>>> m_outstandingReferences;
  bool m_isClosed = false;
};

}  // namespace _winml

#pragma warning(pop)
