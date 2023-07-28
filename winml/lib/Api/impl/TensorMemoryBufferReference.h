// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Tensor.h"
#include <type_traits>

#include <Memorybuffer.h>

namespace _winml {

template <typename T>
struct TensorResources {
  // ITensorNative::GetBuffer
  STDMETHOD(GetBuffer)(std::vector<int64_t> shape, BYTE** value, UINT32* capacity) {
    RETURN_HR_IF_NULL(E_POINTER, value);
    RETURN_HR_IF_NULL(E_POINTER, capacity);

    RETURN_HR_IF_MSG(
      ERROR_INVALID_FUNCTION, (std::is_same_v<T, std::string>), "TensorString objects cannot return byte buffers!"
    );

    try {
      *value = nullptr;
      *capacity = 0;

      // Lazily allocate the cpu resource on call to GetBuffer
      if (cpu_resource_ == nullptr) {
        cpu_resource_ = std::make_shared<_winml::Tensor<T>>(shape);
      }

      // Get the data pointer and size
      auto buffer = cpu_resource_->buffer();

      // Set out parameters
      *capacity = static_cast<uint32_t>(buffer.size_bytes());
      *value = reinterpret_cast<byte*>(buffer.data());
      return S_OK;
    }
    WINML_CATCH_ALL_COM
  }

  virtual ~TensorResources() {}

  // Theses are access directly by TensorMemoryBufferReference<T> and TensorBase
  std::shared_ptr<_winml::Tensor<T>> cpu_resource_;
  winrt::com_ptr<ID3D12Resource> gpu_resource_;
};

// This class holds onto the lifetime of TensorResources<T> so that they can be kept alive by TensorBase AND its active MBRs.
// When the last MBR/Tensor object is destroyed then TensorResources<T> and its associated cpu and gpu resources will be destroyed.
// The source MB (the tensor object) holds weak references to its TensorMemoryBufferReference<T> MBRs to determine whether
//   there are external callers of the API that are actively using native interface access.
// The template parameter <T> is used to determine the type type of the underlying cpu resource (float, int, etc...).
template <typename T>
class TensorMemoryBufferReference : public winrt::implements<
                                      TensorMemoryBufferReference<T>,
                                      wf::IMemoryBufferReference,
                                      wf::IClosable,
                                      Windows::Foundation::IMemoryBufferByteAccess> {
  using ClosedDelegate = wf::TypedEventHandler<wf::IMemoryBufferReference, wf::IInspectable>;

 public:
  // winrt::Windows::Foundation::IMemoryBufferReference
  //
  // Parameters:
  //
  // shape:           The shape of the tensor being referenced
  // tensorResources: An optional shared_ptr to underlying resources (cpu or gpu).
  //                  This will be null when the source Tensor* object has already been closed.
  //                  When the source IMemoryBuffer is closed, the IMemoryBuffer spec requires the
  //                  successful creation of IMemoryBufferReferences in the closed state.
  TensorMemoryBufferReference(std::vector<int64_t> shape, std::shared_ptr<TensorResources<T>> tensorResources)
    : m_shape(shape),
      m_tensorResources(tensorResources),
      m_handlers() {}

  uint32_t Capacity() const try {
    uint32_t uCapacity = 0;

    // Per IMemoryBuffer.CreateReference (https://docs.microsoft.com/en-us/uwp/api/windows.foundation.imemorybuffer.createreference)
    // If the IMemoryBufferReference has been closed (m_tensorResources == nullptr) then
    // "IMemoryBufferReference instance's Capacity property will be zero."
    if (m_tensorResources) {
      BYTE* pCPUTensor;
      WINML_THROW_IF_FAILED(m_tensorResources->GetBuffer(m_shape, reinterpret_cast<BYTE**>(&pCPUTensor), &uCapacity));
    }

    return uCapacity;
  }
  WINML_CATCH_ALL

  winrt::event_token Closed(const ClosedDelegate& handler) try {
    auto token = m_eventTokenCounter++;
    m_handlers[token] = handler;
    return winrt::event_token{token};
  }
  WINML_CATCH_ALL

  void Closed(winrt::event_token const& cookie) try { m_handlers.erase(cookie.value); }
  WINML_CATCH_ALL

  // Windows::Foundation::IClosable
  void Close() try {
    if (m_tensorResources) {
      // This event must be fired before m_tensorResources are released
      // so that callers can access the data one last time.
      FireClosed();

      // When the object is closed, release the reference to the Tensor
      m_tensorResources = nullptr;
    }
  }
  WINML_CATCH_ALL

  STDMETHOD(GetBuffer)
  (_Outptr_result_buffer_(*capacity) BYTE** value, _Out_ UINT32* capacity) try {
    RETURN_HR_IF_NULL(E_POINTER, value);
    RETURN_HR_IF_NULL(E_POINTER, capacity);

    *value = nullptr;
    *capacity = 0;

    // Per IMemoryBuffer.CreateReference (https://docs.microsoft.com/en-us/uwp/api/windows.foundation.imemorybuffer.createreference)
    // If the IMemoryBufferReference has been closed (m_tensorResources == nullptr) then
    // "IMemoryBufferByteAccess::GetBuffer method will always return a null memory pointer and zero capacity."
    RETURN_HR_IF_NULL(S_OK, m_tensorResources);

    return m_tensorResources->GetBuffer(m_shape, value, capacity);
  }
  WINML_CATCH_ALL_COM

 private:
  void FireClosed() {
    wf::IMemoryBufferReference memoryBufferReference = nullptr;
    WINML_THROW_IF_FAILED(this->QueryInterface(
      winrt::guid_of<wf::IMemoryBufferReference>(), reinterpret_cast<void**>(winrt::put_abi(memoryBufferReference))
    ));

    for (auto handler : m_handlers) {
      handler.second(memoryBufferReference, nullptr);
    }
  }

 private:
  std::vector<int64_t> m_shape;
  std::shared_ptr<TensorResources<T>> m_tensorResources;
  std::unordered_map<int64_t, ClosedDelegate> m_handlers;
  int64_t m_eventTokenCounter = 0;
};

}  // namespace _winml
