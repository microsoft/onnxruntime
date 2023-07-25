// Copyright 2019 Microsoft Corporation. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef RANDOM_ACCESS_STREAM_H
#define RANDOM_ACCESS_STREAM_H

#include <wrl.h>
#include <wrl/client.h>

#include <windows.storage.streams.h>
#include <robuffer.h>

#include <istream>

namespace WinMLTest {

struct BufferBackedRandomAccessStreamReadAsync
  : public Microsoft::WRL::RuntimeClass<
      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRtClassicComMix | Microsoft::WRL::InhibitRoOriginateError>,
      __FIAsyncOperationWithProgress_2_Windows__CStorage__CStreams__CIBuffer_UINT32,
      ABI::Windows::Foundation::IAsyncInfo> {
  InspectableClass(L"WinMLTest.BufferBackedRandomAccessStreamReadAsync", BaseTrust)

    Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IBuffer> buffer_;

  Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncOperationWithProgressCompletedHandler<
    ABI::Windows::Storage::Streams::IBuffer*,
    UINT32>>
    completed_handler_;
  Microsoft::WRL::ComPtr<
    ABI::Windows::Foundation::IAsyncOperationProgressHandler<ABI::Windows::Storage::Streams::IBuffer*, UINT32>>
    progress_handler_;

  AsyncStatus status_ = AsyncStatus::Started;

 public:
  virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Id(
    /* [retval][out] */ __RPC__out unsigned __int32* id
  ) override {
    *id = 0; // Do we need to implement this?
    return S_OK;
  }

  virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Status(
    /* [retval][out] */ __RPC__out AsyncStatus* status
  ) override {
    *status = status_;
    return S_OK;
  }

  virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ErrorCode(
    /* [retval][out] */ __RPC__out HRESULT* /*errorCode*/
  ) override {
    return E_NOTIMPL;
  }

  virtual HRESULT STDMETHODCALLTYPE Cancel(void) override { return E_NOTIMPL; }

  virtual HRESULT STDMETHODCALLTYPE Close(void) override { return E_NOTIMPL; }

  HRESULT SetBuffer(ABI::Windows::Storage::Streams::IBuffer* buffer) {
    buffer_ = buffer;
    status_ = AsyncStatus::Completed;
    if (buffer_ != nullptr) {
      if (completed_handler_ != nullptr) {
        completed_handler_->Invoke(this, ABI::Windows::Foundation::AsyncStatus::Completed);
      }
    }
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE put_Progress(
    ABI::Windows::Foundation::IAsyncOperationProgressHandler<ABI::Windows::Storage::Streams::IBuffer*, UINT32>* handler
  ) override {
    progress_handler_ = handler;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE get_Progress(
    ABI::Windows::Foundation::IAsyncOperationProgressHandler<ABI::Windows::Storage::Streams::IBuffer*, UINT32>** handler
  ) override {
    progress_handler_.CopyTo(handler);
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE put_Completed(ABI::Windows::Foundation::IAsyncOperationWithProgressCompletedHandler<
                                                  ABI::Windows::Storage::Streams::IBuffer*,
                                                  UINT32>* handler) override {
    completed_handler_ = handler;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE get_Completed(ABI::Windows::Foundation::IAsyncOperationWithProgressCompletedHandler<
                                                  ABI::Windows::Storage::Streams::IBuffer*,
                                                  UINT32>** handler) override {
    completed_handler_.CopyTo(handler);
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE GetResults(ABI::Windows::Storage::Streams::IBuffer** results) override {
    if (buffer_ == nullptr) {
      return E_FAIL;
    }

    buffer_.CopyTo(results);
    return S_OK;
  }
};

struct RandomAccessStream
  : public Microsoft::WRL::RuntimeClass<
      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRtClassicComMix | Microsoft::WRL::InhibitRoOriginateError>,
      ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType,
      ABI::Windows::Storage::Streams::IContentTypeProvider,
      ABI::Windows::Storage::Streams::IRandomAccessStream,
      ABI::Windows::Storage::Streams::IInputStream,
      ABI::Windows::Storage::Streams::IOutputStream,
      ABI::Windows::Foundation::IClosable> {
  InspectableClass(L"WinMLTest.RandomAccessStream", BaseTrust)

    private : Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IBuffer> buffer_ = nullptr;
  UINT64 position_ = 0;

 public:
  HRESULT RuntimeClassInitialize(ABI::Windows::Storage::Streams::IBuffer* buffer) {
    buffer_ = buffer;
    position_ = 0;
    return S_OK;
  }

  HRESULT RuntimeClassInitialize(ABI::Windows::Storage::Streams::IBuffer* buffer, UINT64 position) {
    buffer_ = buffer;
    position_ = position;
    return S_OK;
  }

    // Content Provider

  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_ContentType(
    /* [retval, out] */ __RPC__deref_out_opt HSTRING* value
  ) override {
    return WindowsCreateString(nullptr, 0, value);
  }

    // IRandomAccessStream

  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_Size(
    /* [retval, out] */ __RPC__out UINT64* value
  ) override {
    *value = 0;
    uint32_t length;
    buffer_->get_Length(&length);
    *value = static_cast<uint64_t>(length);
    return S_OK;
  }

  /* [propput] */ virtual HRESULT STDMETHODCALLTYPE put_Size(
    /* [in] */ UINT64 /*value*/
  ) override {
    return E_NOTIMPL;
  }

  virtual HRESULT STDMETHODCALLTYPE GetInputStreamAt(
    /* [in] */ UINT64 position,
    /* [retval, out] */ __RPC__deref_out_opt ABI::Windows::Storage::Streams::IInputStream** stream
  ) override {
    return Microsoft::WRL::MakeAndInitialize<RandomAccessStream>(stream, buffer_.Get(), position);
  }

  virtual HRESULT STDMETHODCALLTYPE GetOutputStreamAt(
    /* [in] */ UINT64 /*position*/,
    /* [retval, out] */ __RPC__deref_out_opt ABI::Windows::Storage::Streams::IOutputStream** /*stream*/
  ) override {
    return E_NOTIMPL;
  }

  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_Position(
    /* [retval, out] */ __RPC__out UINT64* value
  ) override {
    *value = position_;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE Seek(
    /* [in] */ UINT64 position
  ) override {
    position_ = position;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE CloneStream(
    /* [retval, out] */ __RPC__deref_out_opt ABI::Windows::Storage::Streams::IRandomAccessStream** stream
  ) override {
    return Microsoft::WRL::MakeAndInitialize<RandomAccessStream>(stream, buffer_.Get(), 0);
  }

  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_CanRead(
    /* [retval, out] */ __RPC__out::boolean* value
  ) override {
    UINT32 length;
    buffer_->get_Length(&length);
    *value = buffer_ != nullptr && position_ < static_cast<UINT64>(length);
    return S_OK;
  }

  /* [propget] */ virtual HRESULT STDMETHODCALLTYPE get_CanWrite(
    /* [retval, out] */ __RPC__out::boolean* value
  ) override {
    *value = false;
    return S_OK;
  }

    // IInputStream
  virtual HRESULT STDMETHODCALLTYPE ReadAsync(
    /* [in] */ __RPC__in_opt ABI::Windows::Storage::Streams::IBuffer* buffer,
    /* [in] */ UINT32 count,
    /* [in] */ ABI::Windows::Storage::Streams::InputStreamOptions /*options*/,
        /* [retval, out] */
    __RPC__deref_out_opt __FIAsyncOperationWithProgress_2_Windows__CStorage__CStreams__CIBuffer_UINT32** operation
  ) override {
    auto read_async = Microsoft::WRL::Make<BufferBackedRandomAccessStreamReadAsync>();
    read_async.CopyTo(operation);

        // perform the "async work" which is actually synchronous atm
    Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IBuffer> spBuffer = buffer;
    Microsoft::WRL::ComPtr<Windows::Storage::Streams::IBufferByteAccess> out_buffer_byte_access;
    spBuffer.As<Windows::Storage::Streams::IBufferByteAccess>(&out_buffer_byte_access);
    byte* out_bytes = nullptr;
    out_buffer_byte_access->Buffer(&out_bytes);

    Microsoft::WRL::ComPtr<Windows::Storage::Streams::IBufferByteAccess> in_buffer_byte_access;
    buffer_.As<Windows::Storage::Streams::IBufferByteAccess>(&in_buffer_byte_access);
    byte* in_bytes = nullptr;
    in_buffer_byte_access->Buffer(&in_bytes);

    memcpy(out_bytes, in_bytes + static_cast<uint32_t>(position_), count);

    read_async->SetBuffer(buffer);

    return S_OK;
  }

    // IOutputStream
  virtual HRESULT STDMETHODCALLTYPE WriteAsync(
    /* [in] */ __RPC__in_opt ABI::Windows::Storage::Streams::IBuffer* /*buffer*/,
    /* [retval, out] */ __RPC__deref_out_opt __FIAsyncOperationWithProgress_2_UINT32_UINT32** /*operation*/
  ) override {
    return E_NOTIMPL;
  }

  virtual HRESULT STDMETHODCALLTYPE FlushAsync(
    /* [retval, out] */ __RPC__deref_out_opt __FIAsyncOperation_1_boolean** /*operation*/
  ) override {
    return E_NOTIMPL;
  }

    // IClosable
  virtual HRESULT STDMETHODCALLTYPE Close(void) override {
    buffer_ = nullptr;
    return S_OK;
  }
};

struct BufferBackedRandomAccessStreamReferenceOpenReadAsync
  : public Microsoft::WRL::RuntimeClass<
      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRtClassicComMix | Microsoft::WRL::InhibitRoOriginateError>,
      __FIAsyncOperation_1_Windows__CStorage__CStreams__CIRandomAccessStreamWithContentType,
      ABI::Windows::Foundation::IAsyncInfo> {
  InspectableClass(L"WinMLTest.BufferBackedRandomAccessStreamReferenceOpenReadAsync", BaseTrust) public
    : Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType> ras_;
  Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncOperationCompletedHandler<
    ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType*>>
    completed_handler_;
  AsyncStatus status_ = AsyncStatus::Started;

  HRESULT SetRandomAccessStream(ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType* ras) {
    ras_ = ras;
    status_ = AsyncStatus::Completed;
    if (ras_ != nullptr) {
      if (completed_handler_ != nullptr) {
        completed_handler_->Invoke(this, status_);
      }
    }
    return S_OK;
  }

  virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Id(
    /* [retval][out] */ __RPC__out unsigned __int32* id
  ) override {
    *id = 0; // Do we need to implement this?
    return S_OK;
  }

  virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Status(
    /* [retval][out] */ __RPC__out AsyncStatus* status
  ) override {
    *status = status_;
    return S_OK;
  }

  virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ErrorCode(
    /* [retval][out] */ __RPC__out HRESULT* /*errorCode*/
  ) override {
    return E_NOTIMPL;
  }

  virtual HRESULT STDMETHODCALLTYPE Cancel(void) override { return E_NOTIMPL; }

  virtual HRESULT STDMETHODCALLTYPE Close(void) override { return E_NOTIMPL; }

  virtual HRESULT STDMETHODCALLTYPE
  put_Completed(ABI::Windows::Foundation::IAsyncOperationCompletedHandler<
                ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType*>* handler) override {
    completed_handler_ = handler;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE
  get_Completed(ABI::Windows::Foundation::IAsyncOperationCompletedHandler<
                ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType*>** handler) override {
    completed_handler_.CopyTo(handler);
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE
  GetResults(ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType** results) override {
    if (ras_ == nullptr) {
      return E_FAIL;
    }
    ras_.CopyTo(results);
    return S_OK;
  }
};

struct BufferBackedRandomAccessStreamReference
  : public Microsoft::WRL::RuntimeClass<
      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRtClassicComMix | Microsoft::WRL::InhibitRoOriginateError>,
      ABI::Windows::Storage::Streams::IRandomAccessStreamReference> {
  InspectableClass(L"WinMLTest.BufferBackedRandomAccessStreamReference", BaseTrust)

    Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IBuffer> buffer_ = nullptr;

 public:
  HRESULT RuntimeClassInitialize(ABI::Windows::Storage::Streams::IBuffer* buffer) {
    buffer_ = buffer;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE OpenReadAsync(
    /* [retval, out] */ __RPC__deref_out_opt
      __FIAsyncOperation_1_Windows__CStorage__CStreams__CIRandomAccessStreamWithContentType** operation
  ) override {
    auto open_read_async = Microsoft::WRL::Make<BufferBackedRandomAccessStreamReferenceOpenReadAsync>();
    open_read_async.CopyTo(operation);

    Microsoft::WRL::ComPtr<RandomAccessStream> ras;
    Microsoft::WRL::MakeAndInitialize<RandomAccessStream>(&ras, buffer_.Get());

    Microsoft::WRL::ComPtr<ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType> ras_interface = nullptr;
    ras.As<ABI::Windows::Storage::Streams::IRandomAccessStreamWithContentType>(&ras_interface);

    open_read_async.Get()->SetRandomAccessStream(ras_interface.Get());
    return S_OK;
  }
};

} // namespace WinMLTest

#endif // RANDOM_ACCESS_STREAM_H
