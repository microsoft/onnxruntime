// Copyright 2019 Microsoft Corporation. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef WEAK_BUFFER_H
#define WEAK_BUFFER_H

#include <wrl.h>
#include <wrl/client.h>

#include <windows.storage.streams.h>
#include <robuffer.h>

namespace Microsoft { namespace AI { namespace MachineLearning { namespace Details {

template <typename T>
struct weak_buffer
    : public Microsoft::WRL::RuntimeClass<
          Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRtClassicComMix | Microsoft::WRL::InhibitRoOriginateError>,
          ABI::Windows::Storage::Streams::IBuffer,
          Windows::Storage::Streams::IBufferByteAccess> {
private:
    const T* m_p_begin;
    const T* m_p_end;

public:
    HRESULT RuntimeClassInitialize(_In_ const T* p_begin, _In_ const T* p_end) {
        m_p_begin = p_begin;
        m_p_end = p_end;

        return S_OK;
    }

    virtual HRESULT STDMETHODCALLTYPE get_Capacity(
        UINT32 * value)
    {
        if (value == nullptr) {
            return E_POINTER;
        }

        *value = static_cast<uint32_t>(m_p_end - m_p_begin);
        return S_OK;
    }

    virtual HRESULT STDMETHODCALLTYPE get_Length(
        UINT32 * /*value*/)
    {
        return E_NOTIMPL;
    }

    virtual HRESULT STDMETHODCALLTYPE put_Length(
        UINT32 /*value*/)
    {
        return E_NOTIMPL;
    }

    STDMETHOD(Buffer)(uint8_t** value)
    {
        if (value == nullptr) {
            return E_POINTER;
        }

        *value = reinterpret_cast<uint8_t*>(const_cast<T*>(m_p_begin));
        return S_OK;
    }
};

}}}} // namespace Microsoft::AI::MachineLearning::Details

#endif // WEAK_BUFFER_H
