// Copyright 2019 Microsoft Corporation. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef WEAK_SINGLE_THREADED_ITERABLE_H_
#define WEAK_SINGLE_THREADED_ITERABLE_H_

#include <wrl.h>
#include <wrl/client.h>

namespace Microsoft {
namespace AI {
namespace MachineLearning {
namespace Details {

template <typename T>
struct weak_single_threaded_iterable
  : public Microsoft::WRL::RuntimeClass<
      Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRt | Microsoft::WRL::InhibitRoOriginateError>,
      ABI::Windows::Foundation::Collections::IIterable<T>> {
 private:
  const T* m_p_begin;
  const T* m_p_end;

 public:
  HRESULT RuntimeClassInitialize(_In_ const T* p_begin, _In_ const T* p_end) {
    m_p_begin = p_begin;
    m_p_end = p_end;

    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE
  First(_Outptr_result_maybenull_ ABI::Windows::Foundation::Collections::IIterator<T>** first) {
    if (first == nullptr) {
      return E_POINTER;
    }

    Microsoft::WRL::ComPtr<weak_single_threaded_iterator> iterator;
    auto hr = Microsoft::WRL::MakeAndInitialize<weak_single_threaded_iterator>(&iterator, this);

    if (FAILED(hr)) {
      return hr;
    }

    return iterator.CopyTo(first);
  }

  HRESULT Size(unsigned* p_size) {
    if (p_size == nullptr) {
      return E_POINTER;
    }

    *p_size = static_cast<unsigned>(m_p_end - m_p_begin);
    return S_OK;
  }

  HRESULT At(unsigned index, _Out_ T* p_current) {
    if (p_current == nullptr) {
      return E_POINTER;
    }

    *p_current = *(m_p_begin + index);
    return S_OK;
  }

  HRESULT Has(unsigned index, _Out_ boolean* p_has_current) {
    if (p_has_current == nullptr) {
      return E_POINTER;
    }
    unsigned size;
    auto hr = Size(&size);
    if (FAILED(hr)) {
      return hr;
    }

    *p_has_current = index < size;
    return S_OK;
  }

  struct weak_single_threaded_iterator
    : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::WinRt | Microsoft::WRL::InhibitRoOriginateError>,
        ABI::Windows::Foundation::Collections::IIterator<T>> {
   private:
    Microsoft::WRL::ComPtr<weak_single_threaded_iterable> m_weak_single_threaded_iterable;
    unsigned m_current = 0;

   public:
    HRESULT RuntimeClassInitialize(_In_ weak_single_threaded_iterable* p_weak_single_threaded_iterable) {
      m_weak_single_threaded_iterable = p_weak_single_threaded_iterable;
      return S_OK;
    }

    virtual /* propget */ HRESULT STDMETHODCALLTYPE get_Current(_Out_ T* current) {
      return m_weak_single_threaded_iterable->At(m_current, current);
    }

    virtual /* propget */ HRESULT STDMETHODCALLTYPE get_HasCurrent(_Out_ boolean* hasCurrent) {
      return m_weak_single_threaded_iterable->Has(m_current, hasCurrent);
    }

    virtual HRESULT STDMETHODCALLTYPE MoveNext(_Out_ boolean* hasCurrent) {
      if (SUCCEEDED(m_weak_single_threaded_iterable->Has(m_current, hasCurrent)) && *hasCurrent) {
        m_current++;
        return m_weak_single_threaded_iterable->Has(m_current, hasCurrent);
      }
      return S_OK;
    }
  };
};

}  // namespace Details
}  // namespace MachineLearning
}  // namespace AI
}  // namespace Microsoft

#endif  // WEAK_SINGLE_THREADED_ITERABLE_H_
