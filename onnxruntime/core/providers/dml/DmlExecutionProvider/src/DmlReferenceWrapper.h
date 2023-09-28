// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    template <typename T>
    class DmlReferenceWrapper : public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        DmlReferenceWrapper(T inner) : m_inner(std::move(inner)) {}
        T& GetInner() { return m_inner; }

    private:
        T m_inner;
    };
}
