// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "NoisyReluCpu.h"
#include "ReluCpu.h"
#include <winnt.h>

struct CustomOperatorProvider :
    winrt::implements<
        CustomOperatorProvider,
        winml::ILearningModelOperatorProvider,
        ILearningModelOperatorProviderNative>
{
    HMODULE m_library;
    winrt::com_ptr<IMLOperatorRegistry> m_registry;

    CustomOperatorProvider()
    {
      std::wostringstream dll;
      dll << BINARY_NAME;
      auto winml_dll_name =  dll.str();

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
        m_library = LoadLibraryExW(winml_dll_name.c_str(), nullptr, 0);
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_PC_APP)
        m_library = LoadPackagedLibrary(winml_dll_name.c_str(), 0 /*Reserved*/);
#endif
        WINML_EXPECT_TRUE(m_library != 0);

        using create_registry_delegate = HRESULT WINAPI (_COM_Outptr_ IMLOperatorRegistry** registry);
        auto create_registry = reinterpret_cast<create_registry_delegate*>(GetProcAddress(m_library, "MLCreateOperatorRegistry"));
        if (FAILED(create_registry(m_registry.put())))
        {
            __fastfail(0);
        }

        RegisterSchemas();
        RegisterKernels();
    }   

    ~CustomOperatorProvider()
    {
        FreeLibrary(m_library);
    }

    void RegisterSchemas()
    {
        NoisyReluOperatorFactory::RegisterNoisyReluSchema(m_registry);
    }

    void RegisterKernels()
    {
        // Replace the Relu operator kernel
        ReluOperatorFactory::RegisterReluKernel(m_registry);

        // Add a new operator kernel for Relu
        NoisyReluOperatorFactory::RegisterNoisyReluKernel(m_registry);
    }

    STDMETHOD(GetRegistry)(IMLOperatorRegistry** ppOperatorRegistry)
    {
        m_registry.copy_to(ppOperatorRegistry);
        return S_OK;
    }
};