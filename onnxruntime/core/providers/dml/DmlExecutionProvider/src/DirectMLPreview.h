//  Copyright (c) Microsoft Corporation.  All rights reserved.

#pragma once

#include "DirectML.h"

extern "C" {

enum DML_PREVIEW_OPERATOR_TYPE
{
    DML_PREVIEW_OPERATOR_FIRST = 0xC0000000,
};

enum DML_GRAPH_OPTION
{
    DML_GRAPH_OPTION_ENABLE_QDQ_CLEANUP,
};

interface DML_DECLARE_INTERFACE("95c22a63-716d-4838-ab7d-ad06de5e4017") IDMLDevice2 : IDMLDevice1
{ 
    IFACEMETHOD(CompileGraph1)(
        const DML_GRAPH_DESC* desc,
        DML_EXECUTION_FLAGS flags,
        UINT graphOptionCount,
        _In_reads_(graphOptionCount) const DML_GRAPH_OPTION* graphOptions,
        REFIID riid, // expected: IDMLCompiledOperator
        _COM_Outptr_opt_ void** ppv
        ) = 0;
};

} // extern "C"
