// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "DmlSerializedGraphDesc.h"

struct NodeIndex
{
    uint32_t nodeIndex;
    uint32_t nodeOutputIndex;
};

DmlSerializedGraphDesc DeserializeDmlGraph(
    const uint8_t* flatbufferGraphDescBlob,
    /*out*/ std::vector<std::unique_ptr<std::byte[]>>& rawData);