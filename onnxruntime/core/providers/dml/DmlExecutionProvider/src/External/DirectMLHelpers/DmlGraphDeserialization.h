// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "DmlGraphDesc_generated.h"

struct NodeIndex
{
    uint32_t nodeIndex;
    uint32_t nodeOutputIndex;
};

OperatorFieldVariant CreateAttribute(
    const DML_SCHEMA_FIELD* schemaField,
    const dml::ir::operatorFieldTypes::AttributeDesc* attributeDesc);

DmlSerializedGraphDesc DeserializeDmlGraph(
    const uint8_t* flatbufferGraphDescBlob,
    /*out*/ std::vector<std::unique_ptr<std::byte[]>>& rawData);