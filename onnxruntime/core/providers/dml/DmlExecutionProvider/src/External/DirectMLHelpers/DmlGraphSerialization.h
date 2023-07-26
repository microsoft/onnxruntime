// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "DmlGraphDesc_generated.h"

struct DmlSerializedGraphDesc;

// Need to define in header file because of recursive use.
void SerializeAttributeDescs(
    flatbuffers::FlatBufferBuilder& builder,
    const AbstractOperatorDesc& operatorDesc,
    /*out*/ std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>>& attributeDescs);

flatbuffers::DetachedBuffer SerializeDmlGraph(const DmlSerializedGraphDesc& graphDesc);
