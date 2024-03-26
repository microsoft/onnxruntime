// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "DmlGraphDesc_generated.h"

struct DmlSerializedGraphDesc;

flatbuffers::DetachedBuffer SerializeDmlGraph(const DmlSerializedGraphDesc& graphDesc);
