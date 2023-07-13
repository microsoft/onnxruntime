// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef LIBRARY_EXPORTS
#    define LIBRARY_API __declspec(dllexport)
#else
#    define LIBRARY_API __declspec(dllimport)
#endif

LIBRARY_API flatbuffers::DetachedBuffer SerializeDmlGraph(const DmlSerializedGraphDesc& graphDesc);