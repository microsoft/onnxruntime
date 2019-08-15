//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace GraphTransformerHelpers
{
    void RegisterGraphTransformers(onnxruntime::InferenceSession* lotusSession, bool registerLotusTransforms);
}