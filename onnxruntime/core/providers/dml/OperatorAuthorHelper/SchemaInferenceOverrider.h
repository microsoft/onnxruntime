// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "OperatorHelper.h"
#include "OperatorRegistration.h"

namespace SchemaInferenceOverrider
{
    // Overrides a shape and type inference function within the static ONNX registry with
    // a shape inference function generated from WinML's operator helpers (which are also
    // registered with DML kernels). This should only be used for operators which don't
    // require type inference functions.
    template <typename T>
    void OverrideSchemaInferenceFunction(
        _In_z_ const char* name, 
        int version, 
        bool isLatest, 
        gsl::span<const uint32_t> constantCpuInputs
    )
    {
        Microsoft::WRL::ComPtr<MLOperatorShapeInferrer> shapeInferrer =
            wil::MakeOrThrow<MLOperatorShapeInferrer>(OperatorHelper::ShapeInferenceFunction<T>);

        auto schema = const_cast<onnx::OpSchema*>(onnx::OpSchemaRegistry::Schema(name, version));

        std::vector<uint32_t> constantCpuInputsCapture(constantCpuInputs.begin(), constantCpuInputs.end());
        schema->TypeAndShapeInferenceFunction([=](onnx::InferenceContext& ctx) {
            onnxruntime::OpNodeProtoHelper<onnx::InferenceContext> nodeInfo(&ctx);

            if (Windows::AI::MachineLearning::Adapter::InputTensorShapesDefinedOnNode(nodeInfo))
            {
                // Check that required constant CPU inputs exist
                for (uint32_t inputIndex : constantCpuInputsCapture)
                {
                    if (inputIndex >= ctx.getNumInputs() || !ctx.getInputData(inputIndex))
                    {
                        return;
                    }
                }

                auto abiContext =
                    wil::MakeOrThrow<Windows::AI::MachineLearning::Adapter::MLSchemaInferenceContext>(
                        &nodeInfo, &ctx, constantCpuInputsCapture);

                THROW_IF_FAILED(shapeInferrer->InferOutputShapes(abiContext.Get()));
                abiContext->Close();
            }
        });

        if (isLatest)
        {
            // Assert that this is the latest schema version for the operator, since a new version might need
            // the same treatment.
            const uint32_t maxVersion = 9;
            assert(
                !onnx::OpSchemaRegistry::Schema(name, maxVersion) ||
                onnx::OpSchemaRegistry::Schema(name, maxVersion) == onnx::OpSchemaRegistry::Schema(name, version));
        }
    }

#pragma push_macro("REGISTER_FUSED_OP_SCHEMA")
#define OVERRIDE_SCHEMA(version, isLatest, opName) \
OverrideSchemaInferenceFunction<OperatorHelper::ShapeInferenceHelper_##opName>( \
    #opName, OperatorHelper::OnnxOperatorSet##version##::sc_sinceVer_##opName, isLatest, gsl::span<uint32_t>());
    
#pragma push_macro("OVERRIDE_SCHEMA_EX")
#define OVERRIDE_SCHEMA_EX(version, isLatest, opName, shapeInferenceName, /*CPU constant tensor indices*/ ...) \
OverrideSchemaInferenceFunction<OperatorHelper::ShapeInferenceHelper_##shapeInferenceName>( \
    #opName, OperatorHelper::OnnxOperatorSet##version##::sc_sinceVer_##opName, isLatest, std::vector<uint32_t>({##__VA_ARGS__}));

    inline void OverrideSchemaInferenceFunctions()
    {
        OVERRIDE_SCHEMA(    7,  true,  Conv);
        OVERRIDE_SCHEMA(    7,  true,  Transpose);
        OVERRIDE_SCHEMA(    7,  true,  AveragePool);
        OVERRIDE_SCHEMA(    7,  false, MaxPool);
        OVERRIDE_SCHEMA(    7,  true,  LpPool);
        OVERRIDE_SCHEMA(    7,  true,  Crop);
        OVERRIDE_SCHEMA_EX( 7,  false, Upsample, Upsample7);
        OVERRIDE_SCHEMA_EX( 9,  true,  Upsample, Upsample9, 1);
        OVERRIDE_SCHEMA_EX( 7,  true,  Slice, Slice7);
        OVERRIDE_SCHEMA(    7,  true,  Split);
        OVERRIDE_SCHEMA_EX( 7,  true,  Tile, Tile, 1);
        OVERRIDE_SCHEMA_EX( 8,  true,  Expand, Expand, 1);
        OVERRIDE_SCHEMA(    8,  true,  MaxPool);
        OVERRIDE_SCHEMA_EX( 9,  true,  OneHot, OneHot, 1);
        OVERRIDE_SCHEMA_EX( 10, false, Resize, Resize10, 1);

    }
#pragma pop_macro("OVERRIDE_SCHEMA_EX")
#pragma pop_macro("REGISTER_FUSED_OP_SCHEMA")

}