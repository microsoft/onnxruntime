// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#undef ERROR
#undef OPTIONAL

#include "core/session/inference_session.h"

#include "GraphTransformerHelpers.h"

#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/graph_transformer_level.h"
#include "bn_mul_fusion.h"
#include "bn_add_fusion.h"

namespace GraphTransformerHelpers
{
    void RegisterGraphTransformers(onnxruntime::InferenceSession* lotusSession)
    {
        // Register Lotus graph transformers
        // we were able to combine all of the winml/dml/ort work except for 2 transformers.
        // these 2 are tracked by :
        // Bug 22973884 : Fix issues with BatchNorm + Add and BatchNorm + Mul handling implicit inputs, and move from Winml to ORT
        //
        std::unique_ptr<onnxruntime::RuleBasedGraphTransformer> rule_transformer =
            std::make_unique<onnxruntime::RuleBasedGraphTransformer>("WinmlRuleTransformer");
        ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<onnxruntime::BatchNormalizationMulFusion>()));
        ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<onnxruntime::BatchNormalizationAddFusion>()));
        ORT_THROW_IF_ERROR(lotusSession->RegisterGraphTransformer(std::move(rule_transformer),
                                                                  onnxruntime::TransformerLevel::Level1));
    }
}