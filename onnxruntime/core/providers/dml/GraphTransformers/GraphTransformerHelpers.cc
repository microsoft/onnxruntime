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
    void RegisterGraphTransformers(onnxruntime::InferenceSession* lotusSession, bool registerLotusTransforms)
    {
        // Register Lotus graph transformers
        //
        // TODO: Work out issues controlling graph optimization passes through ORT's optimization level
        // and rule list.  In the meantime (and before new transformers are tested in Winml), passes
        // are registered explicitly, and the optimization level is set to default above (no optimization).
        // 
        // Issues:
        // Why is UnsqueezeElimination not registered by name in ORT?
        // Why are level 2 (default) transformers not run before partitioning, which the DML XP requires?  
        // Why are level2 transformers only enabled on the CPU provider in GenerateTransformers?
        // Why does name filtering only apply to rule based graph transformers?        
        // Why is Matmul+Add not used when contrib ops are disabled?

        if (registerLotusTransforms)
        {
            lotusSession->RegisterGraphTransformer(std::move(std::make_unique<onnxruntime::ConstantFolding>()), onnxruntime::TransformerLevel::Level1);
        }

        std::unique_ptr<onnxruntime::RuleBasedGraphTransformer> rule_transformer =
            std::make_unique<onnxruntime::RuleBasedGraphTransformer>("WinmlRuleTransformer");
        
        if (registerLotusTransforms)
        {
            rule_transformer->Register(std::make_unique<onnxruntime::EliminateIdentity>());
            rule_transformer->Register(std::make_unique<onnxruntime::UnsqueezeElimination>());
            rule_transformer->Register(std::make_unique<onnxruntime::EliminateDropout>());
            rule_transformer->Register(std::make_unique<onnxruntime::EliminateSlice>());
            rule_transformer->Register(std::make_unique<onnxruntime::ConvBNFusion>());
            rule_transformer->Register(std::make_unique<onnxruntime::ConvMulFusion>());
            rule_transformer->Register(std::make_unique<onnxruntime::ConvAddFusion>());
        }

        rule_transformer->Register(std::make_unique<onnxruntime::BatchNormalizationMulFusion>());
        rule_transformer->Register(std::make_unique<onnxruntime::BatchNormalizationAddFusion>());

        lotusSession->RegisterGraphTransformer(std::move(rule_transformer), onnxruntime::TransformerLevel::Level1);

        if (registerLotusTransforms)
        {
            lotusSession->RegisterGraphTransformer(std::move(std::make_unique<onnxruntime::MatMulAddFusion>()), onnxruntime::TransformerLevel::Level1);
        }
    }
}