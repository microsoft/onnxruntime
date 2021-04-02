// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    constexpr float DefaultEpsilon = 0.00001f;

    namespace ActivationHelper
    {
        float GetDefaultAlpha(DML_OPERATOR_TYPE function);
        float GetDefaultBeta(DML_OPERATOR_TYPE function);
        float GetDefaultGamma(DML_OPERATOR_TYPE function);
        float GetDefaultLambda(DML_OPERATOR_TYPE function);
        float GetDefaultBias(DML_OPERATOR_TYPE function);

    } // namespace ActivationHelper

    namespace FusionHelpers
    {
        struct FusedOpProperties
        {
            std::string opType;
            std::string domain;
        };

        // Checks whether a candidate op can be fused with the specified activation and returns information about the
        // combined fused op if true, null otherwise.
        std::optional<FusedOpProperties> TryGetFusedOp(
            std::string_view candidateOpType,
            std::string_view candidateOpDomain,
            int candidateOpSinceVersion,
            int candidateOpInputCount,
            std::string_view activationOpType,
            std::string_view activationOpDomain,
            int activationOpSinceVersion);

        // Returns true if the given activation operator type supports being fused with a fusable operator, false
        // otherwise.
        bool IsFusableActivationOperator(std::string_view opType, std::string_view domain, int sinceVersion);

        std::optional<ActivationOperatorDesc> TryGetFusedActivationDesc(const MLOperatorKernelCreationContext& kernelInfo);

        // Produces names for attributes added to fused kernels. This effectively prepends a string to distinguish ONNX
        // attributes from those added dynamically via operator fusion. For example, this function would be used to
        // produce the attribute for Activation in a fused Conv+Activation kernel.
        std::string GetFusedAttributeName(std::string_view name);

    } // namespace FusionHelpers

    // Given an axis in ONNX axis numbering, return the axis adjusted for DML based on how the sizes have been coerced.
    // Note this function presumes the axis attribute is relative to the first input tensor (which is always the case).
    uint32_t GetDmlAdjustedAxis(int32_t onnxAxis, const MLOperatorKernelCreationContext& kernelCreationContext, uint32_t dmlDimCount);

    uint32_t GetDmlAdjustedAxis(int32_t onnxAxis, uint32_t onnxDimCount, uint32_t dmlDimCount);

    void GetDmlAdjustedAxes(/*inout*/ gsl::span<const int32_t> axes, uint32_t onnxDimCount, uint32_t dmlDimCount, std::vector<uint32_t>& dmlAxes);

    struct NameAndIndex
    {
        const char* name; // Null terminated.
        uint32_t index;
    };

    template<typename T>
    std::optional<T> TryMapStringToIndex(std::string_view mode, gsl::span<const NameAndIndex> nameAndIndexList)
    {
        static_assert(sizeof(T) == sizeof(uint32_t));
        auto result = TryMapStringToIndex(mode, nameAndIndexList);
        return *reinterpret_cast<std::optional<T>*>(std::addressof(result));
    }

    std::optional<uint32_t> TryMapStringToIndex(std::string_view mode, gsl::span<const NameAndIndex> nameAndIndexList);

    DML_INTERPOLATION_MODE MapStringToInteropolationMode(std::string_view mode);

    DML_DEPTH_SPACE_ORDER MapStringToDepthSpaceMode(std::string_view mode);

} // namespace Dml