//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>

class KleidiAIQ4BitGemmStrategy {
public:
    virtual size_t GetMStep() const = 0;

    virtual size_t GetNStep() const = 0;

    virtual size_t GetRHSPackedSize(size_t N, size_t K, size_t BlkLen) const = 0;

    virtual size_t GetLHSPackedSize(size_t M, size_t K) const = 0;

    virtual void RunRHSPack(size_t N, size_t K, size_t BlkLen, const std::byte* QuantBData,
        const float* QuantBScale, std::byte* PackedQuantBData) const = 0;

    virtual void RunLHSPack(size_t M, size_t K, const float* A, std::byte* QuantA) const = 0;

    virtual void RunMatMul(
        size_t BlkLen,
        const std::byte* QuantA,
        const std::byte* PackedQuantBData,
        float* C,
        const size_t RangeStartM,
        const size_t RangeCountM,
        const size_t RangeStartN,
        const size_t RangeCountN,
        size_t CountK,
        size_t ldc
    ) const = 0;
};

const KleidiAIQ4BitGemmStrategy& GetKleidiAIGemmStrategy();
const KleidiAIQ4BitGemmStrategy& GetKleidiAIGemvStrategy();
