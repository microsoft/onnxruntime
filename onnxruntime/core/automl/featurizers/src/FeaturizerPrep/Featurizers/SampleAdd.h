// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------
#pragma once

#include "../Featurizer.h"

namespace Microsoft {
namespace Featurizer {

/////////////////////////////////////////////////////////////////////////
///  \namespace     SampleAdd
///  \brief         A Transformer and Estimator that add values. This is a
///                 sample intended to demonstrate patterns within the
///                 implementation of these types.
///
namespace SampleAdd {

/////////////////////////////////////////////////////////////////////////
///  \class         Transformer
///  \brief         Transformer that adds an integer value to a saved delta
///                 and returns the result.
///
class Transformer : public Microsoft::Featurizer::Transformer<std::uint32_t, std::uint16_t> {
public:
    // ----------------------------------------------------------------------
    // |  Public Methods
    Transformer(std::uint16_t delta=0);
    ~Transformer(void) override = default;

    Transformer(Transformer const &) = delete;
    Transformer & operator =(Transformer const &) = delete;

    Transformer(Transformer &&) = default;
    Transformer & operator =(Transformer &&) = delete;

    return_type transform(arg_type const &arg) const override;

private:
    // ----------------------------------------------------------------------
    // |  Private Data
    std::uint32_t const                     _delta;

    // ----------------------------------------------------------------------
    // |  Private Methods
    template <typename ArchiveT>
    void serialize(ArchiveT &ar, unsigned int const version);
};

/////////////////////////////////////////////////////////////////////////
///  \class         Estimator
///  \brief         Estimator that accumulates a delta value and then
///                 creates a Transformer with than value when requested.
///
class Estimator : public Microsoft::Featurizer::Estimator<std::uint32_t, std::uint16_t> {
public:
    // ----------------------------------------------------------------------
    // |  Public Methods
    Estimator(void) = default;
    ~Estimator(void) override = default;

    Estimator(Estimator const &) = delete;
    Estimator & operator =(Estimator const &) = delete;

    Estimator(Estimator &&) = default;
    Estimator & operator =(Estimator &&) = delete;

private:
    // ----------------------------------------------------------------------
    // |  Private Data
    std::uint32_t                           _accumulated_delta = 0;

    // ----------------------------------------------------------------------
    // |  Private Methods
    template <typename ArchiveT>
    void serialize(ArchiveT &ar, unsigned int const version);

    Estimator & fit_impl(apache_arrow const &data) override;
    TransformerUniquePtr commit_impl(void) override;
};

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// |
// |  Implementation
// |
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
} // namespace SampleAdd

} // namespace Featurizer
} // namespace Microsoft
