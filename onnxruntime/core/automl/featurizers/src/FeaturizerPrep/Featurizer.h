// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------
#pragma once

#include <memory>
#include <tuple>

namespace Microsoft {
namespace Featurizer {

/////////////////////////////////////////////////////////////////////////
///  \class         Transformer
///  \brief         Transforms a single "value" and output the result.
///                 A value can be anything from an integer to a collection
///                 of integers.
///
template <typename ReturnT, typename ArgT>
class Transformer {
public:
    // ----------------------------------------------------------------------
    // |  Public Types
    using return_type                       = ReturnT;
    using arg_type                          = ArgT;
    using transformer_type                  = Transformer<ReturnT, ArgT>;

    // ----------------------------------------------------------------------
    // |  Public Methods
    Transformer(void) = default;
    virtual ~Transformer(void) = default;

    Transformer(Transformer const &) = delete;
    Transformer & operator =(Transformer const &) = delete;

    Transformer(Transformer &&) = default;
    Transformer & operator =(Transformer &&) = delete;

    virtual return_type transform(arg_type const &arg) const = 0;

private:
    // ----------------------------------------------------------------------
    // |  Private Methods
    template <typename ArchiveT>
    void serialize(ArchiveT &, unsigned int const /*version*/);
};

/////////////////////////////////////////////////////////////////////////
///  \class         Estimator
///  \brief         Collects state over a collection of data, then produces
///                 a `Transformer` that is able to operate on that collected
///                 state.
///
template <typename ReturnT, typename ArgT>
class Estimator {
public:
    // ----------------------------------------------------------------------
    // |  Public Types
    using transformer_type                  = Transformer<ReturnT, ArgT>;
    using TransformerUniquePtr              = std::unique_ptr<transformer_type>;

    using estimator_type                    = Estimator<ReturnT, ArgT>;

    using apache_arrow                      = unsigned long; // TODO: Temp type as we figure out what will eventually be here

    // ----------------------------------------------------------------------
    // |  Public Methods
    Estimator(void) = default;
    virtual ~Estimator(void) = default;

    Estimator(Estimator const &) = delete;
    Estimator & operator =(Estimator const &) = delete;

    Estimator(Estimator &&) = default;
    Estimator & operator =(Estimator &&) = delete;

    // This method can be called repeatedly in the support of streaming scenarios
    Estimator & fit(apache_arrow const &data);

    // Calls to `commit` are destructive - all previously generated state should
    // be reset. `Estimator` objects that want to share state prior to calls to commit
    // should implement a `copy` method.
    TransformerUniquePtr commit(void);

private:
    // ----------------------------------------------------------------------
    // |  Private Data
    bool _committed                         = false;

    // ----------------------------------------------------------------------
    // |  Private Methods
    template <typename ArchiveT>
    void serialize(ArchiveT &, unsigned int const /*version*/);

    virtual Estimator & fit_impl(apache_arrow const &data) = 0;
    virtual TransformerUniquePtr commit_impl(void) = 0;
};

template <typename EstimatorT, typename... EstimatorConstructorArgsT>
typename EstimatorT::TransformerUniquePtr fit_and_commit(typename EstimatorT::apache_arrow const &data, EstimatorConstructorArgsT &&...args);

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// |
// |  Implementation
// |
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// |
// |  Transformer
// |
// ----------------------------------------------------------------------
template <typename ReturnT, typename ArgT>
template <typename ArchiveT>
void Transformer<ReturnT, ArgT>::serialize(ArchiveT & /*ar*/, unsigned int const /*version*/) {
}

// ----------------------------------------------------------------------
// |
// |  Estimator
// |
// ----------------------------------------------------------------------
template <typename ReturnT, typename ArgT>
Estimator<ReturnT, ArgT> & Estimator<ReturnT, ArgT>::fit(apache_arrow const &data) {
    if(_committed)
        throw std::runtime_error("This instance has already been committed");

    return fit_impl(data);
}

template <typename ReturnT, typename ArgT>
typename Estimator<ReturnT, ArgT>::TransformerUniquePtr Estimator<ReturnT, ArgT>::commit(void) {
    if(_committed)
        throw std::runtime_error("This instance has already been committed");

    TransformerUniquePtr                    result(commit_impl());

    if(!result)
        throw std::runtime_error("Invalid result");

    _committed = true;
    return result;
}

template <typename ReturnT, typename ArgT>
template <typename ArchiveT>
void Estimator<ReturnT, ArgT>::serialize(ArchiveT & /*ar*/, unsigned int const /*version*/) {
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
template <typename EstimatorT, typename... EstimatorConstructorArgsT>
typename EstimatorT::TransformerUniquePtr fit_and_commit(typename EstimatorT::apache_arrow const &data, EstimatorConstructorArgsT &&...args) {
    return EstimatorT(std::forward<EstimatorConstructorArgsT>(args)...).fit(data).commit();
}

} // namespace Featurizer
} // namespace Microsoft
