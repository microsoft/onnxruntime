// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../Featurizer.h"

class MyTransformer : public Microsoft::Featurizer::Transformer<bool, int> {
public:
    // ----------------------------------------------------------------------
    // |  Public Methods
    MyTransformer(bool true_on_odd=false) :
        _true_on_odd(true_on_odd) {
    }

    ~MyTransformer(void) override = default;

    MyTransformer(MyTransformer const &) = delete;
    MyTransformer & operator =(MyTransformer const &) = delete;

    MyTransformer(MyTransformer &&) = default;
    MyTransformer & operator =(MyTransformer &&) = delete;

    return_type transform(arg_type const &arg) const override {
        bool const                          is_odd(arg & 1);

        return _true_on_odd ? is_odd : !is_odd;
    }

private:
    // ----------------------------------------------------------------------
    // |  Relationships
    friend class boost::serialization::access;

    // ----------------------------------------------------------------------
    // |  Private Data
    bool const                              _true_on_odd;

    // ----------------------------------------------------------------------
    // |  Private Methods
    template <typename ArchiveT>
    void serialize(ArchiveT &ar, unsigned int const /*version*/) {
        ar & boost::serialization::base_object<transformer_type>(*this);
        ar & boost::serialization::make_nvp("true_on_odd", const_cast<bool &>(_true_on_odd));
    }
};

class MyEstimator : public Microsoft::Featurizer::Estimator<bool, int> {
public:
    // ----------------------------------------------------------------------
    // |  Public Methods
    MyEstimator(bool return_invalid_transformer=false) :
        _return_invalid_transformer(return_invalid_transformer) {
    }

    ~MyEstimator(void) override = default;

    MyEstimator(MyEstimator const &) = delete;
    MyEstimator & operator =(MyEstimator const &) = delete;

    MyEstimator(MyEstimator &&) = default;
    MyEstimator & operator =(MyEstimator &&) = delete;

private:
    // ----------------------------------------------------------------------
    // |  Relationships
    friend class boost::serialization::access;

    // ----------------------------------------------------------------------
    // |  Private Data
    bool const                              _return_invalid_transformer;
    bool                                    _true_on_odd_state;

    // ----------------------------------------------------------------------
    // |  Private Methods
    MyEstimator & fit_impl(apache_arrow const &data) override {
        _true_on_odd_state = static_cast<bool>(data);
        return *this;
    }

    TransformerUniquePtr commit_impl(void) override {
        if(_return_invalid_transformer)
            return TransformerUniquePtr();

        return std::make_unique<MyTransformer>(_true_on_odd_state);
    }

    template <typename ArchiveT>
    void serialize(ArchiveT &ar, unsigned int const /*version*/) {
        ar & boost::serialization::base_object<estimator_type>(*this);
        ar & boost::serialization::make_nvp("return_invalid_transformer", const_cast<bool &>(_return_invalid_transformer));
        ar & boost::serialization::make_nvp("true_on_odd_state", const_cast<bool &>(_true_on_odd_state));
    }
};

TEST_CASE("Transformer: Functionality") {
    CHECK(MyTransformer(true).transform(1) == true);
    CHECK(MyTransformer(false).transform(1) == false);
    CHECK(MyTransformer(true).transform(2) == false);
    CHECK(MyTransformer(false).transform(2) == true);
}

TEST_CASE("Estimator: Functionality") {
    CHECK(MyEstimator().fit(1).commit()->transform(1) == true);
    CHECK(MyEstimator().fit(0).commit()->transform(1) == false);
    CHECK(MyEstimator().fit(1).commit()->transform(2) == false);
    CHECK(MyEstimator().fit(0).commit()->transform(2) == true);
}

TEST_CASE("Estimator: Errors") {
    MyEstimator                             e;

    CHECK(e.commit());
    CHECK_THROWS_WITH(e.fit(1), Catch::Contains("has already been committed"));
    CHECK_THROWS_WITH(e.commit(), Catch::Contains("has already been committed"));

    CHECK_THROWS_WITH(MyEstimator(true).commit(), Catch::Matches("Invalid result"));
}

TEST_CASE("fit_and_commit") {
    CHECK(Microsoft::Featurizer::fit_and_commit<MyEstimator>(1, false)->transform(1) == true);
    CHECK(Microsoft::Featurizer::fit_and_commit<MyEstimator>(0, false)->transform(1) == false);
    CHECK(Microsoft::Featurizer::fit_and_commit<MyEstimator>(1, false)->transform(2) == false);
    CHECK(Microsoft::Featurizer::fit_and_commit<MyEstimator>(0, false)->transform(2) == true);
}
