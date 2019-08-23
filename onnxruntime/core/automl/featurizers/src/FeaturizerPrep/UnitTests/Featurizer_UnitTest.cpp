// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include "gtest/gtest.h"
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
    // |  Private Data
    bool const                              _true_on_odd;
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
};

TEST(FeaturizerTests, TransformerFunctionality) {
  ASSERT_TRUE(MyTransformer(true).transform(1));
  ASSERT_FALSE(MyTransformer(false).transform(1));
  ASSERT_FALSE(MyTransformer(true).transform(2));
  ASSERT_TRUE(MyTransformer(false).transform(2));
}

TEST(FeaturizerTests, EstimatorFunctionality) {
  ASSERT_TRUE(MyEstimator().fit(1).commit()->transform(1));
  ASSERT_FALSE(MyEstimator().fit(0).commit()->transform(1));
  ASSERT_FALSE(MyEstimator().fit(1).commit()->transform(2));
  ASSERT_TRUE(MyEstimator().fit(0).commit()->transform(2));
}

TEST(FeaturizerTests, EstimatorErrors) {
    MyEstimator                             e;

    ASSERT_NE(e.commit(), nullptr);
    //CHECK_THROWS_WITH(e.fit(1), Catch::Contains("has already been committed"));
    //CHECK_THROWS_WITH(e.commit(), Catch::Contains("has already been committed"));

    //CHECK_THROWS_WITH(MyEstimator(true).commit(), Catch::Matches("Invalid result"));
}

TEST(FeaturizerTests, EstimatorFitAndCommit) {
  ASSERT_TRUE(Microsoft::Featurizer::fit_and_commit<MyEstimator>(1, false)->transform(1));
  ASSERT_FALSE(Microsoft::Featurizer::fit_and_commit<MyEstimator>(0, false)->transform(1));
  ASSERT_FALSE(Microsoft::Featurizer::fit_and_commit<MyEstimator>(1, false)->transform(2));
  ASSERT_TRUE(Microsoft::Featurizer::fit_and_commit<MyEstimator>(0, false)->transform(2));
}
