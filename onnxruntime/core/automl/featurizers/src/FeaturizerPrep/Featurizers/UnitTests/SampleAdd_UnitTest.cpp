// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include "gtest/gtest.h"

#include "../SampleAdd.h"

TEST(SampleAddTests, Transformer) {
  ASSERT_EQ(Microsoft::Featurizer::SampleAdd::Transformer(10).transform(20), 30U);
  ASSERT_EQ(Microsoft::Featurizer::SampleAdd::Transformer(20).transform(1), 21U);
}

TEST(SampleAddTests, Estimator) {
  ASSERT_EQ(Microsoft::Featurizer::SampleAdd::Estimator().fit(10).commit()->transform(20), 30U);
  ASSERT_EQ(Microsoft::Featurizer::SampleAdd::Estimator().fit(20).commit()->transform(1), 21U);

  ASSERT_EQ(Microsoft::Featurizer::SampleAdd::Estimator().fit(10).fit(20).commit()->transform(20), 50U);
  ASSERT_EQ(Microsoft::Featurizer::SampleAdd::Estimator().fit(10).fit(20).fit(30).commit()->transform(20), 80U);
}
