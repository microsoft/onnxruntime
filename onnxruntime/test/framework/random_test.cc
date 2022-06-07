// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "core/framework/random_generator.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(RandomTest, RandomSeedTest) {
  utils::SetRandomSeed(17);
  ASSERT_EQ(utils::GetRandomSeed(), 17);
  ASSERT_EQ(utils::GetRandomSeed(), 17);

  utils::SetRandomSeed(8211);
  ASSERT_EQ(utils::GetRandomSeed(), 8211);
}

TEST(RandomTest, RandomGeneratorTest) {
  RandomGenerator generator(17);
  ASSERT_EQ(generator.NextSeed(), 17);
  ASSERT_EQ(generator.NextSeed(10), 18);
  ASSERT_EQ(generator.NextSeed(0), 28);
  ASSERT_EQ(generator.NextSeed(), 28);

  generator.SetSeed(17);
  ASSERT_EQ(generator.NextSeed(), 17);
}

TEST(RandomTest, PhiloxGeneratorTest) {
  PhiloxGenerator generator(17);

  auto seeds = generator.NextPhiloxSeeds(1);
  ASSERT_EQ(seeds.first, 17);
  ASSERT_EQ(seeds.second, 0);

  seeds = generator.NextPhiloxSeeds(10);
  ASSERT_EQ(seeds.first, 17);
  ASSERT_EQ(seeds.second, 1);

  seeds = generator.NextPhiloxSeeds(0);
  ASSERT_EQ(seeds.first, 17);
  ASSERT_EQ(seeds.second, 11);

  seeds = generator.NextPhiloxSeeds(1);
  ASSERT_EQ(seeds.first, 17);
  ASSERT_EQ(seeds.second, 11);

  generator.SetSeed(17);

  seeds = generator.NextPhiloxSeeds(1);
  ASSERT_EQ(seeds.first, 17);
  ASSERT_EQ(seeds.second, 0);
}

}  // namespace test
}  // namespace onnxruntime
