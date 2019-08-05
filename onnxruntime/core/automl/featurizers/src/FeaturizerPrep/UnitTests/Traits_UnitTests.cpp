// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../Traits.h"
#include <type_traits>

using namespace Microsoft::Featurizer::Traits;

// Floating point values
static_assert(std::is_same<Traits<float>::nullable_type, float>::value, "Incorrect nullable type for float");
static_assert(std::is_same<Traits<double>::nullable_type, double>::value, "Incorrect nullable type for double");

// Int values
static_assert(std::is_same<Traits<std::int8_t>::nullable_type, boost::optional<std::int8_t>>::value, "Incorrect nullable type for std::int8_t");
static_assert(std::is_same<Traits<std::int16_t>::nullable_type, boost::optional<std::int16_t>>::value, "Incorrect nullable type for std::int16_t");
static_assert(std::is_same<Traits<std::int32_t>::nullable_type, boost::optional<std::int32_t>>::value, "Incorrect nullable type for std::int32_t");
static_assert(std::is_same<Traits<std::int64_t>::nullable_type, boost::optional<std::int64_t>>::value, "Incorrect nullable type for std::int64_t");
static_assert(std::is_same<Traits<std::uint8_t>::nullable_type, boost::optional<std::uint8_t>>::value, "Incorrect nullable type for std::uint8_t");
static_assert(std::is_same<Traits<std::uint16_t>::nullable_type, boost::optional<std::uint16_t>>::value, "Incorrect nullable type for std::uint16_t");
static_assert(std::is_same<Traits<std::uint32_t>::nullable_type, boost::optional<std::uint32_t>>::value, "Incorrect nullable type for std::uint32_t");
static_assert(std::is_same<Traits<std::uint64_t>::nullable_type, boost::optional<std::uint64_t>>::value, "Incorrect nullable type for std::uint64_t");

// Others
static_assert(std::is_same<Traits<std::string>::nullable_type, boost::optional<std::string>>::value, "Incorrect nullable type for std::string");
static_assert(std::is_same<Traits<std::array<char, 4>>::nullable_type, boost::optional<std::array<char, 4>>>::value, "Incorrect nullable type for std::array");
static_assert(std::is_same<Traits<bool>::nullable_type, boost::optional<bool>>::value, "Incorrect nullable type for std::string");
static_assert(std::is_same<Traits<std::map<int,int>>::nullable_type, boost::optional<std::map<int,int>>>::value, "Incorrect nullable type for std::string");
static_assert(std::is_same<Traits<std::vector<int>>::nullable_type, boost::optional<std::vector<int>>>::value, "Incorrect nullable type for std::string");
static_assert(std::is_same<Traits<std::function<int>>::nullable_type, boost::optional<std::function<int>>>::value, "Incorrect nullable type for std::string");
static_assert(std::is_same<Traits<boost::optional<int>>::nullable_type, boost::optional<int>>::value, "Incorrect nullable type for std::string");
static_assert(std::is_same<Traits<std::tuple<int>>::nullable_type, boost::optional<std::tuple<int>>>::value, "Incorrect nullable type for std::string");

// Dummy test so it will compile. Replace this with actual tests.
TEST_CASE("Dummy Test") {
    CHECK(true == true);
}
