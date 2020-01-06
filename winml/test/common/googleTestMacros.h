#include <gtest/gtest.h>

#define TEST_GROUP_BEGIN(group_name)
#define TEST_GROUP_END()

#define WINML_TEST(group_name, test_name, method) \
static void method(); \
TEST_F(group_name, test_name) { \
  method();                     \
}

#define WINML_TEST_CLASS_BEGIN_NO_SETUP(test_class_name) \
  class test_class_name : public ::testing::Test {            \
  };

#define WINML_TEST_CLASS_BEGIN_WITH_SETUP(test_class_name, setup_method) \
static void setup_method(); \
  class test_class_name : public ::testing::Test { \
  protected:                                      \
  void SetUp() override {                        \
    setup_method();                              \
  }                                              \
};

#define WINML_TEST_CLASS_END()

#define WINML_EXPECT_NO_THROW(statement) EXPECT_NO_THROW(statement)