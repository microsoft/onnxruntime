#include "test_utils.h"

#include "gtest/gtest.h"
#include "core/framework/path_lib.h"
#include <string>

#define PATH_EXPECT(X, Y)                                            \
  {                                                                  \
    auto a = ORT_TSTR(X);                                            \
    std::basic_string<ORTCHAR_T> ret;                                \
    auto st = onnxruntime::GetDirNameFromFilePath(ORT_TSTR(Y), ret); \
    ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();                     \
    ASSERT_EQ(a, ret);                                               \
  }

#ifdef _WIN32
TEST(PathTest, simple) {
  PATH_EXPECT("C:\\Windows", "C:\\Windows\\a.txt");
}

TEST(PathTest, trailing_slash) {
  PATH_EXPECT("C:\\Windows", "C:\\Windows\\system32\\");
}

TEST(PathTest, trailing_slash2) {
  PATH_EXPECT("C:\\Windows", "C:\\Windows\\\\system32\\");
}

TEST(PathTest, windows_root) {
  PATH_EXPECT("C:\\", "C:\\");
}
TEST(PathTest, root) {
  PATH_EXPECT("\\", "\\");
}
#else
TEST(PathTest, simple) {
  PATH_EXPECT("/Windows", "/Windows/a.txt");
}

TEST(PathTest, trailing_slash) {
  PATH_EXPECT("/Windows", "/Windows/system32/");
}

TEST(PathTest, trailing_slash2) {
  PATH_EXPECT("/Windows", "/Windows//system32/");
}

TEST(PathTest, root) {
  PATH_EXPECT("/", "/");
}
#endif
TEST(PathTest, single) {
  PATH_EXPECT(".", "abc");
}

TEST(PathTest, dot) {
  PATH_EXPECT(".", ".");
}

TEST(PathTest, dot_dot) {
  PATH_EXPECT(".", "..");
}

