// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <fstream>
using namespace std;

string script = "\n\
def Mul(a):      \n\
    return a * a ";

namespace onnxruntime {
namespace test {

TEST(PyOpTest, PyOpMul) {
  fstream fs("TestPyOp.py");
  ORT_ENFORCE(fs.is_open(), "failed to write to TestPyOp.py");
  fs << script;
  fs.close();
  OpTester test("PyOp", 1, onnxruntime::kMSDomain);
  test.AddAttribute<string>("module",   "TestPyOp");
  test.AddAttribute<string>("function", "Mul");
  test.AddInput  <int32_t> ("data",   {3}, {0,1,2});
  test.AddOutput <int32_t> ("output", {3}, {0,1,4});
  test.Run();
}

}
}
