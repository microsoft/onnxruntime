// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dlfcn.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/external_ops/pyop.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace onnxruntime;

TEST(PyOpTest, unittest_numpy_input_output)
{
    ofstream fs("test.py");
    fs << "def Double(A):"         << endl;
    fs << "    return A+A"         << endl;
    fs << "def Add(A,B):"          << endl;
    fs << "    return A+B"         << endl;
    fs << "def Inc(A):"            << endl;
    fs << "    return A+1,A+2,A+3" << endl; 
    fs.close();

    void* handle = dlopen("./libonnxruntime_pyop.so", RTLD_NOW | RTLD_GLOBAL);
    ORT_ENFORCE(nullptr != handle, dlerror());

    auto Initialize = (INIT*)    dlsym(handle, "Initialize");
    auto Pyfunc     = (PYFUNC*)  dlsym(handle, "CallPythonFunction");
    auto LastError  = (LASTERR*) dlsym(handle, "GetLastErrorMessage"); 
    auto SetSysPath = (SETPATH*) dlsym(handle, "SetSysPath");

    ORT_ENFORCE(nullptr != Initialize, dlerror());
    ORT_ENFORCE(nullptr != Pyfunc,     dlerror());
    ORT_ENFORCE(nullptr != LastError,  dlerror());
    ORT_ENFORCE(nullptr != SetSysPath, dlerror());

    ORT_ENFORCE(Initialize(), LastError());
    SetSysPath(L".");
    vector<const void*> output;
    vector<int32_t> output_size;
    vector<vector<int64_t>> output_dim;

    int32_t A[] = {1,2,3};
    vector<vector<int64_t>> input_dim = {{3}};
    ORT_ENFORCE(Pyfunc("test", "Double", {A}, {5}, input_dim, output, output_size, output_dim), LastError());
    ORT_ENFORCE(output.size() == 1,     "Number of output is incorrect");
    ORT_ENFORCE(((const int32_t*)output[0])[0] == 2, "Number of output is incorrect");
    ORT_ENFORCE(((const int32_t*)output[0])[1] == 4, "Number of output is incorrect");
    ORT_ENFORCE(((const int32_t*)output[0])[2] == 6, "Number of output is incorrect");
    output.clear();
    output_size.clear();
    output_dim.clear();

    int64_t B[] = {0,1,2,3,4};
    int64_t C[] = {5,6,7,8,9};
    input_dim = {{5},{5}};
    ORT_ENFORCE(Pyfunc("test", "Add", {B,C}, {9,9}, input_dim, output, output_size, output_dim), LastError());
    ORT_ENFORCE(output.size() == 1, "Number of output is incorrect");
    ORT_ENFORCE(((const int64_t*)output[0])[0] == 5);
    ORT_ENFORCE(((const int64_t*)output[0])[1] == 7);
    ORT_ENFORCE(((const int64_t*)output[0])[2] == 9);
    ORT_ENFORCE(((const int64_t*)output[0])[3] == 11);
    ORT_ENFORCE(((const int64_t*)output[0])[4] == 13);
    output.clear();
    output_size.clear();
    output_dim.clear();

    float D[] = {123, 345};
    input_dim = {{2}};
    ORT_ENFORCE(Pyfunc("test", "Inc", {D}, {11}, input_dim, output, output_size, output_dim), LastError());
    ORT_ENFORCE(output.size() == 3, "Number of output is incorrect");
    ORT_ENFORCE(((const float*)output[0])[0] == 124);
    ORT_ENFORCE(((const float*)output[0])[1] == 346);
    ORT_ENFORCE(((const float*)output[1])[0] == 125);
    ORT_ENFORCE(((const float*)output[1])[1] == 347);
    ORT_ENFORCE(((const float*)output[2])[0] == 126);
    ORT_ENFORCE(((const float*)output[2])[1] == 348);

    dlclose(handle);
    std::remove("test.py");
}
