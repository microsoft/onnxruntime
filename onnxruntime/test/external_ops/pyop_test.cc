// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dlfcn.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace onnxruntime;

typedef bool INIT();
typedef bool PYFUNC(const char*,
                    const char*,
                    const vector<const void*>&,
                    const vector<int32_t>&,
                    const vector<vector<int64_t>>&,
                    vector<const void*>&,
                    vector<int32_t>&,
                    vector<vector<int64_t>>&);

typedef const char* LASTERR();
typedef void SETPATH(const wchar_t*);

TEST(PyOpTest, unittest_numpy_input)
{
    ofstream fs("test.py");
    fs << "def Double(A):" << endl;
    fs << "    return A+A" << endl;
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

    int32_t data[] = {1,2,3};
    vector<const void*> input  = { data };
    vector<int32_t> input_type = {0};
    vector<vector<int64_t>> input_dim = {{3}};

    vector<const void*> output;
    vector<int32_t> output_type;
    vector<vector<int64_t>> output_dim;

    SetSysPath(L".");
    ORT_ENFORCE(Pyfunc("test", "Double", input, input_type, input_dim, output, output_type, output_dim), LastError());
    ORT_ENFORCE(output.size() == 1, "Number of output is incorrect");
    ORT_ENFORCE(((const int32_t*)output[0])[0] == 2, "Number of output is incorrect");
    ORT_ENFORCE(((const int32_t*)output[0])[1] == 4, "Number of output is incorrect");
    ORT_ENFORCE(((const int32_t*)output[0])[2] == 6, "Number of output is incorrect");

    dlclose(handle);
    std::remove("test.py");
}

