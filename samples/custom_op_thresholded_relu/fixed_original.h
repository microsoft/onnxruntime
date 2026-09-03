/*
Fixed version of the user's original C++ custom operator implementation.

This shows what the user's code should have looked like to work correctly.
*/

#pragma once
#include <onnxruntime_lite_custom_op.h>
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

extern "C" {
#ifdef _WIN32
    __declspec(dllexport)
#endif
    OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
}