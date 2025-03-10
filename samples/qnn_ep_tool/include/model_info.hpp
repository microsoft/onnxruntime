// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_cxx_api.h>

#include <vector>

class OnnxModelInfo {
    public:
        OnnxModelInfo(const OrtApi* g_ort, const OrtSession* session, OrtAllocator* allocator);
        size_t get_num_in_tensors();
        std::vector<char*> get_in_tensor_names();
        std::vector<std::vector<int64_t>> get_in_tensor_dims();
        std::vector<ONNXTensorElementDataType> get_in_tensor_element_types();
        std::vector<int64_t> get_in_tensor_element_nums();
        std::vector<OrtValue*>& get_in_tensors();

        size_t get_num_out_tensors();
        std::vector<char*> get_out_tensor_names();
        std::vector<std::vector<int64_t>> get_out_tensor_dims();
        std::vector<ONNXTensorElementDataType> get_out_tensor_element_types();
        std::vector<int64_t> get_out_tensor_element_nums();
        std::vector<OrtValue*>& get_out_tensors();

        void release_ort_values(const OrtApi* g_ort);
        void PrintOnnxModelInfo();
    private:
        size_t num_in_tensors;
        std::vector<char*> in_tensor_names;
        std::vector<std::vector<int64_t>> in_tensor_dims;
        std::vector<ONNXTensorElementDataType> in_tensor_element_types;
        std::vector<int64_t> in_tensor_element_nums;
        std::vector<OrtValue*> in_tensors;

        size_t num_out_tensors;
        std::vector<char*> out_tensor_names;
        std::vector<std::vector<int64_t>> out_tensor_dims;
        std::vector<ONNXTensorElementDataType> out_tensor_element_types;
        std::vector<int64_t> out_tensor_element_nums;
        std::vector<OrtValue*> out_tensors;
};

size_t GetONNXTypeSize(ONNXTensorElementDataType dtype);
int onnx_element_type_to_tensorproto_dtype(ONNXTensorElementDataType dtype);