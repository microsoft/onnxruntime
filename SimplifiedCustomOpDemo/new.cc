// testapi.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include "onnxruntime_cxx_api.h"
#include <iostream>
#define FUSION_FILTER L"..\\..\\fusion_filter.onnx"

/*
Ideas summarized by dicussion:
1. Provide a init function along with compute function to allow some resource management and reading of attributes (and a deinit?);
2. All compute function to return some status, this is useful on scenarios where exceptions are not supported;
3. Allow compute function to get a handle of context;
4. Allow customer to use their own memories.
*/

/////////////////////////////////// Fuse ////////////////////////////////////////

void Fuse(const Ort::Custom::InputTensor<float>& vector_1,
	const Ort::Custom::InputTensor<float>& vector_2,
	Ort::Custom::OutputTensor<float>& vector_output) {

	const float* floats_1 = vector_1.Data();
	const float* floats_2 = vector_2.Data();

	auto len_1 = vector_1.Shape()[0];
	auto len_2 = vector_1.Shape()[0];

	auto len_output = std::min(len_1, len_2);

	float* floats_out = static_cast<float*>(vector_output.Allocate({ len_output }));

	for (int64_t i = 0; i < len_output; ++i) {
		floats_out[i] = floats_1[i] + floats_2[i];
	}
}

/////////////////////////////////// Select ////////////////////////////////////////

void Select(const Ort::Custom::InputTensor<int32_t>& indices_in,
	Ort::Custom::OutputTensor<int32_t>& indices_out) {

	const int32_t* int_in = indices_in.Data();
	auto len_in = indices_in.Shape()[0];

	std::vector<int32_t> selected_indices;
	for (int64_t i = 0; i < len_in; ++i) {
		if (int_in[i] % 2 == 0) {
			selected_indices.push_back(int_in[i]);
		}
	}

	int32_t* int_out = static_cast<int32_t*>(indices_out.Allocate({ static_cast<int64_t>(selected_indices.size()) }));
	for (size_t j = 0; j < selected_indices.size(); ++j) {
		int_out[j] = selected_indices[j];
	}
}

/////////////////////////////////// Filter ////////////////////////////////////////

void Filter(const Ort::Custom::InputTensor<float>& vector_in,
	Ort::Custom::OutputTensor<float>& vector_out) {

	const float* floats_in = vector_in.Data();
	auto len_in = vector_in.Shape()[0];

	std::vector<float> filter_floats;
	for (int64_t i = 0; i < len_in; ++i) {
		if (floats_in[i] > 1.f) {
			filter_floats.push_back(floats_in[i]);
		}
	}

	float* floats_out = static_cast<float*>(vector_out.Allocate({ static_cast<int64_t>(filter_floats.size()) }));
	for (size_t j = 0; j < filter_floats.size(); ++j) {
		floats_out[j] = filter_floats[j];
	}
}

///////////////////////////////////////////////////////////////////////////////////

void TestNew() {
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CustomOpNew");

	const auto& ortApi = Ort::GetApi();
	OrtCustomOpDomain* v2_domain{};
	ortApi.CreateCustomOpDomain("v2", &v2_domain);
	//ortApi.DeleteCustomOpDomain("v2", &v2_domain);?

	std::unique_ptr<OrtCustomOp> fus_op_ptr{ Ort::Custom::CreateCustomOp("Fuse", "CPUExecutionProvider", Fuse)};
	std::unique_ptr<OrtCustomOp> sel_op_ptr{ Ort::Custom::CreateCustomOp("Select", "CPUExecutionProvider", Select) };
	std::unique_ptr<OrtCustomOp> fil_op_ptr{ Ort::Custom::CreateCustomOp("Filter", "CPUExecutionProvider", Filter) };

	ortApi.CustomOpDomain_Add(v2_domain, fus_op_ptr.get());
	ortApi.CustomOpDomain_Add(v2_domain, sel_op_ptr.get());
	ortApi.CustomOpDomain_Add(v2_domain, fil_op_ptr.get());

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	session_options.SetLogSeverityLevel(0);
	session_options.Add(v2_domain);

	const wchar_t* model_path = FUSION_FILTER;
	Ort::Session session(env, model_path, session_options);

	const char* input_names[] = { "vector_1", "vector_2", "indices" };
	const char* output_names[] = { "vector_filtered" };

	float vector_1_value[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
	int64_t vector_1_dim[] = { 10 };

	float vector_2_value[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f };
	int64_t vector_2_dim[] = { 6 };

	int32_t indices_value[] = { 0,1,2,3,4,5 };
	int64_t indices_dim[] = { 6 };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	Ort::Value input_tensors[3] = {
		Ort::Value::CreateTensor<float>(memory_info, vector_1_value, 10, vector_1_dim, 1),
		Ort::Value::CreateTensor<float>(memory_info, vector_2_value, 6, vector_2_dim, 1),
		Ort::Value::CreateTensor<int32_t>(memory_info, indices_value, 6, indices_dim, 1)
	};

	Ort::RunOptions run_optoins;
	auto output_tensors = session.Run(run_optoins, input_names, input_tensors, 3, output_names, 1);
	const auto& vector_filterred = output_tensors.at(0);
	auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
	size_t num_output = type_shape_info.GetElementCount();
	const float* floats_output = static_cast<const float*>(vector_filterred.GetTensorRawData());

	std::cout << std::endl << "/////////////////////////////// OUTPUT ///////////////////////////////" << std::endl;
	std::copy(floats_output, floats_output + num_output, std::ostream_iterator<float>(std::cout, " "));
	std::cout << std::endl;
}

int main() {
	TestNew();
	std::cout << "done" << std::endl;
}