#include "onnxruntime_cxx_api.h"
#include <iostream>
#define FUSION_FILTER L"..\\..\\varied_types.onnx"

void MulTopComputeFloat(const Ort::Custom::InputTensor<float>& floats_from,
	Ort::Custom::OutputTensor<float>& floats_to) {

	const float* from_floats = floats_from.Data();
	float* to_floats = static_cast<float*>(floats_to.Allocate({ 1 }));
	to_floats[0] = from_floats[0] * from_floats[1];
}

void MulTopComputeInt(const Ort::Custom::InputTensor<int32_t>& ints_from,
	Ort::Custom::OutputTensor<int32_t>& ints_to) {

	const int32_t* from_ints = ints_from.Data();
	int32_t* to_ints = static_cast<int32_t*>(ints_to.Allocate({ 1 }));
	to_ints[0] = from_ints[0] * from_ints[1];
}

void TestVaried() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "testapi");
    const auto& ortApi = Ort::GetApi();

	OrtCustomOpDomain* v2_domain{};
	ortApi.CreateCustomOpDomain("v2", &v2_domain);

	std::unique_ptr<OrtCustomOp> multop_float_ptr{ Ort::Custom::CreateCustomOp("MulTop", "CPUExecutionProvider", MulTopComputeFloat) };
	std::unique_ptr<OrtCustomOp> multop_int_ptr{ Ort::Custom::CreateCustomOp("MulTop", "CPUExecutionProvider", MulTopComputeInt) };

	ortApi.CustomOpDomain_Add(v2_domain, multop_float_ptr.get());
	ortApi.CustomOpDomain_Add(v2_domain, multop_int_ptr.get());

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	session_options.SetLogSeverityLevel(0);
	session_options.Add(v2_domain);

	const wchar_t* model_path = FUSION_FILTER;
	Ort::Session session(env, model_path, session_options);

	const char* input_names[] = { "X" };
	const char* output_names[] = { "Y", "Z" };
	float x_value[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
	int64_t x_dim[] = { 10 };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	Ort::Value input_tensors[1] = {
		Ort::Value::CreateTensor<float>(memory_info, x_value, 10, x_dim, 1),
	};

	Ort::RunOptions run_optoins;
	auto output_tensors = session.Run(run_optoins, input_names, input_tensors, 1, output_names, 2);
	std::cout << "output float: " << *output_tensors[0].GetTensorData<float>() << std::endl;
	std::cout << "output int: " << *output_tensors[1].GetTensorData<int32_t>() << std::endl;
}

int main() {
	TestVaried();
	std::cout << "done" << std::endl;
}