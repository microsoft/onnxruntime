#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#define FUSION_FILTER L"..\\..\\fusion_filter.onnx"

/////////////////////////////////// Fuse ////////////////////////////////////////

struct FuseKernel {
	FuseKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api), info_(info) {}
	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		Ort::ConstValue tensor_in_0 = ctx.GetInput(0);
		Ort::ConstValue tensor_in_1 = ctx.GetInput(1);

		auto l_in_0 = tensor_in_0.GetTensorTypeAndShapeInfo().GetShape()[0];
		auto l_in_1 = tensor_in_1.GetTensorTypeAndShapeInfo().GetShape()[0];

		const float* floats_in_0 = tensor_in_0.GetTensorData<float>();
		const float* floats_in_1 = tensor_in_1.GetTensorData<float>();

		auto l_out_0 = std::min(l_in_0, l_in_1);
		auto tensor_out_0 = ctx.GetOutput(0, &l_out_0, 1);

		float* floats_out_0 = tensor_out_0.GetTensorMutableData<float>();
		for (int64_t i = 0; i < l_out_0; ++i) {
			floats_out_0[i] = floats_in_0[i] + floats_in_1[i];
		}
	}
	const OrtApi& api_;
	const OrtKernelInfo* info_;
};

struct FuseOp : Ort::CustomOpBase<FuseOp, FuseKernel> {
	void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
		return std::make_unique<FuseKernel>(api, info).release();
	};

	const char* GetName() const { return "Fuse"; };
	const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

	size_t GetInputTypeCount() const { return 2; };
	size_t GetOutputTypeCount() const { return 1; };

	ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
	ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

/////////////////////////////////// Select ////////////////////////////////////////

struct SelectKernel {
	SelectKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api), info_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		Ort::ConstValue tensor_in = ctx.GetInput(0);
		const int32_t* ints_in = tensor_in.GetTensorData<int32_t>();

		auto l_in = tensor_in.GetTensorTypeAndShapeInfo().GetShape()[0];
		std::vector<int32_t> selected_indices;

		for (int64_t i = 0; i < l_in; ++i) {
			if (ints_in[i] % 2 == 0) {
				selected_indices.push_back(ints_in[i]);
			}
		}

		int64_t l_out = static_cast<int64_t>(selected_indices.size());
		auto tensor_out = ctx.GetOutput(0, &l_out, 1);
		int32_t* ints_out = tensor_out.GetTensorMutableData<int32_t>();

		for (int64_t j = 0; j < l_out; ++j) {
			ints_out[j] = selected_indices[j];
		}
	}

	const OrtApi& api_;
	const OrtKernelInfo* info_;
};

struct SelectOp : Ort::CustomOpBase<SelectOp, SelectKernel> {

	void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
		return std::make_unique<SelectKernel>(api, info).release();
	};

	const char* GetName() const { return "Select"; };
	const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

	size_t GetInputTypeCount() const { return 1; };
	size_t GetOutputTypeCount() const { return 1; };

	ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };
	ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

};

/////////////////////////////////// Filter ////////////////////////////////////////

struct FilterKernel {
	FilterKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api), info_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);
		Ort::ConstValue tensor_in = ctx.GetInput(0);

		auto l_in = tensor_in.GetTensorTypeAndShapeInfo().GetShape()[0];
		const float* floats_in = tensor_in.GetTensorData<float>();

		std::vector<float> filtered_floats;
		for (int64_t i = 0; i < l_in; ++i) {
			if (floats_in[i] > 1.f) {
				filtered_floats.push_back(floats_in[i]);
			}
		}

		auto l_out = static_cast<int64_t>(filtered_floats.size());
		auto tensor_out = ctx.GetOutput(0, &l_out, 1);
		float* floats_out = tensor_out.GetTensorMutableData<float>();

		for (int64_t i = 0; i < l_out; ++i) {
			floats_out[i] = filtered_floats[i];
		}
	}
	const OrtApi& api_;
	const OrtKernelInfo* info_;
};

struct FilterOp : Ort::CustomOpBase<FilterOp, FilterKernel> {

	void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
		return std::make_unique<FilterKernel>(api, info).release();
	};

	const char* GetName() const { return "Filter"; };
	const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

	size_t GetInputTypeCount() const { return 1; };
	size_t GetOutputTypeCount() const { return 1; };

	ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
	ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

///////////////////////////////////////////////////////////////////////////////////

void TestLegacy() {
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CustomOpLegacy");

	const auto& ortApi = Ort::GetApi();
	OrtCustomOpDomain* v2_domain{};
	ortApi.CreateCustomOpDomain("v2", &v2_domain);

	std::unique_ptr<FuseOp> fuse_op = std::make_unique<FuseOp>();
	std::unique_ptr<SelectOp> selelct_op = std::make_unique<SelectOp>();
	std::unique_ptr<FilterOp> filter_op = std::make_unique<FilterOp>();

	ortApi.CustomOpDomain_Add(v2_domain, fuse_op.get());
	ortApi.CustomOpDomain_Add(v2_domain, selelct_op.get());
	ortApi.CustomOpDomain_Add(v2_domain, filter_op.get());

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

//int main() {
//	TestLegacy();
//	std::cout << "done" << std::endl;
//}