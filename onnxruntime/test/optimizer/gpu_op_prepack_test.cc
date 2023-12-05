// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"

#include "core/mlas/inc/mlas_q4.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/mickey/blk_q4/f16_prepack_sm80.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

#include "test/cuda_host/blkq4_fp16_quant_sm80.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {

extern ProviderInfo_CUDA* TryGetProviderInfo_CUDA();

namespace test {
#ifndef DISABLE_CONTRIB_OPS

std::shared_ptr<IExecutionProvider> LoadCudaEp() {
  try {
    OrtCUDAProviderOptions cuda_options;
    auto factory = onnxruntime::CudaProviderFactoryCreator::Create(&cuda_options);
    if (!factory) {
      return nullptr;
    }
    return factory->CreateProvider();
  } catch (const ::onnxruntime::OnnxRuntimeException& e) {
    std::cerr << "LoadCudaEp: " << e.what() << std::endl;
    return nullptr;
  }
}

/**
 * @brief Testing helper for GPU prepacking logic in the graph transformer.
 * This is an modification of the TransformerTester function from
 *   onnxruntime/test/optimizer/graph_transform_test_builder.cc
 * with:
 *  - the addition of cuda execution provider in the session.
 *  - a different location for the model checker, right after session initialization
 *    as the initializers will be deleted during session run.
 */
void GpuPrepackTester(
    const std::shared_ptr<IExecutionProvider>& cuda_ep,
    const std::function<void(ModelTestBuilder& helper)>& build_test_case,
    const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
    TransformerLevel baseline_level,
    TransformerLevel target_level,
    int opset_version = 12,
    double per_sample_tolerance = 0.001,
    double relative_per_sample_tolerance = 0.001,
    const std::function<void(SessionOptions&)>& add_session_options = {},
    const InlinedHashSet<std::string>& disabled_optimizers = {}) {
  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  ASSERT_TRUE(build_test_case);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    if (level == target_level) {
      // we don't really need the model file, but it seems to be the only way to keep the
      // transformed initializers so that they can be verified.
      session_options.optimized_model_filepath =
          ToPathString("gpu_prepack_test_model_opt_level_" + std::to_string(static_cast<int>(level)) + ".onnx");
    }
    if (add_session_options) {
      add_session_options(session_options);
    }
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(cuda_ep));

    ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
    if (!disabled_optimizers.empty()) {
      ASSERT_STATUS_OK(session.FilterEnabledOptimizers(InlinedHashSet<std::string>{disabled_optimizers}));
    }

    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options,
                                 helper.feeds_,
                                 helper.output_names_,
                                 &fetches));

    if (level == target_level) {
      if (check_transformed_graph) {
        check_transformed_graph(session);
      }
    }
  };

  std::vector<OrtValue> baseline_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(baseline_level, baseline_fetches));

  std::vector<OrtValue> target_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(target_level, target_fetches));

  size_t num_outputs = baseline_fetches.size();
  ASSERT_EQ(num_outputs, target_fetches.size());

  for (size_t i = 0; i < num_outputs; i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(target_fetches[i],
                        baseline_fetches[i],
                        per_sample_tolerance,
                        relative_per_sample_tolerance,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

inline Status GetOrtValue(const NodeArg* arg, const Graph& graph, OrtValue& ort_value) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_RETURN_IF_NOT(graph.GetInitializedTensor(arg->Name(), tensor_proto),
                    "Missing initializer for ", arg->Name());

  return utils::TensorProtoToOrtValue(
      Env::Default(), graph.ModelPath(), *tensor_proto,
      std::make_shared<CPUAllocator>(), ort_value);
}

template <int block_size, bool columnwise_blocking, bool has_offsets>
void MatMulQ4Test(int M, int N, int K, const std::shared_ptr<IExecutionProvider>& cuda_ep) {
  //
  // Type definitions
  //
  using ElementT = MLFloat16;
  using Base = onnxruntime::cuda::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      columnwise_blocking>;

  using QuantBlocking = typename Base::QuantBlocking;
  using ElementW = typename Base::ElementW;
  using LayoutWPack = typename Base::LayoutWPack;
  using ElementQOffset = typename Base::ElementQOffset;
  using LayoutQmeta = typename Base::LayoutQmeta;

  //
  // Generate random inputs
  //
  const auto q_weight_shape = Base::get_quant_weights_shape(K, N);
  const auto meta_shape = Base::get_quant_meta_shape(K, N);

  std::vector<ElementW> q_weights;
  std::vector<ElementT> q_scales;
  std::vector<ElementQOffset> q_zp;
  std::vector<ElementT> dequants;
  blkq4_weights_gen<ElementT, block_size, columnwise_blocking, has_offsets>(
      K, N, dequants, q_weights, q_scales, q_zp);

  // for quantization tool, the input is row major, all outputs are column major
  MatrixRef<ElementW, ColumnMajorLayout, true> tensor_q_weight(
      q_weights, q_weight_shape);
  MatrixRef<ElementT, ColumnMajorLayout, true> tensor_scale(
      q_scales, meta_shape);
  MatrixRef<ElementQOffset, ColumnMajorLayout, true> tensor_offset;
  if constexpr (has_offsets) {
    const auto zp_shape = make_Position((meta_shape[0] + 1) / 2, meta_shape[1]);
    tensor_offset = MatrixRef<ElementQOffset, ColumnMajorLayout, true>(q_zp, zp_shape);
  }

  // Compute prepacked weights
  std::vector<ElementW> packed_w_ref(q_weight_shape.product());
  MatrixRef<ElementW, LayoutWPack, true> tensor_packed_w_ref(
      packed_w_ref, make_Position(K, N / 2));
  onnxruntime::test::sm80_prepack_weights_ref(K, N, tensor_q_weight, tensor_packed_w_ref);

  std::vector<ElementT> packed_scales_ref(meta_shape.product());
  MatrixRef<ElementT, LayoutQmeta, true> tensor_packed_s_ref =
      make_MatrixRef<ElementT, LayoutQmeta, true>(packed_scales_ref, meta_shape);
  if constexpr (Base::ShouldRearrangeMeta) {
    onnxruntime::test::sm80_prepack_quant_scales_ref<ElementT, LayoutQmeta, QuantBlocking>(
        K, N, tensor_scale.const_ref(), tensor_packed_s_ref);
  } else {
    for (int col = 0; col < tensor_packed_s_ref.shape()[1]; ++col) {
      for (int row = 0; row < tensor_packed_s_ref.shape()[0]; ++row) {
        tensor_packed_s_ref.at(row, col) = tensor_scale.at(row, col);
      }
    }
  }

  std::vector<ElementQOffset> packed_zp_ref;
  if constexpr (has_offsets) {
    packed_zp_ref.resize(meta_shape.product());
    MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_packed_zp_ref =
        make_MatrixRef<ElementQOffset, LayoutQmeta, true>(packed_zp_ref, meta_shape);
    if constexpr (Base::ShouldRearrangeMeta) {
      onnxruntime::test::sm80_prepack_quant_offsets_ref<LayoutQmeta, QuantBlocking>(
          K, N, tensor_offset.const_ref(), tensor_packed_zp_ref);
    } else {
      for (int col = 0; col < meta_shape[1]; ++col) {
        for (int row = 0; row < meta_shape[0]; row += 2) {
          uint8_t pair01 = tensor_offset.at(row / 2, col);
          tensor_packed_zp_ref.at(row, col) = pair01 & 0xf;
          if (row + 1 < meta_shape[0]) {
            tensor_packed_zp_ref.at(row + 1, col) = pair01 >> 4;
          }
        }
      }
    }
  }

  auto build_test_case = [&](ModelTestBuilder& builder) {
    size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
    MlasBlockwiseQuantizedBufferSizes(4, block_size, columnwise_blocking, K, N,
                                      q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

    auto* input_arg = builder.MakeInput<MLFloat16>({M, K}, MLFloat16(-2.0f), MLFloat16(2.0f));
    constexpr bool print_input_a = false;
    if constexpr (print_input_a) {
      const auto& act_name = input_arg->Name();
      OrtValue act_val = builder.feeds_[act_name];
      const gsl::span<MLFloat16 const> act_data = act_val.GetMutable<Tensor>()->DataAsSpan<MLFloat16>();
      ASSERT_EQ(act_data.size(), M * K);
      fprintf(stderr, "====== act_data ======:\n");
      for (int act_row = 0; act_row < M; act_row++) {
        for (int act_col = 0; act_col < K; act_col++) {
          fprintf(stderr, "%f, ", static_cast<float>(act_data[act_row * K + act_col]));
        }
        fprintf(stderr, "\n");
      }
    }

    auto* output_arg = builder.MakeOutput();
    auto* weight_arg = builder.MakeInitializer<uint8_t>({q_weight_shape[1], q_weight_shape[0]}, q_weights);
    auto* scale_arg = builder.MakeInitializer<MLFloat16>({static_cast<int64_t>(q_scales.size())}, q_scales);

    std::vector<NodeArg*> input_args{input_arg, weight_arg, scale_arg};
    if constexpr (has_offsets) {
      auto* zero_point_arg = builder.MakeInitializer<uint8_t>({static_cast<int64_t>(q_zp.size())}, q_zp);
      input_args.push_back(zero_point_arg);
    } else {
      ASSERT_TRUE(q_zp.empty());
    }
    Node& node = builder.AddNode("MatMulNBits", input_args, {output_arg}, kMSDomain);
    node.AddAttribute("K", static_cast<int64_t>(K));
    node.AddAttribute("N", static_cast<int64_t>(N));
    node.AddAttribute("block_size", static_cast<int64_t>(block_size));
    node.AddAttribute("bits", static_cast<int64_t>(4));
    node.AddAttribute("column_wise_blocking", static_cast<int64_t>(columnwise_blocking));
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    const auto& graph = session.GetGraph();
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        ASSERT_EQ(node.Domain(), kMSDomain);
        ASSERT_EQ(node.GetAttributes().at("prepacked").i(), 1);
        {
          // Verify prepacked weights
          OrtValue packed_w_val;
          ASSERT_STATUS_OK(GetOrtValue(node.InputDefs()[1], graph, packed_w_val));
          const gsl::span<uint8_t const> weights_data = packed_w_val.GetMutable<Tensor>()->DataAsSpan<uint8_t>();
          ASSERT_EQ(weights_data.size(), packed_w_ref.size());
          for (size_t i = 0; i < packed_w_ref.size(); ++i) {
            int expected = packed_w_ref[i];
            int found = weights_data[i];
            ASSERT_EQ(expected, found) << "prepacked weight mismatch index i = " << i << " shape[" << K << "," << N / 2 << "]";
          }
        }
        {
          // Verify prepacked scales
          OrtValue packed_s_val;
          ASSERT_STATUS_OK(GetOrtValue(node.InputDefs()[2], graph, packed_s_val));
          const gsl::span<MLFloat16 const> scales_data = packed_s_val.GetMutable<Tensor>()->DataAsSpan<MLFloat16>();
          ASSERT_EQ(scales_data.size(), packed_scales_ref.size());
          for (size_t i = 0; i < packed_scales_ref.size(); ++i) {
            float expected = packed_scales_ref[i];
            float found = scales_data[i];
            ASSERT_EQ(expected, found) << "prepacked scale mismatch index i = " << i << " shape[" << meta_shape[0] << "," << meta_shape[1] << "]";
          }
        }
        if constexpr (has_offsets) {
          // Verify prepacked zero points
          OrtValue packed_z_val;
          ASSERT_STATUS_OK(GetOrtValue(node.InputDefs()[3], graph, packed_z_val));
          const gsl::span<uint8_t const> offsets_data = packed_z_val.GetMutable<Tensor>()->DataAsSpan<uint8_t>();
          ASSERT_EQ(offsets_data.size(), packed_zp_ref.size());
          for (size_t i = 0; i < packed_zp_ref.size(); ++i) {
            int expected = packed_zp_ref[i];
            int found = offsets_data[i];
            ASSERT_EQ(expected, found) << "prepacked zero-point mismatch index i = " << i << " shape[" << meta_shape[0] << "," << meta_shape[1] << "]";
          }
        } else {
          ASSERT_LE(node.InputDefs().size(), static_cast<size_t>(3));
        }
      }
    }
  };

  GpuPrepackTester(cuda_ep,
                   build_test_case,
                   check_graph,
                   TransformerLevel::Level2,
                   TransformerLevel::Level3);
}

TEST(GpuOpPrepackTests, MatmulNBits) {
  std::shared_ptr<onnxruntime::IExecutionProvider> provider = LoadCudaEp();
  if (!provider) {
    GTEST_SKIP() << "Skipping tests when CUDA EP is not available";
  }

  //
  // Currently these tests only work on sm_80. Going forward, however,
  // we need a better solution when we may have different tests for different
  // hardware.
  //
  auto* provider_info = TryGetProviderInfo_CUDA();
  int major, minor;
  ORT_ENFORCE(provider_info->GetCurrentGpuDeviceVersion(&major, &minor) == nullptr,
              "Failed to query CUDA device version while prepacking cuda operators.");
  if (major < 8) {
    GTEST_SKIP() << "Skipping tests when CUDA EP is not sm_80";
  }

  //
  // GpuPrepackTester function implements two different verifications.
  // First is the hook check_graph, which we use to verify the prepacked weights, scales and zero points.
  // Second is the comparison of the outputs of the model with and without prepacking, this actually
  // doubles as a verification of kernel correctness and prepacking correctness.
  //
  // We do have other sets of tests for the prepack and kernel correctness, defined in
  // onnxruntime/test/providers/cuda/test_cases/blkq4_fp16_gemm_sm80_test.cc
  //
  // What we are doing here is to verify we correctly connected the prepacking logic in
  // the graph transformer, and that the prepacked weights, scales and zero points are correctly
  // passed to the kernel in MatMulNBits cuda op. Plus the redundant verifications allows us to
  // locate the problem more easily.
  //
  // So we don't need to test all the combinations here, just a few representative ones.
  //

  std::cout << "Testing MatMulQ4Test<64, true, true>(4, 128, 64, provider)" << std::endl;
  MatMulQ4Test<64, true, true>(4, 128, 64, provider);
  std::cout << "Testing MatMulQ4Test<64, false, true>(4, 128, 64, provider)" << std::endl;
  MatMulQ4Test<64, false, true>(4, 128, 64, provider);
  std::cout << "Testing MatMulQ4Test<64, true, false>(8, 128, 64, provider)" << std::endl;
  MatMulQ4Test<64, true, false>(8, 128, 64, provider);
  std::cout << "Testing MatMulQ4Test<64, false, false>(8, 128, 64, provider)" << std::endl;
  MatMulQ4Test<64, false, false>(8, 128, 64, provider);

  std::cout << "Testing MatMulQ4Test<32, true, true>(16, 64, 128, provider)" << std::endl;
  MatMulQ4Test<32, true, true>(16, 64, 128, provider);
  std::cout << "Testing MatMulQ4Test<32, true, false>(16, 64, 128, provider)" << std::endl;
  MatMulQ4Test<32, true, false>(16, 64, 128, provider);
  std::cout << "Testing MatMulQ4Test<32, false, true>(16, 64, 128, provider)" << std::endl;
  MatMulQ4Test<32, false, true>(16, 64, 128, provider);
  std::cout << "Testing MatMulQ4Test<32, false, false>(16, 64, 128, provider)" << std::endl;
  MatMulQ4Test<32, false, false>(16, 64, 128, provider);

  std::cout << "Testing MatMulQ4Test<16, true, true>(32, 96, 128, provider)" << std::endl;
  MatMulQ4Test<16, true, true>(32, 96, 128, provider);
  std::cout << "Testing MatMulQ4Test<16, true, false>(32, 96, 128, provider)" << std::endl;
  MatMulQ4Test<16, true, false>(32, 96, 128, provider);
  std::cout << "Testing MatMulQ4Test<16, false, true>(32, 96, 128, provider)" << std::endl;
  MatMulQ4Test<16, false, true>(32, 96, 128, provider);
  std::cout << "Testing MatMulQ4Test<16, false, false>(32, 96, 128, provider)" << std::endl;
  MatMulQ4Test<16, false, false>(32, 96, 128, provider);
}

#endif

}  // namespace test
}  // namespace onnxruntime
