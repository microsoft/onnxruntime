// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/session/onnxruntime_cxx_api.h>
#include <set>
#include <iostream>
#include <fstream>
#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#include <thread>
#endif
#include "TestResultStat.h"
#include "testenv.h"
#include "runner.h"
#include "sync_api.h"
#include "providers.h"
#include <google/protobuf/stubs/common.h>
#include "core/framework/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"

const OrtApi* g_ort = OrtGetApi(ORT_API_VERSION);
const OrtApi* Ort::g_api = OrtGetApi(ORT_API_VERSION);

using namespace onnxruntime;

namespace {
void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-A : Disable memory arena\n"
      "\t-M : Disable memory pattern\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-r [repeat]: Specifies the number of times to repeat\n"
      "\t-v: verbose\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'mkldnn', 'tensorrt', 'ngraph', 'openvino' or 'nuphar'. "
      "Default: 'cpu'.\n"
      "\t-x: Use parallel executor, default (without -x): sequential executor.\n"
      "\t-o [optimization level]: Default is 1. Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).\n"
      "\t\tPlease see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels. \n"
      "\t-h: help\n"
      "\n"
      "onnxruntime version: %s\n",
      OrtGetVersionString());
}

#ifdef _WIN32
int GetNumCpuCores() {
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    // try GetSystemInfo
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (sysInfo.dwNumberOfProcessors <= 0) {
      ORT_THROW("Fatal error: 0 count processors from GetSystemInfo");
    }
    // This is the number of logical processors in the current group
    return sysInfo.dwNumberOfProcessors;
  }
  int processorCoreCount = 0;
  int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  for (int i = 0; i != count; ++i) {
    if (buffer[i].Relationship == RelationProcessorCore) {
      ++processorCoreCount;
    }
  }
  if (!processorCoreCount) ORT_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
  return processorCoreCount;
}
#else
int GetNumCpuCores() { return static_cast<int>(std::thread::hardware_concurrency()); }
#endif
}  // namespace

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[], Ort::Env& env) {
#else
int real_main(int argc, char* argv[], Ort::Env& env) {
#endif
  // if this var is not empty, only run the tests with name in this list
  std::vector<std::basic_string<PATH_CHAR_TYPE> > whitelisted_test_cases;
  int concurrent_session_runs = GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  bool enable_sequential_execution = true;
  int repeat_count = 1;
  int p_models = GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_mkl = false;
  bool enable_ngraph = false;
  bool enable_nuphar = false;
  bool enable_tensorrt = false;
  bool enable_mem_pattern = true;
  bool enable_openvino = false;
  bool enable_nnapi = false;
  GraphOptimizationLevel graph_optimization_level = ORT_DISABLE_ALL;
  bool user_graph_optimization_level_set = false;

  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  {
    int ch;
    while ((ch = getopt(argc, argv, ORT_TSTR("Ac:hj:Mn:r:e:xvo:"))) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'v':
          logging_level = ORT_LOGGING_LEVEL_INFO;
          break;
        case 'c':
          concurrent_session_runs = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (concurrent_session_runs <= 0) {
            usage();
            return -1;
          }
          break;
        case 'j':
          p_models = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (p_models <= 0) {
            usage();
            return -1;
          }
          break;
        case 'r':
          repeat_count = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (repeat_count <= 0) {
            usage();
            return -1;
          }
          break;
        case 'M':
          enable_mem_pattern = false;
          break;
        case 'n':
          // run only some whitelisted tests
          // TODO: parse name str to an array
          whitelisted_test_cases.emplace_back(optarg);
          break;
        case 'e':
          if (!CompareCString(optarg, ORT_TSTR("cpu"))) {
            // do nothing
          } else if (!CompareCString(optarg, ORT_TSTR("cuda"))) {
            enable_cuda = true;
          } else if (!CompareCString(optarg, ORT_TSTR("mkldnn"))) {
            enable_mkl = true;
          } else if (!CompareCString(optarg, ORT_TSTR("ngraph"))) {
            enable_ngraph = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nuphar"))) {
            enable_nuphar = true;
          } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
            enable_tensorrt = true;
          } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
            enable_openvino = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nnapi"))) {
            enable_nnapi = true;
          } else {
            usage();
            return -1;
          }
          break;
        case 'x':
          enable_sequential_execution = false;
          break;
        case 'o': {
          int tmp = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          switch (tmp) {
            case ORT_DISABLE_ALL:
              graph_optimization_level = ORT_DISABLE_ALL;
              break;
            case ORT_ENABLE_BASIC:
              graph_optimization_level = ORT_ENABLE_BASIC;
              break;
            case ORT_ENABLE_EXTENDED:
              graph_optimization_level = ORT_ENABLE_EXTENDED;
              break;
            case ORT_ENABLE_ALL:
              graph_optimization_level = ORT_ENABLE_ALL;
              break;
            default: {
              if (tmp > ORT_ENABLE_ALL) {  // relax constraint
                graph_optimization_level = ORT_ENABLE_ALL;
              } else {
                fprintf(stderr, "See usage for valid values of graph optimization level\n");
                usage();
                return -1;
              }
            }
          }
          user_graph_optimization_level_set = true;
          break;
        }
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }
  if (concurrent_session_runs > 1 && repeat_count > 1) {
    fprintf(stderr, "when you use '-r [repeat]', please set '-c' to 1\n");
    usage();
    return -1;
  }
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify a test data dir\n");
    usage();
    return -1;
  }

  try {
    env = Ort::Env{logging_level, "Default"};
  } catch (std::exception& ex) {
    fprintf(stderr, "Error creating environment: %s \n", ex.what());
    return -1;
  }

  std::vector<std::basic_string<PATH_CHAR_TYPE> > data_dirs;
  TestResultStat stat;

  for (int i = 0; i != argc; ++i) {
    data_dirs.emplace_back(argv[i]);
  }
  {
    double per_sample_tolerance = 1e-3;
    // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
    // when openvino is enabled, set it to a larger value for resolving MNIST accuracy mismatch
    double relative_per_sample_tolerance = enable_cuda ? 0.017 : enable_openvino ? 0.009 : 1e-3;

    Ort::SessionOptions sf;

    if (enable_cpu_mem_arena)
      sf.EnableCpuMemArena();
    else
      sf.DisableCpuMemArena();
    if (enable_mem_pattern)
      sf.EnableMemPattern();
    else
      sf.DisableMemPattern();
    if (enable_sequential_execution)
      sf.EnableSequentialExecution();
    else
      sf.DisableSequentialExecution();

    if (enable_tensorrt) {
#ifdef USE_TENSORRT
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, 0));
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
      fprintf(stderr, "TensorRT is not supported in this build");
      return -1;
#endif
    }

    if (enable_openvino) {
#ifdef USE_OPENVINO
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_OpenVINO(sf, "CPU"));
#else
      fprintf(stderr, "OpenVINO is not supported in this build");
      return -1;
#endif
    }
    if (enable_cuda) {
#ifdef USE_CUDA
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
      fprintf(stderr, "CUDA is not supported in this build");
      return -1;
#endif
    }
    if (enable_nuphar) {
#ifdef USE_NUPHAR
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, /*allow_unaligned_buffers*/ 1, ""));
#else
      fprintf(stderr, "Nuphar is not supported in this build");
      return -1;
#endif
    }
    if (enable_mkl) {
#ifdef USE_MKLDNN
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "MKL-DNN is not supported in this build");
      return -1;
#endif
    }
    if (enable_ngraph) {  //TODO: Re-order the priority?
#ifdef USE_NGRAPH
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_NGraph(sf, "CPU"));
#else
      fprintf(stderr, "nGraph is not supported in this build");
      return -1;
#endif
    }
    if (enable_nnapi) {
#ifdef USE_NNAPI
      ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf));
#else
      fprintf(stderr, "DNNLibrary/NNAPI is not supported in this build");
      return -1;
#endif
    }

    if (user_graph_optimization_level_set) {
      sf.SetGraphOptimizationLevel(graph_optimization_level);
    }

    std::unordered_set<std::string> cuda_flaky_tests = {
        "fp16_inception_v1", "fp16_shufflenet", "fp16_tiny_yolov2"};

#if (defined(_WIN32) && !defined(_WIN64)) || (defined(__GNUG__) && !defined(__LP64__))
    //Minimize mem consumption
    LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance, [&stat, &sf, enable_cuda, &cuda_flaky_tests, &env](ITestCase* l) {
      std::unique_ptr<ITestCase> test_case_ptr(l);
      if (enable_cuda && cuda_flaky_tests.find(l->GetTestCaseName()) != cuda_flaky_tests.end()) {
        return;
      }
      TestResultStat per_case_stat;
      std::vector<ITestCase*> per_case_tests = {l};
      TestEnv per_case_args(per_case_tests, per_case_stat, env, sf);
      RunTests(per_case_args, 1, 1, 1, GetDefaultThreadPool(Env::Default()));
      stat += per_case_stat;
    });
#else
    std::vector<ITestCase*> tests;
    LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance, [&tests](ITestCase* l) { tests.push_back(l); });
    if (enable_cuda) {
      for (auto it = tests.begin(); it != tests.end();) {
        auto iter = cuda_flaky_tests.find((*it)->GetTestCaseName());
        if (iter != cuda_flaky_tests.end()) {
          delete *it;
          it = tests.erase(it);
        } else {
          ++it;
        }
      }
    }

    TestEnv args(tests, stat, env, sf);
    Status st = RunTests(args, p_models, concurrent_session_runs, static_cast<size_t>(repeat_count),
                         GetDefaultThreadPool(Env::Default()));
    if (!st.IsOK()) {
      fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
      return -1;
    }
    for (ITestCase* l : tests) {
      delete l;
    }
#endif
    std::string res = stat.ToString();
    fwrite(res.c_str(), 1, res.size(), stdout);
  }
  // clang-format off

  struct BrokenTest
  {
    std::string test_name_;
    std::string reason_;
    std::set<std::string> broken_versions_ = {}; // apply to all versions if empty
    BrokenTest(std::string name, std::string reason) : test_name_(std::move(name)), reason_(std::move(reason)) {}
    BrokenTest(std::string name, std::string reason, const std::initializer_list<std::string>& versions) :
      test_name_(std::move(name)), reason_(std::move(reason)), broken_versions_(versions) {}
    bool operator < (const struct BrokenTest& test) const {
        return strcmp(test_name_.c_str(), test.test_name_.c_str()) < 0;
    }
  };

  std::set<BrokenTest> broken_tests = {
      {"constantofshape_float_ones", "test data bug", {"onnx141","onnx150"}},
      {"constantofshape_int_zeros", "test data bug", {"onnx141","onnx150"}},
      {"convtranspose_1d", "1d convtranspose not supported yet"},
      {"convtranspose_3d", "3d convtranspose not supported yet"},
      {"cast_STRING_to_FLOAT", "Linux CI has old ONNX python package with bad test data", {"onnx141"}},
      // Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8. 
      // "e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8,
      // which doesn't match the expected output that was generated with numpy.
      {"cast_FLOAT_to_STRING", "Numpy float to string has unexpected rounding for some results."},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong", {"onnx141"}},
      {"maxpool_with_argmax_2d_precomputed_strides", "ShapeInferenceError"},
      {"tf_inception_v2", "result mismatch"},
      {"mxnet_arcface", "result mismatch"},
      {"top_k", "not implemented yet for opset 11"},
      {"top_k_smallest", "not implemented yet for opset 11"},
      {"top_k_negative_axis", "TopK(11) not implemented yet"},
      {"unique_not_sorted_without_axis", "Expected data for 'Y' is incorrect and in sorted order."},
      {"cumsum_1d_reverse_exclusive", "only failing linux GPU CI. Likely build error."},
      {"det_2d", "not implemented yet"},
      {"det_nd", "not implemented yet"},
      {"resize_downsample_scales_cubic_A_n0p5_exclude_outside", "not implemented yet"},
      {"resize_downsample_scales_cubic_align_corners", "not implemented yet"},
      {"resize_downsample_scales_cubic", "not implemented yet"},
      {"resize_downsample_scales_linear_align_corners", "not implemented yet"},
      {"resize_downsample_scales_linear", "not implemented yet"},
      {"resize_downsample_scales_nearest", "not implemented yet"},
      {"resize_downsample_sizes_cubic", "not implemented yet"},
      {"resize_downsample_sizes_linear_pytorch_half_pixel", "not implemented yet"},
      {"resize_downsample_sizes_nearest", "not implemented yet"},
      {"resize_downsample_sizes_nearest_tf_half_pixel_for_nn", "not implemented yet"},
      {"resize_tf_crop_and_resize", "not implemented yet"},
      {"resize_upsample_scales_cubic_A_n0p5_exclude_outside", "not implemented yet"},
      {"resize_upsample_scales_cubic_align_corners", "not implemented yet"},
      {"resize_upsample_scales_cubic_asymmetric", "not implemented yet"},
      {"resize_upsample_scales_cubic", "not implemented yet"},
      {"resize_upsample_scales_linear_align_corners", "not implemented yet"},
      {"resize_upsample_scales_linear", "not implemented yet"},
      {"resize_upsample_scales_nearest", "not implemented yet"},
      {"resize_upsample_sizes_cubic", "not implemented yet"},
      {"resize_upsample_sizes_nearest_ceil_half_pixel", "not implemented yet"},
      {"resize_upsample_sizes_nearest", "not implemented yet"},
      {"resize_upsample_sizes_nearest_floor_align_corners", "not implemented yet"},
      {"resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", "not implemented yet"},
      {"scatternd", "not implemented yet"},
      {"sequence_model7", "SplitToSequence not implemented yet"},
      {"sequence_model6", "SplitToSequence not implemented yet"},
      {"sequence_model5", "SequenceConstruct not implemented yet"},
      {"sequence_model4", "SequenceConstruct not implemented yet"},
      {"sequence_model3", "SequenceConstruct not implemented yet"},
      {"sequence_model2", "SequenceConstruct not implemented yet"},
      {"sequence_model1", "Sequence* not implemented yet"},
      {"scatter_elements_with_negative_indices", "ScatterElements(11) not implemented yet"},
      {"onehot_without_axis", "OneHot(11) not implemented yet"},
      {"onehot_with_negative_axis", "OneHot(11) not implemented yet"},
      {"onehot_with_axis", "OneHot(11) not implemented yet"},
      {"onehot_negative_indices", "OneHot(11) not implemented yet"},
      {"bitshift_right_uint8", "BitShift(11) uint8 support not enabled currently"},
      {"bitshift_right_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"bitshift_left_uint8", "BitShift(11) uint8 support not enabled currently"},
      {"bitshift_left_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"reflect_pad", "test data type `int32_t` not supported yet, the `float` equivalent is covered via unit tests"},
      {"edge_pad", "test data type `int32_t` not supported yet, the `float` equivalent is covered via unit tests"},
};

#ifdef USE_NGRAPH
  broken_tests.insert({"dequantizelinear", "ambiguity in scalar dimensions [] vs [1]", {"onnx150"}});
  broken_tests.insert({"qlinearconv", "ambiguity in scalar dimensions [] vs [1]"});
  broken_tests.insert({"quantizelinear", "ambiguity in scalar dimensions [] vs [1]", {"onnx150"}});
  broken_tests.insert({"clip_splitbounds", "not implemented yet for opset 11"});
  broken_tests.insert({"clip_outbounds", "not implemented yet for opset 11"});
  broken_tests.insert({"clip_example", "not implemented yet for opset 11"});
  broken_tests.insert({"clip_default_min", "not implemented yet for opset 11"});
  broken_tests.insert({"clip_default_max", "not implemented yet for opset 11"});
  broken_tests.insert({"clip", "not implemented yet for opset 11"});
  broken_tests.insert({"depthtospace_crd_mode_example", "NGraph does not support CRD mode"});
  broken_tests.insert({"depthtospace_crd_mode", "NGraph does not support CRD mode"});
  broken_tests.insert({"argmax_negative_axis_keepdims_example", "not implemented yet for opset 11"});
  broken_tests.insert({"argmax_negative_axis_keepdims_random", "not implemented yet for opset 11"});
  broken_tests.insert({"argmin_negative_axis_keepdims_example", "not implemented yet for opset 11"});
  broken_tests.insert({"argmin_negative_axis_keepdims_random", "not implemented yet for opset 11"});
  broken_tests.insert({"gemm_default_no_bias", "not implemented yet for opset 11"});
  broken_tests.insert({"hardmax_negative_axis", "not implemented yet for opset 11"});
  broken_tests.insert({"flatten_negative_axis1", "not implemented yet for opset 11"});
  broken_tests.insert({"flatten_negative_axis2", "not implemented yet for opset 11"});
  broken_tests.insert({"flatten_negative_axis3", "not implemented yet for opset 11"});
  broken_tests.insert({"flatten_negative_axis4", "not implemented yet for opset 11"});
  broken_tests.insert({"squeeze_negative_axes", "not implemented yet for opset 11"});
  broken_tests.insert({"unsqueeze_negative_axes", "not implemented yet for opset 11"});
  broken_tests.insert({"reduce_sum_square_negative_axes_keepdims_random", "ReduceSumSquare(11) not implemented yet"});
  broken_tests.insert({"reduce_sum_square_negative_axes_keepdims_example", "ReduceSumSquare(11) not implemented yet"});
  broken_tests.insert({"reduce_sum_negative_axes_keepdims_random", "ReduceSum(11) not implemented yet"});
  broken_tests.insert({"reduce_sum_negative_axes_keepdims_example", "ReduceSum(11) not implemented yet"});
  broken_tests.insert({"reduce_prod_negative_axes_keepdims_random", "ReduceProd(11) not implemented yet"});
  broken_tests.insert({"reduce_prod_negative_axes_keepdims_example", "ReduceProd(11) not implemented yet"});
  broken_tests.insert({"reduce_min_negative_axes_keepdims_random", "ReduceMin(11) not implemented yet"});
  broken_tests.insert({"reduce_min_negative_axes_keepdims_example", "ReduceMin(11) not implemented yet"});
  broken_tests.insert({"reduce_mean_negative_axes_keepdims_random", "ReduceMean(11) not implemented yet"});
  broken_tests.insert({"reduce_mean_negative_axes_keepdims_example", "ReduceMean(11) not implemented yet"});
  broken_tests.insert({"reduce_max_negative_axes_keepdims_random", "ReduceMax(11) not implemented yet"});
  broken_tests.insert({"reduce_max_negative_axes_keepdims_example", "ReduceMax(11) not implemented yet"});
  broken_tests.insert({"reduce_log_sum_negative_axes", "ReduceLogSum(11) not implemented yet"});
  broken_tests.insert({"reduce_log_sum_exp_negative_axes_keepdims_random", "ReduceLogSumExp(11) not implemented yet"});
  broken_tests.insert({"reduce_log_sum_exp_negative_axes_keepdims_example", "ReduceLogSumExp(11) not implemented yet"});
  broken_tests.insert({"reduce_l2_negative_axes_keep_dims_random", "ReduceL2(11) not implemented yet"});
  broken_tests.insert({"reduce_l2_negative_axes_keep_dims_example", "ReduceL2(11) not implemented yet"});
  broken_tests.insert({"reduce_l1_negative_axes_keep_dims_random", "ReduceL1(11) not implemented yet"});
  broken_tests.insert({"reduce_l1_negative_axes_keep_dims_example", "ReduceL1(11) not implemented yet"});
  broken_tests.insert({"constant_pad", "not implemented yet for opset 11"});
#endif

#ifdef USE_MKLDNN
  broken_tests.insert({"tf_mobilenet_v2_1.0_224", "result mismatch"});
  broken_tests.insert({"tf_mobilenet_v2_1.4_224", "result mismatch"});
  broken_tests.insert({"tf_mobilenet_v1_1.0_224", "result mismatch"});
  broken_tests.insert({"mobilenetv2-1.0", "result mismatch"});
  broken_tests.insert({"candy", "result mismatch"});
  broken_tests.insert({"range_float_type_positive_delta_expanded", "get unknown exception from MKLDNN EP"});
  broken_tests.insert({"range_int32_type_negative_delta_expanded", "get unknown exception from MKLDNN EP"});
#endif

#ifdef USE_OPENVINO
  broken_tests.insert({"fp16_shufflenet", "accuracy mismatch with fp16 precision"});
  broken_tests.insert({"fp16_inception_v1", "accuracy mismatch with fp16 precision"});
  broken_tests.insert({"fp16_tiny_yolov2", "accuaracy mismatch with fp16 precision"});
#ifdef OPENVINO_CONFIG_GPU_FP32
  broken_tests.insert({"tiny_yolov2", "accuracy mismatch"});
  broken_tests.insert({"div", "will be fixed in the next release"});
#ifdef OPENVINO_CONFIG_GPU_FP16
  broken_tests.insert({"div", "will be fixed in the next release"});
#endif
#endif
#endif

#ifdef USE_NNAPI
  broken_tests.insert({"scan9_sum", "Error with the extra graph"});
  broken_tests.insert({"scan_sum", "Error with the extra graph"});
  broken_tests.insert({"mvn_expanded", "Failed to find kernel for MemcpyFromHost(1) (node Memcpy_1)"});
#endif

#ifdef USE_TENSORRT
  broken_tests.insert({"fp16_shufflenet", "TRT EP bug"});
  broken_tests.insert({"fp16_inception_v1", "TRT EP bug"});
  broken_tests.insert({"fp16_tiny_yolov2", "TRT EP bug"});
  broken_tests.insert({"tf_inception_v3", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_mobilenet_v1_1.0_224", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_mobilenet_v2_1.0_224", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_mobilenet_v2_1.4_224", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_resnet_v1_101", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_resnet_v1_152", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_resnet_v1_50", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_resnet_v2_101", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_resnet_v2_152", "TRT Engine couldn't be created"});
  broken_tests.insert({"tf_resnet_v2_50", "TRT Engine couldn't be created"});
#endif

#ifdef USE_CUDA
  broken_tests.insert({"mxnet_arcface", "result mismatch"});
  broken_tests.insert({"mask_rcnn_keras", "result mismatch"});
  broken_tests.insert({"mlperf_ssd_mobilenet_300", "unknown error"});
  broken_tests.insert({"mlperf_ssd_resnet34_1200", "unknown error"});
  broken_tests.insert({"tf_inception_v1", "flaky test"}); //TODO: Investigate cause for flakiness
#endif
  // clang-format on

#if defined(_WIN32) && !defined(_WIN64)
  broken_tests.insert({"vgg19", "failed: bad allocation"});
#endif

#if defined(__GNUG__) && !defined(__LP64__)
  broken_tests.insert({"nonzero_example", "failed: type mismatch", {"onnx123", "onnx130", "onnx141", "onnx150", "onnxtip"}});
  broken_tests.insert({"slice_neg_steps", "failed: type mismatch"});
  broken_tests.insert({"mod_float_mixed_sign_example", "failed: type mismatch"});
#endif

#ifdef DISABLE_CONTRIB_OPS
  broken_tests.insert({"mask_rcnn_keras", "This model uses contrib ops."});
  broken_tests.insert({"coreml_SqueezeNet_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Permute_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_ReLU_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Padding-Upsampling-Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"fp16_tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Pooling_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Padding_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet_small", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet_large", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_leakyrelu_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_hard_sigmoid_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_elu_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Dense_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Conv2D_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_VGG16_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_Resnet50_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_Inceptionv3_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_FNS-Candy_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_AgeNet_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_ImageNet_large", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_ImageNet_small", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu_default", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_default_axes", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu_example", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_neg failed", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_start_out_of_bounds", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_end_out_of_bounds", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_neg", "This model uses contrib ops."});
  broken_tests.insert({"mvn", "This model uses contrib ops.", {"onnx130"}});
#endif

  int result = 0;
  for (const auto& p : stat.GetFailedTest()) {
    BrokenTest t = {p.first, ""};
    auto iter = broken_tests.find(t);
    if (iter == broken_tests.end() ||
        (p.second != TestModelInfo::unknown_version && !iter->broken_versions_.empty() && iter->broken_versions_.find(p.second) == iter->broken_versions_.end())) {
      fprintf(stderr, "test %s failed, please fix it\n", p.first.c_str());
      result = -1;
    }
  }
  return result;
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  Ort::Env env{nullptr};
  int retval = -1;
  try {
    retval = real_main(argc, argv, env);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  // Release the protobuf library if we failed to create an env (the env will release it automatically on destruction)
  if (!env) {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
  return retval;
}
