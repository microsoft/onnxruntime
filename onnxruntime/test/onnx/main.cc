// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/session/onnxruntime_cxx_api.h>
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
#include "path_lib.h"
#include "sync_api.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"

using namespace onnxruntime;

namespace {
void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-A : Disable memory arena\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-r [repeat]: Specifies the number of times to repeat\n"
      "\t-v: verbose\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda' or 'mkldnn'. Default: 'cpu'.\n"
      "\t-x: Use parallel executor, default (without -x): sequential executor.\n"
      "\t-h: help\n");
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
int GetNumCpuCores() {
  return std::thread::hardware_concurrency();
}
#endif
}  // namespace

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  //if this var is not empty, only run the tests with name in this list
  std::vector<std::basic_string<PATH_CHAR_TYPE> > whitelisted_test_cases;
  int concurrent_session_runs = GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  bool enable_sequential_execution = true;
  int repeat_count = 1;
  int p_models = GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_mkl = false;
  bool enable_nuphar = false;
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  {
    int ch;
    while ((ch = getopt(argc, argv, ORT_TSTR("Ac:hj:m:n:r:e:xv"))) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'v':
          logging_level = ORT_LOGGING_LEVEL_INFO;
          break;
        case 'c':
          concurrent_session_runs = static_cast<int>(MyStrtol<PATH_CHAR_TYPE>(optarg, nullptr, 10));
          if (concurrent_session_runs <= 0) {
            usage();
            return -1;
          }
          break;
        case 'j':
          p_models = static_cast<int>(MyStrtol<PATH_CHAR_TYPE>(optarg, nullptr, 10));
          if (p_models <= 0) {
            usage();
            return -1;
          }
          break;
        case 'r':
          repeat_count = static_cast<int>(MyStrtol<PATH_CHAR_TYPE>(optarg, nullptr, 10));
          if (repeat_count <= 0) {
            usage();
            return -1;
          }
          break;
        case 'm':
          //ignore.
          break;
        case 'n':
          //run only some whitelisted tests
          //TODO: parse name str to an array
          whitelisted_test_cases.emplace_back(optarg);
          break;
        case 'e':
          if (!MyStrCmp(optarg, ORT_TSTR("cpu"))) {
            //do nothing
          } else if (!MyStrCmp(optarg, ORT_TSTR("cuda"))) {
            enable_cuda = true;
          } else if (!MyStrCmp(optarg, ORT_TSTR("mkldnn"))) {
            enable_mkl = true;
          } else if (!MyStrCmp(optarg, ORT_TSTR("nuphar"))) {
            enable_nuphar = true;
          } else {
            usage();
            return -1;
          }
          break;
        case 'x':
          enable_sequential_execution = false;
          break;
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
  std::unique_ptr<OrtEnv> env;
  {
    OrtEnv* t;
    OrtStatus* ost = OrtInitialize(logging_level, "Default", &t);
    if (ost != nullptr) {
      fprintf(stderr, "Error creating environment: %s \n", OrtGetErrorMessage(ost));
      OrtReleaseStatus(ost);
      return -1;
    }
    env.reset(t);
  }
  std::vector<std::basic_string<PATH_CHAR_TYPE> > data_dirs;
  TestResultStat stat;

  std::unique_ptr<OrtAllocator> default_allocator;
  {
    OrtAllocator* p;
    OrtStatus* ost = OrtCreateDefaultAllocator(&p);
    if (ost != nullptr) {
      fprintf(stderr, "Error creating environment: %s \n", OrtGetErrorMessage(ost));
      OrtReleaseStatus(ost);
      return -1;
    }
    default_allocator.reset(p);
  }
  for (int i = 0; i != argc; ++i) {
    data_dirs.emplace_back(argv[i]);
  }
  {
    std::vector<ITestCase*> tests = LoadTests(data_dirs, whitelisted_test_cases, default_allocator.get());
    SessionOptionsWrapper sf(env.get());
    if (enable_cpu_mem_arena)
      sf.EnableCpuMemArena();
    else
      sf.DisableCpuMemArena();
    if (enable_sequential_execution)
      sf.EnableSequentialExecution();
    else
      sf.DisableSequentialExecution();
    if (enable_cuda) {
#ifdef USE_CUDA
      OrtProviderFactoryInterface** f;
      ORT_THROW_ON_ERROR(OrtCreateCUDAExecutionProviderFactory(0, &f));
      sf.AppendExecutionProvider(f);
      OrtReleaseObject(f);
#else
      fprintf(stderr, "CUDA is not supported in this build");
      return -1;
#endif
    }
    if (enable_nuphar) {
#ifdef USE_NUPHAR
      OrtProviderFactoryInterface** f;
      ORT_THROW_ON_ERROR(OrtCreateNupharExecutionProviderFactory(0, "", &f));
      sf.AppendExecutionProvider(f);
      OrtReleaseObject(f);
#else
      fprintf(stderr, "Nuphar is not supported in this build");
      return -1;
#endif
    }
    if (enable_mkl) {
#ifdef USE_MKLDNN
      OrtProviderFactoryInterface** f;
      ORT_THROW_ON_ERROR(OrtCreateMkldnnExecutionProviderFactory(enable_cpu_mem_arena ? 1 : 0, &f));
      sf.AppendExecutionProvider(f);
      OrtReleaseObject(f);
#else
      fprintf(stderr, "MKL-DNN is not supported in this build");
      return -1;
#endif
    }
    TestEnv args(tests, stat, sf);
    Status st = RunTests(args, p_models, concurrent_session_runs, static_cast<size_t>(repeat_count), GetDefaultThreadPool(Env::Default()));
    if (!st.IsOK()) {
      fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
      return -1;
    }

    std::string res = stat.ToString();
    fwrite(res.c_str(), 1, res.size(), stdout);
    for (ITestCase* l : tests) {
      delete l;
    }
  }
  std::map<std::string, std::string> broken_tests{
      {"AvgPool1d", "disable reason"},
      {"AvgPool1d_stride", "disable reason"},
      {"AvgPool2d", "disable reason"},
      {"AvgPool2d_stride", "disable reason"},
      {"AvgPool3d", "disable reason"},
      {"AvgPool3d_stride", "disable reason"},
      {"AvgPool3d_stride1_pad0_gpu_input", "disable reason"},
      {"BatchNorm1d_3d_input_eval", "disable reason"},
      {"BatchNorm2d_eval", "disable reason"},
      {"BatchNorm2d_momentum_eval", "disable reason"},
      {"BatchNorm3d_eval", "disable reason"},
      {"BatchNorm3d_momentum_eval", "disable reason"},
      {"GLU", "disable reason"},
      {"GLU_dim", "disable reason"},
      {"Linear", "disable reason"},
      {"PReLU_1d", "disable reason"},
      {"PReLU_1d_multiparam", "disable reason"},
      {"PReLU_2d", "disable reason"},
      {"PReLU_2d_multiparam", "disable reason"},
      {"PReLU_3d", "disable reason"},
      {"PReLU_3d_multiparam", "disable reason"},
      {"PoissonNLLLLoss_no_reduce", "disable reason"},
      {"Softsign", "disable reason"},
      {"convtranspose_1d", "disable reason"},
      {"convtranspose_3d", "disable reason"},
      {"eyelike_populate_off_main_diagonal", "disable reason"},
      {"eyelike_with_dtype", "disable reason"},
      {"eyelike_without_dtype", "disable reason"},
      {"flatten_axis0", "disable reason"},
      {"flatten_axis1", "disable reason"},
      {"flatten_axis2", "disable reason"},
      {"flatten_axis3", "disable reason"},
      {"flatten_default_axis", "disable reason"},
      {"gemm_broadcast", "disable reason"},
      {"gemm_nobroadcast", "disable reason"},
      {"greater", "disable reason"},
      {"greater_bcast", "disable reason"},
      {"less", "disable reason"},
      {"less_bcast", "disable reason"},
      {"matmul_2d", "disable reason"},
      {"matmul_3d", "disable reason"},
      {"matmul_4d", "disable reason"},
      {"mvn", "disable reason"},
      {"operator_add_broadcast", "disable reason"},
      {"operator_add_size1_broadcast", "disable reason"},
      {"operator_add_size1_right_broadcast", "disable reason"},
      {"operator_add_size1_singleton_broadcast", "disable reason"},
      {"operator_addconstant", "disable reason"},
      {"operator_addmm", "disable reason"},
      {"operator_basic", "disable reason"},
      {"operator_lstm", "disable reason"},
      {"operator_mm", "disable reason"},
      {"operator_non_float_params", "disable reason"},
      {"operator_params", "disable reason"},
      {"operator_pow", "disable reason"},
      {"operator_rnn", "disable reason"},
      {"operator_rnn_single_layer", "disable reason"},
      {"prelu_broadcast", "disable reason"},
      {"prelu_example", "disable reason"},
      {"upsample_nearest", "opset 9 not supported yet"},
      {"sinh_example", "opset 9 not supported yet"},
      {"cosh_example", "opset 9 not supported yet"},
      {"asinh_example", "opset 9 not supported yet"},
      {"acosh_example", "opset 9 not supported yet"},
      {"atanh_example", "opset 9 not supported yet"},
      {"sign_model", "opset 9 not supported yet"},
      {"sign", "opset 9 not supported yet"},
      {"scatter_with_axis", "opset 9 not supported yet"},
      {"scatter_without_axis", "opset 9 not supported yet"},
      {"scan_sum", "opset 9 not supported yet"}};

#ifdef USE_CUDA
  broken_tests["maxpool_2d_default"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_pads"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_precomputed_strides"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_precomputed_pads"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_strides"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_precomputed_same_upper"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_same_upper"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_2d_same_lower"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_3d_default"] = "cudnn pooling only support input dimension >= 3";
  broken_tests["maxpool_1d_default"] = "cudnn pooling only support input dimension >= 3";

  broken_tests["tf_nasnet_large"] = "unknown failure on CUDA";
  broken_tests["tf_inception_resnet_v2"] = "unknown failure on CUDA";
  broken_tests["tf_nasnet_mobile"] = "unknown failure on CUDA";
  broken_tests["tf_resnet_v2_152"] = "unknown failure on CUDA";
  broken_tests["tf_inception_v4"] = "unknown failure on CUDA";
  broken_tests["tf_resnet_v2_101"] = "unknown failure on CUDA";
  broken_tests["tf_pnasnet_large"] = "unknown failure on CUDA";
  broken_tests["tf_inception_v1"] = "unknown failure on CUDA";
  broken_tests["fp16_inception_v1"] = "CUDNN_STATUS_ARCH_MISMATCH";
  broken_tests["fp16_shufflenet"] = "CUDNN_STATUS_ARCH_MISMATCH";
  broken_tests["fp16_tiny_yolov2"] = "CUDNN_STATUS_ARCH_MISMATCH";
#endif

  int result = 0;
  for (const std::string& s : stat.GetFailedTest()) {
    if (broken_tests.find(s) == broken_tests.end()) {
      fprintf(stderr, "test %s failed, please fix it\n", s.c_str());
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
  try {
    return real_main(argc, argv);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    return -1;
  }
}
