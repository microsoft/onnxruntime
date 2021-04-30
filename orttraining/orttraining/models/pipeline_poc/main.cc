// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"

#if defined(USE_CUDA) && defined(USE_MPI)

#include "cxxopts.hpp"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/models/runner/constant.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"
#include "orttraining/models/runner/data_loader.h"

#include "core/providers/cuda/cuda_execution_provider.h"

#include <condition_variable>
#include <mutex>
#include <tuple>

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

struct Parameters {
  size_t num_steps;
  std::string model_stage0_name;
  std::string model_stage1_name;
  std::string model_stage2_name;
};

Status parse_arguments(int argc, char* argv[], Parameters& params) {
  cxxopts::Options options("P2P Runner Options", "Main Program to run P2P graphs.");
  // clang-format off
  options
    .add_options()
      ("num_steps", "The number of times that the pipeline will be executed.", cxxopts::value<size_t>())
      ("model_stage0_name", "First stage.", cxxopts::value<std::string>())
      ("model_stage1_name", "Second stage.", cxxopts::value<std::string>())
      ("model_stage2_name", "Third stage.", cxxopts::value<std::string>());
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);
    params.num_steps = flags["num_steps"].as<size_t>();
    params.model_stage0_name = ToPathString(flags["model_stage0_name"].as<std::string>());
    params.model_stage1_name = ToPathString(flags["model_stage1_name"].as<std::string>());
    params.model_stage2_name = ToPathString(flags["model_stage2_name"].as<std::string>());
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << "\n"
         << options.help() << "\n";
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, msg);
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {
  Parameters params;

  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kERROR,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id,
                                                  -1};

  std::unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(nullptr, env) == Status::OK(), "Enviroment creation fails.");
  ORT_ENFORCE(parse_arguments(argc, argv, params) == Status::OK(), "Parsing command-line argument fails");

  // Set up MPI.
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // setup onnxruntime env
  std::vector<FreeDimensionOverride> overrides = {};
  SessionOptions so = {
      ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
      ExecutionOrder::DEFAULT,           //execution_order
      false,                             //enable_profiling
      ORT_TSTR(""),                      //optimized_model_filepath
      true,                              //enable_mem_pattern
      true,                              //enable_mem_reuse
      true,                              //enable_cpu_mem_arena
      ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
      "",                                //session_logid
      -1,                                //session_log_severity_level
      0,                                 //session_log_verbosity_level
      5,                                 //max_num_graph_transformation_steps
      TransformerLevel::Level1,          //graph_optimization_level
      {},                                //intra_op_param
      {},                                //inter_op_param
      overrides,                         //free_dimension_overrides
      true,                              //use_per_session_threads
      true,                              //thread_pool_allow_spinning
      false,                             //use_deterministic_compute
      {},                                //session_configurations
      {}                                 //initializers_to_share_map
  };

  InferenceSession session_object{so, *env};

  Status st;
  CUDAExecutionProviderInfo xp_info{static_cast<OrtDevice::DeviceId>(world_rank)};
  st = session_object.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info));
  ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());

  std::string model_at_rank;
  if (world_rank == 0) {
    st = session_object.Load(params.model_stage0_name);
    ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());
    st = session_object.Initialize();
    ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());

  } else if (world_rank == 1) {
    st = session_object.Load(params.model_stage1_name);
    ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());
    st = session_object.Initialize();
    ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());
  } else if (world_rank == 2) {
    st = session_object.Load(params.model_stage2_name);
    ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());
    st = session_object.Initialize();
    ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());
  }

  for (size_t round = 0; round < params.num_steps; ++round) {
    if (world_rank == 0) {
      std::cout << "Round " << round << " starts..." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
      // Prepare inputs and outputs.
      OrtValue input;
      std::vector<int64_t> shape = {2, 2};
      std::vector<float> value = {1000.f + round, 2.f, 3.f, 4.f};
      TrainingUtil::CreateCpuMLValue(shape, value, &input);
      NameMLValMap feeds = std::unordered_map<std::string, OrtValue>{{"X1", input}};

      std::vector<std::string> output_names{"X8"};
      std::vector<OrtValue> fetches;

      RunOptions run_options;
      run_options.run_tag = "Session at MPI rank " + std::to_string(world_rank);

      // Run a model.
      st = session_object.Run(run_options, feeds, output_names, &fetches);
      ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());

      for (int j = 0; j < 1; ++j) {
        const Tensor& received = fetches[j].Get<Tensor>();
        const float* received_ptr = received.template Data<float>();
        for (int i = 0; i < received.Shape().Size(); ++i) {
          std::cout << "Round: " << round << ", Rank: " << world_rank << ", Output" << j << ": received[" << i << "]=" << received_ptr[i] << std::endl;
        }
      }
    } else if (world_rank == 1) {
      RunOptions run_options;
      run_options.run_tag = "Session at MPI rank " + std::to_string(world_rank);

      // Prepare inputs and outputs.
      NameMLValMap feeds;
      std::vector<std::string> output_names{"X2", "X4"};
      std::vector<OrtValue> fetches;

      // Run a model.
      st = session_object.Run(run_options, feeds, output_names, &fetches);
      ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());

      for (int j = 0; j < 2; ++j) {
        const Tensor& received = fetches[j].Get<Tensor>();
        const float* received_ptr = received.template Data<float>();
        for (int i = 0; i < received.Shape().Size(); ++i) {
          std::cout << "Round: " << round << ", Rank: " << world_rank << ", Output" << j << ": received[" << i << "]=" << received_ptr[i] << std::endl;
        }
      }
    } else if (world_rank == 2) {
      RunOptions run_options;
      run_options.run_tag = "Session at MPI rank " + std::to_string(world_rank);

      // Prepare inputs and outputs.
      NameMLValMap feeds;
      std::vector<std::string> output_names{"X3"};
      std::vector<OrtValue> fetches;

      // Run a model.
      st = session_object.Run(run_options, feeds, output_names, &fetches);
      ORT_ENFORCE(st == Status::OK(), "MPI rank ", world_rank, ": ", st.ErrorMessage());

      for (int j = 0; j < 1; ++j) {
        const Tensor& received = fetches[j].Get<Tensor>();
        const float* received_ptr = received.template Data<float>();
        for (int i = 0; i < received.Shape().Size(); ++i) {
          std::cout << "Round: " << round << ", Rank: " << world_rank << ", Output" << j << ": received[" << i << "]=" << received_ptr[i] << std::endl;
        }
      }
    }
  }

  MPI_Finalize();

  return 0;
}

#else

int main(int, char*[]) {
  ORT_NOT_IMPLEMENTED("P2P demo currently requires CUDA to run.");
}

#endif
