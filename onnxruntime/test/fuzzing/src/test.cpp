// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "src/mutator.h"
#include "testlog.h"
#include "OnnxPrediction.h"
#include <type_traits>

using userOptions = struct
{
  bool writeModel;
  bool verbose;
  bool stress;
};

void predict(onnx::ModelProto& model_proto, unsigned int seed)
{
    // Create object for prediction
    //
    OnnxPrediction predict{model_proto};

    // Give predict a function to generate the data
    // to run prediction on.
    //
    predict.SetupInput(GenerateDataForInputTypeTensor, seed);

    // Run the prediction on the data
    //
    predict.RunInference();

    // View the output
    //
    predict.PrintOutputValues();
}

void mutateModelTest(onnx::ModelProto& model_proto, 
                    std::wstring mutatedModelDirName,
                    userOptions opt,
                    unsigned int seed = 0)
{
    // Used to initialize all random engines
    //
    std::wstring modelPrefix =  L"/ReproMutateModel_";
    if(seed == 0)
    {
      seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
      modelPrefix =  L"/MutateModel_";
    }
    
    if(opt.stress)
    {
      Logger::testLog.enable();
    }

    Logger::testLog << L"Mutate test seed: " << seed << Logger::endl;
    opt.stress ? Logger::testLog.disable() : Logger::testLog.enable(); 

    // Create mutator
    //
    protobuf_mutator::Mutator mutator;
    
    // Mutate model
    //
    Logger::testLog << L"Model Successfully Initialized" << Logger::endl;
    mutator.Seed(seed);
    mutator.Mutate(&model_proto, model_proto.ByteSizeLong());

    if (opt.writeModel)
    {
      // Create file to store model
      //
      std::wstringstream mutateModelName;
      mutateModelName << mutatedModelDirName << modelPrefix << seed << L".onnx";
      auto mutateModelFileName = mutateModelName.str();

      // Log the model to a file
      //
      std::ofstream outStream(mutateModelFileName);
      model_proto.SerializeToOstream(&outStream);
      Logger::testLog<< "Mutated Model Written to file: " << mutateModelFileName << Logger::endl;

      // Flush the buffer to ensure the
      // mutated model info for reproduction
      // purposes.
      //
      outStream << std::flush;
    }

    // Flush any logs before prediction
    //
    Logger::testLog.flush();

    // run prediction on model
    //
    predict(model_proto, seed);

    // print out all output before next test
    //
    Logger::testLog.flush();
}

void printUsage()
{
  std::cout << "Not enough command line arguments\n";
  std::cout << "usage:\n"
  << "\t\tFor testing: test.exe /t [options] onnx_model_file test_timeout test_time_scale\n"
  << "\t\tFor repro/debugging: test.exe /r onnx_model_file seed\n"
  << "\n\nonnx_model_file: Unmutated onnx model file\n"
  << "options: /m - output mutated models /v - verbose logging /s - stress test"
  << "test_time_scale: h|m|s\n"
  << "test_timeout: Time to run the test in hrs\n"
  << "seed: The seed that generated the mutated model. This value is the decimal digit part of the mutated model name (or can be found in the logs)\n"
  << "\n";
}

enum class timeScale : char
{
  Hrs = 'h', 
  Min = 'm',
  Sec = 's'
};

using runtimeOpt = struct
{
  std::wstring modelFileName{};
  std::wstring mutateModelDirName{};
  Logger::ccstream errStreamBuf{};
  Logger::wcstream werrStreamBuf{};
  bool repoMode{false};
  int testTimeOut{0};
  unsigned int seed{0};
  timeScale scale{timeScale::Sec};
  userOptions userOpt{false, false, false};
};

int processCommandLine(int argc, char* argv[], runtimeOpt& opt)
{
  if (argc <= 1)
  {
    printUsage();
    return 2;
  }
  else
  {
    bool isTest = std::string{argv[1]} == "/t";
    bool isRepo = std::string{argv[1]} == "/r";

    if(isRepo)
    {
      opt.repoMode = true;
      opt.mutateModelDirName = L"./repromodel";
      std::filesystem::path mutateModelDir{opt.mutateModelDirName};
      if ( !std::filesystem::exists(mutateModelDir) )
      {
        std::filesystem::create_directory(mutateModelDir);
      }

      opt.modelFileName = Logger::towstr(argv[2]);
      Logger::testLog<< L"Repo Model file: " << opt.modelFileName << Logger::endl;

      // Get seed
      //
      std::stringstream parser{argv[3]};
      parser >> opt.seed;
      if (parser.bad())
      {
        throw std::exception("Could not parse seed from command line");
      }

      std::wcout << L"seed: " << opt.seed <<  L"\n";
    }
    else if (isTest)
    {
      int index{argc};
      index--;

      // Parse right to left
      //
      std::stringstream parser;
      char desiredScale;
      parser << argv[index--];
      parser >> desiredScale;
      
      if (parser.bad())
      {
        throw std::exception("Could not parse the time scale from the command line");
      }
      
      opt.scale = static_cast<timeScale>(std::tolower(desiredScale));
      switch(opt.scale)
      {
        case timeScale::Hrs:
        case timeScale::Min:
        case timeScale::Sec:
          break;
        default:
          throw std::exception("Could not parse the time scale from the command line");
      }
      
      parser << argv[index--];
      parser >> opt.testTimeOut;
      if (parser.bad())
      {
        throw std::exception("Could not parse the time value from the command line");
      }

      Logger::testLog<< L"Running Test for: " << opt.testTimeOut << desiredScale << Logger::endl;
      opt.modelFileName = Logger::towstr(argv[index--]);
      Logger::testLog<< L"Model file: " << opt.modelFileName << Logger::endl;
      std::filesystem::path modelFileNamePath{opt.modelFileName};
      if (!std::filesystem::exists(modelFileNamePath))
      {
        throw std::exception("Cannot find model file");
      }

      // process options
      //
      while(index > 0)
      {
        auto option{std::string{argv[index]}};
        if ( option == "/m")
        {
          opt.userOpt.writeModel = true;
        }
        else if (option == "/v")
        {
          opt.userOpt.verbose = true;
        }
        else if (option == "/s")
        {
          opt.userOpt.stress = true;
        }
        index--;
      }

      if (opt.userOpt.stress)
      {
        std::cerr.rdbuf(&opt.errStreamBuf);
        std::wcerr.rdbuf(&opt.werrStreamBuf);
        opt.userOpt.writeModel = false;
        opt.userOpt.verbose = false;
        Logger::testLog.disable();
        Logger::testLog.minLog();
      }

      // create directory for mutated model output 
      //
      if(opt.userOpt.writeModel)
      {
        opt.mutateModelDirName = L"./mutatemodel";
        std::filesystem::path mutateModelDir{opt.mutateModelDirName};
        if ( !std::filesystem::exists(mutateModelDir) )
        {
          std::filesystem::create_directory(mutateModelDir);
        }
      }
    }
    else
    {
      printUsage();
      return 2;
    }
  }

  return 0;
}

int main(int argc, char* argv[]) 
{
  runtimeOpt opt{};
  try
  {
    // Initialize the runtime options
    //
    auto canContinue{processCommandLine(argc, argv, opt) == 0};

    if(!canContinue)
    {
      return -1;
    }

    std::wstring& modelFileName{opt.modelFileName};
    std::wstring& mutateModelDirName{opt.mutateModelDirName};
    bool& repoMode{opt.repoMode};
    int& testTimeOut{opt.testTimeOut};
    unsigned int& seed{opt.seed};
    timeScale& scale{opt.scale};
    userOptions& userOpt{opt.userOpt};
    Logger::wcstream& werrStreamBuf{opt.werrStreamBuf};
    
    // Model file
    //
    std::wstring modelFile {modelFileName};
    
    // Create a stream to hold the model
    //
    std::ifstream modelStream{modelFile};

    // Create an onnx protobuf object
    //
    onnx::ModelProto model_proto;

    // Initialize the model
    //
    if( model_proto.ParseFromIstream(&modelStream) )
    {
      if(repoMode)
      {
        Logger::testLog<< L"Running Prediction for: " << modelFileName 
        << L" with seed " << seed << Logger::endl;
        mutateModelTest(model_proto, mutateModelDirName, userOpt, seed);
        Logger::testLog<< L"Finished Prediction for: " << modelFileName 
        << L" with seed " << seed << Logger::endl;
      }
      else
      {
        // Call the mutateModelTest
        //
        std::chrono::system_clock::time_point currTime{ std::chrono::system_clock::now() };
        
        std::chrono::minutes timeInMin{testTimeOut};
        std::chrono::seconds timeInSec{testTimeOut};
        std::chrono::hours timeInHrs{testTimeOut};
        std::chrono::system_clock::time_point endTime{currTime};
        endTime += scale == timeScale::Hrs ? timeInHrs 
                    : scale == timeScale::Min ? timeInMin : timeInSec;       
        Logger::testLog<< "Starting Test" << Logger::endl;
        size_t num_ort_exception{0};
        size_t num_std_exception{0};
        size_t num_unknown_exception{0};
        size_t num_successful_runs{0};
        size_t iteration{0};
        while(currTime < endTime)
        {
          try
          {
            onnx::ModelProto bad_model = model_proto;
            Logger::testLog<< "Starting Test iteration: " << iteration << Logger::endl;
            mutateModelTest(bad_model, mutateModelDirName, userOpt);
            num_successful_runs++;
            Logger::testLog<< "Completed Test iteration: " << iteration++ << Logger::endl;
          }
          catch(const Ort::Exception& ortException)
          {
            num_ort_exception++;
            Logger::testLog<< L"onnx runtime exception: " << ortException.what() << Logger::endl;
            Logger::testLog<< "Failed Test iteration: " << iteration++ << Logger::endl;
          }
          catch(const std::exception& e)
          {
            num_std_exception++;
            Logger::testLog<< L"standard exception: " << e.what() << Logger::endl;
            Logger::testLog<< "Failed Test iteration: " << iteration++ << Logger::endl;
          }
          catch(...)
          {
            num_unknown_exception++;
            Logger::testLog<< L"unknown exception: " << Logger::endl;
            Logger::testLog<< "Failed Test iteration: " << iteration++ << Logger::endl;
            throw;
          }
          
          // Update current time
          //
          currTime = std::chrono::system_clock::now();
        }
        Logger::testLog<< "Ending Test" << Logger::endl;
        
        if(userOpt.stress)
        {
          Logger::testLog.enable();
        }

        Logger::testLog<< L"Total number of exceptions: " << num_unknown_exception+num_std_exception+num_ort_exception << Logger::endl;
        Logger::testLog<< L"Number of Unknown exceptions: " << num_unknown_exception << Logger::endl;
        Logger::testLog<< L"Number of ort exceptions: " << num_ort_exception << Logger::endl;
        Logger::testLog<< L"Number of std exceptions: " << num_std_exception << Logger::endl;
        Logger::testLog << L"Number of unique errors: "<< werrStreamBuf.get_unique_errors() << L"\n";
        
        if(userOpt.stress)
        {
          Logger::testLog.disable();
          Logger::testLog.flush();
        }
      }
    }
    else
    {
      throw std::exception("Unable to initialize the Onnx model in memory");
    }

    return 0;
  }
  catch(const Ort::Exception& ortException)
  {
    Logger::testLog<< L"onnx runtime exception: " << ortException.what() << Logger::endl;
  }
  catch(const std::exception& e)
  {
    Logger::testLog<< L"standard exception: " << e.what() << Logger::endl;
  }
  catch(...)
  {
    Logger::testLog<< L"Something Went very wrong: " << Logger::endl;
  }

  return 1;
}
