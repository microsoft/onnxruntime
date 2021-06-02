// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "host_utils.h"
#include "model.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// Model input for a CoreML model
// All the input onnx tensors values will be converted to MLMultiArray(s)
@interface OnnxTensorFeatureProvider : NSObject <MLFeatureProvider> {
  const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>* inputs_;
  NSSet* featureNames_;
  const onnxruntime::logging::Logger* logger_;
}

- (instancetype)initWithInputs:(const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>&)inputs
                        logger:(const onnxruntime::logging::Logger*)logger;
- (MLFeatureValue*)featureValueForName:(NSString*)featureName API_AVAILABLE_OS_VERSIONS;
- (NSSet<NSString*>*)featureNames;

@end

// Execution for a CoreML model, it performs
// 1. Compile the model by given path for execution
// 2. Predict using given OnnxTensorFeatureProvider input and copy the output data back ORT
// 3. The compiled model will be removed in dealloc or removed using cleanup function
@interface CoreMLExecution : NSObject {
  NSString* coreml_model_path_;
  NSString* compiled_model_path_;
  const onnxruntime::logging::Logger* logger_;
  uint32_t coreml_flags_;
}

- (instancetype)initWithPath:(const std::string&)path
                      logger:(const onnxruntime::logging::Logger&)logger
                coreml_flags:(uint32_t)coreml_flags;
- (void)cleanup;
- (void)dealloc;
- (onnxruntime::common::Status)loadModel API_AVAILABLE_OS_VERSIONS;
- (onnxruntime::common::Status)
    predict:(const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>&)inputs
    outputs:(const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>&)outputs
    API_AVAILABLE_OS_VERSIONS;

@property MLModel* model API_AVAILABLE_OS_VERSIONS;

@end

@implementation OnnxTensorFeatureProvider

- (instancetype)initWithInputs:(const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>&)inputs
                        logger:(const onnxruntime::logging::Logger*)logger {
  if (self = [super init]) {
    inputs_ = &inputs;
    logger_ = logger;
  }
  return self;
}

- (nonnull NSSet<NSString*>*)featureNames {
  if (featureNames_ == nil) {
    NSMutableArray* names = [[NSMutableArray alloc] init];
    for (const auto& input : *inputs_) {
      [names addObject:[NSString stringWithCString:input.first.c_str()
                                          encoding:[NSString defaultCStringEncoding]]];
    }

    featureNames_ = [NSSet setWithArray:names];
  }

  return featureNames_;
}

- (nullable MLFeatureValue*)featureValueForName:(nonnull NSString*)featureName {
  auto it = inputs_->find([featureName cStringUsingEncoding:NSUTF8StringEncoding]);
  if (it != inputs_->end()) {
    auto& input = it->second;
    NSMutableArray* shape = [[NSMutableArray alloc] init];
    for (const auto dim : input.tensor_info.shape) {
      [shape addObject:[NSNumber numberWithLongLong:dim]];
    }

    NSMutableArray* strides = [[NSMutableArray alloc] init];
    int64_t stride = 1;
    for (int i = static_cast<int>(input.tensor_info.shape.size()) - 1; i >= 0; i--) {
      [strides insertObject:[NSNumber numberWithLongLong:stride]
                    atIndex:0];

      stride *= input.tensor_info.shape[i];
    }

    MLMultiArrayDataType data_type = MLMultiArrayDataTypeFloat32;
    if (input.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      LOGS(*logger_, ERROR) << "Input data type is not float, actual type: "
                            << input.tensor_info.data_type;
      return nil;
    }

    NSError* error = nil;
    MLMultiArray* mlArray = [[MLMultiArray alloc] initWithDataPointer:input.buffer
                                                                shape:shape
                                                             dataType:data_type
                                                              strides:strides
                                                          deallocator:(^(void* /* bytes */){
                                                                      })error:&error];
    if (error != nil) {
      LOGS(*logger_, ERROR) << "Failed to create MLMultiArray for feature: " << [featureName UTF8String]
                            << ", error: " << [[error localizedDescription] UTF8String];
      return nil;
    }

    auto* mlFeatureValue = [MLFeatureValue featureValueWithMultiArray:mlArray];
    return mlFeatureValue;
  }

  return nil;
}

@end

@implementation CoreMLExecution

- (instancetype)initWithPath:(const std::string&)path
                      logger:(const onnxruntime::logging::Logger&)logger
                coreml_flags:(uint32_t)coreml_flags {
  if (self = [super init]) {
    coreml_model_path_ = [NSString stringWithUTF8String:path.c_str()];
    logger_ = &logger;
    coreml_flags_ = coreml_flags;
  }
  return self;
}

- (void)cleanup {
  NSError* error = nil;
  if (compiled_model_path_ != nil) {
    [[NSFileManager defaultManager] removeItemAtPath:compiled_model_path_ error:&error];
    if (error != nil) {
      LOGS(*logger_, ERROR) << "Failed cleaning up the compiled model: " << [compiled_model_path_ UTF8String]
                            << ", error message: " << [[error localizedDescription] UTF8String];
    }
    compiled_model_path_ = nil;
  }

  if (coreml_model_path_ != nil) {
    error = nil;
    [[NSFileManager defaultManager] removeItemAtPath:coreml_model_path_ error:&error];
    if (error != nil) {
      LOGS(*logger_, ERROR) << "Failed cleaning up the coreml model: " << [coreml_model_path_ UTF8String]
                            << ", error message: " << [[error localizedDescription] UTF8String];
    }
    coreml_model_path_ = nil;
  }
}

- (void)dealloc {
  [self cleanup];
}

- (onnxruntime::common::Status)loadModel {
  NSError* error = nil;
  NSURL* modelUrl = [NSURL URLWithString:coreml_model_path_];
  NSURL* compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];

  if (error != nil) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error compiling model ",
                           [[error localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding]);
  }

  compiled_model_path_ = [compileUrl path];

  MLModelConfiguration* config = [MLModelConfiguration alloc];
  config.computeUnits = MLComputeUnitsAll;
  _model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:&error];

  if (error != NULL) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error Creating MLModel ",
                           [[error localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding]);
  }

  return onnxruntime::common::Status::OK();
}

- (onnxruntime::common::Status)
    predict:(const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>&)inputs
    outputs:(const std::unordered_map<std::string, onnxruntime::coreml::OnnxTensorData>&)outputs {
  if (_model == nil) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model is not loaded");
  }

  OnnxTensorFeatureProvider* input_feature = [[OnnxTensorFeatureProvider alloc] initWithInputs:inputs
                                                                                        logger:logger_];

  if (input_feature == nil) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "inputFeature is not initialized");
  }

  MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
  options.usesCPUOnly = coreml_flags_ & COREML_FLAG_USE_CPU_ONLY;
  NSError* error = nil;
  id<MLFeatureProvider> output_feature = [_model predictionFromFeatures:input_feature
                                                                options:options
                                                                  error:&error];

  if (error != nil) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error executing model: ",
                           [[error localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding]);
  }

  for (auto& output : outputs) {
    NSString* output_name = [NSString stringWithCString:output.first.c_str()
                                               encoding:[NSString defaultCStringEncoding]];
    MLFeatureValue* output_value =
        [output_feature featureValueForName:output_name];

    if (output_value == nil) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "output_feature has no value for ",
                             [output_name cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    auto* data = [output_value multiArrayValue];
    auto* model_output_data = data.dataPointer;
    if (model_output_data == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model_output_data has no data for ",
                             [output_name cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    auto& output_tensor = output.second;
    size_t num_elements =
        accumulate(output_tensor.tensor_info.shape.begin(),
                   output_tensor.tensor_info.shape.end(),
                   1,
                   std::multiplies<int64_t>());

    size_t output_data_byte_size = 0;
    auto type = output_tensor.tensor_info.data_type;
    switch ( type ) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        output_data_byte_size = num_elements * sizeof(float);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        output_data_byte_size = num_elements * sizeof(int32_t);
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Output data type is not float/int32, actual type: ",
                             output_tensor.tensor_info.data_type);
    }
    memcpy(output_tensor.buffer, model_output_data, output_data_byte_size);
  }

  return onnxruntime::common::Status::OK();
}

@end

namespace onnxruntime {
namespace coreml {

// Internal Execution class
// This class will bridge Model (c++) with CoreMLExecution (objective c++)
class Execution {
 public:
  Execution(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags);
  ~Execution(){};

  Status LoadModel();
  Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                 const std::unordered_map<std::string, OnnxTensorData>& outputs);

 private:
  bool model_loaded{false};
  CoreMLExecution* execution_;
};

Execution::Execution(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags) {
  execution_ = [[CoreMLExecution alloc] initWithPath:path
                                              logger:logger
                                        coreml_flags:coreml_flags];
}

Status Execution::LoadModel() {
  if (model_loaded) {
    return Status::OK();
  }

  if (HAS_VALID_BASE_OS_VERSION) {
    auto status = [execution_ loadModel];
    model_loaded = status.IsOK();
    return status;
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Execution::LoadModel requires macos 10.15+ or ios 13+ ");
}

Status Execution::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                          const std::unordered_map<std::string, OnnxTensorData>& outputs) {
  ORT_RETURN_IF_NOT(model_loaded, "Execution::Predict requires Execution::LoadModel");

  if (HAS_VALID_BASE_OS_VERSION) {
    return [execution_ predict:inputs outputs:outputs];
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Execution::LoadModel requires macos 10.15+ or ios 13+ ");
}

Model::Model(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags)
    : execution_(std::make_unique<Execution>(path, logger, coreml_flags)) {
}

Model::~Model() {}

Status Model::LoadModel() {
  return execution_->LoadModel();
}

Status Model::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                      const std::unordered_map<std::string, OnnxTensorData>& outputs) {
  return execution_->Predict(inputs, outputs);
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

const OnnxTensorInfo& Model::GetInputOutputInfo(const std::string& name) const {
  return input_output_info_.at(name);
}

}  // namespace coreml
}  // namespace onnxruntime