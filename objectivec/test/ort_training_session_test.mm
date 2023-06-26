// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_APIS
#import <XCTest/XCTest.h>

#import "ort_checkpoint.h"
#import "ort_training_session.h"
#import "ort_env.h"
#import "ort_session.h"
#import "ort_value.h"

#import "test/test_utils.h"
#import "test/assertion_utils.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTTrainingSessionTest : XCTestCase
@property(readonly, nullable) ORTEnv* ortEnv;
@end

@implementation ORTTrainingSessionTest

- (void)setUp {
  [super setUp];

  self.continueAfterFailure = NO;

  NSError* err = nil;
  _ortEnv = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                           error:&err];
  ORTAssertNullableResultSuccessful(_ortEnv, err);
}

+ (NSString*)getFilePathFromName:(NSString*)name {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTTrainingSessionTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:name];
  return path;
}

+ (NSMutableData*)loadTensorDataFromFile:(NSString*)filePath skipHeader:(BOOL)skipHeader {
  NSError* error = nil;
  NSString* fileContents = [NSString stringWithContentsOfFile:filePath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
  ORTAssertNullableResultSuccessful(fileContents, error);

  NSArray<NSString*>* lines = [fileContents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];

  if (skipHeader) {
    lines = [lines subarrayWithRange:NSMakeRange(1, lines.count - 1)];
  }

  NSArray<NSString*>* dataArray = [lines[0] componentsSeparatedByCharactersInSet:
                                                [NSCharacterSet characterSetWithCharactersInString:@",[] "]];
  NSMutableData* tensorData = [NSMutableData data];

  for (NSString* str in dataArray) {
    if (str.length > 0) {
      float value = [str floatValue];
      [tensorData appendBytes:&value length:sizeof(float)];
    }
  }

  return tensorData;
}

- (ORTTrainingSession*)makeTrainingSessionWithCheckPoint:(ORTCheckpoint*)checkpoint {
  NSError* error = nil;
  ORTSessionOptions* sessionOptions = [[ORTSessionOptions alloc] initWithError:&error];
  ORTAssertNullableResultSuccessful(sessionOptions, error);

  ORTTrainingSession* session = [[ORTTrainingSession alloc]
             initWithEnv:self.ortEnv
          sessionOptions:sessionOptions
              checkPoint:checkpoint
          trainModelPath:[ORTTrainingSessionTest getFilePathFromName:@"training_model.onnx"]
           evalModelPath:[ORTTrainingSessionTest getFilePathFromName:@"eval_model.onnx"]
      optimizerModelPath:[ORTTrainingSessionTest getFilePathFromName:@"adamw.onnx"]
                   error:&error];

  ORTAssertNullableResultSuccessful(session, error);
  return session;
}

- (void)testInitTrainingSession {
  NSError* error = nil;

  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  // check that inputNames contains input-0
  NSArray<NSString*>* inputNames = [session getTrainInputNamesWithError:&error];
  ORTAssertNullableResultSuccessful(inputNames, error);

  XCTAssertTrue(inputNames.count > 0);
  XCTAssertTrue([inputNames containsObject:@"input-0"]);

  // check that outNames contains onnx::loss::21273
  NSArray<NSString*>* outputNames = [session getTrainOutputNamesWithError:&error];
  ORTAssertNullableResultSuccessful(outputNames, error);

  XCTAssertTrue(outputNames.count > 0);
  XCTAssertTrue([outputNames containsObject:@"onnx::loss::21273"]);
}

- (void)testInitTrainingSessionWithEval {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  // check that inputNames contains input-0
  NSArray<NSString*>* inputNames = [session getEvalInputNamesWithError:&error];
  ORTAssertNullableResultSuccessful(inputNames, error);

  XCTAssertTrue(inputNames.count > 0);
  XCTAssertTrue([inputNames containsObject:@"input-0"]);

  // check that outNames contains onnx::loss::21273
  NSArray<NSString*>* outputNames = [session getEvalOutputNamesWithError:&error];
  ORTAssertNullableResultSuccessful(outputNames, error);

  XCTAssertTrue(outputNames.count > 0);
  XCTAssertTrue([outputNames containsObject:@"onnx::loss::21273"]);
}

- (void)runTrainStepWithSession:(ORTTrainingSession*)session {
  // load input and expected output
  NSError* error = nil;
  NSMutableData* expectedOutput = [ORTTrainingSessionTest loadTensorDataFromFile:[ORTTrainingSessionTest
                                                                                     getFilePathFromName:@"loss_1.out"]
                                                                      skipHeader:YES];

  NSMutableData* input = [ORTTrainingSessionTest loadTensorDataFromFile:[ORTTrainingSessionTest
                                                                            getFilePathFromName:@"input-0.in"]
                                                             skipHeader:YES];

  int32_t labels[] = {1, 1};

  // create ORTValue array for input and labels
  NSMutableArray<ORTValue*>* inputValues = [NSMutableArray array];

  ORTValue* inputTensor = [[ORTValue alloc] initWithTensorData:input
                                                   elementType:ORTTensorElementDataTypeFloat
                                                         shape:@[ @2, @784 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(inputTensor, error);
  [inputValues addObject:inputTensor];

  ORTValue* labelTensor = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithBytes:labels
                                                                                     length:sizeof(labels)]
                                                   elementType:ORTTensorElementDataTypeInt32
                                                         shape:@[ @2 ]
                                                         error:&error];

  ORTAssertNullableResultSuccessful(labelTensor, error);
  [inputValues addObject:labelTensor];

  NSArray<ORTValue*>* outputs = [session trainStepWithInputValues:inputValues error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);
  XCTAssertTrue(outputs.count > 0);

  BOOL result = [session lazyResetGradWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);

  outputs = [session trainStepWithInputValues:inputValues error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);
  XCTAssertTrue(outputs.count > 0);

  ORTValue* outputValue = outputs[0];
  ORTValueTypeInfo* typeInfo = [outputValue typeInfoWithError:&error];
  ORTAssertNullableResultSuccessful(typeInfo, error);
  XCTAssertEqual(typeInfo.type, ORTValueTypeTensor);
  XCTAssertNotNil(typeInfo.tensorTypeAndShapeInfo);

  ORTTensorTypeAndShapeInfo* tensorInfo = [outputValue tensorTypeAndShapeInfoWithError:&error];
  ORTAssertNullableResultSuccessful(tensorInfo, error);
  XCTAssertEqual(tensorInfo.elementType, ORTTensorElementDataTypeFloat);

  NSMutableData* tensorData = [outputValue tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(tensorData, error);
  ORTAssertEqualFloatArrays(test_utils::getFloatArrayFromData(tensorData),
                            test_utils::getFloatArrayFromData(expectedOutput));
}
- (void)testTrainStepOutput {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];
  [self runTrainStepWithSession:session];
}

- (void)testOptimizerStep {
  // load input and expected output
  NSError* error = nil;
  NSMutableData* expectedOutput1 = [ORTTrainingSessionTest loadTensorDataFromFile:[ORTTrainingSessionTest
                                                                                      getFilePathFromName:@"loss_1.out"]
                                                                       skipHeader:YES];

  NSMutableData* expectedOutput2 = [ORTTrainingSessionTest loadTensorDataFromFile:[ORTTrainingSessionTest
                                                                                      getFilePathFromName:@"loss_2.out"]
                                                                       skipHeader:YES];

  NSMutableData* input = [ORTTrainingSessionTest loadTensorDataFromFile:[ORTTrainingSessionTest
                                                                            getFilePathFromName:@"input-0.in"]
                                                             skipHeader:YES];

  int32_t labels[] = {1, 1};

  // create ORTValue array for input and labels
  NSMutableArray<ORTValue*>* inputValues = [NSMutableArray array];

  ORTValue* inputTensor = [[ORTValue alloc] initWithTensorData:input
                                                   elementType:ORTTensorElementDataTypeFloat
                                                         shape:@[ @2, @784 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(inputTensor, error);
  [inputValues addObject:inputTensor];

  ORTValue* labelTensor = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithBytes:labels
                                                                                     length:sizeof(labels)]
                                                   elementType:ORTTensorElementDataTypeInt32
                                                         shape:@[ @2 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(labelTensor, error);
  [inputValues addObject:labelTensor];

  // create session
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  // run train step, optimizer steps and check loss
  NSArray<ORTValue*>* outputs = [session trainStepWithInputValues:inputValues error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);

  NSMutableData* loss = [outputs[0] tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(loss, error);
  ORTAssertEqualFloatArrays(test_utils::getFloatArrayFromData(loss),
                            test_utils::getFloatArrayFromData(expectedOutput1));

  BOOL result = [session lazyResetGradWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);

  outputs = [session trainStepWithInputValues:inputValues error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);

  loss = [outputs[0] tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(loss, error);
  ORTAssertEqualFloatArrays(test_utils::getFloatArrayFromData(loss),
                            test_utils::getFloatArrayFromData(expectedOutput1));

  result = [session optimizerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);

  outputs = [session trainStepWithInputValues:inputValues error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);
  loss = [outputs[0] tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(loss, error);
  ORTAssertEqualFloatArrays(test_utils::getFloatArrayFromData(loss),
                            test_utils::getFloatArrayFromData(expectedOutput2));
}

- (void)testSetLearningRate {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  float learningRate = 0.1f;
  BOOL result = [session setLearningRate:learningRate error:&error];
  ORTAssertBoolResultSuccessful(result, error);
  float actualLearningRate = [session getLearningRateWithError:&error];
  ORTAssertEqualFloatAndNoError(learningRate, actualLearningRate, error);
}

- (void)testLinearLRScheduler {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  float learningRate = 0.1f;
  BOOL result = [session registerLinearLRSchedulerWithWarmupStepCount:2
                                                       totalStepCount:4
                                                            initialLr:learningRate
                                                                error:&error];

  ORTAssertBoolResultSuccessful(result, error);

  result = [session optimizerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertEqualFloatAndNoError(0.05f, [session getLearningRateWithError:&error], error);

  result = [session optimizerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertEqualFloatAndNoError(0.1f, [session getLearningRateWithError:&error], error);

  result = [session optimizerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertEqualFloatAndNoError(0.05f, [session getLearningRateWithError:&error], error);

  result = [session optimizerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertEqualFloatAndNoError(0.0f, [session getLearningRateWithError:&error], error);
}

- (void)testExportModelForInference {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  NSString* inferenceModelPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"inference_model.onnx"];
  XCTAssertNotNil(inferenceModelPath);

  NSArray<NSString*>* graphOutputNames = [NSArray arrayWithObjects:@"output-0", nil];

  BOOL result = [session exportModelForInferenceWithOutputPath:inferenceModelPath
                                              graphOutputNames:graphOutputNames
                                                         error:&error];

  ORTAssertBoolResultSuccessful(result, error);
  XCTAssertTrue([[NSFileManager defaultManager] fileExistsAtPath:inferenceModelPath]);

  [self addTeardownBlock:^{
    NSError* error = nil;
    [[NSFileManager defaultManager] removeItemAtPath:inferenceModelPath error:&error];
  }];
}

- (void)testToBuffer {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  ORTValue* buffer = [session toBufferWithTrainable:YES error:&error];
  ORTAssertNullableResultSuccessful(buffer, error);

  ORTValueTypeInfo* typeInfo = [buffer typeInfoWithError:&error];
  ORTAssertNullableResultSuccessful(typeInfo, error);
  XCTAssertEqual(typeInfo.type, ORTValueTypeTensor);
  XCTAssertNotNil(typeInfo.tensorTypeAndShapeInfo);
}

- (void)testFromBuffer {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest
                                                                      getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  ORTValue* buffer = [session toBufferWithTrainable:YES error:&error];
  ORTAssertNullableResultSuccessful(buffer, error);

  BOOL result = [session fromBufferWithValue:buffer error:&error];
  ORTAssertBoolResultSuccessful(result, error);
}

- (void)testSetSeed {
  ORTSetSeed(2718);
}

- (void)tearDown {
  _ortEnv = nil;

  [super tearDown];
}

@end

NS_ASSUME_NONNULL_END

#endif  // ENABLE_TRAINING_APIS
