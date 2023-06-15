// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_APIS
#import <XCTest/XCTest.h>

#import "ort_checkpoint.h"
#import "ort_training_session.h"
#import "ort_env.h"
#import "ort_session.h"
#import "ort_value.h"

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

+ (NSString*)createTempFileWithName:(NSString*)name extension:(NSString*)extension {
  NSString* tempDir = NSTemporaryDirectory();
  NSString* tempFile = [tempDir stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.%@", name, extension]];

  NSFileManager* fileManager = [NSFileManager defaultManager];
  if ([fileManager createFileAtPath:tempFile contents:nil attributes:nil]) {
    return tempFile;
  } else {
    return nil;
  }
}

+ (NSMutableData*)loadTensorFromFile:(NSString*)filePath skipHeader:(BOOL)skipHeader {
  NSError* error = nil;
  NSString* fileContents = [NSString stringWithContentsOfFile:filePath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
  ORTAssertNullableResultSuccessful(fileContents, error);

  NSArray<NSString*>* lines = [fileContents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];

  if (skipHeader) {
    lines = [lines subarrayWithRange:NSMakeRange(1, lines.count - 1)];
  }

  NSArray<NSString*>* dataArray = [lines[0] componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@",[] "]];
  NSMutableData* tensorData = [NSMutableData data];

  for (NSString* str in dataArray) {
    if (str.length > 0) {
      float value = [str floatValue];
      [tensorData appendBytes:&value length:sizeof(float)];
    }
  }

  return tensorData;
}

+ (float)getFirstValueFromData:(NSData*)data {
  float value;
  [data getBytes:&value length:sizeof(float)];
  return value;
}

- (ORTTrainingSession*)makeTrainingSessionWithCheckPoint:(ORTCheckpoint*)checkpoint {
  NSError* error = nil;
  ORTSessionOptions* sessionOptions = [[ORTSessionOptions alloc] initWithError:&error];
  ORTAssertNullableResultSuccessful(sessionOptions, error);

  ORTTrainingSession* session = [[ORTTrainingSession alloc] initWithEnv:self.ortEnv
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

  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"]
                                                            error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  // check that inputNames contains input-0
  NSArray<NSString*>* inputNames = [session inputNamesWithTraining:YES error:&error];
  ORTAssertNullableResultSuccessful(inputNames, error);

  XCTAssertTrue(inputNames.count > 0);
  XCTAssertTrue([inputNames containsObject:@"input-0"]);

  // check that outNames contains onnx::loss::21273
  NSArray<NSString*>* outputNames = [session outputNamesWithTraining:YES error:&error];
  ORTAssertNullableResultSuccessful(outputNames, error);

  XCTAssertTrue(outputNames.count > 0);
  XCTAssertTrue([outputNames containsObject:@"onnx::loss::21273"]);
}

- (void)testInintTrainingSessionWithEval {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  // check that inputNames contains input-0
  NSArray<NSString*>* inputNames = [session inputNamesWithTraining:NO error:&error];
  ORTAssertNullableResultSuccessful(inputNames, error);

  XCTAssertTrue(inputNames.count > 0);
  XCTAssertTrue([inputNames containsObject:@"input-0"]);

  // check that outNames contains onnx::loss::21273
  NSArray<NSString*>* outputNames = [session outputNamesWithTraining:NO error:&error];
  ORTAssertNullableResultSuccessful(outputNames, error);

  XCTAssertTrue(outputNames.count > 0);
  XCTAssertTrue([outputNames containsObject:@"onnx::loss::21273"]);
}

//-(void)testTrainStep {
//  NSError* error = nil;
//  ORTTrainingSession* session = [self makeTrainingSession];
//
//  BOOL result = [session trainStepWithError:&error];
//  ORTAssertBoolResultSuccessful(result, error);
//}

- (void)runTrainStepWithSession:(ORTTrainingSession*)session {
  // load input and expected output
  NSError* error = nil;
  NSMutableData* expectedOutput = [ORTTrainingSessionTest loadTensorFromFile:[ORTTrainingSessionTest getFilePathFromName:@"loss_1.out"] skipHeader:YES];

  NSMutableData* input = [ORTTrainingSessionTest loadTensorFromFile:[ORTTrainingSessionTest getFilePathFromName:@"input-0.in"] skipHeader:YES];

  int32_t lables[] = {1, 1};

  // create ORTValue array for input and labels
  NSMutableArray<ORTValue*>* pinnedInputs = [NSMutableArray array];

  ORTValue* inputTensor = [[ORTValue alloc] initWithTensorData:input
                                                   elementType:ORTTensorElementDataTypeFloat
                                                         shape:@[ @2, @784 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(inputTensor, error);
  [pinnedInputs addObject:inputTensor];

  ORTValue* labelTensor = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithBytes:lables length:sizeof(lables)]
                                                   elementType:ORTTensorElementDataTypeInt32
                                                         shape:@[ @2 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(labelTensor, error);
  [pinnedInputs addObject:labelTensor];

  NSArray<ORTValue*>* outputs = [session trainStepWithInputValues:pinnedInputs error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);
  XCTAssertTrue(outputs.count > 0);

  BOOL result = [session lazyResetGradWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);

  outputs = [session trainStepWithInputValues:pinnedInputs error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);
  XCTAssertTrue(outputs.count > 0);

  ORTValue* outputBuffer = outputs[0];
  ORTValueTypeInfo* typeInfo = [outputBuffer typeInfoWithError:&error];
  ORTAssertNullableResultSuccessful(typeInfo, error);
  XCTAssertEqual(typeInfo.type, ORTValueTypeTensor);
  XCTAssertNotNil(typeInfo.tensorTypeAndShapeInfo);

  ORTTensorTypeAndShapeInfo* tensorInfo = [outputBuffer tensorTypeAndShapeInfoWithError:&error];
  ORTAssertNullableResultSuccessful(tensorInfo, error);
  XCTAssertEqual(tensorInfo.elementType, ORTTensorElementDataTypeFloat);

  NSMutableData* tensorData = [outputBuffer tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(tensorData, error);
  XCTAssertEqualWithAccuracy([ORTTrainingSessionTest getFirstValueFromData:tensorData], [ORTTrainingSessionTest getFirstValueFromData:expectedOutput], 1e-3f);
}
- (void)testTrainStepOutput {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];
  [self runTrainStepWithSession:session];
}

- (void)testOptimizerStep {
  // load input and expected output
  NSError* error = nil;
  NSMutableData* expectedOutput1 = [ORTTrainingSessionTest loadTensorFromFile:[ORTTrainingSessionTest getFilePathFromName:@"loss_1.out"] skipHeader:YES];

  NSMutableData* expectedOutput2 = [ORTTrainingSessionTest loadTensorFromFile:[ORTTrainingSessionTest getFilePathFromName:@"loss_2.out"] skipHeader:YES];

  NSMutableData* input = [ORTTrainingSessionTest loadTensorFromFile:[ORTTrainingSessionTest getFilePathFromName:@"input-0.in"] skipHeader:YES];

  int32_t lables[] = {1, 1};

  // create ORTValue array for input and labels
  NSMutableArray<ORTValue*>* pinnedInputs = [NSMutableArray array];

  ORTValue* inputTensor = [[ORTValue alloc] initWithTensorData:input
                                                   elementType:ORTTensorElementDataTypeFloat
                                                         shape:@[ @2, @784 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(inputTensor, error);
  [pinnedInputs addObject:inputTensor];

  ORTValue* labelTensor = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithBytes:lables length:sizeof(lables)]
                                                   elementType:ORTTensorElementDataTypeInt32
                                                         shape:@[ @2 ]
                                                         error:&error];
  ORTAssertNullableResultSuccessful(labelTensor, error);
  [pinnedInputs addObject:labelTensor];

  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  NSArray<ORTValue*>* outputs = [session trainStepWithInputValues:pinnedInputs error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);

  NSMutableData* loss = [outputs[0] tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(loss, error);
  XCTAssertEqualWithAccuracy([ORTTrainingSessionTest getFirstValueFromData:loss], [ORTTrainingSessionTest getFirstValueFromData:expectedOutput1], 1e-3f);

  BOOL result = [session lazyResetGradWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);

  outputs = [session trainStepWithInputValues:pinnedInputs error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);

  loss = [outputs[0] tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(loss, error);
  XCTAssertEqualWithAccuracy([ORTTrainingSessionTest getFirstValueFromData:loss], [ORTTrainingSessionTest getFirstValueFromData:expectedOutput1], 1e-3f);

  result = [session optimzerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);

  outputs = [session trainStepWithInputValues:pinnedInputs error:&error];
  ORTAssertNullableResultSuccessful(outputs, error);
  loss = [outputs[0] tensorDataWithError:&error];
  ORTAssertNullableResultSuccessful(loss, error);
  XCTAssertEqualWithAccuracy([ORTTrainingSessionTest getFirstValueFromData:loss], [ORTTrainingSessionTest getFirstValueFromData:expectedOutput2], 1e-3f);
}

- (void)testSetLearningRate {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  float learningRate = 0.1f;
  BOOL result = [session setLearningRate:learningRate error:&error];
  ORTAssertBoolResultSuccessful(result, error);
  float actualLearningRate = [session getLearningRateWithError:&error];
  ORTAssertFloatResultSuccessful(learningRate, actualLearningRate, error);
}

- (void)testLinearLRScheduler {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  float learningRate = 0.1f;
  BOOL result = [session registerLinearLRSchedulerWithWarmupStepCount:2
                                                       totalStepCount:4
                                                            initialLr:learningRate
                                                                error:&error];

  ORTAssertBoolResultSuccessful(result, error);

  result = [session optimzerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertFloatResultSuccessful(0.05f, [session getLearningRateWithError:&error], error);

  result = [session optimzerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertFloatResultSuccessful(0.1f, [session getLearningRateWithError:&error], error);

  result = [session optimzerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  XCTAssertNil(error);
  ORTAssertFloatResultSuccessful(0.05f, [session getLearningRateWithError:&error], error);

  result = [session optimzerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  result = [session schedulerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
  ORTAssertFloatResultSuccessful(0.0f, [session getLearningRateWithError:&error], error);
}

- (void)testExportModelForInference {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  NSString* infernceModelPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"inference_model.onnx"];
  XCTAssertNotNil(infernceModelPath);

  // [ORTTrainingSessionTest createTempFileWithName:@"inference_model" extension:@"onnx"];
  NSArray<NSString*>* graphOutputNames = [NSArray arrayWithObjects:@"output-0", nil];

  BOOL result = [session exportModelForInferenceWithOutputPath:infernceModelPath
                                              graphOutputNames:graphOutputNames
                                                         error:&error];

  ORTAssertBoolResultSuccessful(result, error);
  XCTAssertTrue([[NSFileManager defaultManager] fileExistsAtPath:infernceModelPath]);
}

- (void)testToBuffer {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
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
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getFilePathFromName:@"checkpoint.ckpt"] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);
  ORTTrainingSession* session = [self makeTrainingSessionWithCheckPoint:checkpoint];

  ORTValue* buffer = [session toBufferWithTrainable:YES error:&error];
  ORTAssertNullableResultSuccessful(buffer, error);

  BOOL result = [session fromBufferWithValue:buffer error:&error];
  ORTAssertBoolResultSuccessful(result, error);
}

-(void)testSetSeed {
  ORTSetSeed(2718);
}

- (void)tearDown {
  _ortEnv = nil;

  [super tearDown];
}

@end

NS_ASSUME_NONNULL_END

#endif  // ENABLE_TRAINING_APIS
