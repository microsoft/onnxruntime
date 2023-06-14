// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_APIS
#import <XCTest/XCTest.h>

#import "ort_checkpoint.h"
#import "ort_training_session.h"
#import "ort_env.h"
#import "ort_session.h"

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

+ (NSString*)getCheckpointPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTTrainingSessionTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"checkpoint.ckpt"];
  return path;
}

+ (NSString*)getTrainingModelPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTTrainingSessionTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"training_model.onnx"];
  return path;
}

+ (NSString*)getEvalModelPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTTrainingSessionTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"eval_model.onnx"];
  return path;
}

+ (NSString*)getOptimizerModelPath {
  NSBundle* bundle = [NSBundle bundleForClass:[ORTTrainingSessionTest class]];
  NSString* path = [[bundle resourcePath] stringByAppendingPathComponent:@"adamw.onnx"];
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

- (ORTTrainingSession*)makeTrainingSession {
  NSError* error = nil;
  ORTCheckpoint* checkpoint = [[ORTCheckpoint alloc] initWithPath:[ORTTrainingSessionTest getCheckpointPath] error:&error];
  ORTAssertNullableResultSuccessful(checkpoint, error);

  ORTSessionOptions* sessionOptions = [[ORTSessionOptions alloc] initWithError:&error];
  ORTAssertNullableResultSuccessful(sessionOptions, error);

  ORTTrainingSession* session = [[ORTTrainingSession alloc] initWithEnv:self.ortEnv
                                                         sessionOptions:sessionOptions
                                                             checkPoint:checkpoint
                                                         trainModelPath:[ORTTrainingSessionTest getTrainingModelPath]
                                                          evalModelPath:[ORTTrainingSessionTest getEvalModelPath]
                                                     optimizerModelPath:[ORTTrainingSessionTest getOptimizerModelPath]
                                                                  error:&error];

  ORTAssertNullableResultSuccessful(session, error);
  return session;
}

- (void)testInitTrainingSession {
  NSError* error = nil;
  ORTTrainingSession* session = [self makeTrainingSession];

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
  ORTTrainingSession* session = [self makeTrainingSession];

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

-(void)testTrainStep {
  NSError* error = nil;
  ORTTrainingSession* session = [self makeTrainingSession];

  BOOL result = [session trainStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
}

- (void)testOptimizerStep {
  NSError* error = nil;
  ORTTrainingSession* session = [self makeTrainingSession];

  BOOL result = [session optimzerStepWithError:&error];
  ORTAssertBoolResultSuccessful(result, error);
}

- (void)testSetLearningRate {
  NSError* error = nil;
  ORTTrainingSession* session = [self makeTrainingSession];

  float learningRate = 0.1f;
  BOOL result = [session setLearningRate:learningRate error:&error];
  ORTAssertBoolResultSuccessful(result, error);
  float actualLearningRate = [session getLearningRateWithError:&error];
  ORTAssertFloatResultSuccessful(learningRate, actualLearningRate, error);
}

- (void)testLinearLRScheduler {
  NSError* error = nil;
  ORTTrainingSession* session = [self makeTrainingSession];

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
  ORTTrainingSession* session = [self makeTrainingSession];

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

- (void)tearDown {
  _ortEnv = nil;

  [super tearDown];
}

@end

NS_ASSUME_NONNULL_END

#endif  // ENABLE_TRAINING_APIS
