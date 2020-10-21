// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ViewController.h"

#import "OrtSession.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Safely dispatches the given `block` on the main thread. If already on the main thread, the given
 * block is executed immediately; otherwise, dispatches the block asynchronously on the main thread.
 *
 * @param block The block to dispatch on the main thread.
 */
void TLTSafeDispatchOnMain(dispatch_block_t block) {
  if (block == nil) return;
  if (NSThread.isMainThread) {
    block();
  } else {
    dispatch_async(dispatch_get_main_queue(), block);
  }
}

static NSString *const kModelNameMobileNet = @"mobilenetv2-7";

static NSString *const kModelNameQuantized = @"mobilenetv2-7";

static NSString *const kModelNameNLP = @"nlp";

/** Model resource type. */
static NSString *const kModelType = @"ort";

/** The label for the serial queue for synchronizing runtime calls. */
static const char *kRuntimeSerialQueueLabel = "com.onnxruntime.testapp";

static NSString *const kNilRuntimeError =
    @"Failed to invoke the runtime because the runtime was nil.";
static NSString *const kInvokeRuntimeError = @"Failed to invoke ONNX Runtime due to error: %@.";

/** Model paths. */
static NSArray *arrModelPaths;

@interface ViewController ()

/** Serial queue for synchronizing runtime calls. */
@property(nonatomic) dispatch_queue_t runtimeSerialQueue;

/** ONNXRuntime for the currently selected model. */
@property(nonatomic) OrtMobileSession *runtime;

@property(weak, nonatomic) IBOutlet UISegmentedControl *modelControl;
@property(weak, nonatomic) IBOutlet UIBarButtonItem *invokeButton;
@property(weak, nonatomic) IBOutlet UITextView *resultsTextView;

@end

@implementation ViewController

#pragma mark - NSObject

+ (void)initialize {
  if (self == [ViewController self]) {
    arrModelPaths = @[
      [NSBundle.mainBundle pathForResource:kModelNameMobileNet ofType:kModelType],
      [NSBundle.mainBundle pathForResource:kModelNameQuantized ofType:kModelType],
      [NSBundle.mainBundle pathForResource:kModelNameNLP ofType:kModelType],
    ];
  }
}

#pragma mark - UIViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.runtimeSerialQueue =
      dispatch_queue_create(kRuntimeSerialQueueLabel, DISPATCH_QUEUE_SERIAL);
  self.invokeButton.enabled = NO;
  [self updateResultsText:[NSString stringWithFormat:@"Using ONNXRuntime runtime version %@.", @"1.5.0"]];
  [self loadModel];
}

#pragma mark - IBActions

- (IBAction)modelChanged:(id)sender {
  self.invokeButton.enabled = NO;
  NSString *results = [NSString
      stringWithFormat:@"Switched to the %@ model.",
                       [self.modelControl
                           titleForSegmentAtIndex:self.modelControl.selectedSegmentIndex]];
  [self updateResultsText:results];
  [self loadModel];
    
}

- (IBAction)invokeRuntime:(id)sender {
  switch (self.modelControl.selectedSegmentIndex) {
    case 0:
      [self invokeMobileNet];
      break;
    case 1:
      [self invokeQuantized];
      break;
    case 2:
      [self invokeNLP];
  }
}

#pragma mark - Private

/** Path of the currently selected model. */
- (nullable NSString *)currentModelPath {
  return self.modelControl.selectedSegmentIndex == UISegmentedControlNoSegment
             ? nil
             : arrModelPaths[self.modelControl.selectedSegmentIndex];
}

- (void)loadModel {
  NSString *modelPath = [self currentModelPath];
  if (modelPath.length == 0) {
    [self updateResultsText:@"No model is selected."];
    return;
  }

  __weak typeof(self) weakSelf = self;
  dispatch_async(self.runtimeSerialQueue, ^{
    NSError *error;
    weakSelf.runtime = [[OrtMobileSession alloc] initWithModelPath:modelPath
                                                               error:&error];
    if (weakSelf.runtime == nil || error != nil) {
      NSString *results =
          [NSString stringWithFormat:@"Failed to create the runtime due to error:%@",
                                     error.localizedDescription];
      [weakSelf updateResultsText:results];
    } else {
      TLTSafeDispatchOnMain(^{
        weakSelf.invokeButton.enabled = YES;
      });
    }
  });
}

- (void)invokeMobileNet {
  __weak typeof(self) weakSelf = self;
  dispatch_async(self.runtimeSerialQueue, ^{
    if (weakSelf.runtime == nil) {
      [weakSelf updateResultsText:kNilRuntimeError];
      return;
    }

    NSError* error;
    NSMutableData* data = [NSMutableData alloc];
      NSString *resultMsg = [weakSelf.runtime run:data mname:@"mobilenet" error:&error];
    [weakSelf updateResultsText:resultMsg];

  });
}

- (void)invokeQuantized {
  __weak typeof(self) weakSelf = self;
  dispatch_async(self.runtimeSerialQueue, ^{
    if (weakSelf.runtime == nil) {
      [weakSelf updateResultsText:kNilRuntimeError];
      return;
    }
  });
}

- (void)invokeNLP {
  __weak typeof(self) weakSelf = self;
  dispatch_async(self.runtimeSerialQueue, ^{
    if (weakSelf.runtime == nil) {
      [weakSelf updateResultsText:kNilRuntimeError];
      return;
    }
    NSError* error;
    NSMutableData* data = [NSMutableData alloc];
      NSString *resultMsg = [weakSelf.runtime run:data mname:@"nlp" error:&error];
    [weakSelf updateResultsText:resultMsg];
  });
}

- (void)updateResultsText:(NSString *)text {
  __weak typeof(self) weakSelf = self;
  TLTSafeDispatchOnMain(^{
    weakSelf.resultsTextView.text = text;
  });
}

@end

NS_ASSUME_NONNULL_END
