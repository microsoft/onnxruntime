// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "MNISTDataHandler.h"
#import "OnnxruntimeModule.h"
#import "TensorHelper.h"
#import <Foundation/Foundation.h>
#import <React/RCTLog.h>

NS_ASSUME_NONNULL_BEGIN

@implementation MNISTDataHandler

RCT_EXPORT_MODULE(MNISTDataHandler)

// It returns mode path in local device,
// so that onnxruntime is able to load a model using a given path.
RCT_EXPORT_METHOD(getLocalModelPath : (RCTPromiseResolveBlock)resolve rejecter : (RCTPromiseRejectBlock)reject) {
  @try {
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"mnist" ofType:@"ort"];
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:modelPath]) {
      resolve(modelPath);
    } else {
      reject(@"mnist", @"no such a model", nil);
    }
  } @catch (NSException *exception) {
    reject(@"mnist", @"no such a model", nil);
  }
}

// It returns image path.
RCT_EXPORT_METHOD(getImagePath : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
  @try {
    NSString *imagePath = [[NSBundle mainBundle] pathForResource:@"3" ofType:@"jpg"];
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:imagePath]) {
      resolve(imagePath);
    } else {
      reject(@"mnist", @"no such an image", nil);
    }
  } @catch (NSException *exception) {
    reject(@"mnist", @"no such an image", nil);
  }
}

// It gets raw input data, which can be uri or byte array and others,
// returns cooked data formatted as input of a model.
RCT_EXPORT_METHOD(preprocess
                  : (NSString *)uri resolve
                  : (RCTPromiseResolveBlock)resolve reject
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary *inputDataMap = [self preprocess:uri];
    resolve(inputDataMap);
  } @catch (NSException *exception) {
    reject(@"mnist", @"can't load an image", nil);
  }
}

// It gets a result from onnxruntime and a duration of session time for input data,
// returns output data formatted as React Native map.
RCT_EXPORT_METHOD(postprocess
                  : (NSDictionary *)result resolve
                  : (RCTPromiseResolveBlock)resolve reject
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary *cookedMap = [self postprocess:result];
    resolve(cookedMap);
  } @catch (NSException *exception) {
    reject(@"mnist", @"can't pose-process an image", nil);
  }
}

- (NSDictionary *)preprocess:(NSString *)uri {
  UIImage *image = [UIImage imageNamed:@"3.jpg"];

  CGSize scale = CGSizeMake(28, 28);
  UIGraphicsBeginImageContextWithOptions(scale, NO, 1.0);
  [image drawInRect:CGRectMake(0, 0, scale.width, scale.height)];
  UIImage *scaledImage = UIGraphicsGetImageFromCurrentImageContext();
  UIGraphicsEndImageContext();

  CGImageRef imageRef = [scaledImage CGImage];
  NSUInteger width = CGImageGetWidth(imageRef);
  NSUInteger height = CGImageGetHeight(imageRef);
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

  const NSUInteger rawDataSize = height * width * 4;
  std::vector<unsigned char> rawData(rawDataSize);
  NSUInteger bytesPerPixel = 4;
  NSUInteger bytesPerRow = bytesPerPixel * width;
  CGContextRef context = CGBitmapContextCreate(rawData.data(), width, height, 8, bytesPerRow, colorSpace,
                                               kCGImageAlphaPremultipliedLast | kCGImageByteOrder32Big);
  CGColorSpaceRelease(colorSpace);
  CGContextSetBlendMode(context, kCGBlendModeCopy);
  CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
  CGContextRelease(context);

  const NSInteger dimSize = height * width;
  const NSInteger byteBufferSize = dimSize * sizeof(float);

  unsigned char *byteBuffer = static_cast<unsigned char *>(malloc(byteBufferSize));
  NSData *byteBufferRef = [NSData dataWithBytesNoCopy:byteBuffer length:byteBufferSize];
  float *floatPtr = (float *)[byteBufferRef bytes];
  for (NSUInteger h = 0; h < height; ++h) {
    for (NSUInteger w = 0; w < width; ++w) {
      NSUInteger byteIndex = (bytesPerRow * h) + w * bytesPerPixel;
      *floatPtr++ = rawData[byteIndex];
    }
  }
  floatPtr = (float *)[byteBufferRef bytes];

  NSMutableDictionary *inputDataMap = [NSMutableDictionary dictionary];

  NSMutableDictionary *inputTensorMap = [NSMutableDictionary dictionary];

  // dims
  NSArray *dims = @[
    [NSNumber numberWithInt:1], [NSNumber numberWithInt:static_cast<int>(height)],
    [NSNumber numberWithInt:static_cast<int>(width)]
  ];
  inputTensorMap[@"dims"] = dims;

  // type
  inputTensorMap[@"type"] = JsTensorTypeFloat;

  // encoded data
  NSString *data = [byteBufferRef base64EncodedStringWithOptions:0];
  inputTensorMap[@"data"] = data;

  inputDataMap[@"flatten_2_input"] = inputTensorMap;

  return inputDataMap;
}

- (NSDictionary *)postprocess:(NSDictionary *)result {
  NSMutableString *detectionResult = [NSMutableString string];

  NSDictionary *outputTensor = [result objectForKey:@"Identity"];

  NSString *data = [outputTensor objectForKey:@"data"];
  NSData *buffer = [[NSData alloc] initWithBase64EncodedString:data options:0];
  float *values = (float *)[buffer bytes];
  int count = (int)[buffer length] / 4;

  int argmax = 0;
  float maxValue = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (values[i] > maxValue) {
      maxValue = values[i];
      argmax = i;
    }
  }

  if (maxValue == 0.0f) {
    detectionResult = [NSMutableString stringWithString:@"No match"];
  } else {
    detectionResult = [NSMutableString stringWithFormat:@"I guess, it's %d", argmax];
  }

  NSDictionary *cookedMap = @{@"result" : detectionResult};
  return cookedMap;
}

@end

NS_ASSUME_NONNULL_END
