// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <CommonCrypto/CommonCrypto.h>
#import <Security/Security.h>
#import <XCTest/XCTest.h>

#import "ort_env.h"
#import "ort_ep_context_fixture.h"
#import "ort_session.h"
#import "ort_value.h"
#import "test/assertion_utils.h"

#include <cstring>
#include <stdexcept>

namespace {
NSData* RandomData(size_t size) {
  NSMutableData* bytes = [NSMutableData dataWithLength:size];
  if (SecRandomCopyBytes(kSecRandomDefault, size, bytes.mutableBytes) != errSecSuccess) {
    throw std::runtime_error("Test random generation failed");
  }
  return bytes;
}

NSData* Crypt(CCOperation operation, NSData* input, NSData* key, NSData* iv) {
  NSMutableData* output = [NSMutableData dataWithLength:input.length + kCCBlockSizeAES128];
  size_t size = 0;
  CCCryptorStatus status = CCCrypt(operation, kCCAlgorithmAES, kCCOptionPKCS7Padding,
                                   key.bytes, key.length, iv.bytes, input.bytes, input.length,
                                   output.mutableBytes, output.length, &size);
  if (status != kCCSuccess) {
    throw std::runtime_error("Test AES-CBC operation failed");
  }
  output.length = size;
  return output;
}

NSData* Tag(NSData* record, NSString* name, NSData* key) {
  NSMutableData* authenticated = [[name dataUsingEncoding:NSUTF8StringEncoding] mutableCopy];
  const uint8_t separator = 0;
  [authenticated appendBytes:&separator length:1];
  [authenticated appendData:record];
  uint8_t tag[CC_SHA256_DIGEST_LENGTH];
  CCHmac(kCCHmacAlgSHA256, key.bytes, key.length, authenticated.bytes, authenticated.length, tag);
  return [NSData dataWithBytes:tag length:sizeof(tag)];
}

NSData* Encrypt(NSData* plaintext, NSString* name, NSData* encryptionKey, NSData* authenticationKey) {
  NSData* iv = RandomData(kCCBlockSizeAES128);
  NSMutableData* record = [iv mutableCopy];
  [record appendData:Crypt(kCCEncrypt, plaintext, encryptionKey, iv)];
  [record appendData:Tag(record, name, authenticationKey)];
  return record;
}

NSData* Decrypt(NSData* record, NSString* name, NSData* encryptionKey, NSData* authenticationKey) {
  if (record.length < kCCBlockSizeAES128 * 2 + CC_SHA256_DIGEST_LENGTH) {
    throw std::runtime_error("Encrypted test record is truncated");
  }
  const size_t authenticatedSize = record.length - CC_SHA256_DIGEST_LENGTH;
  NSData* authenticated = [record subdataWithRange:NSMakeRange(0, authenticatedSize)];
  NSData* expected = Tag(authenticated, name, authenticationKey);
  const auto* actual = static_cast<const uint8_t*>(record.bytes) + authenticatedSize;
  const auto* tag = static_cast<const uint8_t*>(expected.bytes);
  uint8_t difference = 0;
  for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
    difference |= actual[i] ^ tag[i];
  }
  if (difference != 0) {
    throw std::runtime_error("Test ciphertext authentication failed");
  }
  return Crypt(kCCDecrypt,
               [record subdataWithRange:NSMakeRange(kCCBlockSizeAES128, authenticatedSize - kCCBlockSizeAES128)],
               encryptionKey, [record subdataWithRange:NSMakeRange(0, kCCBlockSizeAES128)]);
}

NSData* DecryptOrReport(NSData* record, NSString* name, NSData* encryptionKey,
                        NSData* authenticationKey, NSError** error) {
  try {
    return Decrypt(record, name, encryptionKey, authenticationKey);
  } catch (const std::exception& failure) {
    if (error != nullptr) {
      *error = [NSError errorWithDomain:@"ORTEncryptionTest"
                                   code:2
                               userInfo:@{NSLocalizedDescriptionKey : [NSString stringWithUTF8String:failure.what()]}];
    }
    return nil;
  }
}

}  // namespace

@interface ORTEncryptedEpContextTest : XCTestCase
@end

@interface ORTFailingUnregisterFixture : ORTEpContextFixture
@end

@implementation ORTFailingUnregisterFixture
- (void)unregister {
  throw std::runtime_error("Injected fixture unregistration failure");
}
@end

@implementation ORTEncryptedEpContextTest

- (void)cleanupFixture:(ORTEpContextFixture*)fixture directory:(NSString*)directory {
  @try {
    [fixture unregister];
  } @finally {
    NSError* cleanupError = nil;
    XCTAssertTrue([NSFileManager.defaultManager removeItemAtPath:directory error:&cleanupError]);
    XCTAssertNil(cleanupError);
  }
}

- (void)testCleanupRemovesPlaintextWhenUnregisterThrows {
  self.continueAfterFailure = NO;
  NSString* directory = [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
  NSError* error = nil;
  XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtPath:directory
                                        withIntermediateDirectories:NO
                                                         attributes:nil
                                                              error:&error]);
  XCTAssertNil(error);
  @try {
    NSString* plaintext = [directory stringByAppendingPathComponent:@"compiled.onnx"];
    XCTAssertTrue([[@"test plaintext" dataUsingEncoding:NSUTF8StringEncoding] writeToFile:plaintext
                                                                                  options:NSDataWritingAtomic
                                                                                    error:&error]);
    XCTAssertNil(error);
    bool failed = false;
    try {
      [self cleanupFixture:[[ORTFailingUnregisterFixture alloc] init] directory:directory];
    } catch (const std::runtime_error& failure) {
      failed = std::strstr(failure.what(), "Injected fixture unregistration failure") != nullptr;
    }
    XCTAssertTrue(failed);
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:directory]);
  } @finally {
    if ([NSFileManager.defaultManager fileExistsAtPath:directory]) {
      XCTAssertTrue([NSFileManager.defaultManager removeItemAtPath:directory error:&error]);
      XCTAssertNil(error);
    }
  }
}

- (void)testEncryptedCompiledContextInference {
  NSString* plugin = NSProcessInfo.processInfo.environment[@"ORT_ENCRYPTION_PLUGIN_LIBRARY"];
  self.continueAfterFailure = NO;
  XCTAssertGreaterThan(plugin.length, NSUInteger{0}, @"The encryption test target must provision its example EP");
  NSError* error = nil;
  ORTEnv* env = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning error:&error];
  ORTAssertNullableResultSuccessful(env, error);
  NSString* directory = [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
  BOOL created = [NSFileManager.defaultManager createDirectoryAtPath:directory
                                         withIntermediateDirectories:NO
                                                          attributes:nil
                                                               error:&error];
  ORTAssertBoolResultSuccessful(created, error);
  ORTEpContextFixture* fixture = nil;
  @try {
    fixture = [[ORTEpContextFixture alloc] initWithEnv:env pluginPath:plugin];
    NSData* encryptionKey = RandomData(kCCKeySizeAES256);
    NSData* authenticationKey = RandomData(CC_SHA256_DIGEST_LENGTH);
    NSString* modelName = @"compiled.onnx";
    NSString* contextName;
    NSString* encryptedModelPath = [directory stringByAppendingPathComponent:@"model.enc"];
    NSString* encryptedContextPath = [directory stringByAppendingPathComponent:@"context.enc"];
    @autoreleasepool {
      ORTSessionOptions* options = [[ORTSessionOptions alloc] initWithError:&error];
      ORTAssertNullableResultSuccessful(options, error);
      [fixture configureSessionOptions:options];
      NSString* source = [[NSBundle bundleForClass:self.class] pathForResource:@"encrypted_ep_context_mul" ofType:@"onnx"];
      XCTAssertNotNil(source);
      NSData* contextData = nil;
      NSData* model = [fixture compileModelAtPath:source
                                  outputDirectory:directory
                                   sessionOptions:options
                                      contextData:&contextData
                                      contextName:&contextName];
      XCTAssertEqualObjects(contextData, [@"ort-test-mul-float32-v1" dataUsingEncoding:NSUTF8StringEncoding]);
      XCTAssertTrue([Encrypt(model, modelName, encryptionKey, authenticationKey)
          writeToFile:encryptedModelPath
              options:NSDataWritingAtomic
                error:&error]);
      XCTAssertNil(error);
      XCTAssertTrue([Encrypt(contextData, contextName, encryptionKey, authenticationKey)
          writeToFile:encryptedContextPath
              options:NSDataWritingAtomic
                error:&error]);
      XCTAssertNil(error);
    }
    NSArray* persisted = [NSFileManager.defaultManager contentsOfDirectoryAtPath:directory error:&error];
    XCTAssertEqual(persisted.count, NSUInteger{2});
    XCTAssertNil(error);
    NSData* modelRecord = [NSData dataWithContentsOfFile:encryptedModelPath];
    NSData* contextRecord = [NSData dataWithContentsOfFile:encryptedContextPath];
    NSMutableData* wrongKey = [authenticationKey mutableCopy];
    static_cast<uint8_t*>(wrongKey.mutableBytes)[0] ^= 1;
    auto rejects = [&](NSData* record, NSString* name, NSData* key) {
      bool rejected = false;
      try {
        (void)Decrypt(record, name, encryptionKey, key);
      } catch (const std::runtime_error& failure) {
        rejected = std::strstr(failure.what(), "authentication") != nullptr;
      }
      XCTAssertTrue(rejected);
    };
    rejects(modelRecord, modelName, wrongKey);
    rejects(contextRecord, contextName, wrongKey);
    rejects(contextRecord, @"wrong-context-name", authenticationKey);
    for (NSData* record in @[ modelRecord, contextRecord ]) {
      NSMutableData* tampered = [record mutableCopy];
      static_cast<uint8_t*>(tampered.mutableBytes)[tampered.length - CC_SHA256_DIGEST_LENGTH - 1] ^= 1;
      rejects(tampered, record == modelRecord ? modelName : contextName, authenticationKey);
    }
    // The public Objective-C session API only accepts a path, not in-memory model bytes.
    NSString* modelPath = [directory stringByAppendingPathComponent:modelName];
    XCTAssertTrue([Decrypt(modelRecord, modelName, encryptionKey, authenticationKey)
        writeToFile:modelPath
            options:NSDataWritingAtomic
              error:&error]);
    XCTAssertNil(error);
    @autoreleasepool {
      ORTSessionOptions* options = [[ORTSessionOptions alloc] initWithError:&error];
      ORTAssertNullableResultSuccessful(options, error);
      [fixture configureSessionOptions:options];
      __block NSUInteger calls = 0;
      BOOL set = [options
          setEpContextDataReadBlock:^NSData*(NSString* name, NSError** callbackError) {
            ++calls;
            if (![name isEqualToString:contextName]) {
              if (callbackError != nullptr) {
                *callbackError = [NSError errorWithDomain:@"ORTEncryptionTest" code:1 userInfo:nil];
              }
              return nil;
            }
            return DecryptOrReport(contextRecord, name, encryptionKey, authenticationKey, callbackError);
          }
                        maxDataSize:1024
                              error:&error];
      ORTAssertBoolResultSuccessful(set, error);
      ORTSession* session = [[ORTSession alloc] initWithEnv:env modelPath:modelPath sessionOptions:options error:&error];
      ORTAssertNullableResultSuccessful(session, error);
      XCTAssertEqual(calls, NSUInteger{1});
      for (int iteration = 0; iteration < 2; ++iteration) {
        float x[6], y[6];
        for (int i = 0; i < 6; ++i) {
          x[i] = static_cast<float>(i - iteration);
          y[i] = static_cast<float>(iteration + 2 - i);
        }
        ORTValue* a = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithBytes:x length:sizeof(x)]
                                               elementType:ORTTensorElementDataTypeFloat
                                                     shape:@[ @3, @2 ]
                                                     error:&error];
        ORTAssertNullableResultSuccessful(a, error);
        ORTValue* b = [[ORTValue alloc] initWithTensorData:[NSMutableData dataWithBytes:y length:sizeof(y)]
                                               elementType:ORTTensorElementDataTypeFloat
                                                     shape:@[ @3, @2 ]
                                                     error:&error];
        ORTAssertNullableResultSuccessful(b, error);
        NSDictionary* outputs = [session runWithInputs:@{@"x" : a, @"y" : b}
                                           outputNames:[NSSet setWithObject:@"z"]
                                            runOptions:nil
                                                 error:&error];
        ORTAssertNullableResultSuccessful(outputs, error);
        NSData* data = [outputs[@"z"] tensorDataWithError:&error];
        ORTAssertNullableResultSuccessful(data, error);
        XCTAssertEqual(data.length, sizeof(x));
        float actual[6];
        std::memcpy(actual, data.bytes, sizeof(actual));
        for (int i = 0; i < 6; ++i) {
          XCTAssertEqual(actual[i], x[i] * y[i]);
        }
      }
    }
    auto rejectLoad = [&](ORTEpContextDataReadBlock block, NSString* expectedError) {
      @autoreleasepool {
        NSError* loadError = nil;
        ORTSessionOptions* options = [[ORTSessionOptions alloc] initWithError:&loadError];
        ORTAssertNullableResultSuccessful(options, loadError);
        [fixture configureSessionOptions:options];
        if (block != nil) {
          BOOL set = [options setEpContextDataReadBlock:block maxDataSize:1024 error:&loadError];
          ORTAssertBoolResultSuccessful(set, loadError);
        }
        ORTSession* failed = [[ORTSession alloc] initWithEnv:env
                                                   modelPath:modelPath
                                              sessionOptions:options
                                                       error:&loadError];
        ORTAssertNullableResultUnsuccessful(failed, loadError);
        if (expectedError != nil) {
          XCTAssertTrue([loadError.localizedDescription containsString:expectedError], @"%@", loadError);
        }
      }
    };
    rejectLoad(nil, nil);
    rejectLoad(^NSData*(NSString* name, NSError** callbackError) {
      return DecryptOrReport(contextRecord, name, encryptionKey, wrongKey, callbackError);
    },
               @"authentication");
    NSMutableData* tamperedContext = [contextRecord mutableCopy];
    static_cast<uint8_t*>(tamperedContext.mutableBytes)[tamperedContext.length - CC_SHA256_DIGEST_LENGTH - 1] ^= 1;
    rejectLoad(^NSData*(NSString* name, NSError** callbackError) {
      return DecryptOrReport(tamperedContext, name, encryptionKey, authenticationKey, callbackError);
    },
               @"authentication");
    rejectLoad(^NSData*(NSString* name, NSError** callbackError) {
      (void)name;
      (void)callbackError;
      return [@"invalid compiled payload" dataUsingEncoding:NSUTF8StringEncoding];
    },
               @"payload");
  } @finally {
    [self cleanupFixture:fixture directory:directory];
  }
}
@end
