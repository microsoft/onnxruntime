// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <CommonCrypto/CommonCrypto.h>
#import <Security/Security.h>
#import <XCTest/XCTest.h>

#import "ort_env_internal.h"
#import "ort_session_internal.h"
#import "ort_value.h"
#import "test/assertion_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

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

struct Context {
  std::string name;
  std::vector<uint8_t> bytes;
};

OrtStatus* WriteContext(void* state, const char* name, const void* data, size_t size) noexcept {
  try {
    auto& context = *static_cast<Context*>(state);
    if (!context.name.empty()) {
      throw std::runtime_error("Expected exactly one external context write");
    }
    context.name = name;
    const auto* bytes = static_cast<const uint8_t*>(data);
    context.bytes.assign(bytes, bytes + size);
    return nullptr;
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}
}  // namespace

@interface ORTEncryptedEpContextTest : XCTestCase
@end

@implementation ORTEncryptedEpContextTest

- (void)testEncryptedCompiledContextInference {
  NSString* plugin = NSProcessInfo.processInfo.environment[@"ORT_ENCRYPTION_PLUGIN_LIBRARY"];
  XCTSkipIf(plugin.length == 0, @"Set ORT_ENCRYPTION_PLUGIN_LIBRARY to run the example EP integration");
  self.continueAfterFailure = NO;
  NSError* error = nil;
  ORTEnv* env = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning error:&error];
  ORTAssertNullableResultSuccessful(env, error);
  auto& nativeEnv = [env CXXAPIOrtEnv];
  const char* registration = "objc_encryption_test";
  NSString* directory = [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
  BOOL created = [NSFileManager.defaultManager createDirectoryAtPath:directory
                                         withIntermediateDirectories:NO
                                                          attributes:nil
                                                               error:&error];
  ORTAssertBoolResultSuccessful(created, error);
  bool registered = false;
  @try {
    nativeEnv.RegisterExecutionProviderLibrary(registration, plugin.fileSystemRepresentation);
    registered = true;
    auto configure = [&](ORTSessionOptions* options) {
      auto& native = [options CXXAPIOrtSessionOptions];
      auto devices = nativeEnv.GetEpDevices();
      auto device = std::find_if(devices.begin(), devices.end(), [&](const auto& candidate) {
        return std::strcmp(candidate.EpName(), registration) == 0;
      });
      if (device == devices.end()) {
        throw std::runtime_error("Registered test EP device was not found");
      }
      native.AddConfigEntry("ep.example.test_execute_ep_context", "1");
      native.AppendExecutionProvider_V2(nativeEnv, {*device}, {});
    };
    NSData* encryptionKey = RandomData(kCCKeySizeAES256);
    NSData* authenticationKey = RandomData(CC_SHA256_DIGEST_LENGTH);
    NSString* modelName = @"compiled.onnx";
    NSString* contextName;
    NSString* encryptedModelPath = [directory stringByAppendingPathComponent:@"model.enc"];
    NSString* encryptedContextPath = [directory stringByAppendingPathComponent:@"context.enc"];
    @autoreleasepool {
      ORTSessionOptions* options = [[ORTSessionOptions alloc] initWithError:&error];
      ORTAssertNullableResultSuccessful(options, error);
      configure(options);
      Ort::ModelCompilationOptions compilation(nativeEnv, [options CXXAPIOrtSessionOptions]);
      NSString* source = [[NSBundle bundleForClass:self.class] pathForResource:@"encrypted_ep_context_mul" ofType:@"onnx"];
      XCTAssertNotNil(source);
      compilation.SetInputModelPath(source.fileSystemRepresentation);
      compilation.SetEpContextEmbedMode(false);
      compilation.SetEpContextBinaryInformation([[directory stringByAppendingString:@"/"] fileSystemRepresentation],
                                                modelName.fileSystemRepresentation);
      compilation.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
      Context context;
      compilation.SetEpContextDataWriteFunc(WriteContext, &context);
      // The write callback avoids an allocator-owned model buffer on assertion/error paths.
      NSMutableData* model = [NSMutableData data];
      compilation.SetOutputModelWriteFunc(
          [](void* state, const void* data, size_t size) -> OrtStatus* {
            [(__bridge NSMutableData*)state appendBytes:data length:size];
            return nullptr;
          },
          (__bridge void*)model);
      auto status = Ort::CompileModel(nativeEnv, compilation);
      XCTAssertTrue(status.IsOK(), @"%s", status.GetErrorMessage().c_str());
      XCTAssertEqual(context.bytes.size(), std::strlen("ort-test-mul-float32-v1"));
      XCTAssertEqual(std::memcmp(context.bytes.data(), "ort-test-mul-float32-v1", context.bytes.size()), 0);
      contextName = [NSString stringWithUTF8String:context.name.c_str()];
      NSData* contextData = [NSData dataWithBytes:context.bytes.data() length:context.bytes.size()];
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
    XCTAssertEqual(persisted.count, 2);
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
      static_cast<uint8_t*>(tampered.mutableBytes)[tampered.length - 1] ^= 1;
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
      configure(options);
      __block NSUInteger calls = 0;
      BOOL set = [options
          setEpContextDataReadBlock:^NSData*(NSString* name, NSError** callbackError) {
            ++calls;
            if (![name isEqualToString:contextName]) {
              *callbackError = [NSError errorWithDomain:@"ORTEncryptionTest" code:1 userInfo:nil];
              return nil;
            }
            try {
              return Decrypt(contextRecord, name, encryptionKey, authenticationKey);
            } catch (const std::exception& failure) {
              *callbackError = [NSError errorWithDomain:@"ORTEncryptionTest"
                                                   code:2
                                               userInfo:@{NSLocalizedDescriptionKey : [NSString stringWithUTF8String:failure.what()]}];
              return nil;
            }
          }
                        maxDataSize:1024
                              error:&error];
      ORTAssertBoolResultSuccessful(set, error);
      ORTSession* session = [[ORTSession alloc] initWithEnv:env modelPath:modelPath sessionOptions:options error:&error];
      ORTAssertNullableResultSuccessful(session, error);
      XCTAssertEqual(calls, 1);
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
    ORTSessionOptions* missing = [[ORTSessionOptions alloc] initWithError:&error];
    ORTAssertNullableResultSuccessful(missing, error);
    configure(missing);
    ORTSession* failed = [[ORTSession alloc] initWithEnv:env modelPath:modelPath sessionOptions:missing error:&error];
    ORTAssertNullableResultUnsuccessful(failed, error);
  } @finally {
    if (registered) {
      nativeEnv.UnregisterExecutionProviderLibrary(registration);
    }
    NSError* cleanupError = nil;
    XCTAssertTrue([NSFileManager.defaultManager removeItemAtPath:directory error:&cleanupError]);
    XCTAssertNil(cleanupError);
  }
}
@end
