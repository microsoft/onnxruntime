#import "OnnxruntimeJSIHelper.h"
#import <React/RCTBlobManager.h>
#import <React/RCTBridge+Private.h>
#import <jsi/jsi.h>

@implementation OnnxruntimeJSIHelper

RCT_EXPORT_MODULE()

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(install) {
  RCTBridge *bridge = [RCTBridge currentBridge];
  RCTCxxBridge *cxxBridge = (RCTCxxBridge *)bridge;
  if (cxxBridge == nil) {
    return @false;
  }

  using namespace facebook;

  auto jsiRuntime = (jsi::Runtime *)cxxBridge.runtime;
  if (jsiRuntime == nil) {
    return @false;
  }
  auto &runtime = *jsiRuntime;

  auto resolveArrayBuffer = jsi::Function::createFromHostFunction(
      runtime, jsi::PropNameID::forUtf8(runtime, "jsiOnnxruntimeResolveArrayBuffer"), 1,
      [](jsi::Runtime &runtime, const jsi::Value &thisArg, const jsi::Value *args, size_t count) -> jsi::Value {
        if (count != 1) {
          throw jsi::JSError(runtime, "jsiOnnxruntimeResolveArrayBuffer(..) expects one argument (object)!");
        }

        auto data = args[0].asObject(runtime);
        auto blobId = data.getProperty(runtime, "blobId").asString(runtime).utf8(runtime);
        auto size = data.getProperty(runtime, "size").asNumber();
        auto offset = data.getProperty(runtime, "offset").asNumber();

        RCTBlobManager *blobManager = [[RCTBridge currentBridge] moduleForClass:RCTBlobManager.class];
        if (blobManager == nil) {
          throw jsi::JSError(runtime, "RCTBlobManager is not initialized");
        }

        NSString *blobIdStr = [NSString stringWithUTF8String:blobId.c_str()];
        auto blob = [blobManager resolve:blobIdStr offset:(long)offset size:(long)size];

        jsi::Function arrayBufferCtor = runtime.global().getPropertyAsFunction(runtime, "ArrayBuffer");
        jsi::Object o = arrayBufferCtor.callAsConstructor(runtime, (int)blob.length).getObject(runtime);
        jsi::ArrayBuffer buf = o.getArrayBuffer(runtime);
        memcpy(buf.data(runtime), blob.bytes, blob.length);
        [blobManager remove:blobIdStr];
        return buf;
      });
  runtime.global().setProperty(runtime, "jsiOnnxruntimeResolveArrayBuffer", resolveArrayBuffer);

  auto storeArrayBuffer = jsi::Function::createFromHostFunction(
      runtime, jsi::PropNameID::forUtf8(runtime, "jsiOnnxruntimeStoreArrayBuffer"), 1,
      [](jsi::Runtime &runtime, const jsi::Value &thisArg, const jsi::Value *args, size_t count) -> jsi::Value {
        if (count != 1) {
          throw jsi::JSError(runtime, "jsiOnnxruntimeStoreArrayBuffer(..) expects one argument (object)!");
        }

        auto arrayBuffer = args[0].asObject(runtime).getArrayBuffer(runtime);
        auto size = arrayBuffer.length(runtime);
        NSData *data = [NSData dataWithBytesNoCopy:arrayBuffer.data(runtime) length:size freeWhenDone:NO];

        RCTBlobManager *blobManager = [[RCTBridge currentBridge] moduleForClass:RCTBlobManager.class];
        if (blobManager == nil) {
          throw jsi::JSError(runtime, "RCTBlobManager is not initialized");
        }

        NSString *blobId = [blobManager store:data];

        jsi::Object result(runtime);
        auto blobIdString = jsi::String::createFromUtf8(runtime, [blobId cStringUsingEncoding:NSUTF8StringEncoding]);
        result.setProperty(runtime, "blobId", blobIdString);
        result.setProperty(runtime, "offset", jsi::Value(0));
        result.setProperty(runtime, "size", jsi::Value(static_cast<double>(size)));
        return result;
      });

  runtime.global().setProperty(runtime, "jsiOnnxruntimeStoreArrayBuffer", storeArrayBuffer);

  return @true;
}

@end
