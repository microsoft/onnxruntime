# ONNX Runtime Objective-C API

ONNX Runtime provides an Objective-C API.
It can be used from Swift with a bridging header.

The public headers are located [here](https://github.com/microsoft/onnxruntime/tree/master/objectivec/include).
One can import them individually, or import the entire API with [onnxruntime.h](https://github.com/microsoft/onnxruntime/blob/master/objectivec/include/onnxruntime.h).

## Example Usage in Objective-C++

```objectivec
#import <Foundation/Foundation.h>

#import <onnxruntime.h>

// Adds two numbers using ONNX Runtime.
float add(float a, float b) {
  // We will run a simple model which adds two floats.
  // The inputs are named `A` and `B` and the output is named `C` (A + B = C).
  // All inputs and outputs are float tensors with shape [1].
  NSString* const kAddModelPath = @"/path/to/add.ort";

  // ORT APIs take an optional NSError** parameter that will be set if an error occurs.
  // Here, we will omit error handling (i.e., checking results and the NSError object) for brevity.
  NSError* err = nil;

  // First, we create the ORT environment.
  // The environment is required in order to create an ORT session.
  // ORTLoggingLevelWarning should show us only important messages.
  ORTEnv* ortEnv = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning
                                                  error:&err];

  // Next, we will create some ORT values for our input tensors. We have two floats, `a` and `b`.
  auto createOrtValue = [&](float* fp) {
    // `data` will hold the memory of the input ORT value. We set it to refer to the memory of the given float (*fp).
    NSMutableData* data = [[NSMutableData alloc] initWithBytes:fp length:sizeof(float)];
    // This will create a value with a tensor with the given float's data, of type float, and with shape [1].
    ORTValue* ortValue = [[ORTValue alloc] initWithTensorData:data
                                                  elementType:ORTTensorElementDataTypeFloat
                                                        shape:@[ @1 ]
                                                        error:&err];
    return ortValue;
  };

  ORTValue* aInputValue = createOrtValue(&a);
  ORTValue* bInputValue = createOrtValue(&b);

  // Now, we will create an ORT session to run our model.
  // One can configure session options with a session options object (ORTSessionOptions).
  // We use the default options with sessionOptions:nil.
  ORTSession* session = [[ORTSession alloc] initWithEnv:ortEnv
                                              modelPath:kAddModelPath
                                         sessionOptions:nil
                                                  error:&err];

  // With a session and input values, we have what we need to run the model.
  // We provide a dictionary mapping from input name to value and a set of output names.
  // This run method will run the model, allocating the output(s), and return them in a dictionary mapping from output name to value.
  // As with session creation, it is possible to configure run options with a run options object (ORTRunOptions).
  // We use the default options with runOptions:nil.
  NSDictionary<NSString*, ORTValue*>* outputs =
      [session runWithInputs:@{@"A" : aInputValue, @"B" : bInputValue}
                 outputNames:[NSSet setWithArray:@[ @"C" ]]
                  runOptions:nil
                       error:&err];

  // After running the model, we can get the output.
  ORTValue* cOutput = outputs[@"C"];

  // We know the output value is a float tensor with shape [1]. We will just access it directly.
  // It is also possible to query the type information of a value.
  NSData* cData = [cOutput tensorDataWithError:&err];
  float c;
  memcpy(&c, cData.bytes, sizeof(float));

  return c;
}
```
