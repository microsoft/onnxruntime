/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <math.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxTensor.h"

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createTensor
 * Signature: (JJLjava/lang/Object;[JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createTensor
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jobject dataObj,
         jlongArray shape, jint onnxTypeJava) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Convert type to ONNX C enum
    ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeJava);

    // Extract the shape information
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv, shape, NULL);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv, shape);

    // Create the OrtValue
    OrtValue* ortValue = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api,
                                       api->CreateTensorAsOrtValue(
                                           allocator, (int64_t*)shapeArr, shapeLen, onnxType, &ortValue
                                           )
                                       );
    (*jniEnv)->ReleaseLongArrayElements(jniEnv, shape, shapeArr, JNI_ABORT);

    int failed = 0;
    if (code == ORT_OK) {
      // Get a reference to the OrtValue's data
      uint8_t* tensorData = NULL;
      code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData(ortValue, (void**)&tensorData));
      if (code == ORT_OK) {
        // Check if we're copying a scalar or not
        if (shapeLen == 0) {
          // Scalars are passed in as a single element array
          int64_t copied = copyJavaToPrimitiveArray(jniEnv, onnxType, dataObj, tensorData);
          failed = copied == -1 ? 1 : failed;
        } else {
          // Extract the tensor shape information
          JavaTensorTypeShape typeShape;
          code = getTensorTypeShape(jniEnv, &typeShape, api, ortValue);

          if (code == ORT_OK) {
            // Copy the java array into the tensor
            int64_t copied = copyJavaToTensor(jniEnv, onnxType, typeShape.elementCount,
                                             typeShape.dimensions, dataObj, tensorData);
            failed = copied == -1 ? 1 : failed;
          } else {
            failed = 1;
          }
        }
      } else {
        failed = 1;
      }
    }

    if (failed) {
      api->ReleaseValue(ortValue);
      ortValue = NULL;
    }

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createTensorFromBuffer
 * Signature: (JJLjava/nio/Buffer;IJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createTensorFromBuffer
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jobject buffer, jint bufferPos, jlong bufferSize,
         jlongArray shape, jint onnxTypeJava) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    const OrtMemoryInfo* allocatorInfo;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->AllocatorGetInfo(allocator, &allocatorInfo));
    if (code != ORT_OK) {
      return (jlong) NULL;
    }

    // Convert type to ONNX C enum
    ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeJava);

    // Extract the buffer
    char* bufferArr = (char*)(*jniEnv)->GetDirectBufferAddress(jniEnv, buffer);
    // Increment by bufferPos bytes
    bufferArr = bufferArr + bufferPos;

    // Extract the shape information
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv, shape, NULL);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv, shape);

    // Create the OrtValue
    OrtValue* ortValue = NULL;
    checkOrtStatus(jniEnv, api, api->CreateTensorWithDataAsOrtValue(allocatorInfo, bufferArr, bufferSize,
                    (int64_t*)shapeArr, shapeLen, onnxType, &ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv, shape, shapeArr, JNI_ABORT);

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createString
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createString
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jstring input) {
  (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

  // Create the OrtValue
  int64_t shapeArr = 1;
  OrtValue* ortValue = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator, &shapeArr, 0,
                                                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &ortValue));

  if (code == ORT_OK) {
    // Create the buffer for the Java string
    const char* stringBuffer = (*jniEnv)->GetStringUTFChars(jniEnv,input,NULL);

    // Assign the strings into the Tensor
    code = checkOrtStatus(jniEnv, api, api->FillStringTensor(ortValue, &stringBuffer, 1));

    // Release the Java string whether the call succeeded or failed
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, input, stringBuffer);

    // Assignment failed, return null
    if (code != ORT_OK) {
      api->ReleaseValue(ortValue);
      return (jlong) NULL;
    }
  }

  return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createStringTensor
 * Signature: (JJ[Ljava/lang/Object;[J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createStringTensor
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jobjectArray stringArr, jlongArray shape) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Extract the shape information
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv, shape, NULL);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv, shape);

    // Array length
    jsize length = (*jniEnv)->GetArrayLength(jniEnv, stringArr);

    // Create the OrtValue
    OrtValue* ortValue = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator, (int64_t*)shapeArr, shapeLen,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv, shape, shapeArr, JNI_ABORT);

    if (code == ORT_OK) {
      // Create the buffers for the Java strings
      const char** strings = NULL;
      code = checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator, sizeof(char*) * length, (void**)&strings));

      if (code == ORT_OK) {
        // Copy the java strings into the buffers
        for (jsize i = 0; i < length; i++) {
            jobject javaString = (*jniEnv)->GetObjectArrayElement(jniEnv, stringArr, i);
            strings[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaString, NULL);
        }

        // Assign the strings into the Tensor
        code = checkOrtStatus(jniEnv, api, api->FillStringTensor(ortValue, strings, length));

        // Release the Java strings
        for (int i = 0; i < length; i++) {
            jobject javaString = (*jniEnv)->GetObjectArrayElement(jniEnv, stringArr, i);
            (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaString, strings[i]);
        }

        // Release the buffers
        OrtErrorCode freeCode = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, (void*)strings));

        // Assignment failed, return null
        if ((code != ORT_OK) || (freeCode != ORT_OK))  {
          api->ReleaseValue(ortValue);
          return (jlong) NULL;
        }
      }
    }

    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getBuffer
 * Signature: (JJ)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_ai_onnxruntime_OnnxTensor_getBuffer
        (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* ortValue = (OrtValue *) handle;
    JavaTensorTypeShape typeShape;
    OrtErrorCode code = getTensorTypeShape(jniEnv, &typeShape, api, ortValue);

    if (code == ORT_OK) {
      size_t typeSize = onnxTypeSize(typeShape.onnxTypeEnum);
      size_t sizeBytes = typeShape.elementCount * typeSize;

      uint8_t* arr = NULL;
      code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));

      if (code == ORT_OK) {
        return (*jniEnv)->NewDirectByteBuffer(jniEnv, arr, (jlong)sizeBytes);
      }
    }
    return NULL;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getFloat
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_OnnxTensor_getFloat
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxTypeInt) {
  (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeInt);
  if (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    jfloat* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return *arr;
    }
  }
  return NAN;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getDouble
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_ai_onnxruntime_OnnxTensor_getDouble
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    jdouble* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return *arr;
    } else {
      return NAN;
    }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getByte
 * Signature: (JI)B
 */
JNIEXPORT jbyte JNICALL Java_ai_onnxruntime_OnnxTensor_getByte
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxTypeInt) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeInt);
  if ((onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) || (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)) {
    uint8_t* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return (jbyte) *arr;
    }
  }
  return (jbyte) 0;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getShort
 * Signature: (JI)S
 */
JNIEXPORT jshort JNICALL Java_ai_onnxruntime_OnnxTensor_getShort
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxTypeInt) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeInt);
  if ((onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) ||
      (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)  ||
      (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ||
      (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    uint16_t* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return (jshort) *arr;
    }
  }
  return (jshort) 0;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getInt
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OnnxTensor_getInt
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxTypeInt) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeInt);
  if ((onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) || (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
    uint32_t* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return (jint) *arr;
    }
  }
  return (jint) 0;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getLong
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_getLong
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxTypeInt) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeInt);
  if ((onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) || (onnxType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    uint64_t* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return (jlong) *arr;
    }
  }
  return (jlong) 0;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getString
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OnnxTensor_getString
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    // Extract a String array - if this becomes a performance issue we'll refactor later.
    jobjectArray outputArray = createStringArrayFromTensor(jniEnv, api, (OrtValue*) handle);
    if (outputArray != NULL) {
      // Get reference to the string
      jobject output = (*jniEnv)->GetObjectArrayElement(jniEnv, outputArray, 0);

      // Free array
      (*jniEnv)->DeleteLocalRef(jniEnv, outputArray);

      return output;
    } else {
      return NULL;
    }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getBool
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_ai_onnxruntime_OnnxTensor_getBool
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    jboolean* arr = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData((OrtValue*)handle, (void**)&arr));
    if (code == ORT_OK) {
      return *arr;
    } else {
      return 0;
    }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getArray
 * Signature: (JJLjava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxTensor_getArray
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jobject carrier) {
    (void) jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* value = (OrtValue*) handle;
    JavaTensorTypeShape typeShape;
    OrtErrorCode code = getTensorTypeShape(jniEnv, &typeShape, api, value);
    if (code == ORT_OK) {
      if (typeShape.onnxTypeEnum == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        copyStringTensorToArray(jniEnv, api, value, typeShape.elementCount, carrier);
      } else {
        uint8_t* arr = NULL;
        code = checkOrtStatus(jniEnv, api, api->GetTensorMutableData(value, (void**)&arr));
        if (code == ORT_OK) {
          copyTensorToJava(jniEnv, typeShape.onnxTypeEnum, arr, typeShape.elementCount,
                           typeShape.dimensions, (jarray)carrier);
        }
      }
    }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxTensor_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}
