/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxSequence.h"
/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getStringKeys
 * Signature: (JJI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getStringKeys
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jobjectArray output = NULL;
    // Extract element
    OrtValue* element;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));
    if (code == ORT_OK) {
      // Extract keys from element
      OrtValue* keys;
      code = checkOrtStatus(jniEnv,api,api->GetValue(element,0,allocator,&keys));

      if (code == ORT_OK) {
        // Convert to Java String array
        output = createStringArrayFromTensor(jniEnv, api, allocator, keys);
        // Release if valid
        api->ReleaseValue(element);
      }

      // Keys is valid, so release
      api->ReleaseValue(keys);
    }
    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getLongKeys
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxSequence_getLongKeys
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jlongArray output = NULL;
    // Extract element
    OrtValue* element;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));
    if (code == ORT_OK) {
      // Extract keys from element
      OrtValue* keys;
      code = checkOrtStatus(jniEnv,api,api->GetValue(element,0,allocator,&keys));

      if (code == ORT_OK) {
        // Convert to Java long array
        output = createLongArrayFromTensor(jniEnv, api, keys);
        // Release if valid
        api->ReleaseValue(element);
      }

      // Keys is valid, so release
      api->ReleaseValue(keys);
    }
    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getStringValues
 * Signature: (JJI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getStringValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jobjectArray output = NULL;
    // Extract element
    OrtValue* element;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));
    if (code == ORT_OK) {
      // Extract values from element
      OrtValue* values;
      code = checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

      if (code == ORT_OK) {
        // Convert to Java String array
        output = createStringArrayFromTensor(jniEnv, api, allocator, values);
        // Release if valid
        api->ReleaseValue(element);
      }

      // values is valid, so release
      api->ReleaseValue(values);
    }
    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getLongValues
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxSequence_getLongValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jlongArray output = NULL;
    // Extract element
    OrtValue* element;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));
    if (code == ORT_OK) {
      // Extract values from element
      OrtValue* values;
      code = checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

      if (code == ORT_OK) {
        // Convert to Java long array
        output = createLongArrayFromTensor(jniEnv, api, values);
        // Release if valid
        api->ReleaseValue(element);
      }

      // values is valid, so release
      api->ReleaseValue(values);
    }
    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getFloatValues
 * Signature: (JJI)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_OnnxSequence_getFloatValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jfloatArray output = NULL;
    // Extract element
    OrtValue* element;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));
    if (code == ORT_OK) {
      // Extract values from element
      OrtValue* values;
      code = checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

      if (code == ORT_OK) {
        // Convert to Java float array
        output = createFloatArrayFromTensor(jniEnv, api, values);
        // Release if valid
        api->ReleaseValue(element);
      }

      // values is valid, so release
      api->ReleaseValue(values);
    }
    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getDoubleValues
 * Signature: (JJI)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_OnnxSequence_getDoubleValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jdoubleArray output = NULL;
    // Extract element
    OrtValue* element;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));
    if (code == ORT_OK) {
      // Extract values from element
      OrtValue* values;
      code = checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

      if (code == ORT_OK) {
        // Convert to Java double array
        output = createDoubleArrayFromTensor(jniEnv, api, values);
        // Release if valid
        api->ReleaseValue(element);
      }

      // values is valid, so release
      api->ReleaseValue(values);
    }
    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getStrings
 * Signature: (JJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getStrings
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    jobjectArray outputArray = NULL;

    // Get the element count of this sequence
    size_t count;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));
    if (code == ORT_OK) {
      jclass stringClazz = (*jniEnv)->FindClass(jniEnv,"java/lang/String");
      outputArray = (*jniEnv)->NewObjectArray(jniEnv,safecast_size_t_to_jsize(count),stringClazz, NULL);
      for (size_t i = 0; i < count; i++) {
          // Extract element
          OrtValue* element;
          code = checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));
          if (code == ORT_OK) {
            jobject str = createStringFromStringTensor(jniEnv,api,allocator,element);
            if (str == NULL) {
              api->ReleaseValue(element);
              // bail out as exception has been thrown
              return NULL;
            }
            (*jniEnv)->SetObjectArrayElement(jniEnv, outputArray, i, str);

            api->ReleaseValue(element);
          } else {
              // bail out as exception has been thrown
              return NULL;
          }
      }
    }
    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getLongs
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxSequence_getLongs
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jlongArray outputArray = NULL;

    // Get the element count of this sequence
    size_t count;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));
    if (code == ORT_OK) {
      int64_t* values;
      code = checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(int64_t)*count,(void**)&values));
      if (code == ORT_OK) {
        for (size_t i = 0; i < count; i++) {
            // Extract element
            OrtValue* element;
            code = checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));
            if (code == ORT_OK) {
              // Extract the values
              int64_t* arr;
              code = checkOrtStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
              if (code == ORT_OK) {
                values[i] = arr[0];
              } else {
                // bail out as exception has been thrown
                api->ReleaseValue(element);
                checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
                return NULL;
              }

              api->ReleaseValue(element);
            } else {
              // bail out as exception has been thrown
              checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
              return NULL;
            }
        }

        outputArray = (*jniEnv)->NewLongArray(jniEnv,safecast_size_t_to_jsize(count));
        (*jniEnv)->SetLongArrayRegion(jniEnv, outputArray,0,safecast_size_t_to_jsize(count),(jlong*)values);

        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
      }
    }
    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getFloats
 * Signature: (JJ)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_OnnxSequence_getFloats
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jfloatArray outputArray = NULL;

    // Get the element count of this sequence
    size_t count;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));
    if (code == ORT_OK) {
      float* values;
      code = checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(float)*count,(void**)&values));
      if (code == ORT_OK) {
        for (size_t i = 0; i < count; i++) {
            // Extract element
            OrtValue* element;
            code = checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));
            if (code == ORT_OK) {
              // Extract the values
              float* arr;
              code = checkOrtStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
              if (code == ORT_OK) {
                values[i] = arr[0];
              } else {
                // bail out as exception has been thrown
                api->ReleaseValue(element);
                checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
                return NULL;
              }

              api->ReleaseValue(element);
            } else {
              // bail out as exception has been thrown
              checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
              return NULL;
            }
        }

        outputArray = (*jniEnv)->NewFloatArray(jniEnv,safecast_size_t_to_jsize(count));
        (*jniEnv)->SetFloatArrayRegion(jniEnv, outputArray,0,safecast_size_t_to_jsize(count),values);

        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
      }
    }
    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getDoubles
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_OnnxSequence_getDoubles
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    jdoubleArray outputArray = NULL;

    // Get the element count of this sequence
    size_t count;
    OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));
    if (code == ORT_OK) {
      double* values;
      code = checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(double)*count,(void**)&values));
      if (code == ORT_OK) {
        for (size_t i = 0; i < count; i++) {
            // Extract element
            OrtValue* element;
            code = checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));
            if (code == ORT_OK) {
              // Extract the values
              double* arr;
              code = checkOrtStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
              if (code == ORT_OK) {
                values[i] = arr[0];
              } else {
                // bail out as exception has been thrown
                api->ReleaseValue(element);
                checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
                return NULL;
              }

              api->ReleaseValue(element);
            } else {
              // bail out as exception has been thrown
              checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
              return NULL;
            }
        }

        outputArray = (*jniEnv)->NewDoubleArray(jniEnv,safecast_size_t_to_jsize(count));
        (*jniEnv)->SetDoubleArrayRegion(jniEnv, outputArray,0,safecast_size_t_to_jsize(count),values);

        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));
      }
    }
    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxSequence_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}
