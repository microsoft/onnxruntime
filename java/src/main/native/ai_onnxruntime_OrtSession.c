/*
 * Copyright (c) 2019, 2020, 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtSession.h"

const char * const ORTJNI_StringClassName = "java/lang/String";
const char * const ORTJNI_OnnxValueClassName = "ai/onnxruntime/OnnxValue";
const char * const ORTJNI_NodeInfoClassName = "ai/onnxruntime/NodeInfo";
const char * const ORTJNI_MetadataClassName = "ai/onnxruntime/OnnxModelMetadata";


/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    createSession
 * Signature: (JJLjava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_createSession__JJLjava_lang_String_2J(JNIEnv* jniEnv, jclass jclazz, jlong apiHandle, jlong envHandle, jstring modelPath, jlong optsHandle) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSession* session = NULL;

#ifdef _WIN32
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, modelPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, modelPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, modelPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return 0;
  }
  wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
  checkOrtStatus(jniEnv, api,
                 api->CreateSession((OrtEnv*)envHandle, newString, (OrtSessionOptions*)optsHandle, &session));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv, modelPath, cPath);
#else
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, modelPath, NULL);
  checkOrtStatus(jniEnv, api, api->CreateSession((OrtEnv*)envHandle, cPath, (OrtSessionOptions*)optsHandle, &session));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, modelPath, cPath);
#endif

  return (jlong)session;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    createSession
 * Signature: (JJ[BJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_createSession__JJ_3BJ(JNIEnv* jniEnv, jclass jclazz, jlong apiHandle, jlong envHandle, jbyteArray jModelArray, jlong optsHandle) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtEnv* env = (OrtEnv*)envHandle;
  OrtSessionOptions* opts = (OrtSessionOptions*)optsHandle;
  OrtSession* session = NULL;

  size_t modelLength = (*jniEnv)->GetArrayLength(jniEnv, jModelArray);
  if (modelLength == 0) {
    throwOrtException(jniEnv, 2, "Invalid ONNX model, the byte array is zero length.");
    return 0;
  }

  // Get a reference to the byte array elements
  jbyte* modelArr = (*jniEnv)->GetByteArrayElements(jniEnv, jModelArray, NULL);
  checkOrtStatus(jniEnv, api, api->CreateSessionFromArray(env, modelArr, modelLength, opts, &session));
  // Release the C array.
  (*jniEnv)->ReleaseByteArrayElements(jniEnv, jModelArray, modelArr, JNI_ABORT);

  return (jlong)session;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getNumInputs
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getNumInputs(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  size_t numInputs = 0;
  checkOrtStatus(jniEnv, api, api->SessionGetInputCount((OrtSession*)handle, &numInputs));
  return numInputs;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getInputNames
 * Signature: (JJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getInputNames(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  OrtSession* session = (OrtSession*)sessionHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_StringClassName);

  // Get the number of inputs
  size_t numInputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->SessionGetInputCount(session, &numInputs));
  if (code != ORT_OK) {
    return NULL;
  }

  int32_t numInputsInt = (int32_t) numInputs;
  if (numInputs != (size_t) numInputsInt) {
    throwOrtException(jniEnv, 1, "Too many inputs, expected less than 2^31");
  }

  // Allocate the return array
  jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv, numInputsInt, stringClazz, NULL);
  for (int32_t i = 0; i < numInputsInt; i++) {
    // Read out the input name and convert it to a java.lang.String
    char* inputName = NULL;
    code = checkOrtStatus(jniEnv, api, api->SessionGetInputName(session, i, allocator, &inputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
    jstring name = (*jniEnv)->NewStringUTF(jniEnv, inputName);
    (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, name);
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, inputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
  }

  return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getNumOutputs
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getNumOutputs(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  size_t numOutputs = 0;
  checkOrtStatus(jniEnv, api, api->SessionGetOutputCount((OrtSession*)handle, &numOutputs));
  return numOutputs;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getOutputNames
 * Signature: (JJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getOutputNames(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSession* session = (OrtSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_StringClassName);

  // Get the number of outputs
  size_t numOutputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->SessionGetOutputCount(session, &numOutputs));
  if (code != ORT_OK) {
    return NULL;
  }

  int32_t numOutputsInt = (int32_t) numOutputs;
  if (numOutputs != (size_t) numOutputsInt) {
    throwOrtException(jniEnv, 1, "Too many outputs, expected less than 2^31");
  }

  // Allocate the return array
  jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv, numOutputsInt, stringClazz, NULL);
  for (int32_t i = 0; i < numOutputsInt; i++) {
    // Read out the output name and convert it to a java.lang.String
    char* outputName = NULL;
    code = checkOrtStatus(jniEnv, api, api->SessionGetOutputName(session, i, allocator, &outputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
    jstring name = (*jniEnv)->NewStringUTF(jniEnv, outputName);
    (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, name);
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, outputName));
    if (code != ORT_OK) {
      // break out on error, return array and let Java throw the exception.
      break;
    }
  }

  return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getInputInfo
 * Signature: (JJJ)[Lai/onnxruntime/NodeInfo;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getInputInfo(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSession* session = (OrtSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass nodeInfoClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_NodeInfoClassName);
  jmethodID nodeInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, nodeInfoClazz, "<init>",
                                                         "(Ljava/lang/String;Lai/onnxruntime/ValueInfo;)V");

  // Get the number of inputs
  size_t numInputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->SessionGetInputCount(session, &numInputs));
  if (code != ORT_OK) {
    return NULL;
  }

  // Allocate the return array
  jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv, safecast_size_t_to_jsize(numInputs), nodeInfoClazz, NULL);
  for (size_t i = 0; i < numInputs; i++) {
    // Read out the input name and convert it to a java.lang.String
    char* inputName = NULL;
    code = checkOrtStatus(jniEnv, api, api->SessionGetInputName(session, i, allocator, &inputName));
    if (code != ORT_OK) {
      break;
    }
    jstring name = (*jniEnv)->NewStringUTF(jniEnv, inputName);
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, inputName));
    if (code != ORT_OK) {
      break;
    }

    // Create a ValueInfo from the OrtTypeInfo
    OrtTypeInfo* typeInfo = NULL;
    code = checkOrtStatus(jniEnv, api, api->SessionGetInputTypeInfo(session, i, &typeInfo));
    if (code != ORT_OK) {
      break;
    }
    jobject valueInfoJava = convertToValueInfo(jniEnv, api, typeInfo);
    api->ReleaseTypeInfo(typeInfo);
    if (valueInfoJava == NULL) {
      break;
    }

    // Create a NodeInfo and assign into the array
    jobject nodeInfo = (*jniEnv)->NewObject(jniEnv, nodeInfoClazz, nodeInfoConstructor, name, valueInfoJava);
    (*jniEnv)->SetObjectArrayElement(jniEnv, array, safecast_size_t_to_jsize(i), nodeInfo);
  }

  return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getOutputInfo
 * Signature: (JJJ)[Lai/onnxruntime/NodeInfo;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getOutputInfo(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSession* session = (OrtSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass nodeInfoClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_NodeInfoClassName);
  jmethodID nodeInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, nodeInfoClazz, "<init>",
                                                         "(Ljava/lang/String;Lai/onnxruntime/ValueInfo;)V");

  // Get the number of outputs
  size_t numOutputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->SessionGetOutputCount(session, &numOutputs));
  if (code != ORT_OK) {
    return NULL;
  }

  // Allocate the return array
  jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv, safecast_size_t_to_jsize(numOutputs), nodeInfoClazz, NULL);
  for (uint32_t i = 0; i < numOutputs; i++) {
    // Read out the output name and convert it to a java.lang.String
    char* outputName = NULL;
    code = checkOrtStatus(jniEnv, api, api->SessionGetOutputName(session, i, allocator, &outputName));
    if (code != ORT_OK) {
      break;
    }
    jstring name = (*jniEnv)->NewStringUTF(jniEnv, outputName);
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, outputName));
    if (code != ORT_OK) {
      break;
    }

    // Create a ValueInfo from the OrtTypeInfo
    OrtTypeInfo* typeInfo = NULL;
    code = checkOrtStatus(jniEnv, api, api->SessionGetOutputTypeInfo(session, i, &typeInfo));
    if (code != ORT_OK) {
      break;
    }
    jobject valueInfoJava = convertToValueInfo(jniEnv, api, typeInfo);
    api->ReleaseTypeInfo(typeInfo);
    if (valueInfoJava == NULL) {
      break;
    }

    // Create a NodeInfo and assign into the array
    jobject nodeInfo = (*jniEnv)->NewObject(jniEnv, nodeInfoClazz, nodeInfoConstructor, name, valueInfoJava);
    (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, nodeInfo);
  }

  return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    run
 * Signature: (JJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 * private native OnnxValue[] run(long apiHandle, long nativeHandle, long allocatorHandle, String[] inputNamesArray, long[] inputs, long numInputs, String[] outputNamesArray, long numOutputs)
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_run(JNIEnv* jniEnv, jobject jobj, jlong apiHandle,
                                                                  jlong sessionHandle, jlong allocatorHandle,
                                                                  jobjectArray inputNamesArr, jlongArray tensorArr,
                                                                  jlong numInputs, jobjectArray outputNamesArr,
                                                                  jlong numOutputs, jlong runOptionsHandle) {

  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  OrtSession* session = (OrtSession*)sessionHandle;
  OrtRunOptions* runOptions = (OrtRunOptions*)runOptionsHandle;

  jobjectArray outputArray = NULL;

  // Create the buffers for the Java input & output strings, and the input pointers
  const char** inputNames = allocarray(numInputs, sizeof(char*));
  if (inputNames == NULL) {
    // Nothing to cleanup, return and throw exception
    return outputArray;
  }
  const char** outputNames = allocarray(numOutputs, sizeof(char*));
  if (outputNames == NULL) {
    goto cleanup_input_names;
  }
  jobject* javaInputStrings = allocarray(numInputs, sizeof(jobject));
  if (javaInputStrings == NULL) {
    goto cleanup_output_names;
  }
  jobject* javaOutputStrings = allocarray(numOutputs, sizeof(jobject));
  if (javaOutputStrings == NULL) {
    goto cleanup_java_input_strings;
  }
  const OrtValue** inputValuePtrs = allocarray(numInputs, sizeof(OrtValue*));
  if (inputValuePtrs == NULL) {
    goto cleanup_java_output_strings;
  }
  OrtValue** outputValues = allocarray(numOutputs, sizeof(OrtValue*));
  if (outputValues == NULL) {
    goto cleanup_input_values;
  }

  // Extract a C array of longs which are pointers to the input tensors.
  // The Java-side objects store native pointers as 64-bit longs, and on 32-bit systems
  // we cannot cast the long array to a pointer array as they are different sizes,
  // so we copy the longs applying the appropriate cast.
  jlong* inputValueLongs = (*jniEnv)->GetLongArrayElements(jniEnv, tensorArr, NULL);

  // Extract the names and native pointers of the input values.
  for (int i = 0; i < numInputs; i++) {
    javaInputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, inputNamesArr, i);
    inputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaInputStrings[i], NULL);
    inputValuePtrs[i] = (OrtValue*)inputValueLongs[i];
  }

  // Release the java array copy of pointers to the tensors.
  (*jniEnv)->ReleaseLongArrayElements(jniEnv, tensorArr, inputValueLongs, JNI_ABORT);

  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
    outputValues[i] = NULL;
  }

  // Actually score the inputs.
  // ORT_API_STATUS(OrtRun, _Inout_ OrtSession* sess, _In_ OrtRunOptions* run_options,
  // _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len,
  // _In_ const char* const* output_names, size_t output_names_len, _Out_ OrtValue** output);
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->Run(session, runOptions, (const char* const*)inputNames,
                                              (const OrtValue* const*)inputValuePtrs, numInputs,
                                              (const char* const*)outputNames, numOutputs, outputValues));
  if (code != ORT_OK) {
    goto cleanup_output_values;
  }

  // Construct the output array of ONNXValues
  jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, ORTJNI_OnnxValueClassName);
  outputArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_int64_to_jsize(numOutputs), onnxValueClass, NULL);

  // Convert the output tensors into ONNXValues
  for (int i = 0; i < numOutputs; i++) {
    if (outputValues[i] != NULL) {
      jobject onnxValue = convertOrtValueToONNXValue(jniEnv, api, allocator, outputValues[i]);
      if (onnxValue == NULL) {
        break;  // go to cleanup, exception thrown
      }
      (*jniEnv)->SetObjectArrayElement(jniEnv, outputArray, i, onnxValue);
    }
  }

  // Note these gotos are in a specific order so they mirror the allocation pattern above.
  // They must be changed if the allocation code is rearranged.
cleanup_output_values:
  free(outputValues);

  // Release the Java output strings
  for (int i = 0; i < numOutputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaOutputStrings[i], outputNames[i]);
  }

  // Release the Java input strings
  for (int i = 0; i < numInputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaInputStrings[i], inputNames[i]);
  }

  // Release the buffers
cleanup_input_values:
  free((void*)inputValuePtrs);
cleanup_java_output_strings:
  free(javaOutputStrings);
cleanup_java_input_strings:
  free(javaInputStrings);
cleanup_output_names:
  free((void*)outputNames);
cleanup_input_names:
  free((void*)inputNames);

  return outputArray;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getProfilingStartTimeInNs
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getProfilingStartTimeInNs(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSession* session = (OrtSession*)sessionHandle;

  uint64_t timestamp = 0;
  checkOrtStatus(jniEnv, api, api->SessionGetProfilingStartTimeNs(session, &timestamp));
  return (jlong)timestamp;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    endProfiling
 * Signature: (JJJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtSession_endProfiling(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  char* profileStr = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->SessionEndProfiling((OrtSession*)handle, allocator,
                                                                           &profileStr));
  if (code != ORT_OK) {
    return NULL;
  }
  jstring profileOutput = (*jniEnv)->NewStringUTF(jniEnv, profileStr);
  checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, profileStr));
  return profileOutput;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    closeSession
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_closeSession(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  api->ReleaseSession((OrtSession*)handle);
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    constructMetadata
 * Signature: (JJJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtSession_constructMetadata(JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  jobject metadataJava = NULL;
  jstring producerStr = NULL;
  jstring graphStr = NULL;
  jstring graphDescStr = NULL;
  jstring domainStr = NULL;
  jstring descriptionStr = NULL;

  // macro for processing char* into a Java UTF-8 string with error handling inside this function
#define STR_PROCESS(STR_NAME) \
  if (code == ORT_OK) { \
    STR_NAME = (*jniEnv)->NewStringUTF(jniEnv, charBuffer); \
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, charBuffer)); \
    if (code != ORT_OK) { \
      goto release_metadata; \
    } \
  } else { \
    goto release_metadata; \
  }

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_StringClassName);
  jclass metadataClazz = (*jniEnv)->FindClass(jniEnv, ORTJNI_MetadataClassName);
  // OnnxModelMetadata(String producerName, String graphName, String domain, String description,
  //                   long version, String[] customMetadataArray)
  jmethodID metadataConstructor = (*jniEnv)->GetMethodID(
      jniEnv, metadataClazz, "<init>",
      "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;J[Ljava/lang/String;)V");

  // Get metadata
  OrtModelMetadata* metadata = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->SessionGetModelMetadata((OrtSession*)nativeHandle, &metadata));
  if (code != ORT_OK) {
    // Nothing to cleanup, return null as an exception has been thrown
    return NULL;
  }

  // Read out the producer name and convert it to a java.lang.String
  char* charBuffer = NULL;
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetProducerName(metadata, allocator, &charBuffer));
  STR_PROCESS(producerStr)

  // Read out the graph name and convert it to a java.lang.String
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetGraphName(metadata, allocator, &charBuffer));
  STR_PROCESS(graphStr)

  // Read out the graph description and convert it to a java.lang.String
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetGraphDescription(metadata, allocator, &charBuffer));
  STR_PROCESS(graphDescStr)

  // Read out the domain and convert it to a java.lang.String
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetDomain(metadata, allocator, &charBuffer));
  STR_PROCESS(domainStr)

  // Read out the description and convert it to a java.lang.String
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetDescription(metadata, allocator, &charBuffer));
  STR_PROCESS(descriptionStr)

  // Read out the version
  int64_t version = 0;
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetVersion(metadata, &version));
  if (code != ORT_OK) {
    goto release_metadata;
  }

  // Read out the keys, look up the values.
  int64_t numKeys = 0;
  char** keys = NULL;
  code = checkOrtStatus(jniEnv, api, api->ModelMetadataGetCustomMetadataMapKeys(metadata, allocator, &keys, &numKeys));
  if (code != ORT_OK) {
    goto release_metadata;
  }
  jobjectArray customArray = NULL;
  if (numKeys > 0) {
    customArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_int64_to_jsize(numKeys * 2), stringClazz, NULL);

    // Iterate key array to extract the values
    for (int64_t i = 0; i < numKeys; i++) {
      // Create a java.lang.String for the key
      jstring keyJava = (*jniEnv)->NewStringUTF(jniEnv, keys[i]);

      // Extract the value and convert it to a java.lang.String
      code = checkOrtStatus(jniEnv, api, api->ModelMetadataLookupCustomMetadataMap(metadata, allocator, keys[i], &charBuffer));
      jstring valueJava = NULL;
      if (code == ORT_OK) {
        valueJava = (*jniEnv)->NewStringUTF(jniEnv, charBuffer);
        code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, charBuffer));
        if (code != ORT_OK) {
          // Signal that custom metadata extraction failed and break out
          customArray = NULL;
          break;
        }
      } else {
        // Signal that custom metadata extraction failed and break out
        customArray = NULL;
        break;
      }

      // Write the key and value into the array
      (*jniEnv)->SetObjectArrayElement(jniEnv, customArray, safecast_int64_to_jsize(i * 2), keyJava);
      (*jniEnv)->SetObjectArrayElement(jniEnv, customArray, safecast_int64_to_jsize((i * 2) + 1), valueJava);
    }

    // Release key array
    for (int64_t i = 0; i < numKeys; i++) {
      code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, keys[i]));
      if (code != ORT_OK) {
        customArray = NULL;
        break;
      }
    }
    code = checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, keys));
    if (code != ORT_OK) {
      customArray = NULL;
    }
  } else {
    customArray = (*jniEnv)->NewObjectArray(jniEnv, 0, stringClazz, NULL);
  }

  if (customArray != NULL) {
    // If the array is non-null then the custom metadata extraction completed successfully so
    // we invoke the metadata constructor
    // OnnxModelMetadata(String producerName, String graphName, String graphDescription, String domain,
    //                   String description, long version, String[] customMetadataArray)
    metadataJava = (*jniEnv)->NewObject(jniEnv, metadataClazz, metadataConstructor,
                                        producerStr, graphStr, graphDescStr, domainStr, descriptionStr, (jlong)version,
                                        customArray);
  }

release_metadata:
  // Release the metadata
  api->ReleaseModelMetadata(metadata);

  return metadataJava;
}
