/*
 * Copyright (c) 2019, 2020 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtSession.h"

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    createSession
 * Signature: (JJLjava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_createSession__JJLjava_lang_String_2J
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong envHandle, jstring modelPath, jlong optsHandle) {
    (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSession* session;

#ifdef _WIN32
    const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, modelPath, NULL);
    size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, modelPath);
    wchar_t* newString = (wchar_t*)calloc(stringLength+1,sizeof(jchar));
    wcsncpy_s(newString, stringLength+1, (const wchar_t*) cPath, stringLength);
    checkOrtStatus(jniEnv,api,api->CreateSession((OrtEnv*)envHandle, (const wchar_t*)newString, (OrtSessionOptions*)optsHandle, &session));
    free(newString);
    (*jniEnv)->ReleaseStringChars(jniEnv,modelPath,cPath);
#else
    const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, modelPath, NULL);
    checkOrtStatus(jniEnv,api,api->CreateSession((OrtEnv*)envHandle, cPath, (OrtSessionOptions*)optsHandle, &session));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,modelPath,cPath);
#endif

    return (jlong) session;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    createSession
 * Signature: (JJ[BJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_createSession__JJ_3BJ
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong envHandle, jbyteArray jModelArray, jlong optsHandle) {
    (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSession* session;

    // Get a reference to the byte array elements
    jbyte* modelArr = (*jniEnv)->GetByteArrayElements(jniEnv,jModelArray,NULL);
    size_t modelLength = (*jniEnv)->GetArrayLength(jniEnv,jModelArray);
    checkOrtStatus(jniEnv,api,api->CreateSessionFromArray((OrtEnv*)envHandle, modelArr, modelLength, (OrtSessionOptions*)optsHandle, &session));
    // Release the C array.
    (*jniEnv)->ReleaseByteArrayElements(jniEnv,jModelArray,modelArr,JNI_ABORT);

    return (jlong) session;
  }

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getNumInputs
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getNumInputs
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    size_t numInputs;
    checkOrtStatus(jniEnv,api,api->SessionGetInputCount((OrtSession*)handle, &numInputs));
    return numInputs;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getInputNames
 * Signature: (JJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getInputNames
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Setup
    char *stringClassName = "java/lang/String";
    jclass stringClazz = (*jniEnv)->FindClass(jniEnv, stringClassName);

    // Get the number of inputs
    size_t numInputs = Java_ai_onnxruntime_OrtSession_getNumInputs(jniEnv, jobj, apiHandle, sessionHandle);

    // Allocate the return array
    jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv,safecast_size_t_to_jsize(numInputs),stringClazz,NULL);
    for (uint32_t i = 0; i < numInputs; i++) {
        // Read out the input name and convert it to a java.lang.String
        char* inputName;
        checkOrtStatus(jniEnv,api,api->SessionGetInputName((OrtSession*)sessionHandle, i, allocator, &inputName));
        jstring name = (*jniEnv)->NewStringUTF(jniEnv,inputName);
        (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, name);
        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,inputName));
    }

    return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getNumOutputs
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getNumOutputs
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    size_t numOutputs;
    checkOrtStatus(jniEnv,api,api->SessionGetOutputCount((OrtSession*)handle, &numOutputs));
    return numOutputs;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getOutputNames
 * Signature: (JJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getOutputNames
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Setup
    char *stringClassName = "java/lang/String";
    jclass stringClazz = (*jniEnv)->FindClass(jniEnv, stringClassName);

    // Get the number of outputs
    size_t numOutputs = Java_ai_onnxruntime_OrtSession_getNumOutputs(jniEnv, jobj, apiHandle, sessionHandle);

    // Allocate the return array
    jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv,safecast_size_t_to_jsize(numOutputs),stringClazz, NULL);
    for (uint32_t i = 0; i < numOutputs; i++) {
        // Read out the output name and convert it to a java.lang.String
        char* outputName;
        checkOrtStatus(jniEnv,api,api->SessionGetOutputName((OrtSession*)sessionHandle, i, allocator, &outputName));
        jstring name = (*jniEnv)->NewStringUTF(jniEnv,outputName);
        (*jniEnv)->SetObjectArrayElement(jniEnv, array, i, name);
        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,outputName));
    }

    return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getInputInfo
 * Signature: (JJJ)[Lai/onnxruntime/NodeInfo;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getInputInfo
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Setup
    char *nodeInfoClassName = "ai/onnxruntime/NodeInfo";
    jclass nodeInfoClazz = (*jniEnv)->FindClass(jniEnv, nodeInfoClassName);
    jmethodID nodeInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,nodeInfoClazz, "<init>", "(Ljava/lang/String;Lai/onnxruntime/ValueInfo;)V");

    // Get the number of inputs
    size_t numInputs = Java_ai_onnxruntime_OrtSession_getNumInputs(jniEnv, jobj, apiHandle, sessionHandle);

    // Allocate the return array
    jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv,safecast_size_t_to_jsize(numInputs),nodeInfoClazz, NULL);
    for (size_t i = 0; i < numInputs; i++) {
        // Read out the input name and convert it to a java.lang.String
        char* inputName;
        checkOrtStatus(jniEnv,api,api->SessionGetInputName((OrtSession*)sessionHandle, i, allocator, &inputName));
        jstring name = (*jniEnv)->NewStringUTF(jniEnv,inputName);
        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,inputName));

        // Create a ValueInfo from the OrtTypeInfo
        OrtTypeInfo* typeInfo;
        checkOrtStatus(jniEnv,api,api->SessionGetInputTypeInfo((OrtSession*)sessionHandle, i, &typeInfo));
        jobject valueInfoJava = convertToValueInfo(jniEnv,api,typeInfo);
        api->ReleaseTypeInfo(typeInfo);

        // Create a NodeInfo and assign into the array
        jobject nodeInfo = (*jniEnv)->NewObject(jniEnv, nodeInfoClazz, nodeInfoConstructor, name, valueInfoJava);
        (*jniEnv)->SetObjectArrayElement(jniEnv, array,safecast_size_t_to_jsize(i),nodeInfo);
    }

    return array;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getOutputInfo
 * Signature: (JJJ)[Lai/onnxruntime/NodeInfo;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getOutputInfo
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Setup
    char *nodeInfoClassName = "ai/onnxruntime/NodeInfo";
    jclass nodeInfoClazz = (*jniEnv)->FindClass(jniEnv, nodeInfoClassName);
    jmethodID nodeInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, nodeInfoClazz, "<init>", "(Ljava/lang/String;Lai/onnxruntime/ValueInfo;)V");

    // Get the number of outputs
    size_t numOutputs = Java_ai_onnxruntime_OrtSession_getNumOutputs(jniEnv, jobj, apiHandle, sessionHandle);

    // Allocate the return array
    jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv,safecast_size_t_to_jsize(numOutputs),nodeInfoClazz,NULL);
    for (uint32_t i = 0; i < numOutputs; i++) {
        // Read out the output name and convert it to a java.lang.String
        char* outputName;
        checkOrtStatus(jniEnv,api,api->SessionGetOutputName((OrtSession*)sessionHandle, i, allocator, &outputName));
        jstring name = (*jniEnv)->NewStringUTF(jniEnv,outputName);
        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,outputName));

        // Create a ValueInfo from the OrtTypeInfo
        OrtTypeInfo* typeInfo;
        checkOrtStatus(jniEnv,api,api->SessionGetOutputTypeInfo((OrtSession*)sessionHandle, i, &typeInfo));
        jobject valueInfoJava = convertToValueInfo(jniEnv,api,typeInfo);
        api->ReleaseTypeInfo(typeInfo);

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
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_run
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle, jobjectArray inputNamesArr, jlongArray tensorArr, jlong numInputs, jobjectArray outputNamesArr, jlong numOutputs, jlong runOptionsHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    OrtSession* session = (OrtSession*) sessionHandle;
    OrtRunOptions* runOptions = (OrtRunOptions*) runOptionsHandle;

    // Create the buffers for the Java input and output strings
    const char** inputNames;
    checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(char*)*numInputs,(void**)&inputNames));
    const char** outputNames;
    checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(char*)*numOutputs,(void**)&outputNames));
    jobject* javaInputStrings;
    checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(jobject)*numInputs,(void**)&javaInputStrings));
    jobject* javaOutputStrings;
    checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(jobject)*numOutputs,(void**)&javaOutputStrings));

    // Extract the names of the input values.
    for (int i = 0; i < numInputs; i++) {
        javaInputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv,inputNamesArr,i);
        inputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv,javaInputStrings[i],NULL);
    }

    // Extract a C array of longs which are pointers to the input tensors.
    jlong* inputTensors = (*jniEnv)->GetLongArrayElements(jniEnv,tensorArr,NULL);

    // Extract the names of the output values, and allocate their output array.
    OrtValue** outputValues;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(OrtValue*)*numOutputs,(void**)&outputValues));
    for (int i = 0; i < numOutputs; i++) {
        javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv,outputNamesArr,i);
        outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv,javaOutputStrings[i],NULL);
        outputValues[i] = NULL;
    }

    // Actually score the inputs.
    //printf("inputTensors = %p, first tensor = %p, numInputs = %ld, outputValues = %p, numOutputs = %ld\n",inputTensors,(OrtValue*)inputTensors[0],numInputs,outputValues,numOutputs);
    //ORT_API_STATUS(OrtRun, _Inout_ OrtSession* sess, _In_ OrtRunOptions* run_options, _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len, _In_ const char* const* output_names, size_t output_names_len, _Out_ OrtValue** output);
    checkOrtStatus(jniEnv,api,api->Run(session, runOptions, (const char* const*) inputNames, (const OrtValue* const*) inputTensors, numInputs, (const char* const*) outputNames, numOutputs, outputValues));
    // Release the C array of pointers to the tensors.
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,tensorArr,inputTensors,JNI_ABORT);

    // Construct the output array of ONNXValues
    char *onnxValueClassName = "ai/onnxruntime/OnnxValue";
    jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, onnxValueClassName);
    jobjectArray outputArray = (*jniEnv)->NewObjectArray(jniEnv,safecast_int64_to_jsize(numOutputs), onnxValueClass, NULL);

    // Convert the output tensors into ONNXValues and release the output strings.
    for (int i = 0; i < numOutputs; i++) {
        if (outputValues[i] != NULL) {
            jobject onnxValue = convertOrtValueToONNXValue(jniEnv,api,allocator,outputValues[i]);
            (*jniEnv)->SetObjectArrayElement(jniEnv,outputArray,i,onnxValue);
        }
        (*jniEnv)->ReleaseStringUTFChars(jniEnv,javaOutputStrings[i],outputNames[i]);
    }
    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,outputValues));

    // Release the Java input strings
    for (int i = 0; i < numInputs; i++) {
        (*jniEnv)->ReleaseStringUTFChars(jniEnv,javaInputStrings[i],inputNames[i]);
    }

    // Release the buffers
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, (void*)inputNames));
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, (void*)outputNames));
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, javaInputStrings));
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, javaOutputStrings));

    return outputArray;
}


/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getProfilingStartTimeInNs
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getProfilingStartTimeInNs
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle) {
  (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtSession* session = (OrtSession*) sessionHandle;

  uint64_t timestamp = 0;

  checkOrtStatus(jniEnv,api,api->SessionGetProfilingStartTimeNs(session,&timestamp));
  return (jlong) timestamp;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    endProfiling
 * Signature: (JJJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtSession_endProfiling
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

  char* profileStr;
  checkOrtStatus(jniEnv,api,api->SessionEndProfiling((OrtSession*)handle,allocator,&profileStr));
  jstring profileOutput = (*jniEnv)->NewStringUTF(jniEnv,profileStr);
  checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,profileStr));
  return profileOutput;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    closeSession
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_closeSession
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseSession((OrtSession*)handle);
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    constructMetadata
 * Signature: (JJJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtSession_constructMetadata
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle, jlong allocatorHandle) {
  (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

  // Setup
  char* stringClassName = "java/lang/String";
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, stringClassName);
  char *metadataClassName = "ai/onnxruntime/OnnxModelMetadata";
  jclass metadataClazz = (*jniEnv)->FindClass(jniEnv, metadataClassName);
  //OnnxModelMetadata(String producerName, String graphName, String domain, String description, long version, String[] customMetadataArray)
  jmethodID metadataConstructor = (*jniEnv)->GetMethodID(jniEnv, metadataClazz, "<init>",
                                                         "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;J[Ljava/lang/String;)V");

  // Get metadata
  OrtModelMetadata* metadata;
  checkOrtStatus(jniEnv,api,api->SessionGetModelMetadata((OrtSession*)nativeHandle,&metadata));

  // Read out the producer name and convert it to a java.lang.String
  char* charBuffer;
  checkOrtStatus(jniEnv,api,api->ModelMetadataGetProducerName(metadata, allocator, &charBuffer));
  jstring producerStr = (*jniEnv)->NewStringUTF(jniEnv,charBuffer);
  checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,charBuffer));

  // Read out the graph name and convert it to a java.lang.String
  checkOrtStatus(jniEnv,api,api->ModelMetadataGetGraphName(metadata, allocator, &charBuffer));
  jstring graphStr = (*jniEnv)->NewStringUTF(jniEnv,charBuffer);
  checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,charBuffer));

  // Read out the domain and convert it to a java.lang.String
  checkOrtStatus(jniEnv,api,api->ModelMetadataGetDomain(metadata, allocator, &charBuffer));
  jstring domainStr = (*jniEnv)->NewStringUTF(jniEnv,charBuffer);
  checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,charBuffer));

  // Read out the description and convert it to a java.lang.String
  checkOrtStatus(jniEnv,api,api->ModelMetadataGetDescription(metadata, allocator, &charBuffer));
  jstring descriptionStr = (*jniEnv)->NewStringUTF(jniEnv,charBuffer);
  checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,charBuffer));

  // Read out the version
  int64_t version;
  checkOrtStatus(jniEnv,api,api->ModelMetadataGetVersion(metadata, &version));

  // Read out the keys, look up the values.
  int64_t numKeys;
  char** keys;
  checkOrtStatus(jniEnv,api,api->ModelMetadataGetCustomMetadataMapKeys(metadata, allocator, &keys, &numKeys));
  jobjectArray customArray = NULL;
  if (numKeys > 0) {
    customArray = (*jniEnv)->NewObjectArray(jniEnv,safecast_int64_to_jsize(numKeys * 2),stringClazz, NULL);

    // Iterate key array to extract the values
    for (int64_t i = 0; i < numKeys; i++) {
      // Create a java.lang.String for the key
      jstring keyJava = (*jniEnv)->NewStringUTF(jniEnv,keys[i]);

      // Extract the value and convert it to a java.lang.String
      checkOrtStatus(jniEnv,api,api->ModelMetadataLookupCustomMetadataMap(metadata,allocator,keys[i],&charBuffer));
      jstring valueJava = (*jniEnv)->NewStringUTF(jniEnv,charBuffer);
      checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,charBuffer));

      // Write the key and value into the array
      (*jniEnv)->SetObjectArrayElement(jniEnv,customArray,safecast_int64_to_jsize(i*2),keyJava);
      (*jniEnv)->SetObjectArrayElement(jniEnv,customArray,safecast_int64_to_jsize((i * 2) + 1),valueJava);
    }

    // Release key array
    for (int64_t i = 0; i < numKeys; i++) {
      checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,keys[i]));
    }
    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,keys));
  } else {
    customArray = (*jniEnv)->NewObjectArray(jniEnv,0,stringClazz,NULL);
  }

  // Invoke the metadata constructor
  //OnnxModelMetadata(String producerName, String graphName, String domain, String description, long version, String[] customMetadataArray)
  jobject metadataJava = (*jniEnv)->NewObject(jniEnv, metadataClazz, metadataConstructor, producerStr, graphStr, domainStr, descriptionStr, (jlong) version, customArray);

  // Release the metadata
  api->ReleaseModelMetadata(metadata);

  return metadataJava;
}
