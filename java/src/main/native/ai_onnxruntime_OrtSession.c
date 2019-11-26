/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_OrtSession.h"

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    createSession
 * Signature: (JJLjava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_createSession__JJLjava_lang_String_2J
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong envHandle, jstring modelPath, jlong optsHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSession* session;

    jboolean copy;
    const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, modelPath, &copy);
    checkOrtStatus(jniEnv,api,api->CreateSession((OrtEnv*)envHandle, cPath, (OrtSessionOptions*)optsHandle, &session));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,modelPath,cPath);

    return (jlong) session;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    createSession
 * Signature: (JJ[BJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_createSession__JJ_3BJ
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong envHandle, jbyteArray jModelArray, jlong optsHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
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
 * Signature: (J)J
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
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getInputNames
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the number of inputs
    size_t numInputs = Java_ai_onnxruntime_OrtSession_getNumInputs(jniEnv, jobj, apiHandle, sessionHandle);

    char** inputNames;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(char*)*numInputs,(void**)&inputNames));
    for (uint32_t i = 0; i < numInputs; i++) {
        // Read out the input name and convert it to a java.lang.String
        char* inputName;
        checkOrtStatus(jniEnv,api,api->SessionGetInputName((OrtSession*)sessionHandle, i, allocator, &inputName));
        inputNames[i] = inputName;
    }

    return (jlong) inputNames;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getNumOutputs
 * Signature: (J)J
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
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_getOutputNames
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the number of outputs
    size_t numOutputs = Java_ai_onnxruntime_OrtSession_getNumOutputs(jniEnv, jobj, apiHandle, sessionHandle);

    char** outputNames;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(char*)*numOutputs,(void**)&outputNames));
    for (uint32_t i = 0; i < numOutputs; i++) {
        // Read out the output name and convert it to a java.lang.String
        char* outputName;
        checkOrtStatus(jniEnv,api,api->SessionGetOutputName((OrtSession*)sessionHandle, i, allocator, &outputName));
        outputNames[i] = outputName;
    }

    return (jlong) outputNames;
}

/*
 * Class:     ai_onnxruntime_OrtSession
 * Method:    getInputInfo
 * Signature: (JJ)[Lai/onnxruntime/NodeInfo;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getInputInfo
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong inputNamesHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    // Setup
    char *nodeInfoClassName = "ai/onnxruntime/NodeInfo";
    jclass nodeInfoClazz = (*jniEnv)->FindClass(jniEnv, nodeInfoClassName);
    jmethodID nodeInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,nodeInfoClazz, "<init>", "(Ljava/lang/String;Lai/onnxruntime/ValueInfo;)V");

    // Get the number of inputs
    size_t numInputs = Java_ai_onnxruntime_OrtSession_getNumInputs(jniEnv, jobj, apiHandle, sessionHandle);

    // Allocate the return array
    jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv,numInputs,nodeInfoClazz,NULL);
    for (uint32_t i = 0; i < numInputs; i++) {
        // Convert the input name to a java.lang.String
        jstring name = (*jniEnv)->NewStringUTF(jniEnv,((char**)inputNamesHandle)[i]);

        // Create a ValueInfo from the OrtTypeInfo
        OrtTypeInfo* typeInfo;
        checkOrtStatus(jniEnv,api,api->SessionGetInputTypeInfo((OrtSession*)sessionHandle, i, &typeInfo));
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
 * Method:    getOutputInfo
 * Signature: (JJ)[Lai/onnxruntime/NodeInfo;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_getOutputInfo
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong outputNamesHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    // Setup
    char *nodeInfoClassName = "ai/onnxruntime/NodeInfo";
    jclass nodeInfoClazz = (*jniEnv)->FindClass(jniEnv, nodeInfoClassName);
    jmethodID nodeInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, nodeInfoClazz, "<init>", "(Ljava/lang/String;Lai/onnxruntime/ValueInfo;)V");

    // Get the number of outputs
    size_t numOutputs = Java_ai_onnxruntime_OrtSession_getNumOutputs(jniEnv, jobj, apiHandle, sessionHandle);

    // Allocate the return array
    jobjectArray array = (*jniEnv)->NewObjectArray(jniEnv,numOutputs,nodeInfoClazz,NULL);
    for (uint32_t i = 0; i < numOutputs; i++) {
        // Convert the output name to a java.lang.String
        jstring name = (*jniEnv)->NewStringUTF(jniEnv,((char**)outputNamesHandle)[i]);

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
 * Signature: (JJJJ[JJJ)[Lai/onnxruntime/OnnxValue;
 * private native OnnxValue[] run(long nativeHandle, long allocatorHandle, long inputNamesHandle, long numInputs, long[] inputTensorHandles, long outputNamesHandle, long numOutputs) throws OrtException;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtSession_run
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong sessionHandle, jlong allocatorHandle, jlong inputNamesHandle, jlong numInputs, jlongArray tensorArr, jlong outputNamesHandle, jlong numOutputs) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract a C array of longs.
    jlong* inputTensors = (*jniEnv)->GetLongArrayElements(jniEnv,tensorArr,NULL);
    OrtValue** outputValues;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(OrtValue*)*numOutputs,(void**)&outputValues));
    for (int i = 0; i < numOutputs; i++) {
        outputValues[i] = NULL;
    }
    // Actually score the inputs.
    //printf("inputTensors = %p, first tensor = %p, numInputs = %ld, outputValues = %p, numOutputs = %ld\n",inputTensors,(OrtValue*)inputTensors[0],numInputs,outputValues,numOutputs);
    //ORT_API_STATUS(OrtRun, _Inout_ OrtSession* sess, _In_ OrtRunOptions* run_options, _In_ const char* const* input_names, _In_ const OrtValue* const* input, size_t input_len, _In_ const char* const* output_names, size_t output_names_len, _Out_ OrtValue** output);
    checkOrtStatus(jniEnv,api,api->Run((OrtSession*)sessionHandle, NULL, (const char* const*) inputNamesHandle, (const OrtValue* const*) inputTensors, numInputs, (const char* const*) outputNamesHandle, numOutputs, outputValues));
    // Release the C array.
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,tensorArr,inputTensors,JNI_ABORT);

    // Construct the output array of ONNXValues
    char *onnxValueClassName = "ai/onnxruntime/OnnxValue";
    jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, onnxValueClassName);
    jobjectArray outputArray = (*jniEnv)->NewObjectArray(jniEnv,numOutputs,onnxValueClass,NULL);

    // Convert the output tensors into ONNXValues
    for (int i = 0; i < numOutputs; i++) {
        if (outputValues[i] != NULL) {
            jobject onnxValue = convertOrtValueToONNXValue(jniEnv,api,allocator,outputValues[i]);
            (*jniEnv)->SetObjectArrayElement(jniEnv,outputArray,i,onnxValue);
        }
    }
    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,outputValues));

    return outputArray;
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
 * Method:    releaseNamesHandle
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_releaseNamesHandle
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jlong namesHandle, jlong length) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    char** names = (char**) namesHandle;
    for (uint32_t i = 0; i < length; i++) {
        checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,names[i]));
    }
    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,names));
}
