/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include "OrtJniUtil.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "ai_onnxruntime_OrtTrainingSession.h"

#ifdef _WIN32
wchar_t* copyAndPad(JNIEnv * jniEnv, jstring javaStr) {
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  const jchar* charArr = (*jniEnv)->GetStringChars(jniEnv, javaStr, NULL);
  size_t strLength = (*jniEnv)->GetStringLength(jniEnv, javaStr);
  wchar_t* outputStr = (wchar_t*)calloc(strLength + 1, sizeof(wchar_t));
  if (outputStr != NULL) {
    wcsncpy_s(outputStr, strLength + 1, (const wchar_t*)charArr, strLength);
  } else {
    throwOrtException(jniEnv, 1, "Not enough memory");
  }
  (*jniEnv)->ReleaseStringChars(jniEnv, javaStr, charArr);
  return outputStr;
}
#endif

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    createTrainingSession
 * Signature: (JJJJJLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_createTrainingSession
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong trainApiHandle,
     jlong envHandle, jlong optionsHandle, jlong checkpointHandle,
     jstring trainPath, jstring evalPath, jstring  optimizerPath) {
  (void) clazz; // Required JNI parameters not needed by functions which don't need to access their host class.

  // evalPath and optimizerPath could be NULL, as that is used to signal that those models
  // should not be loaded, which induces some juggling to avoid calling JNI methods with a NULL
  // pointer. trainPath cannot be null, as in that case a Java exception is thrown before this
  // method is called.

  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*) trainApiHandle;
  const OrtEnv* env = (const OrtEnv*) envHandle;
  const OrtSessionOptions* options = (const OrtSessionOptions*) optionsHandle;
  OrtCheckpointState* checkpoint = (OrtCheckpointState*) checkpointHandle;

  OrtTrainingSession* session = NULL;

#ifdef _WIN32
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  wchar_t* trainStr = copyAndPad(jniEnv, trainPath);
  if (trainStr == NULL) {
    // nothing to cleanup, return zero as exception has been thrown in Java
    return 0L;
  }
  wchar_t* evalStr = NULL;
  if (evalPath != NULL) {
    evalStr = copyAndPad(jniEnv, evalPath);
    if (evalStr == NULL) {
      // exception has been thrown in Java, go to cleanup and return null.
      goto cleanupTrain;
    }
  }
  wchar_t* optimizerStr = NULL;
  if (optimizerPath == NULL) {
    optimizerStr = copyAndPad(jniEnv, optimizerPath);
    if (optimizerStr == NULL) {
      // exception has been thrown in Java, go to cleanup and return null.
      goto cleanupEval;
    }
  }
  checkOrtStatus(jniEnv, api, trainApi->CreateTrainingSession(env, options, checkpoint, trainStr, evalStr, optimizerStr, &session));
  if (optimizerStr != NULL) {
    free(optimizerStr);
  }
cleanupEval:
  if (evalStr != NULL) {
    free(evalStr);
  }
cleanupTrain:
  free(trainStr);
#else
  // GetStringUTFChars is null terminated, so can be used directly
  const char* trainStr = (*jniEnv)->GetStringUTFChars(jniEnv, trainPath, NULL);
  const char* evalStr = evalPath == NULL ? NULL : (*jniEnv)->GetStringUTFChars(jniEnv, evalPath, NULL);
  const char* optimizerStr = optimizerPath == NULL ? NULL : (*jniEnv)->GetStringUTFChars(jniEnv, optimizerPath, NULL);
  checkOrtStatus(jniEnv, api, trainApi->CreateTrainingSession(env, options, checkpoint, trainStr, evalStr, optimizerStr, &session));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, trainPath, trainStr);
  if (evalPath != NULL) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, evalPath, evalStr);
  }
  if (optimizerPath != NULL) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, optimizerPath, optimizerStr);
  }
#endif

  return (jlong) session;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    closeSession
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_closeSession
    (JNIEnv * jniEnv, jobject jobj, jlong trainHandle, jlong nativeHandle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainHandle;
  trainApi->ReleaseTrainingSession((OrtTrainingSession*)nativeHandle);
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getTrainInputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getTrainInputNames
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  const OrtTrainingSession* trainSession = (const OrtTrainingSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, "java/lang/String");

  // Get the number of inputs
  size_t numInputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetTrainingModelInputCount(trainSession, &numInputs));
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
    code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetTrainingModelInputName(trainSession, i, allocator, &inputName));
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
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getTrainOutputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getTrainOutputNames
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  const OrtTrainingSession* trainSession = (const OrtTrainingSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, "java/lang/String");

  // Get the number of outputs
  size_t numOutputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetTrainingModelOutputCount(trainSession, &numOutputs));
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
    code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetTrainingModelOutputName(trainSession, i, allocator, &outputName));
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
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getEvalInputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getEvalInputNames
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  const OrtTrainingSession* trainSession = (const OrtTrainingSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, "java/lang/String");

  // Get the number of inputs
  size_t numInputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetEvalModelInputCount(trainSession, &numInputs));
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
    code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetEvalModelInputName(trainSession, i, allocator, &inputName));
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
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getEvalOutputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getEvalOutputNames
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong sessionHandle, jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  const OrtTrainingSession* trainSession = (const OrtTrainingSession*)sessionHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  // Setup
  jclass stringClazz = (*jniEnv)->FindClass(jniEnv, "java/lang/String");

  // Get the number of outputs
  size_t numOutputs = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetEvalModelOutputCount(trainSession, &numOutputs));
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
    code = checkOrtStatus(jniEnv, api, trainApi->TrainingSessionGetEvalModelOutputName(trainSession, i, allocator, &outputName));
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
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    lazyResetGrad
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_lazyResetGrad
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->LazyResetGrad(trainSession));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    trainStep
 * Signature: (JJJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_trainStep
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle,
     jlong nativeHandle, jlong allocatorHandle, jobjectArray inputNamesArr, jlongArray inputHandles, jlong numInputs,
     jobjectArray outputNamesArr, jlong numOutputs, jlong runOptionsHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  OrtRunOptions* runOptions = (OrtRunOptions*)runOptionsHandle;

  jobjectArray outputArray = NULL;

  // Create the buffers for the Java input & output strings, and the input pointers
  const char** inputNames = malloc(sizeof(char*) * numInputs);
  if (inputNames == NULL) {
    // Nothing to cleanup, return and throw exception
    return outputArray;
  }
  const char** outputNames = malloc(sizeof(char*) * numOutputs);
  if (outputNames == NULL) {
    goto cleanup_input_names;
  }
  jobject* javaInputStrings = malloc(sizeof(jobject) * numInputs);
  if (javaInputStrings == NULL) {
    goto cleanup_output_names;
  }
  jobject* javaOutputStrings = malloc(sizeof(jobject) * numOutputs);
  if (javaOutputStrings == NULL) {
    goto cleanup_java_input_strings;
  }
  const OrtValue** inputValuePtrs = malloc(sizeof(OrtValue*) * numInputs);
  if (inputValuePtrs == NULL) {
    goto cleanup_java_output_strings;
  }
  OrtValue** outputValues = malloc(sizeof(OrtValue*) * numOutputs);
  if (outputValues == NULL) {
    goto cleanup_input_values;
  }

  // Extract a C array of longs which are pointers to the input tensors.
  // The Java-side objects store native pointers as 64-bit longs, and on 32-bit systems
  // we cannot cast the long array to a pointer array as they are different sizes,
  // so we copy the longs applying the appropriate cast.
  jlong* inputValueLongs = (*jniEnv)->GetLongArrayElements(jniEnv, inputHandles, NULL);

  // Extract the names and native pointers of the input values.
  for (int i = 0; i < numInputs; i++) {
    javaInputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, inputNamesArr, i);
    inputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaInputStrings[i], NULL);
    inputValuePtrs[i] = (OrtValue*)inputValueLongs[i];
  }

  // Release the java array copy of pointers to the tensors.
  (*jniEnv)->ReleaseLongArrayElements(jniEnv, inputHandles, inputValueLongs, JNI_ABORT);

  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
    outputValues[i] = NULL;
  }

  // Actually score the inputs.
  //ORT_API2_STATUS(TrainStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
  //                size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
  //                size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->TrainStep(trainSession, runOptions,
                                                                      numInputs, (const OrtValue* const*)inputValuePtrs,
                                                                      numOutputs, outputValues));
  if (code != ORT_OK) {
    goto cleanup_output_values;
  }

  // Construct the output array of ONNXValues
  jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, "ai/onnxruntime/OnnxValue");
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
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    evalStep
 * Signature: (JJJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_evalStep
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle,
     jlong nativeHandle, jlong allocatorHandle, jobjectArray inputNamesArr, jlongArray inputHandles, jlong numInputs,
     jobjectArray outputNamesArr, jlong numOutputs, jlong runOptionsHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  OrtRunOptions* runOptions = (OrtRunOptions*)runOptionsHandle;

  jobjectArray outputArray = NULL;

  // Create the buffers for the Java input & output strings, and the input pointers
  const char** inputNames = malloc(sizeof(char*) * numInputs);
  if (inputNames == NULL) {
    // Nothing to cleanup, return and throw exception
    return outputArray;
  }
  const char** outputNames = malloc(sizeof(char*) * numOutputs);
  if (outputNames == NULL) {
    goto cleanup_input_names;
  }
  jobject* javaInputStrings = malloc(sizeof(jobject) * numInputs);
  if (javaInputStrings == NULL) {
    goto cleanup_output_names;
  }
  jobject* javaOutputStrings = malloc(sizeof(jobject) * numOutputs);
  if (javaOutputStrings == NULL) {
    goto cleanup_java_input_strings;
  }
  const OrtValue** inputValuePtrs = malloc(sizeof(OrtValue*) * numInputs);
  if (inputValuePtrs == NULL) {
    goto cleanup_java_output_strings;
  }
  OrtValue** outputValues = malloc(sizeof(OrtValue*) * numOutputs);
  if (outputValues == NULL) {
    goto cleanup_input_values;
  }

  // Extract a C array of longs which are pointers to the input tensors.
  // The Java-side objects store native pointers as 64-bit longs, and on 32-bit systems
  // we cannot cast the long array to a pointer array as they are different sizes,
  // so we copy the longs applying the appropriate cast.
  jlong* inputValueLongs = (*jniEnv)->GetLongArrayElements(jniEnv, inputHandles, NULL);

  // Extract the names and native pointers of the input values.
  for (int i = 0; i < numInputs; i++) {
    javaInputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, inputNamesArr, i);
    inputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaInputStrings[i], NULL);
    inputValuePtrs[i] = (OrtValue*)inputValueLongs[i];
  }

  // Release the java array copy of pointers to the tensors.
  (*jniEnv)->ReleaseLongArrayElements(jniEnv, inputHandles, inputValueLongs, JNI_ABORT);

  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
    outputValues[i] = NULL;
  }

  // Actually score the inputs.
  //ORT_API2_STATUS(EvalStep, _In_ const OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
  //                size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
  //                size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);
  OrtErrorCode code = checkOrtStatus(jniEnv, api, trainApi->EvalStep(trainSession, runOptions,
                                                                      numInputs, (const OrtValue* const*)inputValuePtrs,
                                                                      numOutputs, outputValues));
  if (code != ORT_OK) {
    goto cleanup_output_values;
  }

  // Construct the output array of ONNXValues
  jclass onnxValueClass = (*jniEnv)->FindClass(jniEnv, "ai/onnxruntime/OnnxValue");
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
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    setSeed
 * Signature: (JJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_setSeed
    (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong trainApiHandle, jlong seed) {
  (void)clazz;  // Required JNI parameter not needed by functions which don't need to access their host class.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  checkOrtStatus(jniEnv, api, trainApi->SetSeed(seed));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    setLearningRate
 * Signature: (JJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_setLearningRate
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jfloat learningRate) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->SetLearningRate(trainSession, learningRate));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getLearningRate
 * Signature: (JJJ)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_OrtTrainingSession_getLearningRate
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  jfloat learningRate = 0.0f;
  checkOrtStatus(jniEnv, api, trainApi->GetLearningRate(trainSession, &learningRate));
  return learningRate;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    optimizerStep
 * Signature: (JJJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_optimizerStep
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jlong runOptionsHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  const OrtRunOptions* options = (const OrtRunOptions*) runOptionsHandle;
  checkOrtStatus(jniEnv, api, trainApi->OptimizerStep(trainSession, options));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    registerLinearLRScheduler
 * Signature: (JJJJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_registerLinearLRScheduler
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jlong warmupSteps, jlong totalSteps, jfloat initialLearningRate) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->RegisterLinearLRScheduler(trainSession, warmupSteps, totalSteps, initialLearningRate));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    schedulerStep
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_schedulerStep
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;
  checkOrtStatus(jniEnv, api, trainApi->SchedulerStep(trainSession));
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    exportModelForInference
 * Signature: (JJJJLjava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_exportModelForInference
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainApiHandle, jlong nativeHandle, jstring outputPath, jlong numOutputs, jobjectArray outputNamesArr) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*)trainApiHandle;
  OrtTrainingSession* trainSession = (OrtTrainingSession*)nativeHandle;

  // prep output names array
  const char** outputNames = malloc(sizeof(char*) * numOutputs);
  if (outputNames == NULL) {
    throwOrtException(jniEnv, 1, "Not enough memory");
    return;
  }
  jobject* javaOutputStrings = malloc(sizeof(jobject) * numOutputs);
  if (javaOutputStrings == NULL) {
    throwOrtException(jniEnv, 1, "Not enough memory");
    free(outputNames);
    return;
  }
  // Extract the names of the output values.
  for (int i = 0; i < numOutputs; i++) {
    javaOutputStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, outputNamesArr, i);
    outputNames[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaOutputStrings[i], NULL);
  }

#ifdef _WIN32
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  wchar_t* outputStr = copyAndPad(jniEnv, outputPath);
  if (outputStr == NULL) {
    goto cleanup_array;
  }
  checkOrtStatus(jniEnv, api, trainApi->ExportModelForInferencing(trainSession, outputStr, numOutputs, outputNames));
  free(outputStr);
#else
  // GetStringUTFChars is null terminated, so can be used directly
  const char* outputStr = (*jniEnv)->GetStringUTFChars(jniEnv, outputPath, NULL);
  checkOrtStatus(jniEnv, api, trainApi->ExportModelForInferencing(trainSession, outputStr, numOutputs, outputNames));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, outputPath, outputStr);
  goto cleanup_array; // Only used in the WIN32 branch, but gcc complains we don't use this label otherwise
#endif

cleanup_array:
  // Release the Java output strings
  for (int i = 0; i < numOutputs; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaOutputStrings[i], outputNames[i]);
  }
  free(javaOutputStrings);
  free(outputNames);
}
