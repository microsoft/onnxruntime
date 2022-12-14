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

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    createTrainingSession
 * Signature: (JJJJJLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_createTrainingSession
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jlong, jstring, jstring, jstring);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    closeSession
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_closeSession
    (JNIEnv * jniEnv, jobject jobj, jlong trainingHandle, jlong handle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtTrainingApi* api = (const OrtTrainingApi*) trainingHandle;
  api->ReleaseTrainingSession((OrtTrainingSession*)handle);
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    saveCheckpoint
 * Signature: (JJJLjava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_saveCheckpoint
  (JNIEnv *, jobject, jlong, jlong, jlong, jstring, jboolean);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getTrainOutputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getTrainOutputNames
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getEvalOutputNames
 * Signature: (JJJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getEvalOutputNames
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    resetGrad
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_resetGrad
  (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    trainStep
 * Signature: (JJJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_trainStep
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobjectArray, jlongArray, jlong, jobjectArray, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    evalStep
 * Signature: (JJJJ[Ljava/lang/String;[JJ[Ljava/lang/String;JJ)[Lai/onnxruntime/OnnxValue;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_evalStep
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobjectArray, jlongArray, jlong, jobjectArray, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    setLearningRate
 * Signature: (JJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_setLearningRate
  (JNIEnv *, jobject, jlong, jlong, jlong, jfloat);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getLearningRate
 * Signature: (JJJ)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_OrtTrainingSession_getLearningRate
  (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    optimizerStep
 * Signature: (JJJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_optimizerStep
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    registerLinearLRScheduler
 * Signature: (JJJJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_registerLinearLRScheduler
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jlong, jfloat);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    schedulerStep
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_schedulerStep
  (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getParametersSize
 * Signature: (JJJZ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_getParametersSize
  (JNIEnv *, jobject, jlong, jlong, jlong, jboolean);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    exportModelForInference
 * Signature: (JJJLjava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_exportModelForInference
  (JNIEnv *, jobject, jlong, jlong, jlong, jstring, jobjectArray);
