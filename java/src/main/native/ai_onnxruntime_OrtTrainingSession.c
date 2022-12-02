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
 * Signature: (JJJJLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_createTrainingSession
    (JNIEnv *, jclass, jlong, jlong, jlong, jlong, jstring, jstring, jstring);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    closeSession
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_closeSession
    (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    saveCheckpoint
 * Signature: (JJLjava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_saveCheckpoint
    (JNIEnv *, jobject, jlong, jlong, jstring, jboolean);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getTrainOutputNames
 * Signature: (JJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getTrainOutputNames
    (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getEvalOutputNames
 * Signature: (JJJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtTrainingSession_getEvalOutputNames
    (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    resetGrad
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_resetGrad
    (JNIEnv *, jobject, jlong, jlong);

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
 * Signature: (JJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_setLearningRate
    (JNIEnv *, jobject, jlong, jlong, jfloat);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getLearningRate
 * Signature: (JJ)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_OrtTrainingSession_getLearningRate
    (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    optimizerStep
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_optimizerStep
    (JNIEnv *, jobject, jlong, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    registerLinearLRScheduler
 * Signature: (JJJJF)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_registerLinearLRScheduler
    (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jfloat);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    schedulerStep
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_schedulerStep
    (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    getParametersSize
 * Signature: (JJZ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_getParametersSize
    (JNIEnv *, jobject, jlong, jlong, jboolean);

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    exportModelForInference
 * Signature: (JJLjava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_exportModelForInference
    (JNIEnv *, jobject, jlong, jlong, jstring, jobjectArray);