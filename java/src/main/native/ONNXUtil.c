/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 */
#include <jni.h>
#include <stdio.h>
#include "ONNXUtil.h"

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    return JNI_VERSION_1_8;
}

/**
 * Must be kept in sync with ORT_LOGGING_LEVEL and ONNXEnvironment#LoggingLevel
 */
OrtLoggingLevel convertLoggingLevel(jint level) {
    switch (level) {
        case 0:
            return ORT_LOGGING_LEVEL_VERBOSE;
        case 1:
            return ORT_LOGGING_LEVEL_INFO;
        case 2:
            return ORT_LOGGING_LEVEL_WARNING;
        case 3:
            return ORT_LOGGING_LEVEL_ERROR;
        case 4:
            return ORT_LOGGING_LEVEL_FATAL;
        default:
            return ORT_LOGGING_LEVEL_VERBOSE;
    }
}

/**
 * Must be kept in sync with GraphOptimizationLevel and SessionOptions#OptLevel
 */
GraphOptimizationLevel convertOptimizationLevel(jint level) {
    switch (level) {
        case 0:
            return ORT_DISABLE_ALL;
        case 1:
            return ORT_ENABLE_BASIC;
        case 2:
            return ORT_ENABLE_EXTENDED;
        case 99:
            return ORT_ENABLE_ALL;
        default:
            return ORT_DISABLE_ALL;
    }
}

/**
 * Must be kept in sync with convertToONNXDataFormat
 */
jint convertFromONNXDataFormat(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            return 0;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    // maps to c type int8_t
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
            return 3;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   // maps to c type int16_t
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      // maps to c type uint32_t
            return 5;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   // maps to c type int32_t
            return 6;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      // maps to c type uint64_t
            return 7;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   // maps to c type int64_t
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return 9;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   // maps to c type float
            return 10;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      // maps to c type double
            return 11;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  // maps to c++ type std::string
            return 12;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 13;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
            return 14;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
            return 15;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
            return 16;
        default:
            return -1;
    }
}

/**
 * Must be kept in sync with convertFromONNXDataFormat
 */
ONNXTensorElementDataType convertToONNXDataFormat(jint type) {
    switch (type) {
        case 0:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        case 1:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;   // maps to c type uint8_t
        case 2:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;    // maps to c type int8_t
        case 3:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;  // maps to c type uint16_t
        case 4:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;   // maps to c type int16_t
        case 5:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;      // maps to c type uint32_t
        case 6:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;   // maps to c type int32_t
        case 7:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;      // maps to c type uint64_t
        case 8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;   // maps to c type int64_t
        case 9:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case 10:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;   // maps to c type float
        case 11:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;      // maps to c type double
        case 12:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;  // maps to c++ type std::string
        case 13:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case 14:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;   // complex with float32 real and imaginary components
        case 15:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;  // complex with float64 real and imaginary components
        case 16:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;    // Non-IEEE floating-point format based on IEEE754 single-precision
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

size_t onnxTypeSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    // maps to c type int8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   // maps to c type int16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      // maps to c type uint32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   // maps to c type int32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   // maps to c type float
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      // maps to c type uint64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   // maps to c type int64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      // maps to c type double
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  // maps to c++ type std::string
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
        default:
            return 0;
    }
}

jfloat convertHalfToFloat(uint16_t half) {
    return ((half&0x8000)<<16) | (((half&0x7c00)+0x1C000)<<13) | ((half&0x03FF)<<13);
}

jobject convertToValueInfo(JNIEnv *jniEnv, const OrtApi * api, OrtTypeInfo * info) {
    ONNXType type;
    checkONNXStatus(jniEnv,api,api->GetOnnxTypeFromTypeInfo(info,&type));

    switch (type) {
        case ONNX_TYPE_TENSOR: {
            const OrtTensorTypeAndShapeInfo* tensorInfo;
            checkONNXStatus(jniEnv,api,api->CastTypeInfoToTensorInfo(info,&tensorInfo));
            return convertToTensorInfo(jniEnv, api, (const OrtTensorTypeAndShapeInfo *) tensorInfo);
        }
        case ONNX_TYPE_SEQUENCE: {
            return convertToSequenceInfo(jniEnv,api,info);
        }
        case ONNX_TYPE_MAP: {
            return convertToMapInfo(jniEnv,api,info);
        }
        case ONNX_TYPE_UNKNOWN:
        case ONNX_TYPE_OPAQUE:
        case ONNX_TYPE_SPARSETENSOR:
        default: {
            throwONNXException(jniEnv,convertErrorCode(ORT_NOT_IMPLEMENTED),"Invalid ONNXType found.");
            return NULL;
        }
    }
}

jobject convertToTensorInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTensorTypeAndShapeInfo * info) {
    // Extract the information from the info struct.
    ONNXTensorElementDataType onnxType;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(info,&onnxType));
    size_t numDim;
    checkONNXStatus(jniEnv,api,api->GetDimensionsCount(info,&numDim));
    //printf("numDim %d\n",numDim);
    int64_t* dimensions = (int64_t*) malloc(sizeof(int64_t)*numDim);
    checkONNXStatus(jniEnv,api,api->GetDimensions(info, dimensions, numDim));
    jint onnxTypeInt = convertFromONNXDataFormat(onnxType);

    // Create the long array for the shape.
    jlongArray shape = (*jniEnv)->NewLongArray(jniEnv, numDim);
    (*jniEnv)->SetLongArrayRegion(jniEnv, shape, 0, numDim, (jlong*)dimensions);
    // Free the dimensions array
    free(dimensions);
    dimensions = NULL;

    // Create the ONNXTensorType enum
    char *onnxTensorTypeClassName = "com/microsoft/onnxruntime/TensorInfo$ONNXTensorType";
    jclass clazz = (*jniEnv)->FindClass(jniEnv, onnxTensorTypeClassName);
    jmethodID onnxTensorTypeMapFromInt = (*jniEnv)->GetStaticMethodID(jniEnv,clazz, "mapFromInt", "(I)Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;");
    jobject onnxTensorTypeJava = (*jniEnv)->CallStaticObjectMethod(jniEnv,clazz,onnxTensorTypeMapFromInt,onnxTypeInt);
    //printf("ONNXTensorType class %p, methodID %p, object %p\n",clazz,onnxTensorTypeMapFromInt,onnxTensorTypeJava);

    // Create the ONNXJavaType enum
    char *javaDataTypeClassName = "com/microsoft/onnxruntime/ONNXJavaType";
    clazz = (*jniEnv)->FindClass(jniEnv, javaDataTypeClassName);
    jmethodID javaDataTypeMapFromONNXTensorType = (*jniEnv)->GetStaticMethodID(jniEnv,clazz, "mapFromONNXTensorType", "(Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;)Lcom/microsoft/onnxruntime/ONNXJavaType;");
    jobject javaDataType = (*jniEnv)->CallStaticObjectMethod(jniEnv,clazz,javaDataTypeMapFromONNXTensorType,onnxTensorTypeJava);
    //printf("JavaDataType class %p, methodID %p, object %p\n",clazz,javaDataTypeMapFromONNXTensorType,javaDataType);

    // Create the TensorInfo object
    char *tensorInfoClassName = "com/microsoft/onnxruntime/TensorInfo";
    clazz = (*jniEnv)->FindClass(jniEnv, tensorInfoClassName);
    jmethodID tensorInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,clazz, "<init>", "([JLcom/microsoft/onnxruntime/ONNXJavaType;Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;)V");
    //printf("TensorInfo class %p, methodID %p\n",clazz,tensorInfoConstructor);
    jobject tensorInfo = (*jniEnv)->NewObject(jniEnv, clazz, tensorInfoConstructor, shape, javaDataType, onnxTensorTypeJava);
    return tensorInfo;
}

jobject convertToMapInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTypeInfo * info) {
    // Create the ONNXJavaType enum
    char *onnxJavaTypeClassName = "com/microsoft/onnxruntime/ONNXJavaType";
    jclass clazz = (*jniEnv)->FindClass(jniEnv, onnxJavaTypeClassName);
    jmethodID onnxJavaTypeMapFromInt = (*jniEnv)->GetStaticMethodID(jniEnv,clazz, "mapFromInt", "(I)Lcom/microsoft/onnxruntime/ONNXJavaType;");
    jobject unknownType = (*jniEnv)->CallStaticObjectMethod(jniEnv,clazz,onnxJavaTypeMapFromInt,0);

    char *mapInfoClassName = "com/microsoft/onnxruntime/MapInfo";
    clazz = (*jniEnv)->FindClass(jniEnv, mapInfoClassName);
    jmethodID mapInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,clazz,"<init>","(Lcom/microsoft/onnxruntime/ONNXJavaType;Lcom/microsoft/onnxruntime/ONNXJavaType;)V");
    jobject mapInfo = (*jniEnv)->NewObject(jniEnv,clazz,mapInfoConstructor,unknownType,unknownType);

    return mapInfo;
}

jobject convertToSequenceInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTypeInfo * info) {
    // Create the ONNXJavaType enum
    char *onnxJavaTypeClassName = "com/microsoft/onnxruntime/ONNXJavaType";
    jclass clazz = (*jniEnv)->FindClass(jniEnv, onnxJavaTypeClassName);
    jmethodID onnxJavaTypeMapFromInt = (*jniEnv)->GetStaticMethodID(jniEnv,clazz, "mapFromInt", "(I)Lcom/microsoft/onnxruntime/ONNXJavaType;");
    jobject unknownType = (*jniEnv)->CallStaticObjectMethod(jniEnv,clazz,onnxJavaTypeMapFromInt,0);

    char *sequenceInfoClassName = "com/microsoft/onnxruntime/SequenceInfo";
    clazz = (*jniEnv)->FindClass(jniEnv, sequenceInfoClassName);
    jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,clazz,"<init>","(ILcom/microsoft/onnxruntime/ONNXJavaType;)V");
    jobject sequenceInfo = (*jniEnv)->NewObject(jniEnv,clazz,sequenceInfoConstructor,-1,unknownType);

    return sequenceInfo;
}

size_t copyJavaToPrimitiveArray(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, size_t tensorSize, jarray input) {
    uint32_t inputLength = (*jniEnv)->GetArrayLength(jniEnv,input);
    size_t consumedSize = inputLength * onnxTypeSize(onnxType);
    switch (onnxType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {  // maps to c type int8_t
            jbyteArray typedArr = (jbyteArray) input;
            (*jniEnv)->GetByteArrayRegion(jniEnv, typedArr, 0, inputLength, (jbyte * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: { // maps to c type int16_t
            jshortArray typedArr = (jshortArray) input;
            (*jniEnv)->GetShortArrayRegion(jniEnv, typedArr, 0, inputLength, (jshort * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      // maps to c type uint32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: { // maps to c type int32_t
            jintArray typedArr = (jintArray) input;
            (*jniEnv)->GetIntArrayRegion(jniEnv, typedArr, 0, inputLength, (jint * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      // maps to c type uint64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: { // maps to c type int64_t
            jlongArray typedArr = (jlongArray) input;
            (*jniEnv)->GetLongArrayRegion(jniEnv, typedArr, 0, inputLength, (jlong * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            throwONNXException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "16-bit float not supported.");
            return 0;
            /*
            float *floatArr = malloc(sizeof(float) * inputLength);
            uint16_t *halfArr = (uint16_t *) tensor;
            for (uint32_t i = 0; i < inputLength; i++) {
                floatArr[i] = convertHalfToFloat(halfArr[i]);
            }
            jfloatArray typedArr = (jfloatArray) input;
            (*jniEnv)->GetFloatArrayRegion(jniEnv, typedArr, 0, inputLength, floatArr);
            free(floatArr);
            return consumedSize;
            */
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: { // maps to c type float
            jfloatArray typedArr = (jfloatArray) input;
            (*jniEnv)->GetFloatArrayRegion(jniEnv, typedArr, 0, inputLength, (jfloat * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {    // maps to c type double
            jdoubleArray typedArr = (jdoubleArray) input;
            (*jniEnv)->GetDoubleArrayRegion(jniEnv, typedArr, 0, inputLength, (jdouble * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: { // maps to c++ type std::string
            throwONNXException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "String is not supported.");
            return 0;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            jbooleanArray typedArr = (jbooleanArray) input;
            (*jniEnv)->GetBooleanArrayRegion(jniEnv, typedArr, 0, inputLength, (jboolean *) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        default: {
            throwONNXException(jniEnv, convertErrorCode(ORT_INVALID_ARGUMENT), "Invalid tensor element type.");
            return 0;
        }
    }
}

size_t copyJavaToTensor(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, size_t tensorSize,
                        uint32_t dimensionsRemaining, jarray input) {
    if (dimensionsRemaining == 1) {
        // write out 1d array of the respective primitive type
        return copyJavaToPrimitiveArray(jniEnv,onnxType,tensor,tensorSize,input);
    } else {
        // recurse through the dimensions
        // Java arrays are objects until the final dimension
        jobjectArray inputObjArr = (jobjectArray) input;
        uint32_t dimLength = (*jniEnv)->GetArrayLength(jniEnv,inputObjArr);
        size_t sizeConsumed = 0;
        for (uint32_t i = 0; i < dimLength; i++) {
            jarray childArr = (jarray) (*jniEnv)->GetObjectArrayElement(jniEnv,inputObjArr,i);
            sizeConsumed += copyJavaToTensor(jniEnv, onnxType, tensor + sizeConsumed, tensorSize - sizeConsumed, dimensionsRemaining - 1, childArr);
            // Cleanup reference to childArr so it doesn't prevent GC.
            (*jniEnv)->DeleteLocalRef(jniEnv,childArr);
        }
        return sizeConsumed;
    }
}

size_t copyPrimitiveArrayToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, size_t tensorSize, jarray output) {
    uint32_t outputLength = (*jniEnv)->GetArrayLength(jniEnv,output);
    size_t consumedSize = outputLength * onnxTypeSize(onnxType);
    switch (onnxType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {  // maps to c type int8_t
            jbyteArray typedArr = (jbyteArray) output;
            (*jniEnv)->SetByteArrayRegion(jniEnv, typedArr, 0, outputLength, (jbyte * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: { // maps to c type int16_t
            jshortArray typedArr = (jshortArray) output;
            (*jniEnv)->SetShortArrayRegion(jniEnv, typedArr, 0, outputLength, (jshort * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      // maps to c type uint32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: { // maps to c type int32_t
            jintArray typedArr = (jintArray) output;
            (*jniEnv)->SetIntArrayRegion(jniEnv, typedArr, 0, outputLength, (jint * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      // maps to c type uint64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: { // maps to c type int64_t
            jlongArray typedArr = (jlongArray) output;
            (*jniEnv)->SetLongArrayRegion(jniEnv, typedArr, 0, outputLength, (jlong * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            float *floatArr = malloc(sizeof(float) * outputLength);
            uint16_t *halfArr = (uint16_t *) tensor;
            for (uint32_t i = 0; i < outputLength; i++) {
                floatArr[i] = convertHalfToFloat(halfArr[i]);
            }
            jfloatArray typedArr = (jfloatArray) output;
            (*jniEnv)->SetFloatArrayRegion(jniEnv, typedArr, 0, outputLength, floatArr);
            free(floatArr);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: { // maps to c type float
            jfloatArray typedArr = (jfloatArray) output;
            (*jniEnv)->SetFloatArrayRegion(jniEnv, typedArr, 0, outputLength, (jfloat * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {    // maps to c type double
            jdoubleArray typedArr = (jdoubleArray) output;
            (*jniEnv)->SetDoubleArrayRegion(jniEnv, typedArr, 0, outputLength, (jdouble * ) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: { // maps to c++ type std::string
            throwONNXException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "String is not supported.");
            return 0;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            jbooleanArray typedArr = (jbooleanArray) output;
            (*jniEnv)->SetBooleanArrayRegion(jniEnv, typedArr, 0, outputLength, (jboolean *) tensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        default: {
            throwONNXException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "Invalid tensor element type.");
            return 0;
        }
    }
}

size_t copyTensorToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, size_t tensorSize,
                        uint32_t dimensionsRemaining, jarray output) {
    if (dimensionsRemaining == 1) {
        // write out 1d array of the respective primitive type
        return copyPrimitiveArrayToJava(jniEnv,onnxType,tensor,tensorSize,output);
    } else {
        // recurse through the dimensions
        // Java arrays are objects until the final dimension
        jobjectArray outputObjArr = (jobjectArray) output;
        uint32_t dimLength = (*jniEnv)->GetArrayLength(jniEnv,outputObjArr);
        size_t sizeConsumed = 0;
        for (uint32_t i = 0; i < dimLength; i++) {
            jarray childArr = (jarray) (*jniEnv)->GetObjectArrayElement(jniEnv,outputObjArr,i);
            sizeConsumed += copyTensorToJava(jniEnv, onnxType, tensor + sizeConsumed, tensorSize - sizeConsumed, dimensionsRemaining - 1, childArr);
            // Cleanup reference to childArr so it doesn't prevent GC.
            (*jniEnv)->DeleteLocalRef(jniEnv,childArr);
        }
        return sizeConsumed;
    }
}

jobject createStringFromStringTensor(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor) {
    // Get the buffer size needed
    size_t totalStringLength;
    checkONNXStatus(jniEnv,api,api->GetStringTensorDataLength(tensor,&totalStringLength));

    // Create the character and offset buffers
    char * characterBuffer;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(char)*(totalStringLength+1),(void**)&characterBuffer));
    size_t * offsets;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(size_t),(void**)&offsets));

    // Get a view on the String data
    checkONNXStatus(jniEnv,api,api->GetStringTensorContent(tensor,characterBuffer,totalStringLength,offsets,1));

    size_t curSize = (offsets[0]) + 1;
    characterBuffer[curSize-1] = '\0';
    jobject tempString = (*jniEnv)->NewStringUTF(jniEnv,characterBuffer);

    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,characterBuffer));
    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,offsets));

    return tempString;
}

void copyStringTensorToArray(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor, size_t length, jobjectArray outputArray) {
    // Get the buffer size needed
    size_t totalStringLength;
    checkONNXStatus(jniEnv,api,api->GetStringTensorDataLength(tensor,&totalStringLength));

    // Create the character and offset buffers
    char * characterBuffer;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(char)*(totalStringLength+length),(void**)&characterBuffer));
    // length + 1 as we need to write out the final offset
    size_t * offsets;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(size_t)*(length+1),(void**)&offsets));

    // Get a view on the String data
    checkONNXStatus(jniEnv,api,api->GetStringTensorContent(tensor,characterBuffer,totalStringLength,offsets,length));

    // Get the final offset, write to the end of the array.
    checkONNXStatus(jniEnv,api,api->GetStringTensorDataLength(tensor,offsets+length));

    char * tempBuffer = NULL;
    size_t bufferSize = 0;
    for (int i = 0; i < length; i++) {
        size_t curSize = (offsets[i+1] - offsets[i]) + 1;
        if (curSize > bufferSize) {
            checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,tempBuffer));
            checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,curSize,(void**)&tempBuffer));
            bufferSize = curSize;
        }
        memcpy(tempBuffer,characterBuffer+offsets[i],curSize);
        tempBuffer[curSize-1] = '\0';
        jobject tempString = (*jniEnv)->NewStringUTF(jniEnv,tempBuffer);
        (*jniEnv)->SetObjectArrayElement(jniEnv,outputArray,i,tempString);
    }

    if (tempBuffer != NULL) {
        checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,tempBuffer));
    }
    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,characterBuffer));
    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,offsets));
}

jobjectArray createStringArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor) {
    // Extract tensor info
    OrtTensorTypeAndShapeInfo* tensorInfo;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor,&tensorInfo));

    // Get the element count of this tensor
    size_t length;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo,&length));
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);

    // Create the java array
    jclass stringClazz = (*jniEnv)->FindClass(jniEnv,"java/lang/String");
    jobjectArray outputArray = (*jniEnv)->NewObjectArray(jniEnv,length,stringClazz,NULL);

    copyStringTensorToArray(jniEnv, api, allocator, tensor, length, outputArray);

    return outputArray;
}

jlongArray createLongArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
    // Extract tensor type
    OrtTensorTypeAndShapeInfo* tensorInfo;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor,&tensorInfo));
    ONNXTensorElementDataType value;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(tensorInfo,&value));

    // Get the element count of this tensor
    size_t length;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo,&length));
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);

    // Extract the values
    uint8_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)tensor,(void**)&arr));

    // Create the java array and copy to it.
    jlongArray outputArray = (*jniEnv)->NewLongArray(jniEnv,length);
    copyPrimitiveArrayToJava(jniEnv, value, arr, length, outputArray);
    return outputArray;
}

jfloatArray createFloatArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
    // Extract tensor type
    OrtTensorTypeAndShapeInfo* tensorInfo;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor,&tensorInfo));
    ONNXTensorElementDataType value;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(tensorInfo,&value));

    // Get the element count of this tensor
    size_t length;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo,&length));
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);

    // Extract the values
    uint8_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)tensor,(void**)&arr));

    // Create the java array and copy to it.
    jfloatArray outputArray = (*jniEnv)->NewFloatArray(jniEnv,length);
    copyPrimitiveArrayToJava(jniEnv, value, arr, length, outputArray);
    return outputArray;
}

jdoubleArray createDoubleArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
    // Extract tensor type
    OrtTensorTypeAndShapeInfo* tensorInfo;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor,&tensorInfo));
    ONNXTensorElementDataType value;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(tensorInfo,&value));

    // Get the element count of this tensor
    size_t length;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo,&length));
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);

    // Extract the values
    uint8_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)tensor,(void**)&arr));

    // Create the java array and copy to it.
    jdoubleArray outputArray = (*jniEnv)->NewDoubleArray(jniEnv,length);
    copyPrimitiveArrayToJava(jniEnv, value, arr, length, outputArray);
    return outputArray;
}

jobject createJavaTensorFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor) {
    // Extract the type information
    OrtTensorTypeAndShapeInfo* info;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor, &info));

    // Construct the TensorInfo object
    jobject tensorInfo = convertToTensorInfo(jniEnv, api, info);

    // Release the info object
    api->ReleaseTensorTypeAndShapeInfo(info);

    // Construct the ONNXTensor object
    char *tensorClassName = "com/microsoft/onnxruntime/ONNXTensor";
    jclass clazz = (*jniEnv)->FindClass(jniEnv, tensorClassName);
    jmethodID tensorConstructor = (*jniEnv)->GetMethodID(jniEnv,clazz, "<init>", "(JJLcom/microsoft/onnxruntime/TensorInfo;)V");
    jobject javaTensor = (*jniEnv)->NewObject(jniEnv, clazz, tensorConstructor, (jlong) tensor, (jlong) allocator, tensorInfo);

    return javaTensor;
}

jobject createJavaSequenceFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* sequence) {
    // Setup
    // Get the ONNXTensorType enum static method
    char *onnxTensorTypeClassName = "com/microsoft/onnxruntime/TensorInfo$ONNXTensorType";
    jclass onnxTensorTypeClazz = (*jniEnv)->FindClass(jniEnv, onnxTensorTypeClassName);
    jmethodID onnxTensorTypeMapFromInt = (*jniEnv)->GetStaticMethodID(jniEnv,onnxTensorTypeClazz, "mapFromInt", "(I)Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;");

    // Get the ONNXJavaType enum static method
    char *javaDataTypeClassName = "com/microsoft/onnxruntime/ONNXJavaType";
    jclass onnxJavaTypeClazz = (*jniEnv)->FindClass(jniEnv, javaDataTypeClassName);
    jmethodID onnxJavaTypeMapFromONNXTensorType = (*jniEnv)->GetStaticMethodID(jniEnv,onnxJavaTypeClazz, "mapFromONNXTensorType", "(Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;)Lcom/microsoft/onnxruntime/ONNXJavaType;");

    // Get the sequence info class
    char *sequenceInfoClassName = "com/microsoft/onnxruntime/SequenceInfo";
    jclass sequenceInfoClazz = (*jniEnv)->FindClass(jniEnv, sequenceInfoClassName);

    // Get the element count of this sequence
    size_t count;
    checkONNXStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    // Extract the first element
    OrtValue* firstElement;
    checkONNXStatus(jniEnv,api,api->GetValue(sequence,0,allocator,&firstElement));
    ONNXType elementType;
    checkONNXStatus(jniEnv,api,api->GetValueType(firstElement,&elementType));
    jobject sequenceInfo;
    switch (elementType) {
        case ONNX_TYPE_TENSOR: {
            // Figure out element type
            OrtTensorTypeAndShapeInfo* firstElementInfo;
            checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(firstElement,&firstElementInfo));
            ONNXTensorElementDataType element;
            checkONNXStatus(jniEnv,api,api->GetTensorElementType(firstElementInfo,&element));
            api->ReleaseTensorTypeAndShapeInfo(firstElementInfo);

            // Convert element type into ONNXTensorType
            jint onnxTypeInt = convertFromONNXDataFormat(element);
            jobject onnxTensorTypeJava = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxTensorTypeClazz,onnxTensorTypeMapFromInt,onnxTypeInt);
            jobject onnxJavaType = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxJavaTypeClazz,onnxJavaTypeMapFromONNXTensorType,onnxTensorTypeJava);

            // Construct sequence info
            jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,sequenceInfoClazz,"<init>","(ILcom/microsoft/onnxruntime/ONNXJavaType;)V");
            sequenceInfo = (*jniEnv)->NewObject(jniEnv,sequenceInfoClazz,sequenceInfoConstructor,(jint)count,onnxJavaType);
            break;
        }
        case ONNX_TYPE_MAP: {
            // Extract key
            OrtValue* keys;
            checkONNXStatus(jniEnv,api,api->GetValue(firstElement,0,allocator,&keys));

            // Extract key type
            OrtTensorTypeAndShapeInfo* keysInfo;
            checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(keys,&keysInfo));
            ONNXTensorElementDataType key;
            checkONNXStatus(jniEnv,api,api->GetTensorElementType(keysInfo,&key));

            // Get the element count of this map
            size_t mapCount;
            checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(keysInfo,&mapCount));

            api->ReleaseTensorTypeAndShapeInfo(keysInfo);

            // Convert key type to java
            jint onnxTypeKey = convertFromONNXDataFormat(key);
            jobject onnxTensorTypeJavaKey = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxTensorTypeClazz,onnxTensorTypeMapFromInt,onnxTypeKey);
            jobject onnxJavaTypeKey = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxJavaTypeClazz,onnxJavaTypeMapFromONNXTensorType,onnxTensorTypeJavaKey);

            // Extract value
            OrtValue* values;
            checkONNXStatus(jniEnv,api,api->GetValue(firstElement,1,allocator,&values));

            // Extract value type
            OrtTensorTypeAndShapeInfo* valuesInfo;
            checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(values,&valuesInfo));
            ONNXTensorElementDataType value;
            checkONNXStatus(jniEnv,api,api->GetTensorElementType(valuesInfo,&value));
            api->ReleaseTensorTypeAndShapeInfo(valuesInfo);

            // Convert value type to java
            jint onnxTypeValue = convertFromONNXDataFormat(value);
            jobject onnxTensorTypeJavaValue = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxTensorTypeClazz,onnxTensorTypeMapFromInt,onnxTypeValue);
            jobject onnxJavaTypeValue = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxJavaTypeClazz,onnxJavaTypeMapFromONNXTensorType,onnxTensorTypeJavaValue);

            // Get the map info class
            char *mapInfoClassName = "com/microsoft/onnxruntime/MapInfo";
            jclass mapInfoClazz = (*jniEnv)->FindClass(jniEnv, mapInfoClassName);
            // Construct map info
            jmethodID mapInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,mapInfoClazz,"<init>","(ILcom/microsoft/onnxruntime/ONNXJavaType;Lcom/microsoft/onnxruntime/ONNXJavaType;)V");
            jobject mapInfo = (*jniEnv)->NewObject(jniEnv,mapInfoClazz,mapInfoConstructor,(jint)mapCount,onnxJavaTypeKey,onnxJavaTypeValue);

            // Free the intermediate tensors.
            api->ReleaseValue(keys);
            api->ReleaseValue(values);

            // Construct sequence info
            jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,sequenceInfoClazz,"<init>","(ILcom/microsoft/onnxruntime/MapInfo;)V");
            sequenceInfo = (*jniEnv)->NewObject(jniEnv,sequenceInfoClazz,sequenceInfoConstructor,(jint)count,mapInfo);
            break;
        }
        default: {
            throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"Invalid element type found in sequence");
            break;
        }
    }

    // Free the intermediate tensor.
    api->ReleaseValue(firstElement);

    // Construct the ONNXSequence object
    char *sequenceClassName = "com/microsoft/onnxruntime/ONNXSequence";
    jclass sequenceClazz = (*jniEnv)->FindClass(jniEnv, sequenceClassName);
    jmethodID sequenceConstructor = (*jniEnv)->GetMethodID(jniEnv,sequenceClazz, "<init>", "(JJLcom/microsoft/onnxruntime/SequenceInfo;)V");
    jobject javaSequence = (*jniEnv)->NewObject(jniEnv, sequenceClazz, sequenceConstructor, (jlong)sequence, (jlong)allocator, sequenceInfo);

    return javaSequence;
}

jobject createJavaMapFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* map) {
    // Setup
    // Get the ONNXTensorType enum static method
    char *onnxTensorTypeClassName = "com/microsoft/onnxruntime/TensorInfo$ONNXTensorType";
    jclass onnxTensorTypeClazz = (*jniEnv)->FindClass(jniEnv, onnxTensorTypeClassName);
    jmethodID onnxTensorTypeMapFromInt = (*jniEnv)->GetStaticMethodID(jniEnv,onnxTensorTypeClazz, "mapFromInt", "(I)Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;");

    // Get the ONNXJavaType enum static method
    char *javaDataTypeClassName = "com/microsoft/onnxruntime/ONNXJavaType";
    jclass onnxJavaTypeClazz = (*jniEnv)->FindClass(jniEnv, javaDataTypeClassName);
    jmethodID onnxJavaTypeMapFromONNXTensorType = (*jniEnv)->GetStaticMethodID(jniEnv,onnxJavaTypeClazz, "mapFromONNXTensorType", "(Lcom/microsoft/onnxruntime/TensorInfo$ONNXTensorType;)Lcom/microsoft/onnxruntime/ONNXJavaType;");

    // Get the map info class
    char *mapInfoClassName = "com/microsoft/onnxruntime/MapInfo";
    jclass mapInfoClazz = (*jniEnv)->FindClass(jniEnv, mapInfoClassName);

    // Extract key
    OrtValue* keys;
    checkONNXStatus(jniEnv,api,api->GetValue(map,0,allocator,&keys));

    // Extract key type
    OrtTensorTypeAndShapeInfo* keysInfo;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(keys,&keysInfo));
    ONNXTensorElementDataType key;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(keysInfo,&key));

    // Get the element count of this map
    size_t mapCount;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(keysInfo,&mapCount));

    api->ReleaseTensorTypeAndShapeInfo(keysInfo);

    // Convert key type to java
    jint onnxTypeKey = convertFromONNXDataFormat(key);
    jobject onnxTensorTypeJavaKey = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxTensorTypeClazz,onnxTensorTypeMapFromInt,onnxTypeKey);
    jobject onnxJavaTypeKey = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxJavaTypeClazz,onnxJavaTypeMapFromONNXTensorType,onnxTensorTypeJavaKey);

    // Extract value
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue(map,1,allocator,&values));

    // Extract value type
    OrtTensorTypeAndShapeInfo* valuesInfo;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(values,&valuesInfo));
    ONNXTensorElementDataType value;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(valuesInfo,&value));
    api->ReleaseTensorTypeAndShapeInfo(valuesInfo);

    // Convert value type to java
    jint onnxTypeValue = convertFromONNXDataFormat(value);
    jobject onnxTensorTypeJavaValue = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxTensorTypeClazz,onnxTensorTypeMapFromInt,onnxTypeValue);
    jobject onnxJavaTypeValue = (*jniEnv)->CallStaticObjectMethod(jniEnv,onnxJavaTypeClazz,onnxJavaTypeMapFromONNXTensorType,onnxTensorTypeJavaValue);

    // Construct map info
    jmethodID mapInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,mapInfoClazz,"<init>","(ILcom/microsoft/onnxruntime/ONNXJavaType;Lcom/microsoft/onnxruntime/ONNXJavaType;)V");
    jobject mapInfo = (*jniEnv)->NewObject(jniEnv,mapInfoClazz,mapInfoConstructor,(jint)mapCount,onnxJavaTypeKey,onnxJavaTypeValue);

    // Free the intermediate tensors.
    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,keys));
    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    // Construct the ONNXMap object
    char *mapClassName = "com/microsoft/onnxruntime/ONNXMap";
    jclass mapClazz = (*jniEnv)->FindClass(jniEnv, mapClassName);
    jmethodID mapConstructor = (*jniEnv)->GetMethodID(jniEnv,mapClazz, "<init>", "(JJLcom/microsoft/onnxruntime/MapInfo;)V");
    jobject javaMap = (*jniEnv)->NewObject(jniEnv, mapClazz, mapConstructor, (jlong)map, (jlong) allocator, mapInfo);

    return javaMap;
}

jobject convertOrtValueToONNXValue(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* onnxValue) {
    // Note this is the ONNXType C enum
    ONNXType valueType;
    checkONNXStatus(jniEnv,api,api->GetValueType(onnxValue,&valueType));
    switch (valueType) {
        case ONNX_TYPE_TENSOR: {
            return createJavaTensorFromONNX(jniEnv, api, allocator, onnxValue);
        }
        case ONNX_TYPE_SEQUENCE: {
            return createJavaSequenceFromONNX(jniEnv, api, allocator, onnxValue);
            //throwONNXException(jniEnv,"Sequence is not yet implemented.");
            //break;
        }
        case ONNX_TYPE_MAP: {
            return createJavaMapFromONNX(jniEnv, api, allocator, onnxValue);
        }
        case ONNX_TYPE_UNKNOWN:
        case ONNX_TYPE_OPAQUE:
        case ONNX_TYPE_SPARSETENSOR: {
            throwONNXException(jniEnv,convertErrorCode(ORT_NOT_IMPLEMENTED),"These types are unsupported - ONNX_TYPE_UNKNOWN, ONNX_TYPE_OPAQUE, ONNX_TYPE_SPARSETENSOR.");
            break;
        }
    }
    return NULL;
}

jint throwONNXException(JNIEnv *jniEnv, int messageId, const char *message) {
    jstring messageStr = (*jniEnv)->NewStringUTF(jniEnv, message);

    char *className = "com/microsoft/onnxruntime/ONNXException";
    jclass exClazz = (*jniEnv)->FindClass(jniEnv,className);
    jmethodID exConstructor = (*jniEnv)->GetMethodID(jniEnv, exClazz, "<init>", "(ILjava/lang/String;)V");
    jobject javaException = (*jniEnv)->NewObject(jniEnv, exClazz, exConstructor, messageId, messageStr);

    return (*jniEnv)->Throw(jniEnv,javaException);
}

jint convertErrorCode(OrtErrorCode code) {
    switch (code) {
        case ORT_OK:
			return 0;
        case ORT_FAIL:
			return 1;
        case ORT_INVALID_ARGUMENT:
			return 2;
        case ORT_NO_SUCHFILE:
			return 3;
        case ORT_NO_MODEL:
			return 4;
        case ORT_ENGINE_ERROR:
			return 5;
        case ORT_RUNTIME_EXCEPTION:
			return 6;
        case ORT_INVALID_PROTOBUF:
			return 7;
        case ORT_MODEL_LOADED:
			return 8;
        case ORT_NOT_IMPLEMENTED:
			return 9;
        case ORT_INVALID_GRAPH:
			return 10;
        case ORT_EP_FAIL:
			return 11;
    }
}

void checkONNXStatus(JNIEnv *jniEnv, const OrtApi * api, OrtStatus * status) {
    if (status != NULL) {
        const char* message = api->GetErrorMessage(status);
        int len = strlen(message)+1;
        char* copy = malloc(sizeof(char)*len);
        memcpy(copy,message,len);
        int messageId = convertErrorCode(api->GetErrorCode(status));
        api->ReleaseStatus(status);
        throwONNXException(jniEnv,messageId,copy);
    }
}
