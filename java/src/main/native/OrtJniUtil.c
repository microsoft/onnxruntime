/*
 * Copyright (c) 2019, 2023 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <stdio.h>
#include "OrtJniUtil.h"

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    // To silence unused-parameter error.
    // This function must exist according to the JNI spec, but the arguments aren't necessary for the library
    // to request a specific version.
    (void)vm; (void) reserved;
    // Requesting 1.6 to support Android. Will need to be bumped to a later version to call interface default methods
    // from native code, or to access other new Java features.
    return JNI_VERSION_1_6;
}

/**
 * Must be kept in sync with ORT_LOGGING_LEVEL and the OrtLoggingLevel java enum
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
 * Must be kept in sync with ExecutionMode and SessionOptions#ExecutionMode
 */
ExecutionMode convertExecutionMode(jint mode) {
    switch (mode) {
        case 0:
            return ORT_SEQUENTIAL;
        case 1:
            return ORT_PARALLEL;
        default:
            return ORT_SEQUENTIAL;
    }
}

/**
 * Must be kept in sync with OrtSparseFormat and OnnxSparseTensor.SparseTensorType
 * @param format The Java int.
 * @return The enum.
 */
OrtSparseFormat convertToOrtSparseFormat(jint format) {
    switch (format) {
      case 0:
        return ORT_SPARSE_UNDEFINED;
      case 1:
        return ORT_SPARSE_COO;
      case 2:
        return ORT_SPARSE_CSRC;
      case 4:
        return ORT_SPARSE_BLOCK_SPARSE;
      default:
        return ORT_SPARSE_UNDEFINED;
    }
}

/**
 * Must be kept in sync with OrtSparseFormat and OnnxSparseTensor.SparseTensorType
 * @param format The enum.
 * @return The Java int.
 */
jint convertFromOrtSparseFormat(OrtSparseFormat format) {
    switch (format) {
      case ORT_SPARSE_COO:
        return 1;
      case ORT_SPARSE_CSRC:
        return 2;
      case ORT_SPARSE_BLOCK_SPARSE:
        return 4;
      case ORT_SPARSE_UNDEFINED:
      default:
        return 0;
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
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  // maps to c type uint32_t
            return 5;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   // maps to c type int32_t
            return 6;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  // maps to c type uint64_t
            return 7;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   // maps to c type int64_t
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return 9;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   // maps to c type float
            return 10;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  // maps to c type double
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
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
            return 17;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
            return 18;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
            return 19;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
            return 20;
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
        case 17:
          return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
        case 18:
          return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
        case 19:
          return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
        case 20:
          return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

size_t onnxTypeSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    // maps to c type int8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   // maps to c type int16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  // maps to c type uint32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   // maps to c type int32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   // maps to c type float
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  // maps to c type uint64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   // maps to c type int64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  // maps to c type double
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  // maps to c++ type std::string
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
        default:
            return 0;
    }
}

OrtErrorCode getTensorTypeShape(JNIEnv * jniEnv, JavaTensorTypeShape* output, const OrtApi * api, const OrtValue * value) {
  OrtTensorTypeAndShapeInfo* info;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorTypeAndShape(value, &info));
  if (code != ORT_OK) {
    return code;
  }
  code = checkOrtStatus(jniEnv, api, api->GetDimensionsCount(info, &output->dimensions));
  if (code != ORT_OK) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    return code;
  }
  code = checkOrtStatus(jniEnv, api, api->GetTensorShapeElementCount(info, &output->elementCount));
  if (code != ORT_OK) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    return code;
  }
  code = checkOrtStatus(jniEnv, api, api->GetTensorElementType(info, &output->onnxTypeEnum));
  api->ReleaseTensorTypeAndShapeInfo(info);
  if (code != ORT_OK) {
    return code;
  }

  return ORT_OK;
}

typedef union FP32 {
    int intVal;
    float floatVal;
} FP32;

jfloat convertHalfToFloat(const uint16_t half) {
    FP32 output;
    output.intVal = (((half&0x8000)<<16) | (((half&0x7c00)+0x1C000)<<13) | ((half&0x03FF)<<13));
    return output.floatVal;
}

jfloat convertBF16ToFloat(const uint16_t bf16) {
    FP32 output;
    output.intVal = bf16 << 16;
    return output.floatVal;
}

jobject convertToValueInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTypeInfo * info) {
  ONNXType type = ONNX_TYPE_UNKNOWN;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetOnnxTypeFromTypeInfo(info, &type));
  if (code != ORT_OK) {
    return NULL;
  }

  switch (type) {
    case ONNX_TYPE_TENSOR:
    case ONNX_TYPE_SPARSETENSOR: {
      const OrtTensorTypeAndShapeInfo* tensorInfo = NULL;
      code = checkOrtStatus(jniEnv, api, api->CastTypeInfoToTensorInfo(info, &tensorInfo));
      if (code == ORT_OK) {
        return convertToTensorInfo(jniEnv, api, tensorInfo);
      } else {
        return NULL;
      }
    }
    case ONNX_TYPE_SEQUENCE: {
      const OrtSequenceTypeInfo* sequenceInfo = NULL;
      code = checkOrtStatus(jniEnv, api, api->CastTypeInfoToSequenceTypeInfo(info, &sequenceInfo));
      if (code == ORT_OK) {
        return convertToSequenceInfo(jniEnv, api, sequenceInfo);
      } else {
        return NULL;
      }
    }
    case ONNX_TYPE_MAP: {
      const OrtMapTypeInfo* mapInfo = NULL;
      code = checkOrtStatus(jniEnv, api, api->CastTypeInfoToMapTypeInfo(info, &mapInfo));
      if (code == ORT_OK) {
        return convertToMapInfo(jniEnv, api, mapInfo);
      } else {
        return NULL;
      }
    }
    case ONNX_TYPE_UNKNOWN:
    case ONNX_TYPE_OPAQUE:
    default: {
      throwOrtException(jniEnv,convertErrorCode(ORT_NOT_IMPLEMENTED),"Invalid ONNXType found.");
      return NULL;
    }
  }
}

jobject convertToTensorInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTensorTypeAndShapeInfo * info) {
  // Extract the information from the info struct.
  ONNXTensorElementDataType onnxType;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorElementType(info, &onnxType));
  if (code != ORT_OK) {
    return NULL;
  }
  size_t numDim = 0;
  code = checkOrtStatus(jniEnv, api, api->GetDimensionsCount(info, &numDim));
  if (code != ORT_OK) {
    return NULL;
  }
  //printf("numDim %d\n",numDim);
  int64_t* dimensions = (int64_t*) malloc(sizeof(int64_t)*numDim);
  code = checkOrtStatus(jniEnv, api, api->GetDimensions(info, dimensions, numDim));
  if (code != ORT_OK) {
    free((void*) dimensions);
    return NULL;
  }
  jint onnxTypeInt = convertFromONNXDataFormat(onnxType);

  // Create the long array for the shape.
  jlongArray shape = (*jniEnv)->NewLongArray(jniEnv, safecast_size_t_to_jsize(numDim));
  (*jniEnv)->SetLongArrayRegion(jniEnv, shape, 0, safecast_size_t_to_jsize(numDim), (jlong*)dimensions);
  // Free the dimensions array
  free(dimensions);
  dimensions = NULL;

  // Create the TensorInfo object
  static const char *tensorInfoClassName = "ai/onnxruntime/TensorInfo";
  jclass clazz = (*jniEnv)->FindClass(jniEnv, tensorInfoClassName);
  jmethodID tensorInfoConstructor = (*jniEnv)->GetMethodID(jniEnv,clazz, "<init>", "([JI)V");
  //printf("TensorInfo class %p, methodID %p\n",clazz,tensorInfoConstructor);
  jobject tensorInfo = (*jniEnv)->NewObject(jniEnv, clazz, tensorInfoConstructor, shape, onnxTypeInt);
  return tensorInfo;
}

jobject convertToMapInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtMapTypeInfo * info) {
  // Extract the key type
  ONNXTensorElementDataType keyType;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetMapKeyType(info, &keyType));
  if (code != ORT_OK) {
    return NULL;
  }

  // according to include/onnxruntime/core/framework/data_types.h only the following values are supported.
  // string, int64, float, double
  // So extract the value type, then convert it to a tensor type so we can get it's element type.
  OrtTypeInfo* valueTypeInfo = NULL;
  code = checkOrtStatus(jniEnv, api, api->GetMapValueType(info, &valueTypeInfo));
  if (code != ORT_OK) {
    return NULL;
  }
  const OrtTensorTypeAndShapeInfo* tensorValueInfo = NULL;
  code = checkOrtStatus(jniEnv, api, api->CastTypeInfoToTensorInfo(valueTypeInfo, &tensorValueInfo));
  if (code != ORT_OK) {
    api->ReleaseTypeInfo(valueTypeInfo);
    return NULL;
  }
  ONNXTensorElementDataType valueType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  code = checkOrtStatus(jniEnv, api, api->GetTensorElementType(tensorValueInfo, &valueType));
  api->ReleaseTypeInfo(valueTypeInfo);
  tensorValueInfo = NULL;
  valueTypeInfo = NULL;
  if (code != ORT_OK) {
    return NULL;
  }

  // Convert key type to java
  jint onnxTypeKey = convertFromONNXDataFormat(keyType);
  // Convert value type to java
  jint onnxTypeValue = convertFromONNXDataFormat(valueType);

  // Get the map info class
  static const char *mapInfoClassName = "ai/onnxruntime/MapInfo";
  jclass mapInfoClazz = (*jniEnv)->FindClass(jniEnv, mapInfoClassName);
  jmethodID mapInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, mapInfoClazz, "<init>", "(III)V");

  // Construct map info
  jobject mapInfo = (*jniEnv)->NewObject(jniEnv, mapInfoClazz, mapInfoConstructor, (jint)-1, onnxTypeKey, onnxTypeValue);

  return mapInfo;
}

jobject convertToSequenceInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtSequenceTypeInfo * info) {
  // Get the sequence info class
  static const char *sequenceInfoClassName = "ai/onnxruntime/SequenceInfo";
  jclass sequenceInfoClazz = (*jniEnv)->FindClass(jniEnv, sequenceInfoClassName);
  jobject sequenceInfo = NULL;

  // according to include/onnxruntime/core/framework/data_types.h the following values are supported.
  // tensor types, map<string,float> and map<long,float>
  OrtTypeInfo* elementTypeInfo = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetSequenceElementType(info, &elementTypeInfo));
  if (code != ORT_OK) {
    return NULL;
  }
  ONNXType type = ONNX_TYPE_UNKNOWN;
  code = checkOrtStatus(jniEnv, api, api->GetOnnxTypeFromTypeInfo(elementTypeInfo, &type));
  if (code != ORT_OK) {
    goto sequence_cleanup;
  }

  switch (type) {
    case ONNX_TYPE_TENSOR: {
      // Figure out element type
      const OrtTensorTypeAndShapeInfo* elementTensorInfo = NULL;
      code = checkOrtStatus(jniEnv, api, api->CastTypeInfoToTensorInfo(elementTypeInfo, &elementTensorInfo));
      if (code != ORT_OK) {
        goto sequence_cleanup;
      }
      ONNXTensorElementDataType element = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      code = checkOrtStatus(jniEnv, api, api->GetTensorElementType(elementTensorInfo, &element));
      if (code != ORT_OK) {
        goto sequence_cleanup;
      }

      // Convert element type into ONNXTensorType
      jint onnxTypeInt = convertFromONNXDataFormat(element);

      // Construct sequence info
      jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, sequenceInfoClazz, "<init>", "(II)V");
      sequenceInfo = (*jniEnv)->NewObject(jniEnv, sequenceInfoClazz, sequenceInfoConstructor, (jint)-1, onnxTypeInt);
      break;
    }
    case ONNX_TYPE_MAP: {
      // Extract the map info
      const OrtMapTypeInfo* mapInfo = NULL;
      code = checkOrtStatus(jniEnv, api, api->CastTypeInfoToMapTypeInfo(elementTypeInfo, &mapInfo));
      if (code != ORT_OK) {
        goto sequence_cleanup;
      }

      // Convert it using the existing convert function
      jobject javaMapInfo = convertToMapInfo(jniEnv, api, mapInfo);

      // Construct sequence info
      jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, sequenceInfoClazz, "<init>", "(ILai/onnxruntime/MapInfo;)V");
      sequenceInfo = (*jniEnv)->NewObject(jniEnv, sequenceInfoClazz, sequenceInfoConstructor, (jint)-1, javaMapInfo);
      break;
    }
    default: {
      throwOrtException(jniEnv, convertErrorCode(ORT_INVALID_ARGUMENT), "Invalid element type found in sequence");
      break;
    }
  }

sequence_cleanup:
  api->ReleaseTypeInfo(elementTypeInfo);
  elementTypeInfo = NULL;

  return sequenceInfo;
}

int64_t copyJavaToPrimitiveArray(JNIEnv* jniEnv, ONNXTensorElementDataType onnxType, jarray inputArray, uint8_t* outputTensor) {
    int32_t inputLength = (*jniEnv)->GetArrayLength(jniEnv, inputArray);
    int64_t consumedSize = inputLength * onnxTypeSize(onnxType);
    switch (onnxType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {  // maps to c type int8_t
            jbyteArray typedArr = (jbyteArray)inputArray;
            (*jniEnv)->GetByteArrayRegion(jniEnv, typedArr, 0, inputLength, (jbyte * )outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: { // maps to c type int16_t
            jshortArray typedArr = (jshortArray)inputArray;
            (*jniEnv)->GetShortArrayRegion(jniEnv, typedArr, 0, inputLength, (jshort * )outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      // maps to c type uint32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: { // maps to c type int32_t
            jintArray typedArr = (jintArray)inputArray;
            (*jniEnv)->GetIntArrayRegion(jniEnv, typedArr, 0, inputLength, (jint * )outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      // maps to c type uint64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: { // maps to c type int64_t
            jlongArray typedArr = (jlongArray)inputArray;
            (*jniEnv)->GetLongArrayRegion(jniEnv, typedArr, 0, inputLength, (jlong * )outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    // Non-IEEE floating-point format based on IEEE754 single-precision
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "16-bit float not supported.");
            return -1;
            /*
            float *floatArr = malloc(sizeof(float) * inputLength);
            uint16_t *halfArr = (uint16_t *) outputTensor;
            for (uint32_t i = 0; i < inputLength; i++) {
                floatArr[i] = convertHalfToFloat(halfArr[i]);
            }
            jfloatArray typedArr = (jfloatArray) inputArray;
            (*jniEnv)->GetFloatArrayRegion(jniEnv, typedArr, 0, inputLength, floatArr);
            free(floatArr);
            return consumedSize;
            */
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: { // maps to c type float
            jfloatArray typedArr = (jfloatArray)inputArray;
            (*jniEnv)->GetFloatArrayRegion(jniEnv, typedArr, 0, inputLength, (jfloat * )outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {    // maps to c type double
            jdoubleArray typedArr = (jdoubleArray)inputArray;
            (*jniEnv)->GetDoubleArrayRegion(jniEnv, typedArr, 0, inputLength, (jdouble * )outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: { // maps to c++ type std::string
            throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "String is not supported.");
            return -1;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            jbooleanArray typedArr = (jbooleanArray)inputArray;
            (*jniEnv)->GetBooleanArrayRegion(jniEnv, typedArr, 0, inputLength, (jboolean *)outputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   // complex with float32 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  // complex with float64 real and imaginary components
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        default: {
            throwOrtException(jniEnv, convertErrorCode(ORT_INVALID_ARGUMENT), "Invalid outputTensor element type.");
            return -1;
        }
    }
}

int64_t copyJavaToTensor(JNIEnv* jniEnv, ONNXTensorElementDataType onnxType, size_t tensorSize, size_t dimensionsRemaining, jarray inputArray, uint8_t* outputTensor) {
    if (dimensionsRemaining == 1) {
        // write out 1d array of the respective primitive type
        return copyJavaToPrimitiveArray(jniEnv, onnxType, inputArray, outputTensor);
    } else {
        // recurse through the dimensions
        // Java arrays are objects until the final dimension
        jobjectArray inputObjArr = (jobjectArray)inputArray;
        int32_t dimLength = (*jniEnv)->GetArrayLength(jniEnv, inputObjArr);
        int64_t sizeConsumed = 0;
        for (int32_t i = 0; i < dimLength; i++) {
            jarray childArr = (jarray) (*jniEnv)->GetObjectArrayElement(jniEnv, inputObjArr, i);
            int64_t consumed = copyJavaToTensor(jniEnv, onnxType, tensorSize - sizeConsumed, dimensionsRemaining - 1, childArr, outputTensor + sizeConsumed);
            sizeConsumed += consumed;
            // Cleanup reference to childArr so it doesn't prevent GC.
            (*jniEnv)->DeleteLocalRef(jniEnv, childArr);
            // If we failed to copy an array then break and return.
            if (consumed == -1) {
              return -1;
            }
        }
        return sizeConsumed;
    }
}

int64_t copyPrimitiveArrayToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, const uint8_t* inputTensor, jarray outputArray) {
    int32_t outputLength = (*jniEnv)->GetArrayLength(jniEnv, outputArray);
    if (outputLength == 0) return 0;
    int64_t consumedSize = outputLength * onnxTypeSize(onnxType);
    switch (onnxType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   // maps to c type uint8_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {  // maps to c type int8_t
            jbyteArray typedArr = (jbyteArray)outputArray;
            (*jniEnv)->SetByteArrayRegion(jniEnv, typedArr, 0, outputLength, (jbyte * )inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: { // maps to c type int16_t
            jshortArray typedArr = (jshortArray)outputArray;
            (*jniEnv)->SetShortArrayRegion(jniEnv, typedArr, 0, outputLength, (jshort * )inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  // maps to c type uint32_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: { // maps to c type int32_t
            jintArray typedArr = (jintArray)outputArray;
            (*jniEnv)->SetIntArrayRegion(jniEnv, typedArr, 0, outputLength, (jint * )inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  // maps to c type uint64_t
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: { // maps to c type int64_t
            jlongArray typedArr = (jlongArray)outputArray;
            (*jniEnv)->SetLongArrayRegion(jniEnv, typedArr, 0, outputLength, (jlong * )inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: { // stored as a uint16_t
            jfloat *floatArr = malloc(sizeof(jfloat) * outputLength);
            if (floatArr == NULL) {
                throwOrtException(jniEnv, 1, "Not enough memory");
                return -1;
            }
            uint16_t *halfArr = (uint16_t *)inputTensor;
            for (int32_t i = 0; i < outputLength; i++) {
                floatArr[i] = convertHalfToFloat(halfArr[i]);
            }
            jfloatArray typedArr = (jfloatArray)outputArray;
            (*jniEnv)->SetFloatArrayRegion(jniEnv, typedArr, 0, outputLength, floatArr);
            free(floatArr);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: { // stored as a uint16_t
            jfloat *floatArr = malloc(sizeof(jfloat) * outputLength);
            if (floatArr == NULL) {
                throwOrtException(jniEnv, 1, "Not enough memory");
                return -1;
            }
            uint16_t *bf16Arr = (uint16_t *)inputTensor;
            for (int32_t i = 0; i < outputLength; i++) {
                floatArr[i] = convertBF16ToFloat(bf16Arr[i]);
            }
            jfloatArray typedArr = (jfloatArray)outputArray;
            (*jniEnv)->SetFloatArrayRegion(jniEnv, typedArr, 0, outputLength, floatArr);
            free(floatArr);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: { // maps to c type float
            jfloatArray typedArr = (jfloatArray)outputArray;
            (*jniEnv)->SetFloatArrayRegion(jniEnv, typedArr, 0, outputLength, (jfloat * )inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: { // maps to c type double
            jdoubleArray typedArr = (jdoubleArray)outputArray;
            (*jniEnv)->SetDoubleArrayRegion(jniEnv, typedArr, 0, outputLength, (jdouble * )inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: { // maps to c++ type std::string
            // Shouldn't reach here, as it's caught by a different codepath in the initial OnnxTensor.getArray call.
            throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "String is not supported by this codepath, please raise a Github issue as it should not reach here.");
            return -1;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            jbooleanArray typedArr = (jbooleanArray)outputArray;
            (*jniEnv)->SetBooleanArrayRegion(jniEnv, typedArr, 0, outputLength, (jboolean *)inputTensor);
            return consumedSize;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: {
          // complex with float32 real and imaginary components
          throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "Invalid inputTensor element type ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64.");
          return -1;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: {
          // complex with float64 real and imaginary components
          throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "Invalid inputTensor element type ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128.");
          return -1;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        default: {
          throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "Invalid inputTensor element type ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED.");
          return -1;
        }
    }
}

int64_t copyTensorToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, const uint8_t* inputTensor, size_t tensorSize,
                        size_t dimensionsRemaining, jarray outputArray) {
  if (dimensionsRemaining == 1) {
    // write out 1d array of the respective primitive type
    return copyPrimitiveArrayToJava(jniEnv, onnxType, inputTensor, outputArray);
  } else {
    // recurse through the dimensions
    // Java arrays are objects until the final dimension
    jobjectArray outputObjArr = (jobjectArray)outputArray;
    int32_t dimLength = (*jniEnv)->GetArrayLength(jniEnv, outputObjArr);
    int64_t sizeConsumed = 0;
    for (int32_t i = 0; i < dimLength; i++) {
      jarray childArr = (jarray) (*jniEnv)->GetObjectArrayElement(jniEnv, outputObjArr, i);
      int64_t consumed = copyTensorToJava(jniEnv, onnxType, inputTensor + sizeConsumed, tensorSize - sizeConsumed, dimensionsRemaining - 1, childArr);
      sizeConsumed += consumed;
      // Cleanup reference to childArr so it doesn't prevent GC.
      (*jniEnv)->DeleteLocalRef(jniEnv, childArr);
      // If we failed to copy an array then break and return.
      if (consumed == -1) {
        return -1;
      }
    }
    return sizeConsumed;
  }
}

jobject createStringFromStringTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
  jobject tempString = NULL;
  // Get the buffer size needed
  size_t totalStringLength = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetStringTensorDataLength(tensor, &totalStringLength));
  if (code != ORT_OK) {
    return NULL;
  }

  // Create the character and offset buffers, character is one larger to allow zero termination.
  char * characterBuffer = malloc(sizeof(char)*(totalStringLength+1));
  if (characterBuffer == NULL) {
    throwOrtException(jniEnv, 1, "OOM error");
  } else {
    size_t * offsets = malloc(sizeof(size_t));
    if (offsets != NULL) {
      // Get a view on the String data
      code = checkOrtStatus(jniEnv, api, api->GetStringTensorContent(tensor, characterBuffer, totalStringLength, offsets, 1));

      if (code == ORT_OK) {
        size_t curSize = (offsets[0]) + 1;
        characterBuffer[curSize-1] = '\0';
        tempString = (*jniEnv)->NewStringUTF(jniEnv, characterBuffer);
      }

      free((void*)characterBuffer);
      free((void*)offsets);
    }
  }

  return tempString;
}

OrtErrorCode copyStringTensorToArray(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor, size_t length, jobjectArray outputArray) {
  size_t bufferSize = 16;
  char * tempBuffer = malloc(bufferSize);
  if (tempBuffer == NULL) {
    throwOrtException(jniEnv, 1, "Not enough memory");
    return ORT_FAIL;
  }
  // Get the buffer size needed
  size_t totalStringLength = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetStringTensorDataLength(tensor, &totalStringLength));
  if (code != ORT_OK) {
    return code;
  }

  // Create the character and offset buffers
  char * characterBuffer = malloc(sizeof(char)*(totalStringLength+length));
  if (characterBuffer == NULL) {
    throwOrtException(jniEnv, 1, "Not enough memory");
    return ORT_FAIL;
  }
  // length + 1 as we need to write out the final offset
  size_t * offsets = allocarray(sizeof(size_t), length+1);
  if (offsets == NULL) {
    free((void*)characterBuffer);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return ORT_FAIL;
  }

  // Get a view on the String data
  code = checkOrtStatus(jniEnv, api, api->GetStringTensorContent(tensor, characterBuffer, totalStringLength, offsets, length));
  if (code == ORT_OK) {
    // Get the final offset, write to the end of the array.
    code = checkOrtStatus(jniEnv, api, api->GetStringTensorDataLength(tensor, offsets+length));
    if (code == ORT_OK) {
      for (size_t i = 0; i < length; i++) {
        size_t curSize = (offsets[i+1] - offsets[i]) + 1;
        if (curSize > bufferSize) {
          char* oldTempBuffer = tempBuffer;
          tempBuffer = realloc(oldTempBuffer, sizeof(char) * curSize);
          if (tempBuffer == NULL) {
            free(oldTempBuffer);
            throwOrtException(jniEnv, 1, "Not enough memory");
            goto string_tensor_cleanup;
          }
          bufferSize = curSize;
        }
        memcpy(tempBuffer,characterBuffer+offsets[i],curSize);
        tempBuffer[curSize-1] = '\0';
        jobject tempString = (*jniEnv)->NewStringUTF(jniEnv,tempBuffer);
        (*jniEnv)->SetObjectArrayElement(jniEnv,outputArray,safecast_size_t_to_jsize(i),tempString);
      }
    }
  }

string_tensor_cleanup:
  if (tempBuffer != NULL) {
    free((void*)tempBuffer);
  }
  free((void*)offsets);
  free((void*)characterBuffer);
  return code;
}

jobjectArray createStringArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
    // Extract tensor info
    OrtTensorTypeAndShapeInfo* tensorInfo = NULL;
    OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorTypeAndShape(tensor, &tensorInfo));
    if (code != ORT_OK) {
        return NULL;
    }

    // Get the element count of this tensor
    size_t length = 0;
    code = checkOrtStatus(jniEnv, api, api->GetTensorShapeElementCount(tensorInfo, &length));
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);
    if (code != ORT_OK) {
        return NULL;
    }

    // Create the java array
    jclass stringClazz = (*jniEnv)->FindClass(jniEnv, "java/lang/String");
    jobjectArray outputArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_size_t_to_jsize(length), stringClazz, NULL);

    code = copyStringTensorToArray(jniEnv, api, tensor, length, outputArray);
    if (code != ORT_OK) {
        outputArray = NULL;
    }

    return outputArray;
}

jlongArray createLongArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
  jlongArray outputArray = NULL;
  // Extract tensor type
  OrtTensorTypeAndShapeInfo* tensorInfo = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor, &tensorInfo));
  if (code == ORT_OK) {
    ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    code = checkOrtStatus(jniEnv,api,api->GetTensorElementType(tensorInfo, &value));
    if ((code == ORT_OK) && ((value == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) || (value == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64))) {
      // Get the element count of this tensor
      size_t length = 0;
      code = checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo, &length));
      if (code == ORT_OK) {
        // Extract the values
        uint8_t* arr = NULL;
        code = checkOrtStatus(jniEnv,api,api->GetTensorMutableData(tensor, (void**)&arr));
        if (code == ORT_OK) {
          // Create the java array and copy to it.
          outputArray = (*jniEnv)->NewLongArray(jniEnv, safecast_size_t_to_jsize(length));
          int64_t consumed = copyPrimitiveArrayToJava(jniEnv, value, arr, outputArray);
          if (consumed == -1) {
            outputArray = NULL;
          }
        }
      }
    }
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);
  }
  return outputArray;
}

jfloatArray createFloatArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
  jfloatArray outputArray = NULL;
  // Extract tensor type
  OrtTensorTypeAndShapeInfo* tensorInfo = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor, &tensorInfo));
  if (code == ORT_OK) {
    ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    code = checkOrtStatus(jniEnv,api,api->GetTensorElementType(tensorInfo, &value));
    if ((code == ORT_OK) && (value == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        // Get the element count of this tensor
        size_t length = 0;
        code = checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo, &length));
        if (code == ORT_OK) {
            // Extract the values
            uint8_t* arr = NULL;
            code = checkOrtStatus(jniEnv,api,api->GetTensorMutableData(tensor, (void**)&arr));
            if (code == ORT_OK) {
                // Create the java array and copy to it.
                outputArray = (*jniEnv)->NewFloatArray(jniEnv, safecast_size_t_to_jsize(length));
                int64_t consumed = copyPrimitiveArrayToJava(jniEnv, value, arr, outputArray);
                if (consumed == -1) {
                    outputArray = NULL;
                }
            }
        }
    }
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);
  }
  return outputArray;
}

jdoubleArray createDoubleArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor) {
  jdoubleArray outputArray = NULL;
  // Extract tensor type
  OrtTensorTypeAndShapeInfo* tensorInfo = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor, &tensorInfo));
  if (code == ORT_OK) {
    ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    code = checkOrtStatus(jniEnv,api,api->GetTensorElementType(tensorInfo, &value));
    if ((code == ORT_OK) && (value == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
        // Get the element count of this tensor
        size_t length = 0;
        code = checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(tensorInfo, &length));
        if (code == ORT_OK) {
            // Extract the values
            uint8_t* arr = NULL;
            code = checkOrtStatus(jniEnv,api,api->GetTensorMutableData(tensor, (void**)&arr));
            if (code == ORT_OK) {
                // Create the java array and copy to it.
                outputArray = (*jniEnv)->NewDoubleArray(jniEnv, safecast_size_t_to_jsize(length));
                int64_t consumed = copyPrimitiveArrayToJava(jniEnv, value, arr, outputArray);
                if (consumed == -1) {
                    outputArray = NULL;
                }
            }
        }
    }
    api->ReleaseTensorTypeAndShapeInfo(tensorInfo);
  }
  return outputArray;
}

jobject createJavaTensorFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor) {
  // Extract the type information
  OrtTensorTypeAndShapeInfo* info = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetTensorTypeAndShape(tensor, &info));
  if (code != ORT_OK) {
    return NULL;
  }

  // Construct the TensorInfo object
  jobject tensorInfo = convertToTensorInfo(jniEnv, api, info);
  // Release the info object
  api->ReleaseTensorTypeAndShapeInfo(info);
  if (tensorInfo == NULL) {
    return NULL;
  }

  // Construct the ONNXTensor object
  static const char *tensorClassName = "ai/onnxruntime/OnnxTensor";
  jclass clazz = (*jniEnv)->FindClass(jniEnv, tensorClassName);
  jmethodID tensorConstructor = (*jniEnv)->GetMethodID(jniEnv,clazz, "<init>", "(JJLai/onnxruntime/TensorInfo;)V");
  jobject javaTensor = (*jniEnv)->NewObject(jniEnv, clazz, tensorConstructor, (jlong) tensor, (jlong) allocator, tensorInfo);

  return javaTensor;
}

jobject createJavaSparseTensorFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor) {
  // Extract the type information
  OrtTensorTypeAndShapeInfo* info;
  OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape(tensor, &info));
  if (code != ORT_OK) {
    return NULL;
  }

  // Construct the TensorInfo object
  jobject tensorInfo = convertToTensorInfo(jniEnv, api, info);

  // Release the info object
  api->ReleaseTensorTypeAndShapeInfo(info);
  if (tensorInfo == NULL) {
    return NULL;
  }

  // Lookup the sparse tensor type enum
  OrtSparseFormat format;
  code = checkOrtStatus(jniEnv,api,api->GetSparseTensorFormat(tensor, &format));
  if (code != ORT_OK) {
    return NULL;
  }
  jint sparseTensorInt = convertFromOrtSparseFormat(format);

  // Construct the ONNXTensor object
  char *tensorClassName = "ai/onnxruntime/OnnxSparseTensor";
  jclass clazz = (*jniEnv)->FindClass(jniEnv, tensorClassName);
  jmethodID tensorConstructor = (*jniEnv)->GetMethodID(jniEnv, clazz, "<init>", "(JJILai/onnxruntime/TensorInfo;)V");
  jobject javaSparseTensor = (*jniEnv)->NewObject(jniEnv, clazz, tensorConstructor, (jlong) tensor, (jlong) allocator, sparseTensorInt, tensorInfo);

  return javaSparseTensor;
}

jobject createJavaSequenceFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* sequence) {
  // Get the sequence info class
  static const char *sequenceInfoClassName = "ai/onnxruntime/SequenceInfo";
  jclass sequenceInfoClazz = (*jniEnv)->FindClass(jniEnv, sequenceInfoClassName);

  // setup return value
  jobject sequenceInfo = NULL;

  // Get the element count of this sequence
  size_t count = 0;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetValueCount(sequence, &count));
  if (code != ORT_OK) {
    return NULL;
  } else if (count == 0) {
    // Construct empty sequence info
    jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, sequenceInfoClazz, "<init>", "(II)V");
    sequenceInfo = (*jniEnv)->NewObject(jniEnv, sequenceInfoClazz, sequenceInfoConstructor, 0,
                       convertFromONNXDataFormat(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED));
  } else {
    // Extract the first element
    OrtValue* firstElement = NULL;
    code = checkOrtStatus(jniEnv, api, api->GetValue(sequence, 0, allocator, &firstElement));
    if (code != ORT_OK) {
      return NULL;
    }
    ONNXType elementType = ONNX_TYPE_UNKNOWN;
    code = checkOrtStatus(jniEnv, api, api->GetValueType(firstElement, &elementType));
    if (code == ORT_OK) {
      switch (elementType) {
        case ONNX_TYPE_TENSOR: {
          // Figure out element type
          OrtTensorTypeAndShapeInfo* firstElementInfo = NULL;
          code = checkOrtStatus(jniEnv, api, api->GetTensorTypeAndShape(firstElement, &firstElementInfo));
          if (code == ORT_OK) {
            ONNXTensorElementDataType element = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            code = checkOrtStatus(jniEnv, api, api->GetTensorElementType(firstElementInfo, &element));
            api->ReleaseTensorTypeAndShapeInfo(firstElementInfo);
            if (code == ORT_OK) {
              // Convert element type into ONNXTensorType
              jint onnxTypeInt = convertFromONNXDataFormat(element);

              // Construct sequence info
              jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, sequenceInfoClazz, "<init>", "(II)V");
              sequenceInfo = (*jniEnv)->NewObject(jniEnv, sequenceInfoClazz, sequenceInfoConstructor, (jint)count, onnxTypeInt);
            }
          }
          break;
        }
        case ONNX_TYPE_MAP: {
          jobject mapInfo = createMapInfoFromValue(jniEnv, api, allocator, firstElement);
          if (mapInfo != NULL) {
              // Construct sequence info
              jmethodID sequenceInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, sequenceInfoClazz, "<init>", "(ILai/onnxruntime/MapInfo;)V");
              sequenceInfo = (*jniEnv)->NewObject(jniEnv, sequenceInfoClazz, sequenceInfoConstructor, (jint)count, mapInfo);
          }
          break;
        }
        default: {
          throwOrtException(jniEnv, convertErrorCode(ORT_INVALID_ARGUMENT), "Invalid element type found in sequence");
          break;
        }
      }
    }
    // Free the intermediate tensor.
    api->ReleaseValue(firstElement);
  }


  jobject javaSequence = NULL;
  if (sequenceInfo != NULL) {
    // Construct the ONNXSequence object
    static const char *sequenceClassName = "ai/onnxruntime/OnnxSequence";
    jclass sequenceClazz = (*jniEnv)->FindClass(jniEnv, sequenceClassName);
    jmethodID sequenceConstructor = (*jniEnv)->GetMethodID(jniEnv, sequenceClazz, "<init>", "(JJLai/onnxruntime/SequenceInfo;)V");
    javaSequence = (*jniEnv)->NewObject(jniEnv, sequenceClazz, sequenceConstructor, (jlong)sequence, (jlong)allocator, sequenceInfo);
  }

  return javaSequence;
}

jobject createJavaMapFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* map) {
  jobject mapInfo = createMapInfoFromValue(jniEnv, api, allocator, map);
  if (mapInfo == NULL) {
    return NULL;
  }

  // Get the map class & constructor
  static const char *mapClassName = "ai/onnxruntime/OnnxMap";
  jclass mapClazz = (*jniEnv)->FindClass(jniEnv, mapClassName);
  jmethodID mapConstructor = (*jniEnv)->GetMethodID(jniEnv, mapClazz, "<init>", "(JJLai/onnxruntime/MapInfo;)V");

  // Construct the ONNXMap object
  jobject javaMap = (*jniEnv)->NewObject(jniEnv, mapClazz, mapConstructor, (jlong)map, (jlong) allocator, mapInfo);

  return javaMap;
}

jobject createMapInfoFromValue(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator * allocator, const OrtValue * map) {
  // Extract key
  OrtValue* keys = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetValue(map, 0, allocator, &keys));
  if (code != ORT_OK) {
    return NULL;
  }

  JavaTensorTypeShape keyInfo;
  code = getTensorTypeShape(jniEnv, &keyInfo, api, keys);
  api->ReleaseValue(keys);
  if (code != ORT_OK) {
    return NULL;
  }

  // Extract value
  OrtValue* values = NULL;
  code = checkOrtStatus(jniEnv, api, api->GetValue(map, 1, allocator, &values));
  if (code != ORT_OK) {
    return NULL;
  }

  JavaTensorTypeShape valueInfo;
  code = getTensorTypeShape(jniEnv, &valueInfo, api, values);
  api->ReleaseValue(values);
  if (code != ORT_OK) {
    return NULL;
  }

  // Convert key and value type to java
  jint onnxTypeKey = convertFromONNXDataFormat(keyInfo.onnxTypeEnum);
  jint onnxTypeValue = convertFromONNXDataFormat(valueInfo.onnxTypeEnum);

  // Get the map info class & constructor
  static const char *mapInfoClassName = "ai/onnxruntime/MapInfo";
  jclass mapInfoClazz = (*jniEnv)->FindClass(jniEnv, mapInfoClassName);
  jmethodID mapInfoConstructor = (*jniEnv)->GetMethodID(jniEnv, mapInfoClazz, "<init>", "(III)V");

  // Construct map info
  jobject mapInfo = (*jniEnv)->NewObject(jniEnv, mapInfoClazz, mapInfoConstructor, (jint)keyInfo.elementCount, onnxTypeKey, onnxTypeValue);
  return mapInfo;
}

jobject convertOrtValueToONNXValue(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* onnxValue) {
  // Note this is the ONNXType C enum
  ONNXType valueType = ONNX_TYPE_UNKNOWN;
  OrtErrorCode code = checkOrtStatus(jniEnv,api,api->GetValueType(onnxValue,&valueType));
  if (code != ORT_OK) {
    return NULL;
  }
  switch (valueType) {
    case ONNX_TYPE_TENSOR: {
      return createJavaTensorFromONNX(jniEnv, api, allocator, onnxValue);
    }
    case ONNX_TYPE_SEQUENCE: {
      return createJavaSequenceFromONNX(jniEnv, api, allocator, onnxValue);
    }
    case ONNX_TYPE_MAP: {
      return createJavaMapFromONNX(jniEnv, api, allocator, onnxValue);
    }
    case ONNX_TYPE_SPARSETENSOR: {
      return createJavaSparseTensorFromONNX(jniEnv, api, allocator, onnxValue);
    }
    case ONNX_TYPE_UNKNOWN:
    case ONNX_TYPE_OPAQUE:
    case ONNX_TYPE_OPTIONAL:
    default: {
      throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "These types are unsupported - ONNX_TYPE_UNKNOWN, ONNX_TYPE_OPAQUE, ONNX_TYPE_OPTIONAL.");
      return NULL;
    }
  }
}

jint throwOrtException(JNIEnv *jniEnv, int messageId, const char *message) {
  jstring messageStr = (*jniEnv)->NewStringUTF(jniEnv, message);

  static const char *className = "ai/onnxruntime/OrtException";
  jclass exClazz = (*jniEnv)->FindClass(jniEnv, className);
  jmethodID exConstructor = (*jniEnv)->GetMethodID(jniEnv, exClazz, "<init>", "(ILjava/lang/String;)V");
  jobject javaException = (*jniEnv)->NewObject(jniEnv, exClazz, exConstructor, messageId, messageStr);

  return (*jniEnv)->Throw(jniEnv, javaException);
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
        default:
            return -1; // Unknown error code
    }
}

OrtErrorCode checkOrtStatus(JNIEnv *jniEnv, const OrtApi * api, OrtStatus * status) {
    if (status == NULL) {
        return ORT_OK;
    }
    const char* message = api->GetErrorMessage(status);
    OrtErrorCode errCode = api->GetErrorCode(status);
    size_t len = strlen(message)+1;
    char* copy = malloc(sizeof(char)*len);
    if (copy == NULL) {
      api->ReleaseStatus(status);
      throwOrtException(jniEnv, 1, "Not enough memory");
      return ORT_FAIL;
    }
    memcpy(copy,message,len);
    int messageId = convertErrorCode(errCode);
    api->ReleaseStatus(status);
    throwOrtException(jniEnv,messageId,copy);
    return errCode;
}

jsize safecast_size_t_to_jsize(size_t v) {
#ifndef NDEBUG
  jsize result = (jsize)v;
  if (v != (size_t)result) {
    abort();
  }
  return result;
#else
  return (jsize)v;
#endif
}

jsize safecast_int64_to_jsize(int64_t v) {
#ifndef NDEBUG
  jsize result = (jsize)v;
  if (v != (int64_t)result) {
    abort();
  }
  return result;
#else
  return (jsize)v;
#endif
}
