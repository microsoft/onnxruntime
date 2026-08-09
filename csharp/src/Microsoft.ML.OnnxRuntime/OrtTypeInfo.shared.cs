// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


using Microsoft.ML.OnnxRuntime.Tensors;
using System;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This class retrieves Type Information for input/outputs of the model.
    /// </summary>
    public class OrtTypeInfo
    {
        private OrtTensorTypeAndShapeInfo? _tensorTypeAndShape;
        private OrtSequenceOrOptionalTypeInfo? _sequenceOrOptional;
        private OrtMapTypeInfo? _mapTypeInfo;

        internal OrtTypeInfo(IntPtr handle)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOnnxTypeFromTypeInfo(handle, out IntPtr onnxType));
            OnnxType = (OnnxValueType)onnxType;

            switch (OnnxType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                case OnnxValueType.ONNX_TYPE_SPARSETENSOR:
                    {
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToTensorInfo(handle, out IntPtr tensorInfo));
                        if (tensorInfo == IntPtr.Zero)
                        {
                            throw new OnnxRuntimeException(ErrorCode.Fail,
                                "Type Information indicates a tensor, but casting to TensorInfo fails");
                        }
                        _tensorTypeAndShape = new OrtTensorTypeAndShapeInfo(tensorInfo);
                    }
                    break;
                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    {
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToSequenceTypeInfo(handle, out IntPtr sequenceInfo));
                        if (sequenceInfo == IntPtr.Zero)
                        {
                            throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                                   "TypeInfo cast to SequenceTypeInfo failed. The object does not represent a sequence");
                        }
                        _sequenceOrOptional = new OrtSequenceOrOptionalTypeInfo(sequenceInfo);
                    }
                    break;
                case OnnxValueType.ONNX_TYPE_MAP:
                    {
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToMapTypeInfo(handle, out IntPtr mapInfo));
                        if (mapInfo == IntPtr.Zero)
                        {
                            throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                                           "TypeInfo cast to MapTypeInfo failed. The object does not represent a map");
                        }
                        _mapTypeInfo = new OrtMapTypeInfo(mapInfo);
                    }
                    break;
                case OnnxValueType.ONNX_TYPE_OPTIONAL:
                    {
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToOptionalTypeInfo(handle, out IntPtr optionalInfo));
                        if (optionalInfo == IntPtr.Zero)
                        {
                            throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                                           "TypeInfo cast to OptionalTypeInfo failed. The object does not represent a optional");
                        }
                        _sequenceOrOptional = new OrtSequenceOrOptionalTypeInfo(optionalInfo);
                    }
                    break;
                default:
                    throw new OnnxRuntimeException(ErrorCode.NotImplemented, $"OnnxValueType: {OnnxType} is not supported here");
            }
        }

        /// <summary>
        /// Represents OnnxValueType of the OrtTypeInfo
        /// </summary>
        /// <value>OnnxValueType</value>
        public OnnxValueType OnnxType { get; private set; }

        /// <summary>
        /// This property returns the tensor type and shape information from the OrtTypeInfo
        /// iff this OrtTypeInfo represents a tensor.
        /// </summary>
        /// <exception cref="OnnxRuntimeException"></exception>
        /// <value>Instance of OrtTensorTypeAndShapeInfo</value>
        public OrtTensorTypeAndShapeInfo TensorTypeAndShapeInfo
        {
            get
            {
                if (OnnxType != OnnxValueType.ONNX_TYPE_TENSOR &&
                    OnnxType != OnnxValueType.ONNX_TYPE_SPARSETENSOR)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "TypeInfo does not represent a tensor/sparsetensor");
                }
                return _tensorTypeAndShape.Value;
            }
        }

        /// <summary>
        /// Sequence type information from the OrtTypeInfo iff this OrtTypeInfo represents a sequence.
        /// </summary>
        /// <returns>Instance of OrtSequenceTypeInfo</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtSequenceOrOptionalTypeInfo SequenceTypeInfo
        {
            get
            {
                if (OnnxType != OnnxValueType.ONNX_TYPE_SEQUENCE)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "TypeInfo does not represent a sequence");
                }
                return _sequenceOrOptional.Value;
            }
        }

        /// <summary>
        /// Represents MapTypeInfo from the OrtTypeInfo iff this OrtTypeInfo represents a map.
        /// </summary>
        /// <value>Instance of MapTypeInfo</value>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtMapTypeInfo MapTypeInfo
        {
            get
            {
                if (OnnxType != OnnxValueType.ONNX_TYPE_MAP)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "TypeInfo does not represent a map");
                }
                return _mapTypeInfo.Value;
            }
        }

        /// <summary>
        /// Fetches OptionalTypeInfo from the OrtTypeInfo iff this OrtTypeInfo represents a optional type.
        /// </summary>
        /// <returns>Instance of OptionalTypeInfo</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtSequenceOrOptionalTypeInfo OptionalTypeInfo
        {
            get
            {
                if (OnnxType != OnnxValueType.ONNX_TYPE_OPTIONAL)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "TypeInfo does not represent a optional");
                }
                return _sequenceOrOptional.Value;
            }
        }
    }

    /// <summary>
    /// This struct represents type and shape information for a tensor.
    /// It may describe a tensor type that is a model input or output or
    /// an information that can be extracted from a tensor in OrtValue.
    /// 
    /// </summary>
    public struct OrtTensorTypeAndShapeInfo
    {
        internal OrtTensorTypeAndShapeInfo(IntPtr handle)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(handle, out IntPtr elementType));
            ElementDataType = (TensorElementType)elementType;

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(handle, out UIntPtr count));
            ElementCount = (long)count;

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(handle, out UIntPtr dimCount));

            Shape = new long[(uint)dimCount];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(handle, Shape, dimCount));
        }

        /// <summary>
        /// Fetches tensor element data type
        /// </summary>
        /// <value>enum value for the data type</value>
        public TensorElementType ElementDataType { get; private set; }

        /// <summary>
        /// Returns true if this data element is a string
        /// </summary>
        public bool IsString
        {
            get
            {
                return ElementDataType == TensorElementType.String;
            }
        }

        /// <summary>
        /// Fetches tensor element count based on the shape information.
        /// </summary>
        /// <returns>number of typed tensor elements</returns>
        public long ElementCount { get; private set; }

        /// <summary>
        /// Fetches shape dimension count (rank) for the tensor.
        /// </summary>
        /// <returns>dim count</returns>
        public int DimensionsCount { get { return Shape.Length; } }

        /// <summary>
        /// Tensor dimensions.
        /// </summary>
        /// <value>array of dims</value>
        public long[] Shape { get; private set; }
    }

    /// <summary>
    /// Represents Sequence type information.
    /// </summary>
    public struct OrtSequenceOrOptionalTypeInfo
    {
        internal OrtSequenceOrOptionalTypeInfo(IntPtr handle)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetSequenceElementType(handle, out IntPtr typeInfo));
            try
            {
                ElementType = new OrtTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        /// <summary>
        /// Returns type information for the element type of the sequence
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public OrtTypeInfo ElementType { get; private set; }
    }

    /// <summary>
    /// Represents Map input/output information.
    /// 
    /// Maps are represented at run time by a tensor of primitive types
    /// and values are represented either by Tensor/Sequence/Optional or another map.
    /// </summary>
    public struct OrtMapTypeInfo
    {
        internal OrtMapTypeInfo(IntPtr handle)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetMapKeyType(handle, out IntPtr tensorElementType));
            KeyType = (TensorElementType)tensorElementType;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetMapValueType(handle, out IntPtr typeInfo));
            try
            {
                ValueType = new OrtTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        /// <summary>
        /// Returns KeyType which is the data type of the keys tensor
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public TensorElementType KeyType { get; private set; }

        /// <summary>
        /// Returns an instance of OrtTypeInfo describing the value type for the map
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public OrtTypeInfo ValueType { get; private set; }
    }
}
