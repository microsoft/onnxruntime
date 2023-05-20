// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This class retrieves Type Information for input/outputs of the model.
    /// </summary>
    public class OrtTypeInfo : SafeHandle
    {
        internal OrtTypeInfo(IntPtr handle) : base(handle, true)
        {
        }

        internal IntPtr Handle { get { return handle; } }

        /// <summary>
        /// Represents OnnxValueType of the OrtTypeInfo
        /// </summary>
        /// <value>OnnxValueType</value>
        public OnnxValueType OnnxType
        {
            get
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOnnxTypeFromTypeInfo(handle, out IntPtr type));
                return (OnnxValueType)type;
            }
        }

        /// <summary>
        /// This method returns the tensor type and shape information from the OrtTypeInfo
        /// iff this OrtTypeInfo represents a tensor.
        /// </summary>
        /// <exception cref="OnnxRuntimeException"></exception>
        /// <returns>Instance of OrtTensorTypeAndShapeInfo</returns>
        public OrtTensorTypeAndShapeInfo GetTensorTypeAndShapeInfo()
        {
            // The method below never fails, but returns null as if the cast has failed.
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToTensorInfo(handle, out IntPtr tensorInfo));
            if (tensorInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    "TypeInfo cast to TensorTypeInfo failed. The object does not represent a tensor");
            }
            return new OrtTensorTypeAndShapeInfo(tensorInfo, false);
        }

        /// <summary>
        /// Fetches sequence type information from the OrtTypeInfo.
        /// </summary>
        /// <returns>Instance of OrtSequenceTypeInfo</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtSequenceTypeInfo GetSequenceTypeInfo()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToSequenceTypeInfo(handle, out IntPtr sequenceInfo));
            if (sequenceInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                       "TypeInfo cast to SequenceTypeInfo failed. The object does not represent a sequence");
            }
            return new OrtSequenceTypeInfo(sequenceInfo);
        }

        /// <summary>
        /// Fetches MapTypeInfo from the OrtTypeInfo.
        /// </summary>
        /// <returns>Instance of MapTypeInfo</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtMapTypeInfo GetMapTypeInfo()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToMapTypeInfo(handle, out IntPtr mapInfo));
            if (mapInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                               "TypeInfo cast to MapTypeInfo failed. The object does not represent a map");
            }
            return new OrtMapTypeInfo(mapInfo);
        }


        /// <summary>
        /// Fetches OptionalTypeInfo from the OrtTypeInfo.
        /// </summary>
        /// <returns>Instance of OptionalTypeInfo</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtOptionalTypeInfo GetOptionalTypeInfo()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToOptionalTypeInfo(handle, out IntPtr optionalInfo));
            if (optionalInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                               "TypeInfo cast to OptionalTypeInfo failed. The object does not represent a optional");
            }
            return new OrtOptionalTypeInfo(optionalInfo);
        }

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtValue
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseTypeInfo(handle);
            handle = IntPtr.Zero;
            return true;
        }
        #endregion
    }

    /// <summary>
    /// This class represents type and shape information for a tensor.
    /// It may describe a tensor type that is a model input or output or
    /// an information that can be extracted from a tensor in OrtValue.
    /// 
    /// </summary>
    public class OrtTensorTypeAndShapeInfo : SafeHandle
    {
        private readonly bool _owned;

        internal OrtTensorTypeAndShapeInfo(IntPtr handle, bool owned) : base(handle, true)
        {
            _owned = owned;
        }

        internal IntPtr Handle { get { return handle; } }

        /// <summary>
        /// Fetches tensor element data type
        /// </summary>
        /// <value>enum value for the data type</value>
        public TensorElementType ElementDataType
        {
            get
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(Handle, out IntPtr type));
                return (TensorElementType)type;
            }
        }

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
        public long GetElementCount()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(Handle, out UIntPtr count));
            return (long)count;
        }

        /// <summary>
        /// Fetches shape dimension count (rank) for the tensor.
        /// </summary>
        /// <returns>dim count</returns>
        public int GetDimensionsCount()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(Handle, out UIntPtr count));
            return (int)count;
        }

        /// <summary>
        /// Fetches tensor symbolic dimensions. Use GetShape to fetch integer dimensions.
        /// The size of the vector returned is equal to the value returned by GetDimensionsCount.
        /// Positions that do not have symbolic dimensions will have empty strings.
        /// </summary>
        /// <returns>string[]</returns>
        public string[] GetSymbolicDimensions()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(Handle, out UIntPtr dimCount));
            IntPtr[] dimPtrs = new IntPtr[dimCount.ToUInt32()];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetSymbolicDimensions(Handle, dimPtrs, dimCount));
            var symbolicDims = new string[dimCount.ToUInt32()];
            for (int i = 0; i < dimPtrs.Length; ++i)
            {
                symbolicDims[i] = NativeOnnxValueHelper.StringFromNativeUtf8(dimPtrs[i]);
            }
            return symbolicDims;
        }


        /// <summary>
        /// Fetches tensor integer dimensions. Symbolic dimensions are represented as -1.
        /// Use GetSymbolicDimensions to fetch symbolic dimensions.
        /// </summary>
        /// <returns>array of dims</returns>
        public long[] GetShape()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(Handle, out UIntPtr dimCount));
            long[] shape = new long[dimCount.ToUInt32()];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(Handle, shape, dimCount));
            return shape;
        }

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtValue
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            // We have to surrender ownership to some legacy classes
            // Or we never had that ownership to begin with
            if (_owned)
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(Handle);
            }
            handle = IntPtr.Zero;
            return true;
        }
        #endregion
    }

    /// <summary>
    /// Represents Sequence type information. This class never owns
    /// the handle, it is owned by OrtTypeInfo instance, that must be alive
    /// at the time of this instance's use.
    /// </summary>
    public class OrtSequenceTypeInfo
    {
        internal OrtSequenceTypeInfo(IntPtr handle)
        {
            Handle = handle;
        }

        internal IntPtr Handle { get; }

        /// <summary>
        /// Returns type information for the element type of the sequence
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public OrtTypeInfo ElementType
        {
            get
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetSequenceElementType(Handle, out IntPtr typeInfo));
                return new OrtTypeInfo(typeInfo);
            }
        }
    }

    /// <summary>
    /// Represents Optional Type information. This class never owns
    /// the native handle. It is owned by the OrtTypeInfo instance, that must be alive
    /// at the time of this instance use
    /// </summary>
    public class OrtOptionalTypeInfo
    {
        internal OrtOptionalTypeInfo(IntPtr handle)
        {
            Handle = handle;
        }

        internal IntPtr Handle { get; }

        /// <summary>
        /// Returns type information for the element type of the sequence
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public OrtTypeInfo ElementType
        {
            get
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOptionalContainedTypeInfo(Handle, out IntPtr typeInfo));
                return new OrtTypeInfo(typeInfo);
            }
        }
    }

    /// <summary>
    /// Represents Map input/output information. This class never owns
    /// the handle, it is owned by OrtTypeInfo instance, that must be alive
    /// at the time of this instance's use.
    /// 
    /// Maps are represented at run time by a tensor of primitive types
    /// and values are represented either by Tensor/Sequence/Optional or another map.
    /// </summary>
    public class OrtMapTypeInfo
    {
        internal OrtMapTypeInfo(IntPtr handle)
        {
            Handle = handle;
        }

        internal IntPtr Handle { get; }

        /// <summary>
        /// Returns KeyType which is the data type of the keys tensor
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public TensorElementType KeyType
        {
            get
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetMapKeyType(Handle, out IntPtr tensorElementType));
                return (TensorElementType)tensorElementType;
            }
        }

        /// <summary>
        /// Returns an instance of OrtTypeInfo describing the value type for the map
        /// </summary>
        /// <value>OrtTypeInfo</value>
        public OrtTypeInfo ValueType
        {
            get
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetMapValueType(Handle, out IntPtr typeInfo));
                return new OrtTypeInfo(typeInfo);
            }
        }
    }
}
