// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// A type of data that OrtValue encapsulates.
    /// </summary>
    public enum OnnxValueType
    {
        ONNX_TYPE_UNKNOWN = 0, // Not set
        ONNX_TYPE_TENSOR = 1, // It's a Tensor
        ONNX_TYPE_SEQUENCE = 2, // It's an Onnx sequence which may be a sequence of Tensors/Maps/Sequences
        ONNX_TYPE_MAP = 3,  // It's a map
        ONNX_TYPE_OPAQUE = 4, // It's an experiemntal Opaque object
        ONNX_TYPE_SPARSETENSOR = 5, // It's a Sparse Tensor
    }

    /// <summary>
    /// Represents a disposable OrtValue.
    /// This class exposes a native instance of OrtValue.
    /// The class implements IDisposable via SafeHandle and must
    /// be disposed.
    /// </summary>
    public class OrtValue : SafeHandle
    {
        /// <summary>
        /// Use factory methods to instantiate this class
        /// </summary>
        /// <param name="handle">Pointer to a native instance of OrtValue</param>
        /// <param name="owned">Default true, own the raw handle. Otherwise, the handle is owned by another instance
        /// However, we use this class to expose OrtValue that is owned by DisposableNamedOnnxValue
        /// </param>
        internal OrtValue(IntPtr handle, bool owned = true)
            : base(handle, true)
        {
            IsOwned = owned;
        }

        internal IntPtr Handle { get { return handle; } }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #region NamedOnnxValue/DisposableOnnxValue accommodations

        /// <summary>
        /// This internal interface is used to transfer ownership elsewhere.
        /// This instance must still be disposed in case there are other native
        /// objects still owned. This is a convince method to ensure that an underlying
        /// OrtValue is disposed exactly once when exception is thrown.
        /// </summary>
        /// <returns></returns>
        internal IntPtr Disown()
        {
            var ret = Handle;
            handle = IntPtr.Zero;
            IsOwned = false;
            return ret;
        }

        internal bool IsOwned { get; private set; }

        #endregion

        /// <summary>
        /// Factory method to construct an OrtValue of Tensor type on top of pre-allocated memory.
        /// This can be a piece of native memory allocated by OrtAllocator (possibly on a device)
        /// or a piece of pinned managed memory.
        /// 
        /// The resulting OrtValue does not own the underlying memory buffer and will not attempt to
        /// deallocate it.
        /// </summary>
        /// <param name="memInfo">Memory Info. For managed memory it is a default cpu.
        ///                       For Native memory must be obtained from the allocator or OrtMemoryAllocation instance</param>
        /// <param name="elementType">DataType for the Tensor</param>
        /// <param name="shape">Tensor shape</param>
        /// <param name="dataBuffer">Pointer to a raw memory buffer</param>
        /// <param name="bufferLength">Buffer length in bytes</param>
        /// <returns>A disposable instance of OrtValue</returns>
        public static OrtValue CreateTensorValueWithData(OrtMemoryInfo memInfo, TensorElementType elementType,
                                                         long[] shape,
                                                         IntPtr dataBuffer,
                                                         uint bufferLength)
        {
            Type type;
            int width;
            TensorElementTypeConverter.GetTypeAndWidth(elementType, out type, out width);
            if(width == 0)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Unknown tensor type");
            }

            var shapeSize = ArrayUtilities.GetSizeForShape(shape);
            if((shapeSize * width) > bufferLength)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Can not bind the shape to smaller buffer");
            }

            IntPtr ortValueHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                                    memInfo.Pointer,
                                    dataBuffer,
                                    (UIntPtr)bufferLength,
                                    shape,
                                    (UIntPtr)shape.Length,
                                    elementType,
                                    out ortValueHandle
                                ));
            return new OrtValue(ortValueHandle);
        }

        /// <summary>
        /// This is a factory method creates a native Onnxruntime OrtValue containing a tensor.
        /// The method will attempt to pin managed memory so no copying occurs when data is passed down
        /// to native code.
        /// </summary>
        /// <param name="value">Tensor object</param>
        /// <param name="memoryHandle">For all tensor types but string tensors we endeavor to use managed memory
        ///  to avoid additional allocation and copy. This out parameter represents a chunk of pinned memory which will need
        ///  to be disposed when no longer needed. The lifespan of memoryHandle should eclipse the lifespan of the corresponding
        ///  OrtValue.
        /// </param>
        /// <param name="elementType">discovered tensor element type</param>
        /// <returns>And instance of OrtValue constructed on top of the object</returns>
        public static OrtValue CreateFromTensorObject(Object value, out MemoryHandle? memoryHandle,
                                                                    out TensorElementType elementType)
        {
            // Check if this is a Tensor
            if (!(value is TensorBase))
            {
                throw new NotSupportedException("The inference value " + nameof(value) + " is not of a supported type");
            }

            var tensorBase = value as TensorBase;
            var typeInfo = tensorBase.GetTypeInfo();
            if (typeInfo == null)
            {
                throw new OnnxRuntimeException(ErrorCode.RequirementNotRegistered, "BUG Check");
            }

            MemoryHandle? memHandle;
            OrtValue ortValue = null;
            int dataBufferLength = 0;
            long[] shape = null;
            int rank = 0;

            TensorElementType elType = typeInfo.ElementType;
            var typeSize = typeInfo.TypeSize;
            if (typeInfo.IsString)
            {
                ortValue = CreateStringTensor(value as Tensor<string>);
                memHandle = null;
            }
            else
            {
                switch (elType)
                {
                    case TensorElementType.Float:
                        PinAsTensor(value as Tensor<float>, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.Double:
                        PinAsTensor(value as Tensor<double>, typeSize, out memHandle, out dataBufferLength,
                                            out shape, out rank);
                        break;
                    case TensorElementType.Int32:
                        PinAsTensor(value as Tensor<int>, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.UInt32:
                        PinAsTensor(value as Tensor<uint>, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.Int64:
                        PinAsTensor(value as Tensor<long>, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.UInt64:
                        PinAsTensor(value as Tensor<ulong>, typeSize, out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    case TensorElementType.Int16:
                        PinAsTensor(value as Tensor<short>, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;

                    case TensorElementType.UInt16:
                        PinAsTensor(value as Tensor<ushort>, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);

                        break;
                    case TensorElementType.UInt8:
                        PinAsTensor(value as Tensor<byte>, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    case TensorElementType.Int8:
                        PinAsTensor(value as Tensor<sbyte>, typeSize,
                            out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.Bool:
                        PinAsTensor(value as Tensor<bool>, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    case TensorElementType.Float16:
                        PinAsTensor(value as Tensor<Float16>, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    case TensorElementType.BFloat16:
                        PinAsTensor(value as Tensor<BFloat16>, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    default:
                        throw new NotSupportedException("Element type: " + elType + " is not of a supported type");
                }

                try
                {
                    Debug.Assert(memHandle.HasValue);
                    IntPtr dataBufferPointer = IntPtr.Zero;
                    unsafe
                    {
                        dataBufferPointer = (IntPtr)((MemoryHandle)memHandle).Pointer;
                    }

                    IntPtr nativeValue;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                        OrtMemoryInfo.DefaultInstance.Pointer,
                        dataBufferPointer,
                        (UIntPtr)(dataBufferLength),
                        shape,
                        (UIntPtr)rank,
                        elType,
                        out nativeValue));

                    ortValue = new OrtValue(nativeValue);
                }
                catch (Exception e)
                {
                    memHandle?.Dispose();
                    throw e;
                }
            }
            memoryHandle = memHandle;
            elementType = elType;
            return ortValue;
        }

        private static void PinAsTensor<T>(
                                            Tensor<T> tensor,
                                            int elementSize,
                                            out MemoryHandle? pinnedHandle,
                                            out int dataBufferLength,
                                            out long[] shape,
                                            out int rank)
        {
            if (tensor == null)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "Cast to Tensor<T> failed. BUG check!");
            }

            if (tensor.IsReversedStride)
            {
                //TODO: not sure how to support reverse stride. may be able to calculate the shape differently
                throw new NotSupportedException(nameof(Tensor<T>) + " of reverseStride is not supported");
            }

            DenseTensor<T> dt = null;
            if (tensor is DenseTensor<T>)
            {
                dt = tensor as DenseTensor<T>;
            }
            else
            {
                dt = tensor.ToDenseTensor();
            }

            pinnedHandle = dt.Buffer.Pin();
            dataBufferLength = dt.Buffer.Length * elementSize;
            shape = new long[dt.Dimensions.Length];
            for (int i = 0; i < dt.Dimensions.Length; ++i)
            {
                shape[i] = dt.Dimensions[i];
            }
            rank = dt.Rank;
        }

        private static OrtValue CreateStringTensor(Tensor<string> tensor)
        {
            if (tensor == null)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "Cast to Tensor<string> failed. BUG check!");
            }

            int totalLength = 0;
            for (int i = 0; i < tensor.Length; i++)
            {
                totalLength += System.Text.Encoding.UTF8.GetByteCount(tensor.GetValue(i));
            }

            long[] shape = new long[tensor.Dimensions.Length];
            for (int i = 0; i < tensor.Dimensions.Length; i++)
            {
                shape[i] = tensor.Dimensions[i];
            }

            // allocate the native tensor
            IntPtr valueHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorAsOrtValue(
                                OrtAllocator.DefaultInstance.Pointer,
                                shape,
                                (UIntPtr)(shape.Length),
                                TensorElementType.String,
                                out valueHandle
                                ));

            var ortValue = new OrtValue(valueHandle);
            try
            {

                // fill the native tensor, using GetValue(index) from the Tensor<string>
                var len = tensor.Length;
                var nativeStrings = new IntPtr[len];
                using (var pinnedHandles = new DisposableList<PinnedGCHandle>((int)len))
                {
                    for (int i = 0; i < len; i++)
                    {
                        var utf8str = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(tensor.GetValue(i));
                        var gcHandle = GCHandle.Alloc(utf8str, GCHandleType.Pinned);
                        nativeStrings[i] = gcHandle.AddrOfPinnedObject();
                        pinnedHandles.Add(new PinnedGCHandle(gcHandle));
                    }

                    using (var pinnedStrings = new PinnedGCHandle(GCHandle.Alloc(nativeStrings, GCHandleType.Pinned)))
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtFillStringTensor(ortValue.Handle, nativeStrings, (UIntPtr)len));
                }
            }
            catch (OnnxRuntimeException e)
            {
                ortValue.Dispose();
                throw e;
            }
            return ortValue;
        }

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtValue
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            // We have to surrender ownership to some legacy classes
            // Or we never had that ownership to begin with
            if (IsOwned)
            {
                NativeMethods.OrtReleaseValue(handle);
            }
            // Prevent use after disposal
            handle = IntPtr.Zero;
            return true;
        }
        // No need for the finalizer
        #endregion
    }
}
