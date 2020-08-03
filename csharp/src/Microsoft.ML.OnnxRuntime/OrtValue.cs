// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    public enum OnnxValueType
    {
        ONNX_TYPE_UNKNOWN = 0,
        ONNX_TYPE_TENSOR = 1,
        ONNX_TYPE_SEQUENCE = 2,
        ONNX_TYPE_MAP = 3,
        ONNX_TYPE_OPAQUE = 4,
        ONNX_TYPE_SPARSETENSOR = 5,
    }

    /// <summary>
    /// Represents a disposable OrtValue
    /// </summary>
    public class OrtValue : IDisposable
    {
        /// <summary>
        /// Use factory methods to instantiate
        /// </summary>
        /// <param name="handle"></param>
        internal OrtValue(IntPtr handle)
        {
            Handle = handle;
        }

        internal IntPtr Handle { get; private set; }

        /// <summary>
        /// This internal interface is used to transfer ownership elsewhere.
        /// This instance must still be disposed in case there are other native
        /// objects still owned.
        /// </summary>
        /// <returns></returns>
        internal IntPtr Disown()
        {
            var handle = Handle;
            Handle = IntPtr.Zero;
            return handle;
        }

        /// <summary>
        /// Factory method to construct an OrtValue of Tensor type on top of pre-allocated memory.
        /// This can be a piece of native memory allocated by OrtAllocator (possibly on a device)
        /// or a piece of pinned managed memory.
        /// 
        /// The resulting OrtValue does not own the underlying memory buffer and will not attempt to
        /// deallocated it.
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
        /// This is a factory method that ta
        /// </summary>
        /// <param name="value">Tensor object</param>
        /// <param name="memoryHandle">For all tensor types but string tensors we endevour to use managed memory
        ///  to avoid additional allocation and copy. This out parameter represents a chunk of pinned memory
        /// </param>
        /// <param name="elementType">discovered tensor element type</param>
        /// <returns></returns>
        public static OrtValue CreateFromTensorObject(Object value, out MemoryHandle memoryHandle,
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

            MemoryHandle memHandle = default;
            OrtValue ortValue = null;
            int dataBufferLength = 0;
            long[] shape = null;
            int rank = 0;

            TensorElementType elType = typeInfo.ElementType;
            var typeSize = typeInfo.TypeSize;
            if (typeInfo.IsString)
            {
                ortValue = CreateStringTensor(value as Tensor<string>);
                memHandle = default;
            }
            else
            {
                switch (elType)
                {
                    case TensorElementType.Float:
                        PinAsTensor(value as Tensor<float>, elType, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.Double:
                        PinAsTensor(value as Tensor<double>, elType, typeSize, out memHandle, out dataBufferLength,
                                            out shape, out rank);
                        break;
                    case TensorElementType.Int32:
                        PinAsTensor(value as Tensor<int>, elType, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.UInt32:
                        PinAsTensor(value as Tensor<uint>, elType, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.Int64:
                        PinAsTensor(value as Tensor<long>, elType, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.UInt64:
                        PinAsTensor(value as Tensor<ulong>, elType, typeSize, out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    case TensorElementType.Int16:
                        PinAsTensor(value as Tensor<short>, elType, typeSize, out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;

                    case TensorElementType.UInt16:
                        PinAsTensor(value as Tensor<ushort>, elType, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);

                        break;
                    case TensorElementType.UInt8:
                        PinAsTensor(value as Tensor<byte>, elType, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    case TensorElementType.Int8:
                        PinAsTensor(value as Tensor<sbyte>, elType, typeSize,
                            out memHandle, out dataBufferLength,
                            out shape, out rank);
                        break;
                    case TensorElementType.Bool:
                        PinAsTensor(value as Tensor<bool>, elType, typeSize,
                                    out memHandle, out dataBufferLength,
                                    out shape, out rank);
                        break;
                    default:
                        throw new NotSupportedException("Element type: " + elType + " is not of a supported type");
                }

                try
                {
                    IntPtr dataBufferPointer = IntPtr.Zero;
                    unsafe
                    {
                        dataBufferPointer = (IntPtr)memHandle.Pointer;
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
                    memHandle.Dispose();
                    throw e;
                }
            }
            memoryHandle = memHandle;
            elementType = elType;
            return ortValue;
        }

        private static void PinAsTensor<T>(
                                            Tensor<T> tensor,
                                            TensorElementType nativeElementType,
                                            int elementSize,
                                            out MemoryHandle pinnedHandle,
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
                        pinnedHandles.Add(new PinnedGCHandle(gcHandle));
                        nativeStrings[i] = gcHandle.AddrOfPinnedObject();
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

        #region Disposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // We have to surrender ownership to some legacy classes
                if (Handle != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseValue(Handle);
                    // Prevent use after disposal
                    Handle = IntPtr.Zero;
                }
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        // No need for the finalizer
        #endregion
    }
}
