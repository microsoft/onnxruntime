// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

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
        ONNX_TYPE_OPAQUE = 4, // It's an experimental Opaque object
        ONNX_TYPE_SPARSETENSOR = 5, // It's a Sparse Tensor
        ONNX_TYPE_OPTIONAL = 6, // It's an optional type that designates anything above (except UNKOWN)
    }

    /// <summary>
    /// Represents a disposable OrtValue.
    /// This class exposes a native instance of OrtValue.
    /// The class implements IDisposable and must
    /// be disposed of, otherwise native resources will leak
    /// and will eventually cause the application to slow down or crash.
    /// 
    /// If the OrtValue instance is constructed over a managed memory, and it is not
    /// disposed properly, the pinned memory will continue to be pinned and interfere
    /// with GC operation.
    /// </summary>
    public class OrtValue : IOrtValueOwner, IDisposable
    {
        // OrtValues that are members of Sequences or Maps that map. They potentially map managed memory and we need to keep them around.
        // this exists only when we deal with compose ML types.
        private DisposableList<OrtValue> _compositeMembers;
        private IntPtr _handle;
        private MemoryHandle? _memHandle; // Present when the OrtValue is created on top of managed memory
        private bool _disposed;

        internal OrtValue(IntPtr handle)
        {
            _handle = handle;
            InitOnnxType();
        }

        /// <summary>
        /// Constructor. The newly constructed OrtValue takes ownership of the native OrtValue instance
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="onnxValueType"></param>
        /// <exception cref="ArgumentException">thrown when onnxValue type is not known</exception>
        internal OrtValue(IntPtr handle, OnnxValueType onnxValueType)
        {
            if (onnxValueType == OnnxValueType.ONNX_TYPE_UNKNOWN)
            {
                throw new ArgumentException("onnxValueType argument is passed as unknown");
            }

            _handle = handle;
            OnnxType = onnxValueType;
        }

        /// <summary>
        /// Constructor. The newly constructed OrtValue takes ownership of the native OrtValue instance
        /// and disposes of it when the OrtValue instance is disposed. The instance will take ownership and will
        /// dispose of compositeMembers instances.
        /// 
        /// This constructor can only throw if OnnxType is not specified.
        /// </summary>
        /// <param name="handle">native ortValue handle</param>
        /// <param name="onnxValueType">must one of the valid types</param>
        /// <param name="compositeMembers">For composite types this contains dependent ortValues such as members of a sequence
        /// or keys/values for the map, that may have been created on top of the managed memory and must be disposed
        /// with the new ortValue. This container will be taken the ownership of and the argument will be set to null.</param>
        /// <exception cref="ArgumentException">throws when onnxValueType is not specified</exception>
        internal OrtValue(IntPtr handle, OnnxValueType onnxValueType, ref DisposableList<OrtValue> compositeMembers)
        {
            if (onnxValueType == OnnxValueType.ONNX_TYPE_UNKNOWN)
            {
                throw new ArgumentException("onnxValueType argument is passed as unknown");
            }

            _handle = handle;
            OnnxType = onnxValueType;
            _compositeMembers = compositeMembers;
            compositeMembers = null;
        }

        /// <summary>
        /// Constructor to construct OrtValue over managed memory.
        /// We pin the memory and unpin it at the disposal time.
        /// The newly constructed OrtValue takes ownership of the native OrtValue instance
        /// and disposes of it when the OrtValue instance is disposed.
        /// </summary>
        /// <param name="handle">Pointer to a native instance of OrtValue</param>
        /// <param name="memHandle">memory handle to a pinned user supplied (usually managed) memory
        /// It is disposed of (unpinned) when OrtValue is disposed.
        /// </param>
        private OrtValue(IntPtr handle, MemoryHandle memHandle)
        {
            _handle = handle;
            _memHandle = memHandle;
            // OrtValue on top of the pinned memory is always a tensor
            OnnxType = OnnxValueType.ONNX_TYPE_TENSOR;
        }

        /// <summary>
        /// Native handle to OrtValue for internal use.
        /// </summary>
        internal IntPtr Handle { get { return _handle; } }

        /// <summary>
        /// Implement IOrtValueOwner interface
        /// </summary>
        /// <value>returns this</value>
        public OrtValue Value => this;

        /// <summary>
        /// Fetches OrtValue type if it has one.
        /// </summary>
        /// <value>OnnxValueType</value>
        public OnnxValueType OnnxType { get; private set; }

        private void InitOnnxType()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueType(Handle, out IntPtr onnxType));
            OnnxType = (OnnxValueType)onnxType;
        }

        /// <summary>
        /// Returns true if OrtValue contains a tensor
        /// </summary>
        /// <returns>true if tensor</returns>
        public bool IsTensor
        {
            get
            {
                return OnnxType == OnnxValueType.ONNX_TYPE_TENSOR;
            }
        }

        /// <summary>
        /// Returns true if OrtValue contains a sparse tensor
        /// </summary>
        /// <returns>true if sparse tensor</returns>
        public bool IsSparseTensor
        {
            get
            {
                return OnnxType == OnnxValueType.ONNX_TYPE_SPARSETENSOR;
            }
        }

        /// <summary>
        /// Valid for composite ML types like map, sequence.
        /// Returns 2 for map (keys, values) and N for sequence, where N is the number of elements
        /// in the sequence.
        /// </summary>
        /// <returns>Element count</returns>
        public int GetValueCount()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValueCount(Handle, out IntPtr count));
            return (int)count;
        }

        /// <summary>
        /// For non tensors return OrtValue element at the specified index.
        /// For maps only indices 0 and 1 are valid. For sequences, [0..N) are valid.
        /// See GetValueCount() to determine the valid range.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="allocator">allocator to use</param>
        /// <returns>OrtValue disposable instance that points to the corresponding element of the composite type</returns>
        public OrtValue GetValue(int index, OrtAllocator allocator)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetValue(Handle, index,
                allocator.Pointer, out IntPtr ortValueHandle));
            return new OrtValue(ortValueHandle);
        }

        /// <summary>
        /// Returns a ReadOnlySpan<typeparamref name="T"/> over tensor native buffer that
        /// provides a read-only view.
        /// 
        /// Note, that the memory may be device allocated and, therefore, not accessible from the CPU.
        /// To get memory descriptor use GetTensorMemoryInfo().
        /// 
        /// OrtValue must contain a non-string tensor.
        /// The span is valid as long as the OrtValue instance is alive (not disposed).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns>ReadOnlySpan<typeparamref name="T"/></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
        {
            var byteSpan = GetTensorBufferRawData(typeof(T));
            return MemoryMarshal.Cast<byte, T>(byteSpan);
        }

        /// <summary>
        /// Returns a Span<typeparamref name="T"/> over tensor native buffer.
        /// This enables you to safely and efficiently modify the underlying
        /// native buffer in a type-safe manner. This is useful for example in IOBinding scenarios
        /// where you want to modify results of the inference and feed it back as input.
        /// 
        /// Note, that the memory may be device allocated.
        /// To get memory descriptor use GetTensorMemoryInfo().
        /// 
        /// OrtValue must contain a non-string tensor.
        /// The span is valid as long as the OrtValue instance is alive (not disposed).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns>Typed Span over the native buffer</returns>
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
        {
            var byteSpan = GetTensorBufferRawData(typeof(T));
            return MemoryMarshal.Cast<byte, T>(byteSpan);
        }

        /// <summary>
        /// Provides mutable raw native buffer access.
        /// </summary>
        /// <returns>Span over the native buffer bytes</returns>
        public Span<byte> GetTensorMutableRawData()
        {
            return GetTensorBufferRawData(typeof(byte));
        }

        /// <summary>
        /// Fetch string tensor element buffer pointer at the specified index,
        /// convert/copy to UTF-16 char[] and return a ReadOnlyMemory<char> instance.
        /// 
        /// Obtain TensorTypeAndShape to get shape and element count.
        /// </summary>
        /// <param name="index">flat string tensor element index</param>
        /// <returns>ReadOnlyMemory<char> backed by a managed char[]. Its lifespan is not
        /// tied to the native buffer of OrtValue.</returns>
        public ReadOnlyMemory<char> GetStringElementAsMemory(int index)
        {
            var chars = GetStringTensorElementChars(index);
            if (chars.Length == 0)
            {
                return ReadOnlyMemory<char>.Empty;
            }
            return new ReadOnlyMemory<char>(chars);
        }

        /// <summary>
        /// Fetch string tensor element buffer pointer at the specified index,
        /// copy/convert UTF-8 into a UTF-16 string and return it.
        /// 
        /// Obtain TensorTypeAndShape to get shape and element count.
        /// </summary>
        /// <param name="index">flat string tensor element index</param>
        /// <returns>UTF-16 string instance</returns>
        public string GetStringElement(int index)
        {
            var chars = GetStringTensorElementChars(index);
            if (chars.Length == 0)
            {
                return string.Empty;
            }
            return new string(chars);
        }


        /// <summary>
        /// Get a span over the native memory of the string tensor element.
        /// The span is valid as long as the OrtValue is valid.
        /// 
        /// This is useful if you want to perform your own UTF-8 decoding or
        /// you do not care about decoding.
        /// Obtain TensorTypeAndShape to get shape and element count.
        /// </summary>
        /// <param name="index">flat element index</param>
        /// <returns>ReadOnlySpan over UTF-8 bytes of the string tensor element</returns>
        public ReadOnlySpan<byte> GetStringElementAsSpan(int index)
        {
            GetStringTensorElementBuffer((UIntPtr)index, out uint bytesLen, out IntPtr bufferPtr);
            if (bytesLen == 0)
            {
                return ReadOnlySpan<byte>.Empty;
            }
            unsafe
            {
                return new ReadOnlySpan<byte>((bufferPtr).ToPointer(), (int)bytesLen);
            }
        }

        /// <summary>
        /// Convenience method to obtain all string tensor elements as a string array.
        /// </summary>
        /// <returns>string[]</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public string[] GetStringTensorAsArray()
        {
            GetTensorElementTypeAndCount(out long count, out TensorElementType elementType);
            if (elementType != TensorElementType.String)
            {
                throw new OnnxRuntimeException(
                            ErrorCode.Fail,
                            $"GetStringTensorAsArray() is only supported for string tensors. This OrtValue contains a {elementType} tensor.");
            }

            var strings = new string[count];
            for (int i = 0; i < count; i++)
            {
                strings[i] = GetStringElement(i);
            }
            return strings;
        }

        /// <summary>
        /// Creates and fetches Type information about the contained OnnxValue.
        /// </summary>
        /// <returns>a disposable instance of OrtTypeInfo</returns>
        public OrtTypeInfo GetTypeInfo()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTypeInfo(Handle, out IntPtr typeInfo));
            try
            {
                return new OrtTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        /// <summary>
        /// Obtains Tensor And Type Information from the OrtValue iff it contains a tensor.
        /// Valid only for OrtValues that contain a tensor.
        /// </summary>
        /// <returns>A disposable instance of OrtTensorTypeAndShapeInfo</returns>
        public OrtTensorTypeAndShapeInfo GetTensorTypeAndShape()
        {
            var onnxType = OnnxType;
            if (onnxType != OnnxValueType.ONNX_TYPE_TENSOR &&
                onnxType != OnnxValueType.ONNX_TYPE_SPARSETENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"This OrtValue type contains: {onnxType}, not a tensor or sparse tensor");
            }

            NativeMethods.OrtGetTensorTypeAndShape(Handle, out IntPtr typeAndShapeInfo);
            try
            {
                return new OrtTensorTypeAndShapeInfo(typeAndShapeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShapeInfo);
            }
        }

        /// <summary>
        /// Returns OrtMemoryInfo iff this OrtValue contains a tensor or a sparse tensor.
        /// </summary>
        /// <returns>OrtMemoryInfo that describes the underlying memory allocation</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public OrtMemoryInfo GetTensorMemoryInfo()
        {
            var onnxType = OnnxType;
            if (onnxType != OnnxValueType.ONNX_TYPE_TENSOR &&
                onnxType != OnnxValueType.ONNX_TYPE_SPARSETENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                       $"This OrtValue type contains: {onnxType}, not a tensor or sparse tensor");
            }
            NativeMethods.OrtGetTensorMemoryInfo(Handle, out IntPtr memoryInfo);
            return new OrtMemoryInfo(memoryInfo, false);
        }

        private void GetTensorElementTypeAndCount(out long count, out TensorElementType elementType)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(Handle, out IntPtr typeAndShapeInfo));
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShapeInfo, out IntPtr elType));
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(typeAndShapeInfo, out UIntPtr cnt));
                elementType = (TensorElementType)elType;
                count = (long)cnt;
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShapeInfo);
            }
        }

        private char[] GetStringTensorElementChars(int index)
        {
            GetStringTensorElementBuffer((UIntPtr)index, out uint bytesLen, out IntPtr bufferPtr);
            if (bytesLen == 0)
            {
                return Array.Empty<char>();
            }

            unsafe
            {
                int charCount = Encoding.UTF8.GetCharCount((byte*)(bufferPtr).ToPointer(), (int)bytesLen);
                var chars = new char[charCount];
                fixed (char* ch = chars)
                {
                    Encoding.UTF8.GetChars((byte*)(bufferPtr).ToPointer(), (int)bytesLen, (char*)ch, charCount);
                }
                return chars;
            }
        }

        private void GetStringTensorElementBuffer(UIntPtr index, out uint bytesLen, out IntPtr bufferPtr)
        {
            // Length is in UTF-8 bytes. Strings are not zero terminated, so length is required.
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetStringTensorElementLength(Handle, index, out UIntPtr bufferLen));

            bytesLen = (uint)bufferLen;

            if (bytesLen == 0)
            {
                bufferPtr = IntPtr.Zero;
                return;
            }

            // XXX: We lack the API (at the moment) that simply gives access to string element buffer. So we get the resized one
            // to the same length which leaves it unchanged.
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetResizedStringTensorElementBuffer(Handle,
                   (UIntPtr)index, bufferLen, out bufferPtr));
        }

        private Span<byte> GetTensorBufferRawData(Type requestedType)
        {
            if (OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"This OrtValue type contains: {OnnxType}, not a tensor");
            }

            GetTensorElementTypeAndCount(out long count, out TensorElementType elementType);

            if (elementType == TensorElementType.String)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Strings are not supported by this API");
            }

            var typeInfo = TensorBase.GetElementTypeInfo(elementType) ??
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, $"Element type: {elementType} is not registered type.");

            // We are always Ok with byte
            if (requestedType != typeof(byte) && requestedType != typeInfo.TensorType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Requested type: {requestedType} does not match the actual type: {typeInfo.TensorType}");
            }

            if (count == 0)
            {
                return Span<byte>.Empty;
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorMutableData(Handle, out IntPtr tensorData));

            var bufferLenInBytes = count * typeInfo.TypeSize;

            unsafe
            {
                return new Span<byte>(tensorData.ToPointer(), (int)bufferLenInBytes);
            }
        }

        /// <summary>
        /// Factory method to construct an OrtValue of Tensor type on top of pre-allocated memory.
        /// This can be a piece of arbitrary memory that may be allocated by OrtAllocator (possibly on a device),
        /// a chunk of managed memory (must be pinned for the duration of OrtValue lifetime) or a memory that is allocated
        /// natively allocated using Marshal.AllocHGlobal(), stackalloc or other means (may be on a device).
        /// 
        /// The resulting OrtValue does not own the underlying memory buffer and will not attempt to
        /// deallocate it. The caller must make sure that the memory remains valid for the duration of OrtValue lifetime.
        /// </summary>
        /// <param name="memInfo">Memory Info. For managed memory its default is cpu.
        ///                       For other kinds of memory, one must construct as appropriate.</param>
        /// <param name="elementType">DataType for the Tensor</param>
        /// <param name="shape">shape of the tensor to create. The size required by the shape
        /// must be less of equal of the memory.Length</param>
        /// <param name="dataBufferPtr">Pointer to a raw memory buffer which may reside on a device</param>
        /// <param name="bufferLengthInBytes">Buffer length in bytes</param>
        /// <returns>A disposable instance of OrtValue</returns>
        public static OrtValue CreateTensorValueWithData(OrtMemoryInfo memInfo, TensorElementType elementType,
                                                         long[] shape,
                                                         IntPtr dataBufferPtr,
                                                         long bufferLengthInBytes)
        {
            var typeInfo = TensorBase.GetElementTypeInfo(elementType) ?? throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                       $"Tensor element type: {elementType} is not supported");
            if (typeInfo.IsString)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    "Cannot map managed strings buffer to native OrtValue. Use string specific interfaces");
            }

            var shapeSize = ShapeUtils.GetSizeForShape(shape);
            var requiredBufferSizeInBytes = shapeSize * typeInfo.TypeSize;

            // We allow creating a tensor over part of the buffer
            if (requiredBufferSizeInBytes > bufferLengthInBytes)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Shape: {shape} has: {shapeSize} elements requires a buffer of at least {requiredBufferSizeInBytes} bytes. Provided: {bufferLengthInBytes} bytes");
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                                    memInfo.Pointer,
                                    dataBufferPtr,
                                    (UIntPtr)bufferLengthInBytes,
                                    shape,
                                    (UIntPtr)shape.Length,
                                    elementType,
                                    out IntPtr ortValueHandle
                                ));
            return new OrtValue(ortValueHandle, OnnxValueType.ONNX_TYPE_TENSOR);
        }

        /// <summary>
        /// This is a factory method that creates an OrtValue of Tensor type on top of Memory<typeparamref name="T"/> memory.
        /// The API pins the memory for the duration of the OrtValue lifetime.
        /// It is unpinned at disposal time.
        /// </summary>
        /// <typeparam name="T">T must be one of the supported types</typeparam>
        /// <param name="memoryInfo">Memory information that describes memory location</param>
        /// <param name="memory">contiguous region of memory</param>
        /// <param name="shape">shape of the tensor to create. The size required by the shape
        /// must be less of equal of the memory.Length</param>
        /// <returns>A disposable OrtValue instance</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public static OrtValue CreateTensorValueFromMemory<T>(OrtMemoryInfo memoryInfo, Memory<T> memory, long[] shape)
            where T : unmanaged
        {
            var typeInfo = TensorBase.GetTypeInfo(typeof(T)) ??
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, $"Tensor of type: {typeof(T)} is not supported");

            if (typeInfo.IsString)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    "Cannot map managed strings buffer to native OrtValue. Use string specific interfaces.");
            }

            var shapeSize = ShapeUtils.GetSizeForShape(shape);
            // We allow creating a tensor over part of the buffer
            if (shapeSize > memory.Length)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Managed memory size: {memory.Length} elements is less than shape size: {shapeSize} elements");
            }

            var bufferLengthInBytes = memory.Length * typeInfo.TypeSize;
            var memoryHandle = memory.Pin();
            try
            {
                IntPtr bufferPtr;
                unsafe
                {
                    bufferPtr = new IntPtr(memoryHandle.Pointer);
                }

                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                                        memoryInfo.Pointer,
                                        bufferPtr,
                                        (UIntPtr)bufferLengthInBytes,
                                        shape,
                                        (UIntPtr)shape.Length,
                                        typeInfo.ElementType,
                                        out IntPtr ortValueHandle
                                    ));
                return new OrtValue(ortValueHandle, memoryHandle);
            }
            catch (Exception)
            {
                memoryHandle.Dispose();
                throw;
            }

        }

        /// <summary>
        /// This is a factory method that creates an OrtValue of Tensor type on top managed data array.
        /// The API pins the memory for the duration of the OrtValue lifetime.
        /// It is unpinned at disposal time.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data">managed data buffer</param>
        /// <param name="shape">shape that describes the buffer</param>
        /// <returns>A disposable OrtValue instance</returns>
        public static OrtValue CreateTensorValueFromMemory<T>(T[] data, long[] shape) where T : unmanaged
        {
            return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<T>(data), shape);
        }

        /// <summary>
        /// The factory API creates an OrtValue with memory allocated using the given allocator
        /// according to the specified shape and element type. The memory will be released when OrtValue
        /// is disposed. Use GetTensorMutableDataAsSpan&lt;T&gt;() API to fill in the data.
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="elementType"></param>
        /// <param name="shape"></param>
        /// <returns>A disposable OrtValue</returns>
        public static OrtValue CreateAllocatedTensorValue(OrtAllocator allocator, TensorElementType elementType,
                                                         long[] shape)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorAsOrtValue(allocator.Pointer, shape,
                (UIntPtr)shape.Length, elementType, out IntPtr ortValueHandle));
            return new OrtValue(ortValueHandle, OnnxValueType.ONNX_TYPE_TENSOR);
        }

        /// <summary>
        /// This is a factory method creates a native Onnxruntime OrtValue containing a tensor.
        /// The method will attempt to pin managed memory so no copying occurs when data is passed down
        /// to native code.
        /// </summary>
        /// <param name="value">Tensor object</param>
        /// <param name="elementType">discovered tensor element type</param>
        /// <returns>And instance of OrtValue constructed on top of the object</returns>
        internal static OrtValue CreateFromTensorObject(TensorBase value, out TensorElementType elementType)
        {
            var typeInfo = value.GetTypeInfo();
            OrtValue ortValue = null;

            TensorElementType elType = typeInfo.ElementType;
            var typeSize = typeInfo.TypeSize;
            if (typeInfo.IsString)
            {
                ortValue = CreateFromStringTensor(value as Tensor<string>);
            }
            else
            {
                int dataBufferLength;
                long[] shape;
                int rank;

                MemoryHandle memHandle;
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
                    IntPtr dataBufferPointer = IntPtr.Zero;
                    unsafe
                    {
                        dataBufferPointer = (IntPtr)memHandle.Pointer;
                    }

                    NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                        OrtMemoryInfo.DefaultInstance.Pointer,
                        dataBufferPointer,
                        (UIntPtr)(dataBufferLength),
                        shape,
                        (UIntPtr)rank,
                        elType,
                        out IntPtr nativeValue));

                    ortValue = new OrtValue(nativeValue, memHandle);
                }
                catch (Exception)
                {
                    memHandle.Dispose();
                    throw;
                }
            }

            elementType = elType;
            return ortValue;
        }

        /// <summary>
        /// Creates an OrtValue that contains a string tensor of specified shape, and
        /// containing empty strings. String tensors are always on CPU.
        /// Use StringTensorSetElementAt to assign individual elements values.
        /// </summary>
        /// <param name="allocator"></param>
        /// <returns>disposable OrtValue</returns>
        /// <param name="shape">tensor shape</param>
        public static OrtValue CreateTensorWithEmptyStrings(OrtAllocator allocator, long[] shape)
        {
            // allocate the native tensor
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorAsOrtValue(
                                allocator.Pointer,
                                shape,
                                (UIntPtr)(shape.Length),
                                TensorElementType.String,
                                out IntPtr valueHandle
                                ));
            return new OrtValue(valueHandle, OnnxValueType.ONNX_TYPE_TENSOR);
        }

        /// <summary>
        /// Converts the string argument represented by ReadOnlySpan to UTF-8,
        /// allocates space in the native tensor and copies it into the native tensor memory.
        /// Typically, this is used to populate a new empty string tensor element.
        /// 
        /// The number of elements is according to the shape supplied to CreateTensorWithEmptyStrings().
        /// However, this API can also be used to overwrite any existing element within the string tensor.
        /// 
        /// In general, to obtain the number of elements for any tensor, use GetTensorTypeAndShape() which
        /// would return a disposable instance of TensorTypeAndShapeInfo. 
        /// Then call GetElementCount() or GetShape().
        /// </summary>
        /// <param name="str">ReadOnlySpan over chars</param>
        /// <param name="index">index of the string element within the tensor
        /// must be within bounds of [0, N)</param>
        public void StringTensorSetElementAt(ReadOnlySpan<char> str, int index)
        {
            unsafe
            {
                fixed (char* strPtr = str)
                {
                    FillStringTensorElement(strPtr, str.Length, index);
                }
            }
        }

        /// <summary>
        /// Converts the string argument represented by ReadOnlyMemory to UTF-8,
        /// allocates space in the native tensor and copies it into the native tensor memory.
        /// Typically, this is used to populate a new empty string tensor element.
        /// 
        /// The number of elements is according to the shape supplied to CreateTensorWithEmptyStrings().
        /// However, this API can also be used to overwrite any existing element within the string tensor.
        /// 
        /// In general, to obtain the number of elements for any tensor, use GetTensorTypeAndShape() which
        /// would return a disposable instance of TensorTypeAndShapeInfo. 
        /// Then call GetElementCount() or GetShape().
        ///
        /// </summary>
        /// <param name="rom">ReadOnlyMemory instance over an array of chars</param>
        /// <param name="index">index of the string element within the tensor
        /// must be within bounds of [0, N)</param>
        public void StringTensorSetElementAt(ReadOnlyMemory<char> rom, int index)
        {
            StringTensorSetElementAt(rom.Span, index);
        }

        /// <summary>
        /// This API resizes String Tensor element to the requested amount of bytes (UTF-8)
        /// and copies the bytes from the supplied ReadOnlySpan into the native tensor memory (resized buffer).
        /// 
        /// The API is useful for quick loading of utf8 data into the native tensor memory.
        /// </summary>
        /// <param name="utf8Bytes">read only span of bytes</param>
        /// <param name="index">flat index of the element in the string tensor</param>
        public void StringTensorSetElementAt(ReadOnlySpan<byte> utf8Bytes, int index)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetResizedStringTensorElementBuffer(Handle,
                                  (UIntPtr)index, (UIntPtr)utf8Bytes.Length, out IntPtr buffer));

            if (utf8Bytes.Length == 0)
            {
                return;
            }

            unsafe
            {
                var destSpan = new Span<byte>(buffer.ToPointer(), utf8Bytes.Length);
                utf8Bytes.CopyTo(destSpan);
            }
        }

        /// <summary>
        /// Creates an OrtValue that contains a string tensor.
        /// String tensors are always allocated on CPU.
        /// String data will be converted to UTF-8 and copied to native memory.
        /// 
        /// Note, this is different from creating an OrtValue from other primitive data types
        /// where memory is pinned (if necessary) and the OrtValue points to that chunk of memory.
        /// </summary>
        /// <param name="tensor">Tensor<string></param>
        /// <returns>A disposable OrtValue instance</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        public static OrtValue CreateFromStringTensor(Tensor<string> tensor)
        {
            if (tensor == null)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Expecting a valid string tensor");
            }

            long[] shape = Array.ConvertAll<int, long>(tensor.Dimensions.ToArray(), Convert.ToInt64);

            var ortValue = CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
            try
            {
                var len = tensor.Length;
                for (int i = 0; i < len; ++i)
                {
                    var str = tensor.GetValue(i) ?? throw new ArgumentNullException($"Tensor<string> contains null reference at index:{i}");
                    unsafe
                    {
                        fixed (char* strPtr = str)
                        {
                            ortValue.FillStringTensorElement(strPtr, str.Length, i);
                        }
                    }
                }
            }
            catch (Exception)
            {
                ortValue.Dispose();
                throw;
            }
            return ortValue;
        }

        /// <summary>
        /// Creates a sequence of OrtValues from a collection of OrtValues.
        /// All OrtValues in the collection must be of the same Onnx type.
        /// I.e. (Tensor, SparseTensor, Map, Sequence, etc.)
        /// 
        /// The ortValues that are passed as argument are taken possession of by the newly
        /// created OrtValue. The caller should not dispose them, unless this call fails.
        /// 
        /// The ortValues would be empty on successful return.
        /// </summary>
        /// <param name="ortValues">a collection of OrtValues. On success the ortValues contained in the list
        /// are taken ownership of and the list is cleared.</param>
        /// <returns>A disposable instance of OrtValues</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public static OrtValue CreateSequence(ICollection<OrtValue> ortValues)
        {
            if (ortValues is null)
            {
                throw new ArgumentNullException(nameof(ortValues));
            }

            if (ortValues.IsReadOnly)
            {
                throw new ArgumentException("ortValues argument can not be a readonly collection");
            }

            var compositeMembers = new DisposableList<OrtValue>(ortValues);
            try
            {
                var result = CreateSequence(ref compositeMembers);
                Debug.Assert(compositeMembers is null, "Must be null on success");
                ortValues.Clear();
                return result;
            }
            catch (Exception)
            {
                // The caller is responsible for disposing the ortValues
                compositeMembers?.Clear();
                throw;
            }
        }

        /// <summary>
        /// Creates a sequence from the values in compositeMembers
        /// The argument is taken possession of and is nullified on successful return.
        /// </summary>
        /// <param name="compositeMembers">sequence ortValues</param>
        /// <returns>OrtValue instance representing a Sequence</returns>
        internal static OrtValue CreateSequence(ref DisposableList<OrtValue> compositeMembers)
        {
            var handles = new IntPtr[compositeMembers.Count];
            for (int i = 0; i < compositeMembers.Count; i++)
            {
                handles[i] = compositeMembers[i].Handle;
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateValue(handles,
                    (UIntPtr)handles.Length, (IntPtr)OnnxValueType.ONNX_TYPE_SEQUENCE,
                    out IntPtr sequenceHandle));

            return new OrtValue(sequenceHandle, OnnxValueType.ONNX_TYPE_SEQUENCE, ref compositeMembers);
        }

        /// <summary>
        /// A delegate type that is expected to process each OrtValue in a sequence.
        /// </summary>
        /// <param name="ortValue">OrtValue that holds sequence element</param>
        /// <param name="index">ordinal of the value</param>
        public delegate void SequenceElementVisitor(OrtValue ortValue, int index);

        /// <summary>
        /// Feeds each OrtValue in a sequence to the visitor delegate.
        /// This helps users to avoid dealing each value life-span
        /// </summary>
        /// <param name="visitor">visitor delegate</param>
        /// <param name="allocator">allocator to use for intermediate ort values</param>
        /// <exception cref="OnnxRuntimeException"></exception>
        public void ProcessSequence(SequenceElementVisitor visitor, OrtAllocator allocator)
        {
            if (OnnxType != OnnxValueType.ONNX_TYPE_SEQUENCE)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"OrtValue.OnnxType of {OnnxType} is not a sequence");
            }

            int count = GetValueCount();
            for (int i = 0; i < count; i++)
            {
                using var ortValue = GetValue(i, allocator);
                visitor(ortValue, i);
            }
        }

        /// <summary>
        /// Creates a map OrtValue with keys and values.
        /// On a high level the Onnxruntime representation of the map always consists of two
        /// OrtValues, keys and values.
        /// 
        /// According to ONNX standard map keys can be unmanaged types only (or strings).
        /// Those keys are contained in a single tensor within OrtValue keys.
        /// 
        /// Map values, on the other hand, can be composite types. The values parameter
        /// can either contain a single tensor with unmanaged map values with the same number of
        /// elements as the keys, or it can be a sequence of OrtValues,
        /// each of those can be a composite type (tensor, sequence, map). If it is a sequence,
        /// then the number of elements must match the number of elements in keys.
        /// 
        /// Keys and values must be in the same order.
        /// 
        /// ORT supports only a subset of types for keys and values, however, this API does not
        /// restrict it.
        /// 
        /// The ortValues that are passed as argument are taken possession of by the newly
        /// created OrtValue. The caller should not dispose them, unless this call fails.
        /// 
        /// Keys and values arguments will be set to null on success.
        /// </summary>
        /// <param name="keys">Contains keys</param>
        /// <param name="values">Contains values</param>
        /// <returns>A disposable OrtValue</returns>
        /// <exception cref="ArgumentNullException"></exception>
        public static OrtValue CreateMap(ref OrtValue keys, ref OrtValue values)
        {
            if (keys is null || values is null)
            {
                throw new ArgumentNullException("keys or/and values are null");
            }

            IntPtr[] handles = { keys.Handle, values.Handle };
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtCreateValue(handles, (UIntPtr)handles.Length, (IntPtr)OnnxValueType.ONNX_TYPE_MAP,
                               out IntPtr mapHandle));

            var compositeMembers = new DisposableList<OrtValue>
            {
                keys,
                values
            };

            keys = null;
            values = null;

            // This constructor will not throw.
            return new OrtValue(mapHandle, OnnxValueType.ONNX_TYPE_MAP, ref compositeMembers);
        }

        /// <summary>
        /// This API helps to quickly creates a map OrtValue with unmanaged (primitive) keys and values specified as arrays.
        /// This helps the user not to create OrtValues for keys and values separately and deal only with the final result.
        /// The map would consist of two tensors, one for keys and one for values.
        /// 
        /// The OrtValues would be created on top of the managed memory arrays and use it directly.
        /// The number of elements in keys and values must be the same and they must be in order.
        /// 
        /// The types must be unmanaged.
        /// </summary>
        /// <typeparam name="K">keys type</typeparam>
        /// <typeparam name="V">values type</typeparam>
        /// <param name="keys">array of keys of K type</param>
        /// <param name="values">array of values of V type</param>
        /// <returns>OrtValue instance</returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public static OrtValue CreateMap<K, V>(K[] keys, V[] values) where K : unmanaged where V : unmanaged
        {
            if (keys is null || values is null)
            {
                throw new ArgumentNullException("Keys or/and values are null");
            }

            if (keys.Length != values.Length)
            {
                throw new ArgumentException("Expecting keys and values same len. " +
                    $"Received keys: {keys.Length}, Values: {values.Length}");
            }

            long[] shape = { keys.Length };
            Span<OrtValue> ortValues = new OrtValue[2];
            var disposableGuard = new DisposableArray<OrtValue>(ortValues);
            try
            {
                ortValues[0] = CreateTensorValueFromMemory(keys, shape);
                ortValues[1] = CreateTensorValueFromMemory(values, shape);
                return CreateMap(ref ortValues[0], ref ortValues[1]);
            }
            catch (Exception)
            {
                disposableGuard.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Creates a map OrtValue with string keys and non-string values.
        /// This helps the user not to create OrtValues for keys and values separately.
        /// The number of elements in keys and values must be the same and they must be in order.
        /// The map would consist of two tensors, one for keys and one for values.
        /// 
        /// string keys would be converted to UTF-8 encoding and copied to an allocated native memory.
        /// The OrtValue for values would be created on top of the managed memory using it directly.
        /// 
        /// The values type must be unmanaged.
        /// </summary>
        /// <typeparam name="V"></typeparam>
        /// <param name="keys">Collection of strings</param>
        /// <param name="values"></param>
        /// <returns>OrtValue instance</returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public static OrtValue CreateMapWithStringKeys<V>(IReadOnlyCollection<string> keys, V[] values) where V : unmanaged
        {
            if (keys is null || values is null)
            {
                throw new ArgumentNullException("Keys or/and values are null");
            }

            if (keys.Count != values.Length)
            {
                throw new ArgumentException("Expecting keys and values same len. " +
                    $"Received keys: {keys.Count}, Values: {values.Length}");
            }

            long[] shape = { keys.Count };

            Span<OrtValue> ortValues = new OrtValue[2];
            var disposableGuard = new DisposableArray<OrtValue>(ortValues);
            try
            {
                ortValues[0] = CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
                int count = 0;
                foreach (var key in keys)
                {
                    ortValues[0].StringTensorSetElementAt(key.AsSpan(), count++);
                }

                ortValues[1] = CreateTensorValueFromMemory(values, shape);
                return CreateMap(ref ortValues[0], ref ortValues[1]);
            }
            catch (Exception)
            {
                disposableGuard.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Creates a map OrtValue with non-string keys and string values.
        /// 
        /// This helps the user not to create OrtValues for keys and values separately.
        /// The number of elements in keys and values must be the same and they must be in order.
        /// 
        /// The OrtValue for keys would be created on top of the managed memory using it directly.
        /// string values would be converted to UTF-8 encoding and copied to an allocated native memory.
        /// 
        /// </summary>
        /// <typeparam name="K">unmanaged type of keys</typeparam>
        /// <param name="keys"></param>
        /// <param name="values">collection of string values</param>
        /// <returns>Instance of OrtValue</returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public static OrtValue CreateMapWithStringValues<K>(K[] keys, IReadOnlyCollection<string> values) where K : unmanaged
        {
            if (keys is null || values is null)
            {
                throw new ArgumentNullException("Keys or/and values are null");
            }

            if (keys.Length != values.Count)
            {
                throw new ArgumentException("Expecting keys and values same len. " +
                    $"Received keys: {keys.Length}, Values: {values.Count}");
            }

            long[] shape = { keys.Length };
            Span<OrtValue> ortValues = new OrtValue[2];
            var disposableGuard = new DisposableArray<OrtValue>(ortValues);
            try
            {
                ortValues[0] = CreateTensorValueFromMemory(keys, shape);
                ortValues[1] = CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape);
                int count = 0;
                foreach (var value in values)
                {
                    ortValues[1].StringTensorSetElementAt(value.AsSpan(), count++);
                }
                return CreateMap(ref ortValues[0], ref ortValues[1]);
            }
            catch (Exception)
            {
                disposableGuard.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A public delegate that will be invoked once with map keys and values.
        /// The delegate helps not to deal with the lifespan of intermediate OrtValues.
        /// Typically, when one uses GetValue() API, it creates a copy of OrtValue
        /// that points to the same buffer as keys or values. This API helps to deal with those
        /// temporary instances and avoid leaks.
        /// 
        /// According to ONNX standard map keys can be unmanaged types only (or strings).
        /// Those keys are contained in a single tensor within OrtValue keys. So you can query those
        /// directly from keys argument.
        /// 
        /// Map values, on the other hand, can be composite types. The values parameter
        /// can either contain a single tensor with unmanaged map values with the same number of
        /// elements as the keys, or it can be a sequence of OrtValues,
        /// each of those can be a composite type (tensor, sequence, map). If it is a sequence,
        /// then the number of elements must match the number of elements in keys.
        /// 
        /// Depending on the structure of the values, one will either directly query a single tensor
        /// from values, or will have to iterate over the sequence of OrtValues and visit each of those
        /// resulting in a recursive visitation.
        /// </summary>
        /// <param name="keys">This would always represent a tensor</param>
        /// <param name="values">Can be any of the Onnx types, but they would all reduce to tensors eventually</param>
        public delegate void MapVisitor(OrtValue keys, OrtValue values);

        /// <summary>
        /// This API helps the user to process a map OrtValue without
        /// having to deal with the lifespan of intermediate OrtValues.
        /// 
        /// each API value is fed to the vistor functor.
        /// </summary>
        /// <param name="visitor">visitor function</param>
        /// <param name="allocator">Allocator to use for intermediate values</param>
        /// <exception cref="OnnxRuntimeException"></exception>
        public void ProcessMap(MapVisitor visitor, OrtAllocator allocator)
        {
            if (OnnxType != OnnxValueType.ONNX_TYPE_MAP)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "This OrtValue does not represent a map");
            }

            using var keys = GetValue(0, allocator);
            using var values = GetValue(1, allocator);
            visitor(keys, values);
        }

        private unsafe void FillStringTensorElement(char* strPtr, int strLength, int index)
        {
            IntPtr buffer;
            if (strLength == 0)
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetResizedStringTensorElementBuffer(Handle,
                                                  (UIntPtr)index, UIntPtr.Zero, out buffer));
                return;
            }

            var bytesCount = Encoding.UTF8.GetByteCount(strPtr, strLength);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetResizedStringTensorElementBuffer(Handle,
                                              (UIntPtr)index, (UIntPtr)bytesCount, out buffer));
            NativeOnnxValueHelper.StringToUtf8NativeMemory(strPtr, strLength, buffer, bytesCount);
        }

        private static void PinAsTensor<T>(
                                        Tensor<T> tensor,
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

            DenseTensor<T> dt = tensor as DenseTensor<T> ?? tensor.ToDenseTensor();
            shape = Array.ConvertAll<int, long>(dt.Dimensions.ToArray(), Convert.ToInt64);
            rank = dt.Rank;

            dataBufferLength = dt.Buffer.Length * elementSize;
            pinnedHandle = dt.Buffer.Pin();
        }

        #region IDisposable Support

        ~OrtValue()
        {
            Dispose(false);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                _memHandle?.Dispose();
                _memHandle = null;
                _compositeMembers?.Dispose();
                _compositeMembers = null;
            }

            Debug.Assert(_handle != IntPtr.Zero);
            NativeMethods.OrtReleaseValue(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
        }

        #endregion
    }
}
