// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// A non-owning view of named EPContext data passed to a write delegate.
    /// The view is valid only for the duration of the delegate invocation.
    /// </summary>
    public sealed class OrtEpContextData
    {
        internal OrtEpContextData(IntPtr pointer, UIntPtr size)
        {
            _pointer = pointer;
            Size = size.ToUInt64();
            _isValid = true;
        }

        /// <summary>
        /// Size of the data in bytes.
        /// </summary>
        public ulong Size { get; }

        /// <summary>
        /// Returns all data as a read-only span. For data larger than <see cref="int.MaxValue"/>,
        /// use <see cref="GetSpan(ulong, int)"/> to process it in chunks.
        /// </summary>
        public ReadOnlySpan<byte> GetSpan()
        {
            ThrowIfInvalid();
            if (Size > int.MaxValue)
            {
                throw new InvalidOperationException("EPContext data is too large for a single Span<byte>. Read it in chunks.");
            }

            return GetSpan(0, checked((int)Size));
        }

        /// <summary>
        /// Returns a read-only span for a range of the data.
        /// </summary>
        /// <param name="offset">Byte offset into the data.</param>
        /// <param name="length">Number of bytes in the returned span.</param>
        public unsafe ReadOnlySpan<byte> GetSpan(ulong offset, int length)
        {
            ThrowIfInvalid();
            ValidateRange(offset, length);
            if (length == 0)
            {
                return ReadOnlySpan<byte>.Empty;
            }

            return new ReadOnlySpan<byte>(AddOffset(_pointer, offset).ToPointer(), length);
        }

        internal void Invalidate()
        {
            _isValid = false;
            _pointer = IntPtr.Zero;
        }

        internal static IntPtr AddOffset(IntPtr pointer, ulong offset)
        {
            if (UIntPtr.Size == 4)
            {
                ulong address = unchecked((uint)pointer.ToInt32());
                if (offset > uint.MaxValue || address + offset > uint.MaxValue)
                {
                    throw new ArgumentOutOfRangeException(nameof(offset));
                }

                return new IntPtr(unchecked((int)(uint)(address + offset)));
            }

            ulong address64 = unchecked((ulong)pointer.ToInt64());
            if (offset > ulong.MaxValue - address64)
            {
                throw new ArgumentOutOfRangeException(nameof(offset));
            }

            return new IntPtr(unchecked((long)(address64 + offset)));
        }

        private void ThrowIfInvalid()
        {
            if (!_isValid)
            {
                throw new ObjectDisposedException(nameof(OrtEpContextData),
                    "EPContext data is valid only during the write delegate invocation.");
            }
        }

        private void ValidateRange(ulong offset, int length)
        {
            if (length < 0 || offset > Size || (ulong)length > Size - offset)
            {
                throw new ArgumentOutOfRangeException(nameof(length));
            }

            if (length != 0 && _pointer == IntPtr.Zero)
            {
                throw new InvalidOperationException("EPContext data has a null buffer for a non-empty range.");
            }
        }

        private IntPtr _pointer;
        private bool _isValid;
    }

    /// <summary>
    /// Allocator-backed output supplied to an EPContext read delegate.
    /// Allocate and fill the buffer during the delegate invocation. On success, ownership is transferred to ORT.
    /// </summary>
    public sealed class OrtEpContextDataBuffer : IDisposable
    {
        internal OrtEpContextDataBuffer(IntPtr allocator, ulong maxDataSize)
        {
            _allocator = new OrtAllocator(allocator, false);
            _maxDataSize = maxDataSize;
        }

        /// <summary>
        /// Size of the allocated buffer in bytes.
        /// </summary>
        public ulong Size { get; private set; }

        /// <summary>
        /// Whether the delegate has allocated an output buffer. A zero-byte allocation is valid.
        /// </summary>
        public bool IsAllocated { get; private set; }

        /// <summary>
        /// Allocates an output buffer and returns it as a writable span.
        /// For buffers larger than <see cref="int.MaxValue"/>, use <see cref="Allocate(ulong)"/>
        /// followed by chunked calls to <see cref="GetSpan(ulong, int)"/>.
        /// </summary>
        public Span<byte> Allocate(int size)
        {
            if (size < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size));
            }

            Allocate((ulong)size);
            return GetSpan();
        }

        /// <summary>
        /// Allocates an output buffer of the requested size.
        /// </summary>
        public void Allocate(ulong size)
        {
            ThrowIfDisposed();
            if (IsAllocated)
            {
                throw new InvalidOperationException("The EPContext data buffer has already been allocated.");
            }

            if (size > _maxDataSize)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "EPContext data exceeds the configured maximum size.");
            }

            UIntPtr nativeSize = ToUIntPtr(size);
            IntPtr pointer = IntPtr.Zero;
            if (size != 0)
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtAllocatorAlloc(_allocator.Pointer, nativeSize, out pointer));
                if (pointer == IntPtr.Zero)
                {
                    throw new InvalidOperationException("The ORT allocator returned a null pointer for a non-empty allocation.");
                }
            }

            _pointer = pointer;
            Size = size;
            IsAllocated = true;
        }

        /// <summary>
        /// Returns the entire allocated buffer as a writable span.
        /// </summary>
        public Span<byte> GetSpan()
        {
            if (Size > int.MaxValue)
            {
                throw new InvalidOperationException("EPContext data is too large for a single Span<byte>. Fill it in chunks.");
            }

            return GetSpan(0, checked((int)Size));
        }

        /// <summary>
        /// Returns a writable span for a range of the allocated buffer.
        /// </summary>
        public unsafe Span<byte> GetSpan(ulong offset, int length)
        {
            ThrowIfDisposed();
            if (!IsAllocated)
            {
                throw new InvalidOperationException("Allocate must be called before accessing the EPContext data buffer.");
            }

            if (length < 0 || offset > Size || (ulong)length > Size - offset)
            {
                throw new ArgumentOutOfRangeException(nameof(length));
            }

            if (length == 0)
            {
                return Span<byte>.Empty;
            }

            return new Span<byte>(OrtEpContextData.AddOffset(_pointer, offset).ToPointer(), length);
        }

        internal void Detach(out IntPtr pointer, out UIntPtr size)
        {
            ThrowIfDisposed();
            if (!IsAllocated)
            {
                throw new InvalidOperationException("The EPContext data read delegate must allocate its output buffer.");
            }

            pointer = _pointer;
            size = ToUIntPtr(Size);
            _pointer = IntPtr.Zero;
            _disposed = true;
            _allocator.Dispose();
            _allocator = null;
        }

        /// <summary>
        /// Releases an allocation that was not transferred to ORT because the delegate failed.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_pointer != IntPtr.Zero)
            {
                _allocator.FreeMemory(_pointer);
                _pointer = IntPtr.Zero;
            }

            _allocator.Dispose();
            _allocator = null;
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        private static UIntPtr ToUIntPtr(ulong value)
        {
            if (UIntPtr.Size == 4 && value > uint.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(value));
            }

            return new UIntPtr(value);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(OrtEpContextDataBuffer));
            }
        }

        private OrtAllocator _allocator;
        private readonly ulong _maxDataSize;
        private IntPtr _pointer;
        private bool _disposed;
    }
}
