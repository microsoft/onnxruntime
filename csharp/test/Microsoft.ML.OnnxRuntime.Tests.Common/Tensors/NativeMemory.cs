// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/tests/NativeMemory.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;
using System;

namespace Microsoft.ML.OnnxRuntime.Tensors.Tests
{
    public class NativeMemory<T> : MemoryManager<T>
    {
        private bool disposed = false;
        private int refCount = 0;
        private IntPtr memory;
        private int length;

        public NativeMemory(IntPtr memory, int length)
        {
            this.memory = memory;
            this.length = length;
        }

        public unsafe NativeMemory(void* memory, int length)
        {
            this.memory = (IntPtr)memory;
            this.length = length;
        }

        // A finalizer to allow debugging in test code is allowable.
        // https://learn.microsoft.com/en-us/dotnet/fundamentals/code-analysis/quality-rules/ca2015
#pragma warning disable CA2015
        ~NativeMemory()
        {
            Dispose(false);
        }
#pragma warning restore CA2015

        public static NativeMemory<T> Allocate(int length)
        {
            // typically this would call into a native method appropriate for the platform
            // or the constructors above would be used to wrap the native pointer
            IntPtr memory = Marshal.AllocHGlobal(Marshal.SizeOf<T>() * length);
            return new NativeMemory<T>(memory, length);
        }

        public bool IsDisposed => disposed;

        public unsafe override Span<T> GetSpan() => new Span<T>((void*)memory, length);

        protected bool IsRetained => refCount > 0;

        public override MemoryHandle Pin(int elementIndex = 0)
        {
            unsafe
            {
                Retain();
                if ((uint)elementIndex > length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
                void* pointer = Unsafe.Add<T>((void*)memory, elementIndex);
                return new MemoryHandle(pointer, default, this);
            }
        }

        public bool Release()
        {
            int newRefCount = Interlocked.Decrement(ref refCount);

            if (newRefCount < 0)
            {
                throw new InvalidOperationException("Unmatched Release/Retain");
            }

            return newRefCount != 0;
        }

        public void Retain()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(NativeMemory<T>));
            }

            Interlocked.Increment(ref refCount);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }

            // typically this would call into a native method appropriate for the platform
            Marshal.FreeHGlobal(memory);
            memory = IntPtr.Zero;

            disposed = true;
        }

        protected override bool TryGetArray(out ArraySegment<T> arraySegment)
        {
            // cannot expose managed array
            arraySegment = default;
            return false;
        }

        public override void Unpin()
        {
            Release();
        }
    }
}
