// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    internal class NativeMemoryInfo : SafeHandle
    {
        protected static readonly Lazy<NativeMemoryInfo> _defaultCpuAllocInfo = new Lazy<NativeMemoryInfo>(CreateCpuMemoryInfo);

        private static NativeMemoryInfo CreateCpuMemoryInfo()
        {
            IntPtr allocInfo = IntPtr.Zero;
            try
            {
                IntPtr status = NativeMethods.OrtCreateCpuMemoryInfo(NativeMethods.AllocatorType.DeviceAllocator, NativeMethods.MemoryType.Cpu, out allocInfo);
                NativeApiStatus.VerifySuccess(status);
            }
            catch (Exception e)
            {
                if (allocInfo != IntPtr.Zero)
                {
                    Delete(allocInfo);
                }
                throw e;
            }
            return new NativeMemoryInfo(allocInfo);
        }

        internal static NativeMemoryInfo DefaultInstance
        {
            get
            {
                return _defaultCpuAllocInfo.Value;
            }
        }

        internal IntPtr Handle  // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return handle;
            }
        }

        public override bool IsInvalid
        {
            get
            {
                return (handle == IntPtr.Zero);
            }
        }

        private NativeMemoryInfo(IntPtr allocInfo)
            : base(IntPtr.Zero, true)   //set 0 as invalid pointer
        {
            handle = allocInfo;
        }


        private static void Delete(IntPtr nativePtr)
        {
            NativeMethods.OrtReleaseMemoryInfo(nativePtr);
        }

        protected override bool ReleaseHandle()
        {
            Delete(handle);
            return true;
        }
    }


    internal class NativeMemoryAllocator : SafeHandle
    {
        protected static readonly Lazy<NativeMemoryAllocator> _defaultInstance = new Lazy<NativeMemoryAllocator>(GetDefaultCpuAllocator);

        private static NativeMemoryAllocator GetDefaultCpuAllocator()
        {
            IntPtr allocator = IntPtr.Zero;
            try
            {
                IntPtr status = NativeMethods.OrtGetAllocatorWithDefaultOptions(out allocator);
                NativeApiStatus.VerifySuccess(status);
            }
            catch (Exception e)
            {
                throw e;
            }

            return new NativeMemoryAllocator(allocator);
        }

        static internal NativeMemoryAllocator DefaultInstance // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return _defaultInstance.Value;
            }
        }

        /// <summary>
        /// Releases native memory previously allocated by the allocator
        /// </summary>
        /// <param name="memory"></param>
        internal void FreeMemory(IntPtr memory)
        {
            NativeMethods.OrtAllocatorFree(handle, memory);
        }

        public override bool IsInvalid
        {
            get
            {
                return (this.handle == IntPtr.Zero);
            }
        }

        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        protected NativeMemoryAllocator(IntPtr allocator)
            : base(IntPtr.Zero, true)
        {
            this.handle = allocator;
        }

        protected override bool ReleaseHandle()
        {
            return true;
        }
    }

}
