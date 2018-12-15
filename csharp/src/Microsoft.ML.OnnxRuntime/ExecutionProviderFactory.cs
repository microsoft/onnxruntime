using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{

    internal class CpuExecutionProviderFactory: NativeOnnxObjectHandle
    {
        protected static readonly Lazy<CpuExecutionProviderFactory> _default = new Lazy<CpuExecutionProviderFactory>(() => new CpuExecutionProviderFactory());

        public CpuExecutionProviderFactory(bool useArena=true)
            :base(IntPtr.Zero)
        {
            int useArenaInt = useArena ? 1 : 0;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCpuExecutionProviderFactory(useArenaInt, out handle));
            }
            catch(OnnxRuntimeException e)
            {
                if (IsInvalid)
                {
                    ReleaseHandle();
                    handle = IntPtr.Zero;
                }
                throw e;
            }
        }

        public static CpuExecutionProviderFactory Default
        {
            get
            {
                return _default.Value;
            }
        }
    }

    internal class MklDnnExecutionProviderFactory : NativeOnnxObjectHandle
    {
        protected static readonly Lazy<MklDnnExecutionProviderFactory> _default = new Lazy<MklDnnExecutionProviderFactory>(() => new MklDnnExecutionProviderFactory());

        public MklDnnExecutionProviderFactory(bool useArena = true)
            :base(IntPtr.Zero)
        {
            int useArenaInt = useArena ? 1 : 0;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateMkldnnExecutionProviderFactory(useArenaInt, out handle));
            }
            catch (OnnxRuntimeException e)
            {
                if (IsInvalid)
                {
                    ReleaseHandle();
                    handle = IntPtr.Zero;
                }
                throw e;
            }
        }

        public static MklDnnExecutionProviderFactory Default
        {
            get
            {
                return _default.Value;
            }
        }
    }

    internal class CudaExecutionProviderFactory : NativeOnnxObjectHandle
    {
        protected static readonly Lazy<CudaExecutionProviderFactory> _default = new Lazy<CudaExecutionProviderFactory>(() => new CudaExecutionProviderFactory());

        public CudaExecutionProviderFactory(int deviceId = 0)
            : base(IntPtr.Zero)
        {
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCUDAExecutionProviderFactory(deviceId, out handle));
            }
            catch (OnnxRuntimeException e)
            {
                if (IsInvalid)
                {
                    ReleaseHandle();
                    handle = IntPtr.Zero;
                }
                throw e;
            }
        }

        public static CudaExecutionProviderFactory Default
        {
            get
            {
                return _default.Value;
            }
        }
    }



}
