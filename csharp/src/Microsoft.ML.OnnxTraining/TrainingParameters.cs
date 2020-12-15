using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// Holds the training parameters used by the training runner.
    /// </summary>
    public class TrainingParameters : SafeHandle
    {
        private OrtDataGetBatchCallback m_fnGetTrainingData;
        private OrtDataGetBatchCallback m_fnGetTestingData;
        private OrtErrorFunctionCallback m_fnErrorFunction;
        private OrtEvaluationFunctionCallback m_fnEvaluateFunction;
        public event EventHandler<ErrorFunctionArgs> OnErrorFunction;
        public event EventHandler<EvaluationFunctionArgs> OnEvaluationFunction;
        public event EventHandler<DataBatchArgs> OnGetTrainingDataBatch;
        public event EventHandler<DataBatchArgs> OnGetTestingDataBatch;
        DisposableList<IDisposable> m_rgCleanUpList = new DisposableList<IDisposable>();

        /// <summary>
        /// Constructs a default TrainingParameters
        /// </summary>
        public TrainingParameters()
            : base(IntPtr.Zero, true)
        {
            m_fnGetTrainingData = getTrainingDataFn;
            m_fnGetTestingData = getTestingDataFn;
            m_fnErrorFunction = errorFn;
            m_fnEvaluateFunction = evaluationFn;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtCreateTrainingParameters(out handle));
        }

        public IntPtr Handle
        {
            get { return handle; }
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        #region Public Properties

        public void SetTrainingParameter(OrtTrainingStringParameter key, string strVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetParameter_string(handle, key, NativeMethods.GetPlatformSerializedString(strVal)));
        }

        public string GetTrainingParameter(OrtTrainingStringParameter key)
        {
            string str = null;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetParameter_string(handle, key, allocator.Pointer, out valHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(valHandle);
            }
            return str;
        }

        public void SetTrainingParameter(OrtTrainingBooleanParameter key, bool bVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetParameter_bool(handle, key, bVal));
        }

        public bool GetTrainingParameter(OrtTrainingBooleanParameter key)
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetParameter_bool(handle, key, out val));

            if ((ulong)val == 0)
                return false;
            else
                return true;
        }

        public void SetTrainingParameter(OrtTrainingLongParameter key, long lVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetParameter_long(handle, key, lVal));
        }

        public long GetTrainingParameter(OrtTrainingLongParameter key)
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetParameter_long(handle, key, out val));

            return (long)val;
        }

        public void SetTrainingParameter(OrtTrainingNumericParameter key, double dfVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetNumericParameter(handle, key, dfVal));
        }

        public double GetTrainingParameter(OrtTrainingNumericParameter key)
        {
            string str = null;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetNumericParameter(handle, key, allocator.Pointer, out valHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(valHandle);
            }
            return double.Parse(str);
        }

        public void SetTrainingOptimizer(OrtTrainingOptimizer opt)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetTrainingOptimizer(handle, opt));
        }

        public OrtTrainingOptimizer GetTrainingOptimizer()
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetTrainingOptimizer(handle, out val));

            switch (((OrtTrainingOptimizer)(int)val))
            {
                case OrtTrainingOptimizer.ORT_TRAINING_OPTIMIZER_SGD:
                    return OrtTrainingOptimizer.ORT_TRAINING_OPTIMIZER_SGD;

                default:
                    throw new Exception("Unknown optimizer '" + val.ToString() + "'!");
            }
        }

        public void SetTrainingLossFunction(OrtTrainingLossFunction loss)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetTrainingLossFunction(handle, loss));
        }

        public OrtTrainingLossFunction GetTrainingLossFunction()
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetTrainingLossFunction(handle, out val));

            switch (((OrtTrainingLossFunction)(int)val))
            {
                case OrtTrainingLossFunction.ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY:
                    return OrtTrainingLossFunction.ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY;

                default:
                    throw new Exception("Unknown loss function '" + val.ToString() + "'!");
            }
        }

        public void errorFn(UIntPtr count, IntPtr colVal)
        {
            if (OnErrorFunction == null)
                return;

            List<DisposableNamedOnnxValue> rgVal = new List<DisposableNamedOnnxValue>();
            List<string> rgNames = new List<string>();
            OrtValueCollection col = new OrtValueCollection(colVal);

            int nCount = col.Count;
            for (int i = 0; i < nCount; i++)
            {
                string strName;
                OrtValue val = col.GetAt(i, out strName);
                rgVal.Add(DisposableNamedOnnxValue.CreateTensorFromOnnxValue(strName, val));
            }

            OnErrorFunction(this, new ErrorFunctionArgs(rgVal));

            // Clean-up the data used during this batch.
            foreach (IDisposable iDispose in m_rgCleanUpList)
            {
                iDispose.Dispose();
            }

            m_rgCleanUpList.Clear();
        }

        public void evaluationFn(long lNumSamples, long lStep)
        {
            if (OnEvaluationFunction == null)
                return;

            OnEvaluationFunction(this, new EvaluationFunctionArgs(lNumSamples, lStep));
        }

        public void SetupTrainingParameters()
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetupTrainingParameters(handle, m_fnErrorFunction, m_fnEvaluateFunction));
        }

        public void getTrainingDataFn(UIntPtr batch_size, UIntPtr count, IntPtr colVal)
        {
            if (OnGetTrainingDataBatch == null)
                return;

            int nBatchSize = (int)batch_size;
            int nCount = (int)count;
            DataBatchArgs args = new DataBatchArgs(nBatchSize);
            OnGetTrainingDataBatch(this, args);
            handleGetDataFn(args, nCount, colVal);
        }

        public void getTestingDataFn(UIntPtr batch_size, UIntPtr count, IntPtr colVal)
        {
            if (OnGetTestingDataBatch == null)
                return;

            int nBatchSize = (int)batch_size;
            int nCount = (int)count;
            DataBatchArgs args = new DataBatchArgs(nBatchSize);
            OnGetTestingDataBatch(this, args);
            handleGetDataFn(args, nCount, colVal);
        }

        private void handleGetDataFn(DataBatchArgs args, int nCount, IntPtr colVal)
        {
            OrtValueCollection col = new OrtValueCollection(colVal);

            for (int i = 0; i < args.Values.Count; i++)
            {
                MemoryHandle? memHandle;
                OrtValue val = args.Values[i].ToOrtValue(out memHandle);

                if (memHandle.HasValue)
                    m_rgCleanUpList.Add(memHandle);

                m_rgCleanUpList.Add(val);
              
                col.SetAt(i, val, args.Values[i].Name);
            }
        }

        public void SetupTrainingData(List<string> rgstrFeedNames)
        {
            string strFeedNames = "";
            
            for (int i = 0; i < rgstrFeedNames.Count; i++)
            {
                strFeedNames += rgstrFeedNames[i];
                strFeedNames += ";";
            }

            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetupTrainingData(handle, m_fnGetTrainingData, m_fnGetTestingData, NativeMethods.GetPlatformSerializedString(strFeedNames)));
        }

        #endregion

        #region SafeHandle

        protected override bool ReleaseHandle()
        {
            NativeMethodsTraining.OrtReleaseTrainingParameters(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }

    public class ErrorFunctionArgs : EventArgs
    {
        List<DisposableNamedOnnxValue> m_rgVal;

        public ErrorFunctionArgs(List<DisposableNamedOnnxValue> rgVal)
        {
            m_rgVal = rgVal;
        }

        public List<DisposableNamedOnnxValue> Values
        {
            get { return m_rgVal; }
        }

        public DisposableNamedOnnxValue Find(string strName)
        {
            foreach (DisposableNamedOnnxValue val in m_rgVal)
            {
                if (val.Name == strName)
                    return val;
            }

            return null;
        }
    }

    public class EvaluationFunctionArgs : EventArgs 
    {
        long m_lNumSamples;
        long m_lStep;

        public EvaluationFunctionArgs(long lNumSamples, long lStep)
        {
            m_lNumSamples = lNumSamples;
            m_lStep = lStep;
        }

        public long NumSamples
        {
            get { return m_lNumSamples; }
        }

        public long Step
        {
            get { return m_lStep; }
        }
    }

    public class DataBatchArgs : EventArgs
    {
        int m_nBatchSize;
        List<NamedOnnxValue> m_rgValues = new List<NamedOnnxValue>();

        public DataBatchArgs(int nBatchSize)
        {
            m_nBatchSize = nBatchSize;
        }

        public List<NamedOnnxValue> Values
        {
            get { return m_rgValues; }
        }

        public int BatchSize
        {
            get { return m_nBatchSize; }
        }
    }
}
