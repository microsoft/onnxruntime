using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    public class TrainingSession
    {
        TrainingParameters m_param = new TrainingParameters();

        public TrainingSession()
        {
        }

        public TrainingParameters Parameters
        {
            get { return m_param; }
        }

        public void Initialize(LogLevel logLevel = LogLevel.Warning)
        {
            IntPtr hEnv;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnv(logLevel, @"CSharpOnnxRuntime.Training", out hEnv));
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtInitializeTraining(hEnv, m_param.Handle));
        }

        public void RunTraining()
        {           
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtRunTraining(m_param.Handle));
        }

        public void EndTraining()
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtEndTraining(m_param.Handle));
        }
    }
}
