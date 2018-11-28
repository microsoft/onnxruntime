using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Scoring;
using System.Diagnostics;

namespace Microsoft.ML.OnnxRuntime.PerfTool
{
    public class SonomaRunner
    {
        public static void RunModelSonoma(string modelPath, string inputPath, int iteration, DateTime[] timestamps)
        {
            if (timestamps.Length != (int)TimingPoint.TotalCount)
            {
                throw new ArgumentException("Timestamps array must have " + (int)TimingPoint.TotalCount + " size");
            }

            timestamps[(int)TimingPoint.Start] = DateTime.Now;

            var modelName = "lotusrt_squeezenet";
            using (var modelManager = new ModelManager(modelPath, true))
            {
                modelManager.InitOnnxModel(modelName, int.MaxValue);
                timestamps[(int)TimingPoint.ModelLoaded] = DateTime.Now;

                Tensor[] inputs = new Tensor[1];
                var inputShape = new long[] { 1, 3, 224, 224 };   // hardcoded values
                
                float[] inputData0 = Program.LoadTensorFromFile(inputPath);
                inputs[0] = Tensor.Create(inputData0, inputShape);
                string[] inputNames = new string[] {"data_0"};
                string[] outputNames = new string[] { "softmaxout_1" };

                timestamps[(int)TimingPoint.InputLoaded] = DateTime.Now;

                for (int i = 0; i < iteration; i++)
                {
                    var outputs = modelManager.RunModel(
                                                    modelName,
                                                    int.MaxValue,
                                                    inputNames,
                                                    inputs,
                                                    outputNames
                                                    );
                    Debug.Assert(outputs != null);
                    Debug.Assert(outputs.Length == 1);
                }

                timestamps[(int)TimingPoint.RunComplete] = DateTime.Now;
            }



        }


    }
}
