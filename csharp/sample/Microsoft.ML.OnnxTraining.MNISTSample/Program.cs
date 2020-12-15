using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxTraining;
using MyCaffe.data;

namespace Microsoft.ML.OnnxTraining.MNISTSample
{
    class Program
    {
        static int m_nTrueCount = 0;
        static float m_fTotalLoss = 0;
        static int m_nIterations = 0;
        static List<Tuple<int, float>> m_rgPrecision = new List<Tuple<int, float>>();
        static List<Tuple<int, float>> m_rgAveLoss = new List<Tuple<int, float>>();
        static List<Tuple<byte[], int>> m_rgTrainingData;
        static List<Tuple<byte[], int>> m_rgTestingData;
        static int m_nTrainingDataIdx;
        static int m_nTestingDataIdx;

        static void Main(string[] args)
        {
            TrainingSession session = new TrainingSession();

            session.Parameters.OnErrorFunction += OnErrorFunction;
            session.Parameters.OnEvaluationFunction += OnEvaluationFunction;
            session.Parameters.OnGetTrainingDataBatch += OnGetTrainingDataBatch;
            session.Parameters.OnGetTestingDataBatch += OnGetTestingDataBatch;

            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_MODEL_PATH, "mnist");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_INPUT_LABELS, "labels");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_OUTPUT_LOSS, "loss");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_OUTPUT_PREDICTIONS, "predictions");
            session.Parameters.SetTrainingParameter(OrtTrainingStringParameter.ORT_TRAINING_LOG_PATH, "c:\\temp");
            session.Parameters.SetTrainingParameter(OrtTrainingBooleanParameter.ORT_TRAINING_USE_CUDA, true);
            session.Parameters.SetTrainingParameter(OrtTrainingBooleanParameter.ORT_TRAINING_SHUFFLE_DATA, false);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_EVAL_BATCH_SIZE, 100);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_TRAIN_BATCH_SIZE, 100);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_NUM_TRAIN_STEPS, 2000);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_EVAL_PERIOD, 1);
            session.Parameters.SetTrainingParameter(OrtTrainingLongParameter.ORT_TRAINING_DISPLAY_LOSS_STEPS, 400);
            session.Parameters.SetTrainingParameter(OrtTrainingNumericParameter.ORT_TRAINING_LEARNING_RATE, 0.01);
            session.Parameters.SetTrainingOptimizer(OrtTrainingOptimizer.ORT_TRAINING_OPTIMIZER_SGD);
            session.Parameters.SetTrainingLossFunction(OrtTrainingLossFunction.ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY);

            session.Parameters.SetupTrainingParameters();
            session.Parameters.SetupTrainingData(new List<string>() { "X", "labels" });

            MnistDataLoaderLite dataLoader = new MnistDataLoaderLite("C:\\Users\\winda\\Downloads\\DeepLearning\\Datasets\\mnist");
            dataLoader.ExtractImages(out m_rgTrainingData, out m_rgTestingData);
            m_nTrainingDataIdx = 0;
            m_nTestingDataIdx = 0;

            session.Initialize(LogLevel.Error);
            session.RunTraining();
            session.EndTraining();

            Console.WriteLine("Done!");
        }

        private static void OnGetTestingDataBatch(object sender, DataBatchArgs e)
        {
            loadBatch(e, m_rgTrainingData, ref m_nTrainingDataIdx);
        }

        private static void OnGetTrainingDataBatch(object sender, DataBatchArgs e)
        {
            loadBatch(e, m_rgTestingData, ref m_nTestingDataIdx);
        }

        private static void loadBatch(DataBatchArgs e, List<Tuple<byte[], int>> rgData, ref int nIdx)
        {
            float[] rgRawData = new float[rgData[0].Item1.Length * e.BatchSize];
            int[] rgDimData = new int[] { e.BatchSize, 784 };
            float[] rgRawLabels = new float[e.BatchSize * 10];
            int[] rgDimLabels = new int[] { e.BatchSize, 10 };
            int nDataOffset = 0;
            int nLabelOffset = 0;

            for (int i = 0; i < e.BatchSize; i++)
            {
                int nLabel = rgData[nIdx].Item2;
                rgRawLabels[nLabelOffset + nLabel] = 1;

                for (int j = 0; j < rgData[nIdx].Item1.Length; j++)
                {
                    float fVal = rgData[nIdx].Item1[j];

                    // Binarize data
                    if (fVal > 0)
                        fVal = 1.0f;

                    rgRawData[nDataOffset + j] = fVal;
                }

                nDataOffset += rgData[nIdx].Item1.Length;
                nLabelOffset += 10;

                nIdx++;

                if (nIdx == rgData.Count)
                    nIdx = 0;
            }

            DenseTensor<float> tensorData = new DenseTensor<float>(rgRawData, rgDimData);
            DenseTensor<float> tensorLabels = new DenseTensor<float>(rgRawLabels, rgDimLabels);

            e.Values.Add(NamedOnnxValue.CreateFromTensor<float>("X", tensorData));
            e.Values.Add(NamedOnnxValue.CreateFromTensor<float>("labels", tensorLabels));
        }

        private static void OnErrorFunction(object sender, ErrorFunctionArgs e)
        {
            Tensor<float> labels = e.Find("labels").AsTensor<float>();
            Tensor<float> predict = e.Find("predictions").AsTensor<float>();
            Tensor<float> loss = e.Find("loss").AsTensor<float>();

            if (labels.Dimensions[0] != predict.Dimensions[0])
                throw new Exception("Predict and Labels should have same batch size!");

            if (predict.Dimensions.Length < 2 || predict.Dimensions[1] != 10)
                throw new Exception("The prediction should have at least 2 dimensions.");


            int nBatchSize = predict.Dimensions[0];
            int nOffset = 0;

            for (int n = 0; n < nBatchSize; n++)
            {
                int nPredictMaxIdx = 0;
                float fPredictMax = 0;
                int nLabelMaxIdx = 0;
                float fLabelMax = 0;

                for (int j = 0; j < predict.Dimensions[1]; j++)
                {
                    float fPredict = predict.GetValue(nOffset + j);
                    if (fPredict > fPredictMax)
                    {
                        nPredictMaxIdx = j;
                        fPredictMax = fPredict;
                    }

                    float fLabel = labels.GetValue(nOffset + j);
                    if (fLabel > fLabelMax)
                    {
                        nLabelMaxIdx = j;
                        fLabelMax = fLabel;
                    }
                }

                if (nPredictMaxIdx == nLabelMaxIdx)
                    m_nTrueCount++;

                nOffset += predict.Dimensions[1];
            }

            m_fTotalLoss += loss.GetValue(0);
            m_nIterations++;
        }

        private static void OnEvaluationFunction(object sender, EvaluationFunctionArgs e)
        {
            float fPrecision = (float)m_nTrueCount / e.NumSamples;
            float fAveLoss = m_fTotalLoss / (float)e.NumSamples;

            Console.WriteLine("Step: " + e.Step.ToString() + ", #examples: " + e.NumSamples.ToString() + ", #correct: " + m_nTrueCount.ToString() + ", precision: " + fPrecision.ToString() + ", loss: " + fAveLoss.ToString());

            m_nTrueCount = 0;
            m_fTotalLoss = 0.0f;
        }
    }
}
