using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Microsoft.ML.OnnxRuntime.ResNet50v2Sample
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Read paths
            string modelFilePath = args[0];
            string imageFilePath = args[1];

            // Read image
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < image.Width; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }

            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", input)
            };

            // Run inference
            using var session = new InferenceSession(modelFilePath);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get softmax vector
            IEnumerable<float> output = results.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract top 10 predicted classes
            IEnumerable<Prediction> top10 = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                               .OrderByDescending(x => x.Confidence)
                               .Take(10);

            // Print results to console
            Console.WriteLine("Top 10 predictions for ResNet50 v2...");
            Console.WriteLine("--------------------------------------------------------------");
            foreach (var t in top10)
            {
                Console.WriteLine($"Label: {t.Label}, Confidence: {t.Confidence}");
            }
        }
    }
}
