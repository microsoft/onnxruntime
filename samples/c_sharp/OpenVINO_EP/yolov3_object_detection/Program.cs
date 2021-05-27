/*
Copyright (C) 2021, Intel Corporation
SPDX-License-Identifier: Apache-2.0
*/

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;

namespace yolov3
{
    class Program
    {
        static void Main(string[] args)
        {
            // string is null or empty 
            if (args == null || args.Length < 3)
            {
                Console.WriteLine("Usage information: dotnet run model.onnx input.jpg output.jpg");
                return;
            } else
            {
                if(!(File.Exists(args[0])))
                {
                    Console.WriteLine("Model Path does not exist");
                    return;
                }
                if (!(File.Exists(args[1])))
                {
                    Console.WriteLine("Input Path does not exist");
                    return;
                }
            }

            // Read paths
            string modelFilePath = args[0];
            string imageFilePath = args[1];
            string outImageFilePath = args[2];

            using Image imageOrg = Image.Load(imageFilePath, out IImageFormat format);

            //Letterbox image
            var iw = imageOrg.Width;
            var ih = imageOrg.Height;
            var w = 416;
            var h = 416;

            if ((iw == 0) || (ih == 0))
            {
                Console.WriteLine("Math error: Attempted to divide by Zero");
                return;
            }

            float width = (float)w / iw;
            float height = (float)h / ih;

            float scale = Math.Min(width, height);

            var nw = (int)(iw * scale);
            var nh = (int)(ih * scale);

            var pad_dims_w = (w - nw) / 2;
            var pad_dims_h = (h - nh) / 2;

            // Resize image using default bicubic sampler 
            var image = imageOrg.Clone(x => x.Resize((nw), (nh)));

            var clone = new Image<Rgb24>(w, h);
            clone.Mutate(i => i.Fill(Color.Gray));
            clone.Mutate(o => o.DrawImage(image, new Point(pad_dims_w, pad_dims_h), 1f)); // draw the first one top left

            //Preprocessing image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, h, w });
            for (int y = 0; y < clone.Height; y++)
            {
                Span<Rgb24> pixelSpan = clone.GetPixelRowSpan(y);
                for (int x = 0; x < clone.Width; x++)
                {
                    input[0, 0, y, x] = pixelSpan[x].B / 255f;
                    input[0, 1, y, x] = pixelSpan[x].G / 255f;
                    input[0, 2, y, x] = pixelSpan[x].R / 255f;
                }
            }

            //Get the Image Shape
            var image_shape = new DenseTensor<float>(new[] { 1, 2 });
            image_shape[0, 0] = ih;
            image_shape[0, 1] = iw;

            // Setup inputs and outputs
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("input_1", input));
            container.Add(NamedOnnxValue.CreateFromTensor("image_shape", image_shape));

            // Session Options
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.AppendExecutionProvider_OpenVINO(@"MYRIAD_FP16");
            options.AppendExecutionProvider_CPU(1);

            // Run inference
            using var session = new InferenceSession(modelFilePath,options);
            
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(container);

            Console.WriteLine("Inference done");

            //Post Processing Steps
            var resultsArray = results.ToArray();
            Tensor<float> boxes = resultsArray[0].AsTensor<float>();
            Tensor<float> scores = resultsArray[1].AsTensor<float>();
            int[] indices = resultsArray[2].AsTensor<int>().ToArray();

            var len = indices.Length / 3;
            var out_classes = new int[len];
            float[] out_scores = new float[len];
            
            var predictions = new List<Prediction>();
            var count = 0;
            for (int i = 0; i < indices.Length; i = i + 3)
            {
                out_classes[count] = indices[i + 1];
                out_scores[count] = scores[indices[i], indices[i + 1], indices[i + 2]];
                predictions.Add(new Prediction
                {
                       Box = new Box(boxes[indices[i], indices[i + 2], 1],
                                     boxes[indices[i], indices[i + 2], 0],
                                     boxes[indices[i], indices[i + 2], 3],
                                     boxes[indices[i], indices[i + 2], 2]),
                        Class = LabelMap.Labels[out_classes[count]],
                        Score = out_scores[count]
                });
                count++;
            }

            // Put boxes, labels and confidence on image and save for viewing
            using var outputImage = File.OpenWrite(outImageFilePath);
            Font font = SystemFonts.CreateFont("Arial", 16);
            foreach (var p in predictions)
            {
                imageOrg.Mutate(x =>
                {
                    x.DrawLines(Color.Red, 2f, new PointF[] {

                        new PointF(p.Box.Xmin, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymin),

                        new PointF(p.Box.Xmax, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymax),

                        new PointF(p.Box.Xmax, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymax),

                        new PointF(p.Box.Xmin, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymin)
                    });
                    x.DrawText($"{p.Class}, {p.Score:0.00}", font, Color.White, new PointF(p.Box.Xmin, p.Box.Ymin));
                });
            }
            imageOrg.Save(outputImage, format);

        }
    }
}
