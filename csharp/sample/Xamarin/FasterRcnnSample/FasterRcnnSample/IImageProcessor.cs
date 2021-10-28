using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FasterRcnnSample
{
    public interface IImageProcessor<T>
    {
        T PreprocessSourceImage(byte[] sourceImage);
        Tensor<float> GetTensorForImage(T image);
        byte[] ApplyPredictionsToImage(IList<Prediction> predictions, T image);
    }
}