namespace Microsoft.ML.OnnxRuntime.FasterRcnnSample
{
    public class Prediction
    {
        public Box Box { get; set; }
        public string Label { get; set; }
        public float Confidence { get; set; }
    }

    public class Box
    {
        public float Xmin { get; set; }
        public float Ymin { get; set; }
        public float Xmax { get; set; }
        public float Ymax { get; set; }

        public Box(float xmin, float ymin, float xmax, float ymax)
        {
            Xmin = xmin;
            Ymin = ymin;
            Xmax = xmax;
            Ymax = ymax;

        }
    }
}