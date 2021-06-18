/*
Copyright (C) 2021, Intel Corporation
SPDX-License-Identifier: Apache-2.0
*/

namespace yolov3
{
    public class Prediction
    {
        public Box Box { get; set; }
        public string Class { get; set; }
        public float Score { get; set; }
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