Ort_perf_view.html is to help rendering summarized profiling statistics in the form of treemaps.
By feeding a json file, the view will calculate overall latencies per operator for both cpu and gpu.
Please refer to [performance tuning](https://onnxruntime.ai/docs/performance/tune-performance.html) for the generation of statistics.


To render the view, please first open the json file:

![image](https://user-images.githubusercontent.com/48490400/148128966-a77bb856-f856-4dd0-9e60-40011d5612f0.png)

One will see:

![image](https://user-images.githubusercontent.com/48490400/148128781-ae81e9b3-9d7d-48b4-9324-01335097f963.png)

And each op is clickable to see detailed calling instances:

![image](https://user-images.githubusercontent.com/48490400/148128887-87c402eb-8864-4e37-be60-7aea9b767e01.png)
