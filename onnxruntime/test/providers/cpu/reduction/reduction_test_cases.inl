// Please don't manually edit this file. Generated from reduction_test_cases_generator.py
// Optimizations are disabled in this file to improve build throughput
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#pragma optimize ("", off)
#elif defined(__GNUC__)
#if defined(__clang__)
	#pragma clang optimize off
#else
	#pragma GCC push_options
	#pragma GCC optimize ("O0")
#endif
#endif
ReductionTestCases testcases = {
// input_data
{
0.548814f,
0.715189f,
0.602763f,
0.544883f,
0.423655f,
0.645894f,
0.437587f,
0.891773f,
0.963663f,
0.383442f,
0.791725f,
0.528895f,
0.568045f,
0.925597f,
0.071036f,
0.087129f,
0.020218f,
0.832620f,
0.778157f,
0.870012f,
0.978618f,
0.799159f,
0.461479f,
0.780529f,
0.118274f,
0.639921f,
0.143353f,
0.944669f,
0.521848f,
0.414662f,
0.264556f,
0.774234f,
0.456150f,
0.568434f,
0.018790f,
0.617635f,
0.612096f,
0.616934f,
0.943748f,
0.681820f,
0.359508f,
0.437032f,
0.697631f,
0.060225f,
0.666767f,
0.670638f,
0.210383f,
0.128926f,
0.315428f,
0.363711f,
0.570197f,
0.438602f,
0.988374f,
0.102045f,
0.208877f,
0.161310f,
0.653108f,
0.253292f,
0.466311f,
0.244426f,
0.158970f,
0.110375f,
0.656330f,
0.138183f,
0.196582f,
0.368725f,
0.820993f,
0.097101f,
0.837945f,
0.096098f,
0.976459f,
0.468651f,
},
// input_dims
{2, 3, 2, 2, 3},
  // map_op_attribute_expected
{
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{3.481198f,
3.997084f,
2.504645f,
4.667954f,
2.782728f,
2.699799f,
3.651138f,
2.434570f,
2.778356f,
1.987323f,
1.629165f,
3.297248f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{1.438602f,
1.723071f,
1.373210f,
1.944476f,
1.334432f,
1.257952f,
1.559183f,
1.202303f,
1.317791f,
0.915176f,
0.813585f,
1.602483f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{1.247377f,
1.385565f,
0.918147f,
1.540721f,
1.023432f,
0.993177f,
1.295039f,
0.889770f,
1.021859f,
0.686788f,
0.488067f,
1.193088f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{2.376074f,
2.483345f,
2.279583f,
2.581571f,
2.297269f,
2.270968f,
2.418173f,
2.234973f,
2.294850f,
2.138665f,
2.082937f,
2.401533f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{0.715189f,
0.963663f,
0.925597f,
0.978618f,
0.944669f,
0.774234f,
0.943748f,
0.697631f,
0.988374f,
0.653108f,
0.656330f,
0.976459f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{0.580200f,
0.666181f,
0.417441f,
0.777992f,
0.463788f,
0.449966f,
0.608523f,
0.405762f,
0.463059f,
0.331220f,
0.271527f,
0.549541f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{0.423655f,
0.383442f,
0.020218f,
0.461479f,
0.118274f,
0.018790f,
0.359508f,
0.060225f,
0.102045f,
0.161310f,
0.110375f,
0.096098f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{0.035275f,
0.060379f,
0.000055f,
0.190713f,
0.002218f,
0.000616f,
0.038177f,
0.000510f,
0.002894f,
0.000635f,
0.000115f,
0.002938f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{3.481198f,
3.997084f,
2.504645f,
4.667954f,
2.782728f,
2.699799f,
3.651138f,
2.434570f,
2.778356f,
1.987323f,
1.629165f,
3.297248f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2},
 // expected values
{2.069576f,
2.968973f,
1.885706f,
3.780986f,
1.780707f,
1.582444f,
2.431051f,
1.445532f,
1.736572f,
0.837547f,
0.661920f,
2.567952f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{3.481198f,
3.997084f,
2.504645f,
4.667954f,
2.782728f,
2.699799f,
3.651138f,
2.434570f,
2.778356f,
1.987323f,
1.629165f,
3.297248f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{1.438602f,
1.723071f,
1.373210f,
1.944476f,
1.334432f,
1.257952f,
1.559183f,
1.202303f,
1.317791f,
0.915176f,
0.813585f,
1.602483f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{1.247377f,
1.385565f,
0.918147f,
1.540721f,
1.023432f,
0.993177f,
1.295039f,
0.889770f,
1.021859f,
0.686788f,
0.488067f,
1.193088f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{2.376074f,
2.483345f,
2.279583f,
2.581571f,
2.297269f,
2.270968f,
2.418173f,
2.234973f,
2.294850f,
2.138665f,
2.082937f,
2.401533f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{0.715189f,
0.963663f,
0.925597f,
0.978618f,
0.944669f,
0.774234f,
0.943748f,
0.697631f,
0.988374f,
0.653108f,
0.656330f,
0.976459f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{0.580200f,
0.666181f,
0.417441f,
0.777992f,
0.463788f,
0.449966f,
0.608523f,
0.405762f,
0.463059f,
0.331220f,
0.271527f,
0.549541f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{0.423655f,
0.383442f,
0.020218f,
0.461479f,
0.118274f,
0.018790f,
0.359508f,
0.060225f,
0.102045f,
0.161310f,
0.110375f,
0.096098f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{0.035275f,
0.060379f,
0.000055f,
0.190713f,
0.002218f,
0.000616f,
0.038177f,
0.000510f,
0.002894f,
0.000635f,
0.000115f,
0.002938f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{3.481198f,
3.997084f,
2.504645f,
4.667954f,
2.782728f,
2.699799f,
3.651138f,
2.434570f,
2.778356f,
1.987323f,
1.629165f,
3.297248f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{-1, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 1, 1},
 // expected values
{2.069576f,
2.968973f,
1.885706f,
3.780986f,
1.780707f,
1.582444f,
2.431051f,
1.445532f,
1.736572f,
0.837547f,
0.661920f,
2.567952f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{1.914725f,
2.822342f,
2.741215f,
2.232489f,
2.277307f,
2.662803f,
1.895933f,
1.954793f,
1.631801f,
2.662185f,
1.247050f,
2.176473f,
1.216198f,
1.979705f,
1.569775f,
1.214244f,
1.380518f,
2.331651f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.967783f,
1.453639f,
1.410276f,
1.254770f,
1.351674f,
1.505065f,
1.139954f,
1.132085f,
0.884331f,
1.332661f,
0.746821f,
1.242118f,
0.632181f,
1.163028f,
0.906549f,
0.853013f,
1.006841f,
1.220046f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.649574f,
1.037567f,
1.008401f,
0.803117f,
0.822993f,
0.979379f,
0.639711f,
0.670284f,
0.489684f,
0.979147f,
0.220781f,
0.777706f,
0.195730f,
0.682948f,
0.450933f,
0.194122f,
0.322459f,
0.846577f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{1.867469f,
2.106419f,
2.086068f,
1.981963f,
2.017188f,
2.106127f,
1.911890f,
1.912408f,
1.808374f,
2.052358f,
1.719563f,
1.974530f,
1.694135f,
1.930749f,
1.804090f,
1.739433f,
1.806803f,
1.985549f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.548814f,
0.891773f,
0.963663f,
0.799159f,
0.925597f,
0.978618f,
0.944669f,
0.774234f,
0.617635f,
0.697631f,
0.616934f,
0.943748f,
0.438602f,
0.988374f,
0.653108f,
0.820993f,
0.976459f,
0.837945f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.478681f,
0.705586f,
0.685304f,
0.558122f,
0.569327f,
0.665701f,
0.473983f,
0.488698f,
0.407950f,
0.665546f,
0.311762f,
0.544118f,
0.304050f,
0.494926f,
0.392444f,
0.303561f,
0.345130f,
0.582913f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.383442f,
0.423655f,
0.528895f,
0.087129f,
0.020218f,
0.071036f,
0.118274f,
0.018790f,
0.143353f,
0.612096f,
0.060225f,
0.128926f,
0.208877f,
0.161310f,
0.102045f,
0.096098f,
0.097101f,
0.368725f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.050176f,
0.213925f,
0.198428f,
0.030778f,
0.007514f,
0.045178f,
0.016802f,
0.004858f,
0.016747f,
0.195255f,
0.002810f,
0.035456f,
0.007320f,
0.027040f,
0.009289f,
0.001733f,
0.002057f,
0.095036f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{1.914725f,
2.822342f,
2.741215f,
2.232489f,
2.277307f,
2.662803f,
1.895933f,
1.954793f,
1.631801f,
2.662185f,
1.247050f,
2.176473f,
1.216198f,
1.979705f,
1.569775f,
1.214244f,
1.380518f,
2.331651f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 3},
 // expected values
{0.936604f,
2.113067f,
1.988879f,
1.574448f,
1.827022f,
2.265222f,
1.299495f,
1.281615f,
0.782041f,
1.775985f,
0.557741f,
1.542857f,
0.399652f,
1.352635f,
0.821832f,
0.727631f,
1.013729f,
1.488512f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{1.914725f,
2.822342f,
2.741215f,
2.232489f,
2.277307f,
2.662803f,
1.895933f,
1.954793f,
1.631801f,
2.662185f,
1.247050f,
2.176473f,
1.216198f,
1.979705f,
1.569775f,
1.214244f,
1.380518f,
2.331651f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.967783f,
1.453639f,
1.410276f,
1.254770f,
1.351674f,
1.505065f,
1.139954f,
1.132085f,
0.884331f,
1.332661f,
0.746821f,
1.242118f,
0.632181f,
1.163028f,
0.906549f,
0.853013f,
1.006841f,
1.220046f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.649574f,
1.037567f,
1.008401f,
0.803117f,
0.822993f,
0.979379f,
0.639711f,
0.670284f,
0.489684f,
0.979147f,
0.220781f,
0.777706f,
0.195730f,
0.682948f,
0.450933f,
0.194122f,
0.322459f,
0.846577f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{1.867469f,
2.106419f,
2.086068f,
1.981963f,
2.017188f,
2.106127f,
1.911890f,
1.912408f,
1.808374f,
2.052358f,
1.719563f,
1.974530f,
1.694135f,
1.930749f,
1.804090f,
1.739433f,
1.806803f,
1.985549f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.548814f,
0.891773f,
0.963663f,
0.799159f,
0.925597f,
0.978618f,
0.944669f,
0.774234f,
0.617635f,
0.697631f,
0.616934f,
0.943748f,
0.438602f,
0.988374f,
0.653108f,
0.820993f,
0.976459f,
0.837945f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.478681f,
0.705586f,
0.685304f,
0.558122f,
0.569327f,
0.665701f,
0.473983f,
0.488698f,
0.407950f,
0.665546f,
0.311762f,
0.544118f,
0.304050f,
0.494926f,
0.392444f,
0.303561f,
0.345130f,
0.582913f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.383442f,
0.423655f,
0.528895f,
0.087129f,
0.020218f,
0.071036f,
0.118274f,
0.018790f,
0.143353f,
0.612096f,
0.060225f,
0.128926f,
0.208877f,
0.161310f,
0.102045f,
0.096098f,
0.097101f,
0.368725f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.050176f,
0.213925f,
0.198428f,
0.030778f,
0.007514f,
0.045178f,
0.016802f,
0.004858f,
0.016747f,
0.195255f,
0.002810f,
0.035456f,
0.007320f,
0.027040f,
0.009289f,
0.001733f,
0.002057f,
0.095036f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{1.914725f,
2.822342f,
2.741215f,
2.232489f,
2.277307f,
2.662803f,
1.895933f,
1.954793f,
1.631801f,
2.662185f,
1.247050f,
2.176473f,
1.216198f,
1.979705f,
1.569775f,
1.214244f,
1.380518f,
2.331651f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 1, 3},
 // expected values
{0.936604f,
2.113067f,
1.988879f,
1.574448f,
1.827022f,
2.265222f,
1.299495f,
1.281615f,
0.782041f,
1.775985f,
0.557741f,
1.542857f,
0.399652f,
1.352635f,
0.821832f,
0.727631f,
1.013729f,
1.488512f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{10.747742f,
9.385667f,
8.551745f,
7.226055f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{2.813811f,
2.480093f,
2.342597f,
2.047636f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{2.374696f,
2.239184f,
2.146135f,
1.977693f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{3.527107f,
3.445045f,
3.404396f,
3.330581f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{0.978618f,
0.944669f,
0.943748f,
0.988374f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{0.597097f,
0.521426f,
0.475097f,
0.401447f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{0.071036f,
0.018790f,
0.060225f,
0.096098f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{0.000002f,
0.000000f,
0.000000f,
0.000000f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{10.747742f,
9.385667f,
8.551745f,
7.226055f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 2},
 // expected values
{7.917535f,
6.150859f,
5.487762f,
4.192813f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{10.747742f,
9.385667f,
8.551745f,
7.226055f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{2.813811f,
2.480093f,
2.342597f,
2.047636f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{2.374696f,
2.239184f,
2.146135f,
1.977693f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{3.527107f,
3.445045f,
3.404396f,
3.330581f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{0.978618f,
0.944669f,
0.943748f,
0.988374f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{0.597097f,
0.521426f,
0.475097f,
0.401447f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{0.071036f,
0.018790f,
0.060225f,
0.096098f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{0.000002f,
0.000000f,
0.000000f,
0.000000f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{10.747742f,
9.385667f,
8.551745f,
7.226055f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2, 1, 4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 1, 1, 2, 1},
 // expected values
{7.917535f,
6.150859f,
5.487762f,
4.192813f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{1.646994f,
1.634261f,
1.879291f,
1.405027f,
1.783159f,
1.757001f,
1.423772f,
1.515039f,
1.506836f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{1.521024f,
1.403494f,
1.592839f,
1.237994f,
1.448567f,
1.442811f,
1.134680f,
1.204566f,
1.377115f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{2.657328f,
2.624730f,
2.725001f,
2.541516f,
2.668049f,
2.659616f,
2.522522f,
2.554146f,
2.594027f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.697631f,
0.891773f,
0.963663f,
0.799159f,
0.988374f,
0.978618f,
0.944669f,
0.976459f,
0.837945f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.572114f,
0.508674f,
0.614711f,
0.431086f,
0.532126f,
0.529072f,
0.388772f,
0.416914f,
0.495431f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.383442f,
0.060225f,
0.128926f,
0.087129f,
0.020218f,
0.071036f,
0.096098f,
0.018790f,
0.143353f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.009797f,
0.000601f,
0.007035f,
0.000225f,
0.000203f,
0.000420f,
0.000029f,
0.000010f,
0.001592f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{2.712588f,
2.670808f,
3.531736f,
1.974101f,
3.179657f,
3.087053f,
2.027126f,
2.295344f,
2.270554f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{1.646994f,
1.634261f,
1.879291f,
1.405027f,
1.783159f,
1.757001f,
1.423772f,
1.515039f,
1.506836f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{1.521024f,
1.403494f,
1.592839f,
1.237994f,
1.448567f,
1.442811f,
1.134680f,
1.204566f,
1.377115f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{2.657328f,
2.624730f,
2.725001f,
2.541516f,
2.668049f,
2.659616f,
2.522522f,
2.554146f,
2.594027f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.697631f,
0.891773f,
0.963663f,
0.799159f,
0.988374f,
0.978618f,
0.944669f,
0.976459f,
0.837945f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.572114f,
0.508674f,
0.614711f,
0.431086f,
0.532126f,
0.529072f,
0.388772f,
0.416914f,
0.495431f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.383442f,
0.060225f,
0.128926f,
0.087129f,
0.020218f,
0.071036f,
0.096098f,
0.018790f,
0.143353f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.009797f,
0.000601f,
0.007035f,
0.000225f,
0.000203f,
0.000420f,
0.000029f,
0.000010f,
0.001592f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, -2, -3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{2.712588f,
2.670808f,
3.531736f,
1.974101f,
3.179657f,
3.087053f,
2.027126f,
2.295344f,
2.270554f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{1.646994f,
1.634261f,
1.879291f,
1.405027f,
1.783159f,
1.757001f,
1.423772f,
1.515039f,
1.506836f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{1.521024f,
1.403494f,
1.592839f,
1.237994f,
1.448567f,
1.442811f,
1.134680f,
1.204566f,
1.377115f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{2.657328f,
2.624730f,
2.725001f,
2.541516f,
2.668049f,
2.659616f,
2.522522f,
2.554146f,
2.594027f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.697631f,
0.891773f,
0.963663f,
0.799159f,
0.988374f,
0.978618f,
0.944669f,
0.976459f,
0.837945f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.572114f,
0.508674f,
0.614711f,
0.431086f,
0.532126f,
0.529072f,
0.388772f,
0.416914f,
0.495431f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.383442f,
0.060225f,
0.128926f,
0.087129f,
0.020218f,
0.071036f,
0.096098f,
0.018790f,
0.143353f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{0.009797f,
0.000601f,
0.007035f,
0.000225f,
0.000203f,
0.000420f,
0.000029f,
0.000010f,
0.001592f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
0 ,
},
 // expected dims
{3, 3},
 // expected values
{2.712588f,
2.670808f,
3.531736f,
1.974101f,
3.179657f,
3.087053f,
2.027126f,
2.295344f,
2.270554f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{1.646994f,
1.634261f,
1.879291f,
1.405027f,
1.783159f,
1.757001f,
1.423772f,
1.515039f,
1.506836f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{1.521024f,
1.403494f,
1.592839f,
1.237994f,
1.448567f,
1.442811f,
1.134680f,
1.204566f,
1.377115f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{2.657328f,
2.624730f,
2.725001f,
2.541516f,
2.668049f,
2.659616f,
2.522522f,
2.554146f,
2.594027f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.697631f,
0.891773f,
0.963663f,
0.799159f,
0.988374f,
0.978618f,
0.944669f,
0.976459f,
0.837945f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.572114f,
0.508674f,
0.614711f,
0.431086f,
0.532126f,
0.529072f,
0.388772f,
0.416914f,
0.495431f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.383442f,
0.060225f,
0.128926f,
0.087129f,
0.020218f,
0.071036f,
0.096098f,
0.018790f,
0.143353f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{0.009797f,
0.000601f,
0.007035f,
0.000225f,
0.000203f,
0.000420f,
0.000029f,
0.000010f,
0.001592f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{4.576911f,
4.069392f,
4.917688f,
3.448687f,
4.257011f,
4.232579f,
3.110177f,
3.335311f,
3.963452f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0, 2, 3},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 1, 1, 3},
 // expected values
{2.712588f,
2.670808f,
3.531736f,
1.974101f,
3.179657f,
3.087053f,
2.027126f,
2.295344f,
2.270554f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{1.160909f,
1.332123f,
1.546511f,
1.226703f,
0.783163f,
1.082926f,
1.135218f,
0.951998f,
1.630429f,
1.054079f,
1.002108f,
0.657821f,
0.883473f,
1.289307f,
0.641233f,
0.525731f,
1.008592f,
0.934665f,
0.987034f,
1.031322f,
1.631727f,
1.052450f,
0.927790f,
1.024955f,
0.277244f,
0.750296f,
0.799683f,
1.082852f,
0.718431f,
0.783387f,
1.085549f,
0.871335f,
1.294095f,
0.664532f,
0.995249f,
1.086287f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.822105f,
0.944512f,
1.119814f,
0.872798f,
0.555634f,
0.779856f,
0.823512f,
0.893804f,
1.171846f,
0.772517f,
0.819200f,
0.544382f,
0.649746f,
0.994492f,
0.574605f,
0.447172f,
0.988581f,
0.838850f,
0.805703f,
0.884840f,
1.176539f,
0.838338f,
0.656056f,
0.817906f,
0.198142f,
0.649370f,
0.671803f,
0.954722f,
0.557647f,
0.554890f,
0.862566f,
0.780299f,
0.954057f,
0.576500f,
0.976640f,
0.775311f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.149204f,
0.286774f,
0.436002f,
0.204330f,
-0.244415f,
0.079667f,
0.126825f,
-0.049192f,
0.488843f,
0.052668f,
0.002105f,
-0.418822f,
-0.123895f,
0.254105f,
-0.444363f,
-0.642966f,
0.008556f,
-0.067567f,
-0.013051f,
0.030841f,
0.489639f,
0.051121f,
-0.074950f,
0.024648f,
-1.282857f,
-0.287287f,
-0.223540f,
0.079598f,
-0.330686f,
-0.244128f,
0.082086f,
-0.137729f,
0.257812f,
-0.408672f,
-0.004762f,
0.082765f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{1.274102f,
1.360415f,
1.480867f,
1.308841f,
1.085243f,
1.240053f,
1.269186f,
1.253199f,
1.519340f,
1.230462f,
1.235864f,
1.041923f,
1.142839f,
1.376757f,
1.044591f,
0.971375f,
1.310300f,
1.225764f,
1.226638f,
1.270319f,
1.522197f,
1.256165f,
1.157045f,
1.241128f,
0.831976f,
1.102946f,
1.125527f,
1.313763f,
1.065529f,
1.085104f,
1.274135f,
1.185066f,
1.358306f,
1.053046f,
1.301283f,
1.239063f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.612096f,
0.715189f,
0.943748f,
0.681820f,
0.423655f,
0.645894f,
0.697631f,
0.891773f,
0.963663f,
0.670638f,
0.791725f,
0.528895f,
0.568045f,
0.925597f,
0.570197f,
0.438602f,
0.988374f,
0.832620f,
0.778157f,
0.870012f,
0.978618f,
0.799159f,
0.466311f,
0.780529f,
0.158970f,
0.639921f,
0.656330f,
0.944669f,
0.521848f,
0.414662f,
0.820993f,
0.774234f,
0.837945f,
0.568434f,
0.976459f,
0.617635f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.580455f,
0.666062f,
0.773256f,
0.613352f,
0.391581f,
0.541463f,
0.567609f,
0.475999f,
0.815215f,
0.527040f,
0.501054f,
0.328911f,
0.441736f,
0.644654f,
0.320616f,
0.262865f,
0.504296f,
0.467332f,
0.493517f,
0.515661f,
0.815863f,
0.526225f,
0.463895f,
0.512477f,
0.138622f,
0.375148f,
0.399841f,
0.541426f,
0.359215f,
0.391694f,
0.542774f,
0.435667f,
0.647048f,
0.332266f,
0.497625f,
0.543143f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.548814f,
0.616934f,
0.602763f,
0.544883f,
0.359508f,
0.437032f,
0.437587f,
0.060225f,
0.666767f,
0.383442f,
0.210383f,
0.128926f,
0.315428f,
0.363711f,
0.071036f,
0.087129f,
0.020218f,
0.102045f,
0.208877f,
0.161310f,
0.653108f,
0.253292f,
0.461479f,
0.244426f,
0.118274f,
0.110375f,
0.143353f,
0.138183f,
0.196582f,
0.368725f,
0.264556f,
0.097101f,
0.456150f,
0.096098f,
0.018790f,
0.468651f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.335926f,
0.441225f,
0.568857f,
0.371512f,
0.152307f,
0.282276f,
0.305274f,
0.053707f,
0.642538f,
0.257150f,
0.166565f,
0.068188f,
0.179177f,
0.336649f,
0.040505f,
0.038215f,
0.019983f,
0.084965f,
0.162539f,
0.140341f,
0.639144f,
0.202420f,
0.215193f,
0.190781f,
0.018802f,
0.070631f,
0.094087f,
0.130537f,
0.102586f,
0.152896f,
0.217198f,
0.075179f,
0.382229f,
0.054626f,
0.018347f,
0.289456f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{1.160909f,
1.332123f,
1.546511f,
1.226703f,
0.783163f,
1.082926f,
1.135218f,
0.951998f,
1.630429f,
1.054079f,
1.002108f,
0.657821f,
0.883473f,
1.289307f,
0.641233f,
0.525731f,
1.008592f,
0.934665f,
0.987034f,
1.031322f,
1.631727f,
1.052450f,
0.927790f,
1.024955f,
0.277244f,
0.750296f,
0.799683f,
1.082852f,
0.718431f,
0.783387f,
1.085549f,
0.871335f,
1.294095f,
0.664532f,
0.995249f,
1.086287f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.675857f,
0.892103f,
1.253984f,
0.761777f,
0.308729f,
0.608176f,
0.678172f,
0.798886f,
1.373224f,
0.596783f,
0.671089f,
0.296352f,
0.422170f,
0.989015f,
0.330170f,
0.199963f,
0.977292f,
0.703669f,
0.649157f,
0.782942f,
1.384244f,
0.702811f,
0.430409f,
0.668970f,
0.039260f,
0.421682f,
0.451319f,
0.911494f,
0.310970f,
0.307903f,
0.744020f,
0.608866f,
0.910225f,
0.332352f,
0.953826f,
0.601108f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{1.160909f,
1.332123f,
1.546511f,
1.226703f,
0.783163f,
1.082926f,
1.135218f,
0.951998f,
1.630429f,
1.054079f,
1.002108f,
0.657821f,
0.883473f,
1.289307f,
0.641233f,
0.525731f,
1.008592f,
0.934665f,
0.987034f,
1.031322f,
1.631727f,
1.052450f,
0.927790f,
1.024955f,
0.277244f,
0.750296f,
0.799683f,
1.082852f,
0.718431f,
0.783387f,
1.085549f,
0.871335f,
1.294095f,
0.664532f,
0.995249f,
1.086287f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.822105f,
0.944512f,
1.119814f,
0.872798f,
0.555634f,
0.779856f,
0.823512f,
0.893804f,
1.171846f,
0.772517f,
0.819200f,
0.544382f,
0.649746f,
0.994492f,
0.574605f,
0.447172f,
0.988581f,
0.838850f,
0.805703f,
0.884840f,
1.176539f,
0.838338f,
0.656056f,
0.817906f,
0.198142f,
0.649370f,
0.671803f,
0.954722f,
0.557647f,
0.554890f,
0.862566f,
0.780299f,
0.954057f,
0.576500f,
0.976640f,
0.775311f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.149204f,
0.286774f,
0.436002f,
0.204330f,
-0.244415f,
0.079667f,
0.126825f,
-0.049192f,
0.488843f,
0.052668f,
0.002105f,
-0.418822f,
-0.123895f,
0.254105f,
-0.444363f,
-0.642966f,
0.008556f,
-0.067567f,
-0.013051f,
0.030841f,
0.489639f,
0.051121f,
-0.074950f,
0.024648f,
-1.282857f,
-0.287287f,
-0.223540f,
0.079598f,
-0.330686f,
-0.244128f,
0.082086f,
-0.137729f,
0.257812f,
-0.408672f,
-0.004762f,
0.082765f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{1.274102f,
1.360415f,
1.480867f,
1.308841f,
1.085243f,
1.240053f,
1.269186f,
1.253199f,
1.519340f,
1.230462f,
1.235864f,
1.041923f,
1.142839f,
1.376757f,
1.044591f,
0.971375f,
1.310300f,
1.225764f,
1.226638f,
1.270319f,
1.522197f,
1.256165f,
1.157045f,
1.241128f,
0.831976f,
1.102946f,
1.125527f,
1.313763f,
1.065529f,
1.085104f,
1.274135f,
1.185066f,
1.358306f,
1.053046f,
1.301283f,
1.239063f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.612096f,
0.715189f,
0.943748f,
0.681820f,
0.423655f,
0.645894f,
0.697631f,
0.891773f,
0.963663f,
0.670638f,
0.791725f,
0.528895f,
0.568045f,
0.925597f,
0.570197f,
0.438602f,
0.988374f,
0.832620f,
0.778157f,
0.870012f,
0.978618f,
0.799159f,
0.466311f,
0.780529f,
0.158970f,
0.639921f,
0.656330f,
0.944669f,
0.521848f,
0.414662f,
0.820993f,
0.774234f,
0.837945f,
0.568434f,
0.976459f,
0.617635f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.580455f,
0.666062f,
0.773256f,
0.613352f,
0.391581f,
0.541463f,
0.567609f,
0.475999f,
0.815215f,
0.527040f,
0.501054f,
0.328911f,
0.441736f,
0.644654f,
0.320616f,
0.262865f,
0.504296f,
0.467332f,
0.493517f,
0.515661f,
0.815863f,
0.526225f,
0.463895f,
0.512477f,
0.138622f,
0.375148f,
0.399841f,
0.541426f,
0.359215f,
0.391694f,
0.542774f,
0.435667f,
0.647048f,
0.332266f,
0.497625f,
0.543143f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.548814f,
0.616934f,
0.602763f,
0.544883f,
0.359508f,
0.437032f,
0.437587f,
0.060225f,
0.666767f,
0.383442f,
0.210383f,
0.128926f,
0.315428f,
0.363711f,
0.071036f,
0.087129f,
0.020218f,
0.102045f,
0.208877f,
0.161310f,
0.653108f,
0.253292f,
0.461479f,
0.244426f,
0.118274f,
0.110375f,
0.143353f,
0.138183f,
0.196582f,
0.368725f,
0.264556f,
0.097101f,
0.456150f,
0.096098f,
0.018790f,
0.468651f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.335926f,
0.441225f,
0.568857f,
0.371512f,
0.152307f,
0.282276f,
0.305274f,
0.053707f,
0.642538f,
0.257150f,
0.166565f,
0.068188f,
0.179177f,
0.336649f,
0.040505f,
0.038215f,
0.019983f,
0.084965f,
0.162539f,
0.140341f,
0.639144f,
0.202420f,
0.215193f,
0.190781f,
0.018802f,
0.070631f,
0.094087f,
0.130537f,
0.102586f,
0.152896f,
0.217198f,
0.075179f,
0.382229f,
0.054626f,
0.018347f,
0.289456f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{1.160909f,
1.332123f,
1.546511f,
1.226703f,
0.783163f,
1.082926f,
1.135218f,
0.951998f,
1.630429f,
1.054079f,
1.002108f,
0.657821f,
0.883473f,
1.289307f,
0.641233f,
0.525731f,
1.008592f,
0.934665f,
0.987034f,
1.031322f,
1.631727f,
1.052450f,
0.927790f,
1.024955f,
0.277244f,
0.750296f,
0.799683f,
1.082852f,
0.718431f,
0.783387f,
1.085549f,
0.871335f,
1.294095f,
0.664532f,
0.995249f,
1.086287f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.675857f,
0.892103f,
1.253984f,
0.761777f,
0.308729f,
0.608176f,
0.678172f,
0.798886f,
1.373224f,
0.596783f,
0.671089f,
0.296352f,
0.422170f,
0.989015f,
0.330170f,
0.199963f,
0.977292f,
0.703669f,
0.649157f,
0.782942f,
1.384244f,
0.702811f,
0.430409f,
0.668970f,
0.039260f,
0.421682f,
0.451319f,
0.911494f,
0.310970f,
0.307903f,
0.744020f,
0.608866f,
0.910225f,
0.332352f,
0.953826f,
0.601108f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{0},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.986401f,
1.606962f,
1.566426f,
0.928325f,
1.215380f,
1.174789f,
1.346201f,
1.795609f,
1.049654f,
0.886288f,
0.481698f,
1.613149f,
0.382830f,
1.414155f,
0.599504f,
1.513103f,
0.540638f,
1.032297f,
1.309727f,
0.677159f,
1.610515f,
1.352458f,
0.569890f,
0.565958f,
0.524305f,
0.525020f,
1.223305f,
0.691893f,
1.454685f,
0.346470f,
0.979963f,
0.207476f,
1.494274f,
0.234281f,
1.173042f,
0.837376f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.701911f,
1.143134f,
1.136648f,
0.666277f,
0.897949f,
0.834811f,
0.963433f,
1.270295f,
0.981193f,
0.803894f,
0.461922f,
1.141263f,
0.289790f,
1.004458f,
0.478146f,
1.102505f,
0.522186f,
0.743921f,
0.928090f,
0.619867f,
1.155525f,
0.956365f,
0.416541f,
0.455652f,
0.378318f,
0.397877f,
0.866992f,
0.506486f,
1.092853f,
0.264872f,
0.836242f,
0.147008f,
1.064387f,
0.168313f,
0.996051f,
0.596316f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{-0.013693f,
0.474346f,
0.448797f,
-0.074374f,
0.195057f,
0.161089f,
0.297287f,
0.585344f,
0.048461f,
-0.120713f,
-0.730438f,
0.478188f,
-0.960164f,
0.346532f,
-0.511653f,
0.414162f,
-0.615005f,
0.031787f,
0.269819f,
-0.389848f,
0.476554f,
0.301924f,
-0.562311f,
-0.569235f,
-0.645681f,
-0.644318f,
0.201556f,
-0.368324f,
0.374789f,
-1.059958f,
-0.020241f,
-1.572738f,
0.401641f,
-1.451232f,
0.159600f,
-0.177482f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{1.187893f,
1.500521f,
1.492554f,
1.160564f,
1.317677f,
1.282252f,
1.371756f,
1.591338f,
1.317586f,
1.198369f,
0.958140f,
1.500061f,
0.887235f,
1.402478f,
1.005080f,
1.467289f,
0.994772f,
1.214437f,
1.348925f,
1.069977f,
1.507964f,
1.369392f,
0.980870f,
0.987946f,
0.956718f,
0.960769f,
1.305659f,
1.043380f,
1.454178f,
0.868914f,
1.236941f,
0.796907f,
1.444402f,
0.810509f,
1.353842f,
1.113083f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.548814f,
0.891773f,
0.963663f,
0.544883f,
0.791725f,
0.645894f,
0.778157f,
0.925597f,
0.978618f,
0.799159f,
0.461479f,
0.832620f,
0.264556f,
0.774234f,
0.456150f,
0.944669f,
0.521848f,
0.617635f,
0.697631f,
0.616934f,
0.943748f,
0.681820f,
0.359508f,
0.437032f,
0.315428f,
0.363711f,
0.653108f,
0.438602f,
0.988374f,
0.244426f,
0.820993f,
0.110375f,
0.837945f,
0.138183f,
0.976459f,
0.468651f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.493200f,
0.803481f,
0.783213f,
0.464162f,
0.607690f,
0.587395f,
0.673101f,
0.897804f,
0.524827f,
0.443144f,
0.240849f,
0.806575f,
0.191415f,
0.707077f,
0.299752f,
0.756551f,
0.270319f,
0.516149f,
0.654863f,
0.338580f,
0.805257f,
0.676229f,
0.284945f,
0.282979f,
0.262153f,
0.262510f,
0.611653f,
0.345947f,
0.727342f,
0.173235f,
0.489981f,
0.103738f,
0.747137f,
0.117141f,
0.586521f,
0.418688f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.437587f,
0.715189f,
0.602763f,
0.383442f,
0.423655f,
0.528895f,
0.568045f,
0.870012f,
0.071036f,
0.087129f,
0.020218f,
0.780529f,
0.118274f,
0.639921f,
0.143353f,
0.568434f,
0.018790f,
0.414662f,
0.612096f,
0.060225f,
0.666767f,
0.670638f,
0.210383f,
0.128926f,
0.208877f,
0.161310f,
0.570197f,
0.253292f,
0.466311f,
0.102045f,
0.158970f,
0.097101f,
0.656330f,
0.096098f,
0.196582f,
0.368725f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.240154f,
0.637787f,
0.580861f,
0.208931f,
0.335418f,
0.341610f,
0.442028f,
0.805280f,
0.069517f,
0.069630f,
0.009330f,
0.649884f,
0.031290f,
0.495448f,
0.065391f,
0.536982f,
0.009805f,
0.256110f,
0.427017f,
0.037155f,
0.629260f,
0.457255f,
0.075634f,
0.056345f,
0.065886f,
0.058670f,
0.372400f,
0.111094f,
0.460889f,
0.024942f,
0.130513f,
0.010718f,
0.549968f,
0.013279f,
0.191955f,
0.172803f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.986401f,
1.606962f,
1.566426f,
0.928325f,
1.215380f,
1.174789f,
1.346201f,
1.795609f,
1.049654f,
0.886288f,
0.481698f,
1.613149f,
0.382830f,
1.414155f,
0.599504f,
1.513103f,
0.540638f,
1.032297f,
1.309727f,
0.677159f,
1.610515f,
1.352458f,
0.569890f,
0.565958f,
0.524305f,
0.525020f,
1.223305f,
0.691893f,
1.454685f,
0.346470f,
0.979963f,
0.207476f,
1.494274f,
0.234281f,
1.173042f,
0.837376f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.492679f,
1.306755f,
1.291970f,
0.443925f,
0.806312f,
0.696909f,
0.928203f,
1.613650f,
0.962740f,
0.646246f,
0.213372f,
1.302482f,
0.083979f,
1.008937f,
0.228623f,
1.215517f,
0.272679f,
0.553418f,
0.861350f,
0.384235f,
1.335238f,
0.914634f,
0.173507f,
0.207619f,
0.143125f,
0.158306f,
0.751675f,
0.256528f,
1.194329f,
0.070157f,
0.699301f,
0.021611f,
1.132920f,
0.028329f,
0.992118f,
0.355592f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 3},
 // expected values
{1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.986401f,
1.606962f,
1.566426f,
0.928325f,
1.215380f,
1.174789f,
1.346201f,
1.795609f,
1.049654f,
0.886288f,
0.481698f,
1.613149f,
0.382830f,
1.414155f,
0.599504f,
1.513103f,
0.540638f,
1.032297f,
1.309727f,
0.677159f,
1.610515f,
1.352458f,
0.569890f,
0.565958f,
0.524305f,
0.525020f,
1.223305f,
0.691893f,
1.454685f,
0.346470f,
0.979963f,
0.207476f,
1.494274f,
0.234281f,
1.173042f,
0.837376f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.701911f,
1.143134f,
1.136648f,
0.666277f,
0.897949f,
0.834811f,
0.963433f,
1.270295f,
0.981193f,
0.803894f,
0.461922f,
1.141263f,
0.289790f,
1.004458f,
0.478146f,
1.102505f,
0.522186f,
0.743921f,
0.928090f,
0.619867f,
1.155525f,
0.956365f,
0.416541f,
0.455652f,
0.378318f,
0.397877f,
0.866992f,
0.506486f,
1.092853f,
0.264872f,
0.836242f,
0.147008f,
1.064387f,
0.168313f,
0.996051f,
0.596316f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{-0.013693f,
0.474346f,
0.448797f,
-0.074374f,
0.195057f,
0.161089f,
0.297287f,
0.585344f,
0.048461f,
-0.120713f,
-0.730438f,
0.478188f,
-0.960164f,
0.346532f,
-0.511653f,
0.414162f,
-0.615005f,
0.031787f,
0.269819f,
-0.389848f,
0.476554f,
0.301924f,
-0.562311f,
-0.569235f,
-0.645681f,
-0.644318f,
0.201556f,
-0.368324f,
0.374789f,
-1.059958f,
-0.020241f,
-1.572738f,
0.401641f,
-1.451232f,
0.159600f,
-0.177482f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{1.187893f,
1.500521f,
1.492554f,
1.160564f,
1.317677f,
1.282252f,
1.371756f,
1.591338f,
1.317586f,
1.198369f,
0.958140f,
1.500061f,
0.887235f,
1.402478f,
1.005080f,
1.467289f,
0.994772f,
1.214437f,
1.348925f,
1.069977f,
1.507964f,
1.369392f,
0.980870f,
0.987946f,
0.956718f,
0.960769f,
1.305659f,
1.043380f,
1.454178f,
0.868914f,
1.236941f,
0.796907f,
1.444402f,
0.810509f,
1.353842f,
1.113083f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.548814f,
0.891773f,
0.963663f,
0.544883f,
0.791725f,
0.645894f,
0.778157f,
0.925597f,
0.978618f,
0.799159f,
0.461479f,
0.832620f,
0.264556f,
0.774234f,
0.456150f,
0.944669f,
0.521848f,
0.617635f,
0.697631f,
0.616934f,
0.943748f,
0.681820f,
0.359508f,
0.437032f,
0.315428f,
0.363711f,
0.653108f,
0.438602f,
0.988374f,
0.244426f,
0.820993f,
0.110375f,
0.837945f,
0.138183f,
0.976459f,
0.468651f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.493200f,
0.803481f,
0.783213f,
0.464162f,
0.607690f,
0.587395f,
0.673101f,
0.897804f,
0.524827f,
0.443144f,
0.240849f,
0.806575f,
0.191415f,
0.707077f,
0.299752f,
0.756551f,
0.270319f,
0.516149f,
0.654863f,
0.338580f,
0.805257f,
0.676229f,
0.284945f,
0.282979f,
0.262153f,
0.262510f,
0.611653f,
0.345947f,
0.727342f,
0.173235f,
0.489981f,
0.103738f,
0.747137f,
0.117141f,
0.586521f,
0.418688f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.437587f,
0.715189f,
0.602763f,
0.383442f,
0.423655f,
0.528895f,
0.568045f,
0.870012f,
0.071036f,
0.087129f,
0.020218f,
0.780529f,
0.118274f,
0.639921f,
0.143353f,
0.568434f,
0.018790f,
0.414662f,
0.612096f,
0.060225f,
0.666767f,
0.670638f,
0.210383f,
0.128926f,
0.208877f,
0.161310f,
0.570197f,
0.253292f,
0.466311f,
0.102045f,
0.158970f,
0.097101f,
0.656330f,
0.096098f,
0.196582f,
0.368725f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.240154f,
0.637787f,
0.580861f,
0.208931f,
0.335418f,
0.341610f,
0.442028f,
0.805280f,
0.069517f,
0.069630f,
0.009330f,
0.649884f,
0.031290f,
0.495448f,
0.065391f,
0.536982f,
0.009805f,
0.256110f,
0.427017f,
0.037155f,
0.629260f,
0.457255f,
0.075634f,
0.056345f,
0.065886f,
0.058670f,
0.372400f,
0.111094f,
0.460889f,
0.024942f,
0.130513f,
0.010718f,
0.549968f,
0.013279f,
0.191955f,
0.172803f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.986401f,
1.606962f,
1.566426f,
0.928325f,
1.215380f,
1.174789f,
1.346201f,
1.795609f,
1.049654f,
0.886288f,
0.481698f,
1.613149f,
0.382830f,
1.414155f,
0.599504f,
1.513103f,
0.540638f,
1.032297f,
1.309727f,
0.677159f,
1.610515f,
1.352458f,
0.569890f,
0.565958f,
0.524305f,
0.525020f,
1.223305f,
0.691893f,
1.454685f,
0.346470f,
0.979963f,
0.207476f,
1.494274f,
0.234281f,
1.173042f,
0.837376f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.492679f,
1.306755f,
1.291970f,
0.443925f,
0.806312f,
0.696909f,
0.928203f,
1.613650f,
0.962740f,
0.646246f,
0.213372f,
1.302482f,
0.083979f,
1.008937f,
0.228623f,
1.215517f,
0.272679f,
0.553418f,
0.861350f,
0.384235f,
1.335238f,
0.914634f,
0.173507f,
0.207619f,
0.143125f,
0.158306f,
0.751675f,
0.256528f,
1.194329f,
0.070157f,
0.699301f,
0.021611f,
1.132920f,
0.028329f,
0.992118f,
0.355592f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{2},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 1, 2, 3},
 // expected values
{1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{1.866766f,
1.614432f,
2.293023f,
1.704061f,
1.564677f,
0.939968f,
2.626787f,
2.041167f,
0.901549f,
1.881179f,
1.494940f,
1.204859f,
2.172778f,
1.478360f,
1.424623f,
1.009947f,
1.249336f,
1.529020f,
1.023295f,
0.964028f,
0.925674f,
0.703490f,
1.756039f,
1.541209f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{1.084443f,
0.945283f,
1.383975f,
1.026443f,
1.088324f,
0.837410f,
1.523202f,
1.208654f,
0.666362f,
1.156144f,
0.936750f,
0.839609f,
1.282938f,
0.886071f,
0.966899f,
0.714589f,
0.746261f,
1.086125f,
0.704415f,
0.584248f,
0.684268f,
0.440111f,
1.177119f,
1.087355f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{0.624208f,
0.478983f,
0.829871f,
0.533015f,
0.447680f,
-0.061910f,
0.965762f,
0.713522f,
-0.103641f,
0.631899f,
0.402086f,
0.186363f,
0.776006f,
0.390933f,
0.353907f,
0.009898f,
0.222612f,
0.424627f,
0.023027f,
-0.036635f,
-0.077233f,
-0.351701f,
0.563061f,
0.432567f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{1.723290f,
1.640865f,
1.888538f,
1.681123f,
1.679301f,
1.484055f,
1.977571f,
1.790545f,
1.429359f,
1.752838f,
1.619342f,
1.534500f,
1.835318f,
1.601044f,
1.613229f,
1.464930f,
1.521288f,
1.676130f,
1.465301f,
1.425334f,
1.439054f,
1.337973f,
1.738067f,
1.677796f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{0.715189f,
0.645894f,
0.963663f,
0.791725f,
0.925597f,
0.832620f,
0.978618f,
0.799159f,
0.639921f,
0.944669f,
0.774234f,
0.617635f,
0.943748f,
0.681820f,
0.697631f,
0.670638f,
0.570197f,
0.988374f,
0.653108f,
0.466311f,
0.656330f,
0.368725f,
0.837945f,
0.976459f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{0.622255f,
0.538144f,
0.764341f,
0.568020f,
0.521559f,
0.313323f,
0.875596f,
0.680389f,
0.300516f,
0.627060f,
0.498313f,
0.401620f,
0.724259f,
0.492787f,
0.474874f,
0.336649f,
0.416445f,
0.509673f,
0.341098f,
0.321343f,
0.308558f,
0.234497f,
0.585346f,
0.513736f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{0.548814f,
0.423655f,
0.437587f,
0.383442f,
0.071036f,
0.020218f,
0.778157f,
0.461479f,
0.118274f,
0.414662f,
0.264556f,
0.018790f,
0.612096f,
0.359508f,
0.060225f,
0.128926f,
0.315428f,
0.102045f,
0.161310f,
0.244426f,
0.110375f,
0.138183f,
0.097101f,
0.096098f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{0.236588f,
0.149100f,
0.376049f,
0.160562f,
0.037349f,
0.001467f,
0.662530f,
0.287855f,
0.010850f,
0.204418f,
0.093432f,
0.006597f,
0.356381f,
0.107125f,
0.028014f,
0.018190f,
0.065416f,
0.044237f,
0.022006f,
0.028870f,
0.011516f,
0.010016f,
0.066801f,
0.043976f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{1.866766f,
1.614432f,
2.293023f,
1.704061f,
1.564677f,
0.939968f,
2.626787f,
2.041167f,
0.901549f,
1.881179f,
1.494940f,
1.204859f,
2.172778f,
1.478360f,
1.424623f,
1.009947f,
1.249336f,
1.529020f,
1.023295f,
0.964028f,
0.925674f,
0.703490f,
1.756039f,
1.541209f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{1.176016f,
0.893560f,
1.915388f,
1.053586f,
1.184450f,
0.701256f,
2.320143f,
1.460843f,
0.444038f,
1.336670f,
0.877501f,
0.704944f,
1.645929f,
0.785122f,
0.934894f,
0.510638f,
0.556905f,
1.179667f,
0.496201f,
0.341346f,
0.468223f,
0.193697f,
1.385610f,
1.182342f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{1.000000f,
2.000000f,
2.000000f,
1.000000f,
1.000000f,
2.000000f,
2.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
2.000000f,
2.000000f,
0.000000f,
0.000000f,
0.000000f,
2.000000f,
1.000000f,
2.000000f,
1.000000f,
2.000000f,
2.000000f,
2.000000f,
1.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
0 ,
},
 // expected dims
{2, 3, 2, 2},
 // expected values
{0.000000f,
1.000000f,
0.000000f,
0.000000f,
2.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
2.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
2.000000f,
0.000000f,
2.000000f,
1.000000f,
2.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{1.866766f,
1.614432f,
2.293023f,
1.704061f,
1.564677f,
0.939968f,
2.626787f,
2.041167f,
0.901549f,
1.881179f,
1.494940f,
1.204859f,
2.172778f,
1.478360f,
1.424623f,
1.009947f,
1.249336f,
1.529020f,
1.023295f,
0.964028f,
0.925674f,
0.703490f,
1.756039f,
1.541209f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{1.084443f,
0.945283f,
1.383975f,
1.026443f,
1.088324f,
0.837410f,
1.523202f,
1.208654f,
0.666362f,
1.156144f,
0.936750f,
0.839609f,
1.282938f,
0.886071f,
0.966899f,
0.714589f,
0.746261f,
1.086125f,
0.704415f,
0.584248f,
0.684268f,
0.440111f,
1.177119f,
1.087355f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{0.624208f,
0.478983f,
0.829871f,
0.533015f,
0.447680f,
-0.061910f,
0.965762f,
0.713522f,
-0.103641f,
0.631899f,
0.402086f,
0.186363f,
0.776006f,
0.390933f,
0.353907f,
0.009898f,
0.222612f,
0.424627f,
0.023027f,
-0.036635f,
-0.077233f,
-0.351701f,
0.563061f,
0.432567f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{1.723290f,
1.640865f,
1.888538f,
1.681123f,
1.679301f,
1.484055f,
1.977571f,
1.790545f,
1.429359f,
1.752838f,
1.619342f,
1.534500f,
1.835318f,
1.601044f,
1.613229f,
1.464930f,
1.521288f,
1.676130f,
1.465301f,
1.425334f,
1.439054f,
1.337973f,
1.738067f,
1.677796f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{0.715189f,
0.645894f,
0.963663f,
0.791725f,
0.925597f,
0.832620f,
0.978618f,
0.799159f,
0.639921f,
0.944669f,
0.774234f,
0.617635f,
0.943748f,
0.681820f,
0.697631f,
0.670638f,
0.570197f,
0.988374f,
0.653108f,
0.466311f,
0.656330f,
0.368725f,
0.837945f,
0.976459f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{0.622255f,
0.538144f,
0.764341f,
0.568020f,
0.521559f,
0.313323f,
0.875596f,
0.680389f,
0.300516f,
0.627060f,
0.498313f,
0.401620f,
0.724259f,
0.492787f,
0.474874f,
0.336649f,
0.416445f,
0.509673f,
0.341098f,
0.321343f,
0.308558f,
0.234497f,
0.585346f,
0.513736f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{0.548814f,
0.423655f,
0.437587f,
0.383442f,
0.071036f,
0.020218f,
0.778157f,
0.461479f,
0.118274f,
0.414662f,
0.264556f,
0.018790f,
0.612096f,
0.359508f,
0.060225f,
0.128926f,
0.315428f,
0.102045f,
0.161310f,
0.244426f,
0.110375f,
0.138183f,
0.097101f,
0.096098f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{0.236588f,
0.149100f,
0.376049f,
0.160562f,
0.037349f,
0.001467f,
0.662530f,
0.287855f,
0.010850f,
0.204418f,
0.093432f,
0.006597f,
0.356381f,
0.107125f,
0.028014f,
0.018190f,
0.065416f,
0.044237f,
0.022006f,
0.028870f,
0.011516f,
0.010016f,
0.066801f,
0.043976f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{1.866766f,
1.614432f,
2.293023f,
1.704061f,
1.564677f,
0.939968f,
2.626787f,
2.041167f,
0.901549f,
1.881179f,
1.494940f,
1.204859f,
2.172778f,
1.478360f,
1.424623f,
1.009947f,
1.249336f,
1.529020f,
1.023295f,
0.964028f,
0.925674f,
0.703490f,
1.756039f,
1.541209f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{1.176016f,
0.893560f,
1.915388f,
1.053586f,
1.184450f,
0.701256f,
2.320143f,
1.460843f,
0.444038f,
1.336670f,
0.877501f,
0.704944f,
1.645929f,
0.785122f,
0.934894f,
0.510638f,
0.556905f,
1.179667f,
0.496201f,
0.341346f,
0.468223f,
0.193697f,
1.385610f,
1.182342f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{1.000000f,
2.000000f,
2.000000f,
1.000000f,
1.000000f,
2.000000f,
2.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
2.000000f,
2.000000f,
0.000000f,
0.000000f,
0.000000f,
2.000000f,
1.000000f,
2.000000f,
1.000000f,
2.000000f,
2.000000f,
2.000000f,
1.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{4},
 // keep_dims_
1 ,
},
 // expected dims
{2, 3, 2, 2, 1},
 // expected values
{0.000000f,
1.000000f,
0.000000f,
0.000000f,
2.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
2.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
2.000000f,
0.000000f,
2.000000f,
1.000000f,
2.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{35.911209f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{4.873291f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{3.581049f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{4.815600f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{0.988374f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{0.498767f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{0.018790f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{0.000000f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{35.911209f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{},
 // expected values
{23.748968f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
0 ,
},
 // expected dims
{3, 2, 2, 3},
 // expected values
{0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
})},
  {"ReduceL1",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{35.911209f,
})},
  {"ReduceL2",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{4.873291f,
})},
  {"ReduceLogSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{3.581049f,
})},
  {"ReduceLogSumExp",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{4.815600f,
})},
  {"ReduceMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{0.988374f,
})},
  {"ReduceMean",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{0.498767f,
})},
  {"ReduceMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{0.018790f,
})},
  {"ReduceProd",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{0.000000f,
})},
  {"ReduceSum",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{35.911209f,
})},
  {"ReduceSumSquare",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 1, 1, 1, 1},
 // expected values
{23.748968f,
})},
  {"ArgMax",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
})},
  {"ArgMin",
OpAttributesResult(
    // ReductionAttribute
      {
 // axes_
{
},
 // keep_dims_
1 ,
},
 // expected dims
{1, 3, 2, 2, 3},
 // expected values
{0.000000f,
1.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
0.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
1.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
0.000000f,
1.000000f,
})},
}
};
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
	#pragma optimize ("", on)
#elif defined(__GNUC__)
#if defined(__clang__)
	#pragma clang optimize on
#else
	#pragma GCC pop_options
#endif
#endif
