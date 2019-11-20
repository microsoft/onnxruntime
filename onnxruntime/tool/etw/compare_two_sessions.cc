// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "eparser.h"
#include "TraceSession.h"

#ifdef _WIN32
#include <tchar.h>
#else
#define TCHAR char
#define _tmain main
#endif

#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#include <thread>
#endif

static const GUID OrtProviderGuid = {0x54d81939, 0x62a0, 0x4dc0, {0xbf, 0x32, 0x3, 0x5e, 0xbd, 0xc7, 0xbc, 0xe9}};

int fetch_data(TCHAR* filename, ProfilingInfo& context) {
  TraceSession session;
  session.AddHandler(OrtProviderGuid, OrtEventHandler, &context);
  session.InitializeEtlFile(filename, nullptr);
  ULONG status = ProcessTrace(&session.traceHandle_, 1, 0, 0);
  if (status != ERROR_SUCCESS && status != ERROR_CANCELLED) {
    std::cout << "OpenTrace failed with " << status << std::endl;
    session.Finalize();
    return -1;
  }
  session.Finalize();
  return 0;
}

template <typename T>
std::pair<double, double> CalcMeanAndStdSquare(const T* input, size_t input_len) {
  T sum = 0;
  T sum_square = 0;
  const size_t N = input_len;
  for (size_t i = 0; i != N; ++i) {
    T t = input[i];
    sum += t;
    sum_square += t * t;
  }
  double mean = ((double)sum) / N;
  double std = (sum_square - N * mean * mean) / (N - 1);
  return std::make_pair(mean, std);
}
// see: "Statistical Distributions", 4th Edition, by Catherine Forbes, Merran Evans, Nicholas Hastings and Brian
// Peacock. Chapter 42: "Student’s t Distribution". I only implemented when v is even.
double TDistributionCDF(int v, double x) {
  assert(v >= 2 && (v & 1) == 0);
  double t = x / (2 * std::sqrt(v + x * x));
  double sum1 = 0;
  double b_j = 1;
  for (int j = 0; j <= (v - 2) / 2; ++j) {
    sum1 += b_j / std::pow(1 + x * x / v, j);
    b_j *= static_cast<double>(2 * j + 1) / (2 * j + 2);
  }
  return 0.5 + t * sum1;
}

struct TTestResult {
  double mean1, mean2;
  double std1, std2;
  double tvalue;
};

template <typename T>
TTestResult CalcTValue(const T* input1, size_t input1_len, const T* input2, size_t input2_len) {
  TTestResult result;
  auto p1 = CalcMeanAndStdSquare(input1, input1_len);
  result.mean1 = p1.first;
  result.std1 = std::sqrt(p1.second);
  auto p2 = CalcMeanAndStdSquare(input2, input2_len);
  result.mean2 = p2.first;
  result.std2 = std::sqrt(p2.second);
  auto diff_mean = p1.first - p2.first;
  size_t n1 = input1_len;
  size_t n2 = input2_len;
  auto sdiff = ((n1 - 1) * p1.second + (n2 - 1) * p2.second) / (n1 + n2 - 2);
  sdiff *= ((double)1) / n1 + ((double)1) / n2;
  result.tvalue = diff_mean / std::sqrt(sdiff);
  return result;
}

int real_main(int argc, TCHAR* argv[]) {
  if (argc < 3) {
    printf("error\n");
    return -1;
  }

  ProfilingInfo context1;
  int ret = fetch_data(argv[1], context1);
  if (ret != 0) return ret;
  ProfilingInfo context2;
  ret = fetch_data(argv[2], context2);
  if (ret != 0) return ret;
  size_t n1 = context1.time_per_run.size();
  size_t n2 = context2.time_per_run.size();
  if (n1 <= 10 || n2 <= 10) {
    printf("samples are too few, please try to gather more\n");
    return -1;
  }
  // ignore the first run
  --n1;
  --n2;
  if (((n1 + n2) & 1) != 0) {
    if (n1 > n2)
      n1--;
    else
      n2--;
  }
  TTestResult tresult = CalcTValue(context1.time_per_run.data() + 1, n1, context2.time_per_run.data() + 1, n2);
  size_t freedom = n1 + n2 - 2;
  double p = TDistributionCDF(static_cast<int>(freedom), std::abs(tresult.tvalue));
  std::cout << "Mean1: " << tresult.mean1 << " std1: " << tresult.std1 << "\n"
            << "Mean2: " << tresult.mean2 << " std2: " << tresult.std2 << "\n"
            << "H0:	Mean1 =  Mean2\n"
            << "H1:	Mean1 != Mean2\n"
            << "Test statistic:  T = " << tresult.tvalue << "\n"
            << "Degrees of Freedom: v = " << freedom << "\n"
            << "Significance level:" << (1 - p) * 2 << ". The lower the more likely to reject H0\n";
  if (p > 0.99995) {
    std::cout << "The two population means are different at the 0.0001 significance level." << std::endl;
    return -1;
  } else {
    std::cout << "They don't have significant statistical difference." << std::endl;
    return 0;
  }
}

int _tmain(int argc, TCHAR* argv[]) {
  int retval = -1;
  try {
    retval = real_main(argc, argv);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  return retval;
}