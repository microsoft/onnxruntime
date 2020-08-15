// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "BetaDistribution.h"
#include <iostream>

void unittestBetaDistribution()
{
  const int nrolls=10000;  // number of experiments
  const int nstars=100;    // maximum number of stars to distribute
  const int nintervals=10; // number of intervals

  std::default_random_engine generator;
  BetaDistribution<double> distribution;

  int p[nintervals]={};

  for (int i=0; i<nrolls; ++i) {
    double number = distribution(generator);
    if (number<1.0) ++p[int(nintervals*number)];
  }

  std::cout << "BetaDistribution (0.5, 0.5, 0, 1):" << std::endl;
  std::cout << std::fixed; std::cout.precision(1);

  for (int i=0; i<nintervals; ++i) {
    std::cout << float(i)/nintervals << "-" << float(i+1)/nintervals << ": ";
    std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
  }
}

void unittestGenerateRandomData()
{
  unsigned int seed { static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())};
  auto seq = GenerateRandomData(0.0f, 10000, seed);
  std::map<int, int> count;
  for(auto& v : seq)
  {
    v /= (std::numeric_limits<float>::max() - std::numeric_limits<float>::min());
    v *= 100;
    int index = static_cast<int>(v);
    count[index] = (count[index] <= 0) ? 1: count[index] + 1;
  }
  std::cout << "\n";
  for(auto& v : count)
  {
    std::cout << v.first << ":" << std::string(v.second, '*') << "\n";
  }
}


