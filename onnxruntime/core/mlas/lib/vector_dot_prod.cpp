#include "mlasi.h"

#include <iostream>

#pragma warning(disable: 4100)
void 
MLASCALL
MlasVectorDotProduct(
    const float* A,
    const float* B,
    float* C,
    size_t M, size_t N)
{
    std::cerr << "Hi from MlasVectorDotProduct\n";
}
#pragma warning(default: 4100)
