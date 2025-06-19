#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    int n_devices = 0;
    int rc = cudaGetDeviceCount(&n_devices);
    if (rc != cudaSuccess)
    {
        cudaError_t error = cudaGetLastError();
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return rc;
    }

    std::vector<std::pair<int, int>> arch(n_devices);
    for (int cd = 0; cd < n_devices; ++cd)
    {
        cudaDeviceProp dev;
        int rc = cudaGetDeviceProperties(&dev, cd);
        if (rc != cudaSuccess)
        {
            cudaError_t error = cudaGetLastError();
            std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return rc;
        }
        else
        {
            arch[cd] = {dev.major, dev.minor};
        }
    }

    std::pair<int, int> best_cc = *std::max_element(begin(arch), end(arch));
    std::cout << best_cc.first << best_cc.second;

    return 0;
}
