#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {
    
__inline__ __device__
float warpReduceSum(float val, int warpSize=32) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
    }
    return val;
}

__inline__ __device__
float warpReduceMax(float val, int warpSize = 32) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float incoming = __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
        if (val < incoming) val = incoming;
    }
    return val;
}

template <int blockSize>
__inline__ __device__
float blockReduceSum(float val) {
    constexpr int warpSize = 32;

    static __shared__ float shared[blockSize/warpSize]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    //read from shared memory only if that warp existed
    for (int iter_count = blockSize; iter_count >= warpSize; ) {
        val = warpReduceSum(val, warpSize);
        if (lane==0) shared[wid]=val; // Write reduced value to shared memory
        __syncthreads();              // Wait for all partial reductions
        
        iter_count = (iter_count + warpSize - 1) / warpSize;
        val = (threadIdx.x < iter_count) ? shared[threadIdx.x] : 0;
    }

    if (wid==0) {
        val = warpReduceSum(val, warpSize); //Final reduce within first warp
    }

    return val;
}

}
}
}
