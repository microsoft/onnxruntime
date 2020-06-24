#include <cooperative_groups.h>

const int SUPPORTED_REDUCTION_DIM [] = {768, 512, 1024, 1536, 2048, 2560}; 

// Custom fused bias add with layer normalization
template <typename T, typename U>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     U epsilon,
                                     int64_t n1,
                                     int64_t n2,
                                     U* invvars,
                                     U* means);

