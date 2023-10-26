#pragma once

template<typename T>
class MLAS_GEMM_POSTPROCESSOR
{
   public:
    virtual void Process(T* C,                  /**< the address of matrix to process */
                         size_t RangeStartM,    /**< the start row index of matrix */
                         size_t RangeStartN,    /**< the start col index of matrix */
                         size_t RangeCountM,    /**< the element count per row to process */
                         size_t RangeCountN,    /**< the element count per col to process */
                         size_t ldc             /**< the leading dimension of matrix */
    ) const = 0;

    virtual ~MLAS_GEMM_POSTPROCESSOR() {}
};
