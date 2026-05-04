#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "mlasi_sve.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

extern "C" {

size_t MLASCALL
MlasGemmU8X8KernelUmmlaZero(const uint8_t* A,
                            const uint8_t* B,
                            int32_t* C,
                            size_t PackedCountK,
                            size_t CountM,
                            size_t CountN,
                            size_t ldc,
                            const int32_t* RowSumVector,
                            const int32_t* ColumnSumVector,
                            const int32_t* ZeroPointB);

size_t MLASCALL
MlasGemmU8X8KernelUmmlaAdd(const uint8_t* A,
                           const uint8_t* B,
                           int32_t* C,
                           size_t PackedCountK,
                           size_t CountM,
                           size_t CountN,
                           size_t ldc,
                           const int32_t* RowSumVector,
                           const int32_t* ColumnSumVector,
                           const int32_t* ZeroPointB);
}

struct MLAS_GEMM_U8X8_KERNEL_UMMLA {
    using PackedAType = uint8_t;
    using PackedBType = uint8_t;
};

size_t Process1RowTest256(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool isZeroPointB)
{
    (void)CountM;
    (void)ldc;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = svwhilelt_b32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = svptrue_b32();  // full accumulator lanes
    svbool_t pg_b8_32 = svptrue_b8(); // loads 32 bytes from A/B
    svuint32_t acc03, acc47;

    const int32_t rowsum = RowSumBuffer[0];
    const svint32_t rowSumVec = svdup_s32(rowsum);

    const uint8_t* A_ptr;
    int32_t* C_ptr = C;

    //------------------------------------------------------------------
    // Loop over N in chunks of 8 columns
    //------------------------------------------------------------------
    for (size_t col = 0; col < CountN; )
    {
        size_t remaining = CountN - col;
        size_t cols_this = std::min<size_t>(remaining, 8);

        A_ptr = A;

        //------------------------------------------------------------------
        // Create ZPB and ColumnSum interleaved groups:
        //   VL=256: 8 lanes → represent 4 output columns at once
        //------------------------------------------------------------------
        // svint32_t acc00 = svdup_s32(0); // for columns 0–3
        // svint32_t acc01 = svdup_s32(0); // for columns 4–7

        svint32_t zpb_0_3, col_0_3, zpb_4_7, col_4_7;
        svint64_t zpb64_0_3, col64_0_3, zpb64_4_7, col64_4_7;
        // --- Columns 0–3 ---
        if (ZeroPointB){
            zpb_0_3 = svld1_s32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = svreinterpret_s64_s32(zpb_0_3);
            zpb_0_3 = svreinterpret_s32_s64(svzip1_s64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = svdup_s32(1);
        col_0_3 = svld1_s32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = svreinterpret_s64_s32(col_0_3);
        col_0_3 = svreinterpret_s32_s64(svzip1_s64(col64_0_3, col64_0_3));
        acc03 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec, zpb_0_3), col_0_3));
        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = svld1_s32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = svreinterpret_s64_s32(zpb_4_7);
                zpb_4_7 = svreinterpret_s32_s64(svzip1_s64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = svdup_s32(1);
            }
            col_4_7 = svld1_s32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = svreinterpret_s64_s32(col_4_7);
            col_4_7 = svreinterpret_s32_s64(svzip1_s64(col64_4_7, col64_4_7));
            // acc47 = svreinterpret_u32_s32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a  = svld1rq_u8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t b0 = svld1_u8(pg_b8_32, B);
            svuint8_t b1 = svld1_u8(pg_b8_32, B + 32);

            acc03 = svmmla_u32(acc03, a, b0);
            if (cols_this > 4)
                acc47 = svmmla_u32(acc47, a, b1);

            A_ptr += 8;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx = svdupq_n_u32(0, 1, 4, 5);
        acc03 = svtbl_u32(acc03, idx);
        if (cols_this > 4)
            acc47 = svtbl_u32(acc47, idx);
        // First 4 columns
        // 
        if(cols_this > 6){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C_ptr, svreinterpret_s32_u32(acc03));
                if(cols_this >= 8){
                    svst1_s32(pg_b32_4, C_ptr + 4, svreinterpret_s32_u32(acc47));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C_ptr + 4, svreinterpret_s32_u32(acc47));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C_ptr);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03));
                svst1_s32(pg_b32_4, C_ptr, sum);

                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = svld1_s32(pg_b32_4, C_ptr + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47));
                    svst1_s32(pg_b32_4, C_ptr + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0,3), C_ptr + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47));
                    svst1_s32(svwhilelt_b32(0,3), C_ptr + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C_ptr, svreinterpret_s32_u32(acc03));
                if(cols_this >= 6){
                    svst1_s32(svwhilelt_b32(0, 2), C_ptr + 4, svreinterpret_s32_u32(acc47));
                }
                else{
                    svst1_s32(svwhilelt_b32(0, 1), C_ptr + 4, svreinterpret_s32_u32(acc47));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C_ptr);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03));
                svst1_s32(pg_b32_4, C_ptr, sum);
                if(cols_this >= 6){
                    prev = svld1_s32(svwhilelt_b32(0, 2), C_ptr + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47));
                    svst1_s32(svwhilelt_b32(0, 2), C_ptr + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0, 1), C_ptr + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47));
                    svst1_s32(svwhilelt_b32(0, 1), C_ptr + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(isZeroPointB){
                if(cols_this >= 4){
                    svst1_s32(pg_b32_4, C_ptr, svreinterpret_s32_u32(acc03));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C_ptr, svreinterpret_s32_u32(acc03));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(pg_b32_4, C_ptr);
                    svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03));
                    svst1_s32(pg_b32_4, C_ptr, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(svwhilelt_b32(0,3), C_ptr);
                    svint32_t sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03));
                    svst1_s32(svwhilelt_b32(0,3), C_ptr, sum);
                }   
            }
        }
        else{
            if(isZeroPointB){
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C_ptr, svreinterpret_s32_u32(acc03));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C_ptr);
                svint32_t sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C_ptr, sum);
            }
        }
        // Last 4 columns

        //------------------------------------------------------------------
        // Advance
        //------------------------------------------------------------------
        C_ptr            += cols_this;
        ColumnSumBuffer  += cols_this;
        col              += cols_this;
        if (ZeroPointB) {
            ZeroPointB += cols_this;
        }
    }
    return 1;
}

size_t Process2RowsTest256(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool isZeroPointB)
{
    (void)CountM;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = svwhilelt_b32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = svptrue_b32();  // full accumulator lanes
    svbool_t pg_b8_32 = svptrue_b8(); // loads 32 bytes from A/B
    svuint32_t acc03, acc47;

    // const int32_t rowsum = RowSumBuffer[0];
    svint32_t rowSumVec, r0, r1;
    r0 = svdup_s32(RowSumBuffer[0]);
    r1 = svdup_s32(RowSumBuffer[1]);
    rowSumVec = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r0), svreinterpret_s64_s32(r1)));
    const uint8_t* A_ptr;
    int32_t *C0, *C1;
    C0 = C;
    C1 = C0 + ldc;

    //------------------------------------------------------------------
    // Loop over N in chunks of 8 columns
    //------------------------------------------------------------------
    for (size_t col = 0; col < CountN; )
    {
        size_t remaining = CountN - col;
        size_t cols_this = std::min<size_t>(remaining, 8);

        A_ptr = A;

        svint32_t zpb_0_3, col_0_3, zpb_4_7, col_4_7;
        svint64_t zpb64_0_3, col64_0_3, zpb64_4_7, col64_4_7;
        // --- Columns 0–3 ---
        if (ZeroPointB){
            zpb_0_3 = svld1_s32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = svreinterpret_s64_s32(zpb_0_3);
            zpb_0_3 = svreinterpret_s32_s64(svzip1_s64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = svdup_s32(1);

        col_0_3 = svld1_s32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = svreinterpret_s64_s32(col_0_3);
        col_0_3 = svreinterpret_s32_s64(svzip1_s64(col64_0_3, col64_0_3));
        acc03 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec, zpb_0_3), col_0_3));
        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = svld1_s32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = svreinterpret_s64_s32(zpb_4_7);
                zpb_4_7 = svreinterpret_s32_s64(svzip1_s64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = svdup_s32(1);
            }
            col_4_7 = svld1_s32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = svreinterpret_s64_s32(col_4_7);
            col_4_7 = svreinterpret_s32_s64(svzip1_s64(col64_4_7, col64_4_7));
            // acc47 = svreinterpret_u32_s32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a  = svld1rq_u8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t b0 = svld1_u8(pg_b8_32, B);
            svuint8_t b1 = svld1_u8(pg_b8_32, B + 32);

            acc03 = svmmla_u32(acc03, a, b0);
            if (cols_this > 4)
                acc47 = svmmla_u32(acc47, a, b1); 

            A_ptr += 16;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx1 = svdupq_n_u32(0, 1, 4, 5);
        const svuint32_t idx2 = svdupq_n_u32(2, 3, 6, 7);
        svuint32_t acc03_c0, acc03_c1, acc47_c0, acc47_c1;
        acc03_c0 = svtbl_u32(acc03, idx1);
        acc03_c1 = svtbl_u32(acc03, idx2);
        if (cols_this > 4){
            acc47_c0 = svtbl_u32(acc47, idx1);
            acc47_c1 = svtbl_u32(acc47, idx2);
        }
        // First 4 columns
        // 
        if(cols_this > 6){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                if(cols_this >= 8){
                    svst1_s32(pg_b32_4, C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(pg_b32_4, C1 + 4, svreinterpret_s32_u32(acc47_c1));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0,3), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C0);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C0, sum);
                prev = svld1_s32(pg_b32_4, C1);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C1, sum);
                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = svld1_s32(pg_b32_4, C0 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(pg_b32_4, C0 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C1 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(pg_b32_4, C1 + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0,3), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0,3), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0,3), C1 + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                if(cols_this >= 6){
                    svst1_s32(svwhilelt_b32(0, 2), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 2), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                }
                else{
                    svst1_s32(svwhilelt_b32(0, 1), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 1), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C0);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C0, sum);
                prev = svld1_s32(pg_b32_4, C1);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C1, sum);
                if(cols_this >= 6){
                    prev = svld1_s32(svwhilelt_b32(0, 2), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 2), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 2), C1 + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0, 1), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 1), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 1), C1 + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(isZeroPointB){
                if(cols_this >= 4){
                    svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C0, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(svwhilelt_b32(0,3), C1, svreinterpret_s32_u32(acc03_c1));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(pg_b32_4, C0);
                    svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(pg_b32_4, C0, sum);
                    prev = svld1_s32(pg_b32_4, C1);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(pg_b32_4, C1, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(svwhilelt_b32(0,3), C0);
                    svint32_t sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(svwhilelt_b32(0,3), C0, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C1);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(svwhilelt_b32(0,3), C1, sum);
                }   
            }
        }
        else{
            if(isZeroPointB){
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1, svreinterpret_s32_u32(acc03_c1));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0);
                svint32_t sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1, sum);
            }
        }
        // Last 4 columns

        //------------------------------------------------------------------
        // Advance
        //------------------------------------------------------------------
        C0            += cols_this;
        C1            += cols_this;
        ColumnSumBuffer  += cols_this;
        col              += cols_this;
        if (ZeroPointB) {
            ZeroPointB += cols_this;
        }
    }
    return 2;
}

size_t Process4RowsTest256(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool isZeroPointB)
{
    (void)CountM;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = svwhilelt_b32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = svptrue_b32();  // full accumulator lanes
    svbool_t pg_b8_32 = svptrue_b8(); // loads 32 bytes from A/B
    svuint32_t acc03_0, acc03_1, acc47_0, acc47_1;

    // const int32_t rowsum = RowSumBuffer[0];
    svint32_t rowSumVec01, rowSumVec23, r0, r1, r2, r3;
    r0 = svdup_s32(RowSumBuffer[0]);
    r1 = svdup_s32(RowSumBuffer[1]);
    rowSumVec01 = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r0), svreinterpret_s64_s32(r1)));
    r2 = svdup_s32(RowSumBuffer[2]);
    r3 = svdup_s32(RowSumBuffer[3]);
    rowSumVec23 = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r2), svreinterpret_s64_s32(r3)));
    const uint8_t* A_ptr;
    int32_t *C0, *C1, *C2, *C3;
    C0 = C;
    C1 = C0 + ldc;
    C2 = C1 + ldc;
    C3 = C2 + ldc;

    //------------------------------------------------------------------
    // Loop over N in chunks of 8 columns
    //------------------------------------------------------------------
    for (size_t col = 0; col < CountN; )
    {
        size_t remaining = CountN - col;
        size_t cols_this = std::min<size_t>(remaining, 8);

        A_ptr = A;

        svint32_t zpb_0_3, col_0_3, zpb_4_7, col_4_7;
        svint64_t zpb64_0_3, col64_0_3, zpb64_4_7, col64_4_7;
        // --- Columns 0–3 ---
        if (ZeroPointB){
            zpb_0_3 = svld1_s32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = svreinterpret_s64_s32(zpb_0_3);
            zpb_0_3 = svreinterpret_s32_s64(svzip1_s64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = svdup_s32(1);
        col_0_3 = svld1_s32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = svreinterpret_s64_s32(col_0_3);
        col_0_3 = svreinterpret_s32_s64(svzip1_s64(col64_0_3, col64_0_3));

        acc03_0 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec01, zpb_0_3), col_0_3));
        acc03_1 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec23, zpb_0_3), col_0_3));

        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = svld1_s32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = svreinterpret_s64_s32(zpb_4_7);
                zpb_4_7 = svreinterpret_s32_s64(svzip1_s64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = svdup_s32(1);
            }
            col_4_7 = svld1_s32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = svreinterpret_s64_s32(col_4_7);
            col_4_7 = svreinterpret_s32_s64(svzip1_s64(col64_4_7, col64_4_7));
            // acc47 = svreinterpret_u32_s32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47_0 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec01, zpb_4_7), col_4_7));
            acc47_1 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec23, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a0  = svld1rq_u8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t a1  = svld1rq_u8(pg_b8_32, A_ptr + 16); //load and replicate
            svuint8_t b0 = svld1_u8(pg_b8_32, B);
            svuint8_t b1 = svld1_u8(pg_b8_32, B + 32);

            acc03_0 = svmmla_u32(acc03_0, a0, b0);
            acc03_1 = svmmla_u32(acc03_1, a1, b0);
            if (cols_this > 4){
                acc47_0 = svmmla_u32(acc47_0, a0, b1);
                acc47_1 = svmmla_u32(acc47_1, a1, b1);
            }

            A_ptr += 32;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx1 = svdupq_n_u32(0, 1, 4, 5);
        const svuint32_t idx2 = svdupq_n_u32(2, 3, 6, 7);
        svuint32_t acc03_c0, acc03_c1, acc47_c0, acc47_c1, acc03_c2, acc03_c3, acc47_c2, acc47_c3;
        acc03_c0 = svtbl_u32(acc03_0, idx1);
        acc03_c1 = svtbl_u32(acc03_0, idx2);
        acc03_c2 = svtbl_u32(acc03_1, idx1);
        acc03_c3 = svtbl_u32(acc03_1, idx2);
        if (cols_this > 4){
            acc47_c0 = svtbl_u32(acc47_0, idx1);
            acc47_c1 = svtbl_u32(acc47_0, idx2);
            acc47_c2 = svtbl_u32(acc47_1, idx1);
            acc47_c3 = svtbl_u32(acc47_1, idx2);
        }
        // First 4 columns
        // 
        if(cols_this > 6){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C2, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C3, svreinterpret_s32_u32(acc03_c3));
                if(cols_this >= 8){
                    svst1_s32(pg_b32_4, C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(pg_b32_4, C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(pg_b32_4, C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(pg_b32_4, C3 + 4, svreinterpret_s32_u32(acc47_c3));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0,3), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0,3), C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0,3), C3 + 4, svreinterpret_s32_u32(acc47_c3));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C0);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C0, sum);
                prev = svld1_s32(pg_b32_4, C1);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C1, sum);
                prev = svld1_s32(pg_b32_4, C2);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C2, sum);
                prev = svld1_s32(pg_b32_4, C3);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(pg_b32_4, C3, sum);
                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = svld1_s32(pg_b32_4, C0 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(pg_b32_4, C0 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C1 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(pg_b32_4, C1 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C2 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(pg_b32_4, C2 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C3 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(pg_b32_4, C3 + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0,3), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0,3), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0,3), C1 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C2 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0,3), C2 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C3 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0,3), C3 + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C2, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C3, svreinterpret_s32_u32(acc03_c3));
                if(cols_this >= 6){
                    svst1_s32(svwhilelt_b32(0, 2), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 2), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 2), C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 2), C3 + 4, svreinterpret_s32_u32(acc47_c3));
                }
                else{
                    svst1_s32(svwhilelt_b32(0, 1), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 1), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 1), C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 1), C3 + 4, svreinterpret_s32_u32(acc47_c3));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C0);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C0, sum);
                prev = svld1_s32(pg_b32_4, C1);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C1, sum);
                prev = svld1_s32(pg_b32_4, C2);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C2, sum);
                prev = svld1_s32(pg_b32_4, C3);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(pg_b32_4, C3, sum);
                if(cols_this >= 6){
                    prev = svld1_s32(svwhilelt_b32(0, 2), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 2), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 2), C1 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C2 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 2), C2 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C3 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0, 2), C3 + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0, 1), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 1), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 1), C1 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C2 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 1), C2 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C3 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0, 1), C3 + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(isZeroPointB){
                if(cols_this >= 4){
                    svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(pg_b32_4, C2, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(pg_b32_4, C3, svreinterpret_s32_u32(acc03_c3));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C0, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(svwhilelt_b32(0,3), C1, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(svwhilelt_b32(0,3), C2, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(svwhilelt_b32(0,3), C3, svreinterpret_s32_u32(acc03_c3));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(pg_b32_4, C0);
                    svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(pg_b32_4, C0, sum);
                    prev = svld1_s32(pg_b32_4, C1);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(pg_b32_4, C1, sum);
                    prev = svld1_s32(pg_b32_4, C2);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(pg_b32_4, C2, sum);
                    prev = svld1_s32(pg_b32_4, C3);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c3));
                    svst1_s32(pg_b32_4, C3, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(svwhilelt_b32(0,3), C0);
                    svint32_t sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(svwhilelt_b32(0,3), C0, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C1);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(svwhilelt_b32(0,3), C1, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C2);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(svwhilelt_b32(0,3), C2, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C3);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c3));
                    svst1_s32(svwhilelt_b32(0,3), C3, sum);
                }   
            }
        }
        else{
            if(isZeroPointB){
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C2, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C3, svreinterpret_s32_u32(acc03_c3));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0);
                svint32_t sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C2);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C2, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C3);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C3, sum);
            }
        }
        // Last 4 columns

        //------------------------------------------------------------------
        // Advance
        //------------------------------------------------------------------
        C0            += cols_this;
        C1            += cols_this;
        C2            += cols_this;
        C3            += cols_this;
        ColumnSumBuffer  += cols_this;
        col              += cols_this;
        if (ZeroPointB) {
            ZeroPointB += cols_this;
        }
    }
    return 4;
}

size_t Process8RowsTest256(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool isZeroPointB)
{
    (void)CountM;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = svwhilelt_b32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = svptrue_b32();  // full accumulator lanes
    svbool_t pg_b8_32 = svptrue_b8(); // loads 32 bytes from A/B
    svuint32_t acc03_0, acc03_1, acc03_2, acc03_3, acc47_0, acc47_1, acc47_2, acc47_3;

    // const int32_t rowsum = RowSumBuffer[0];
    svint32_t rowSumVec01, rowSumVec23, rowSumVec45, rowSumVec67, r0, r1, r2, r3, r4, r5, r6, r7;
    r0 = svdup_s32(RowSumBuffer[0]);
    r1 = svdup_s32(RowSumBuffer[1]);
    rowSumVec01 = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r0), svreinterpret_s64_s32(r1)));
    r2 = svdup_s32(RowSumBuffer[2]);
    r3 = svdup_s32(RowSumBuffer[3]);
    rowSumVec23 = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r2), svreinterpret_s64_s32(r3)));
    r4 = svdup_s32(RowSumBuffer[4]);
    r5 = svdup_s32(RowSumBuffer[5]);
    rowSumVec45 = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r4), svreinterpret_s64_s32(r5)));
    r6 = svdup_s32(RowSumBuffer[6]);
    r7 = svdup_s32(RowSumBuffer[7]);
    rowSumVec67 = svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(r6), svreinterpret_s64_s32(r7)));
    const uint8_t* A_ptr;
    int32_t *C0, *C1, *C2, *C3, *C4, *C5, *C6, *C7;
    C0 = C;
    C1 = C0 + ldc;
    C2 = C1 + ldc;
    C3 = C2 + ldc;
    C4 = C3 + ldc;
    C5 = C4 + ldc;
    C6 = C5 + ldc;
    C7 = C6 + ldc;
    //------------------------------------------------------------------
    // Loop over N in chunks of 8 columns
    //------------------------------------------------------------------
    for (size_t col = 0; col < CountN; )
    {
        size_t remaining = CountN - col;
        size_t cols_this = std::min<size_t>(remaining, 8);

        A_ptr = A;

        svint32_t zpb_0_3, col_0_3, zpb_4_7, col_4_7;
        svint64_t zpb64_0_3, col64_0_3, zpb64_4_7, col64_4_7;
        // --- Columns 0–3 ---
        if (ZeroPointB){
            zpb_0_3 = svld1_s32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = svreinterpret_s64_s32(zpb_0_3);
            zpb_0_3 = svreinterpret_s32_s64(svzip1_s64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = svdup_s32(1);
        col_0_3 = svld1_s32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = svreinterpret_s64_s32(col_0_3);
        col_0_3 = svreinterpret_s32_s64(svzip1_s64(col64_0_3, col64_0_3));

        acc03_0 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec01, zpb_0_3), col_0_3));
        acc03_1 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec23, zpb_0_3), col_0_3));
        acc03_2 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec45, zpb_0_3), col_0_3));
        acc03_3 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec67, zpb_0_3), col_0_3));
        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = svld1_s32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = svreinterpret_s64_s32(zpb_4_7);
                zpb_4_7 = svreinterpret_s32_s64(svzip1_s64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = svdup_s32(1);
            }
            col_4_7 = svld1_s32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = svreinterpret_s64_s32(col_4_7);
            col_4_7 = svreinterpret_s32_s64(svzip1_s64(col64_4_7, col64_4_7));
            // acc47 = svreinterpret_u32_s32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47_0 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec01, zpb_4_7), col_4_7));
            acc47_1 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec23, zpb_4_7), col_4_7));
            acc47_2 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec45, zpb_4_7), col_4_7));
            acc47_3 = svreinterpret_u32_s32(svadd_s32_x(pg_b32_8, svmul_s32_x(pg_b32_8, rowSumVec67, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a0  = svld1rq_u8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t a1  = svld1rq_u8(pg_b8_32, A_ptr + 16); //load and replicate
            svuint8_t a2  = svld1rq_u8(pg_b8_32, A_ptr + 32); //load and replicate
            svuint8_t a3  = svld1rq_u8(pg_b8_32, A_ptr + 48); //load and replicate
            svuint8_t b0 = svld1_u8(pg_b8_32, B);
            svuint8_t b1 = svld1_u8(pg_b8_32, B + 32);

            acc03_0 = svmmla_u32(acc03_0, a0, b0);
            acc03_1 = svmmla_u32(acc03_1, a1, b0);
            acc03_2 = svmmla_u32(acc03_2, a2, b0);
            acc03_3 = svmmla_u32(acc03_3, a3, b0);
            if (cols_this > 4){
                acc47_0 = svmmla_u32(acc47_0, a0, b1);
                acc47_1 = svmmla_u32(acc47_1, a1, b1);
                acc47_2 = svmmla_u32(acc47_2, a2, b1);
                acc47_3 = svmmla_u32(acc47_3, a3, b1);
            }

            A_ptr += 64;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx1 = svdupq_n_u32(0, 1, 4, 5);
        const svuint32_t idx2 = svdupq_n_u32(2, 3, 6, 7);
        svuint32_t acc03_c0, acc03_c1, acc47_c0, acc47_c1, acc03_c2, acc03_c3, acc47_c2, acc47_c3, acc03_c4, acc03_c5, acc47_c4, acc47_c5, acc03_c6, acc03_c7, acc47_c6, acc47_c7;
        acc03_c0 = svtbl_u32(acc03_0, idx1);
        acc03_c1 = svtbl_u32(acc03_0, idx2);
        acc03_c2 = svtbl_u32(acc03_1, idx1);
        acc03_c3 = svtbl_u32(acc03_1, idx2);
        acc03_c4 = svtbl_u32(acc03_2, idx1);
        acc03_c5 = svtbl_u32(acc03_2, idx2);
        acc03_c6 = svtbl_u32(acc03_3, idx1);
        acc03_c7 = svtbl_u32(acc03_3, idx2);
        if (cols_this > 4){
            acc47_c0 = svtbl_u32(acc47_0, idx1);
            acc47_c1 = svtbl_u32(acc47_0, idx2);
            acc47_c2 = svtbl_u32(acc47_1, idx1);
            acc47_c3 = svtbl_u32(acc47_1, idx2);
            acc47_c4 = svtbl_u32(acc47_2, idx1);
            acc47_c5 = svtbl_u32(acc47_2, idx2);
            acc47_c6 = svtbl_u32(acc47_3, idx1);
            acc47_c7 = svtbl_u32(acc47_3, idx2);
        }
        // First 4 columns
        // 
        if(cols_this > 6){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C2, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C3, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(pg_b32_4, C4, svreinterpret_s32_u32(acc03_c4));
                svst1_s32(pg_b32_4, C5, svreinterpret_s32_u32(acc03_c5));
                svst1_s32(pg_b32_4, C6, svreinterpret_s32_u32(acc03_c6));
                svst1_s32(pg_b32_4, C7, svreinterpret_s32_u32(acc03_c7));
                if(cols_this >= 8){
                    svst1_s32(pg_b32_4, C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(pg_b32_4, C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(pg_b32_4, C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(pg_b32_4, C3 + 4, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(pg_b32_4, C4 + 4, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(pg_b32_4, C5 + 4, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(pg_b32_4, C6 + 4, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(pg_b32_4, C7 + 4, svreinterpret_s32_u32(acc47_c7));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0,3), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0,3), C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0,3), C3 + 4, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0,3), C4 + 4, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(svwhilelt_b32(0,3), C5 + 4, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(svwhilelt_b32(0,3), C6 + 4, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(svwhilelt_b32(0,3), C7 + 4, svreinterpret_s32_u32(acc47_c7));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C0);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C0, sum);
                prev = svld1_s32(pg_b32_4, C1);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C1, sum);
                prev = svld1_s32(pg_b32_4, C2);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C2, sum);
                prev = svld1_s32(pg_b32_4, C3);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(pg_b32_4, C3, sum);
                prev = svld1_s32(pg_b32_4, C4);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c4));
                svst1_s32(pg_b32_4, C4, sum);
                prev = svld1_s32(pg_b32_4, C5);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c5));
                svst1_s32(pg_b32_4, C5, sum);
                prev = svld1_s32(pg_b32_4, C6);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c6));
                svst1_s32(pg_b32_4, C6, sum);
                prev = svld1_s32(pg_b32_4, C7);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c7));
                svst1_s32(pg_b32_4, C7, sum);
                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = svld1_s32(pg_b32_4, C0 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(pg_b32_4, C0 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C1 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(pg_b32_4, C1 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C2 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(pg_b32_4, C2 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C3 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(pg_b32_4, C3 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C4 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(pg_b32_4, C4 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C5 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(pg_b32_4, C5 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C6 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(pg_b32_4, C6 + 4, sum);
                    prev = svld1_s32(pg_b32_4, C7 + 4);
                    sum  = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc47_c7));
                    svst1_s32(pg_b32_4, C7 + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0,3), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0,3), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0,3), C1 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C2 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0,3), C2 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C3 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0,3), C3 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C4 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(svwhilelt_b32(0,3), C4 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C5 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(svwhilelt_b32(0,3), C5 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C6 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(svwhilelt_b32(0,3), C6 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C7 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc47_c7));
                    svst1_s32(svwhilelt_b32(0,3), C7 + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(isZeroPointB){
                svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C2, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C3, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(pg_b32_4, C4, svreinterpret_s32_u32(acc03_c4));
                svst1_s32(pg_b32_4, C5, svreinterpret_s32_u32(acc03_c5));
                svst1_s32(pg_b32_4, C6, svreinterpret_s32_u32(acc03_c6));
                svst1_s32(pg_b32_4, C7, svreinterpret_s32_u32(acc03_c7));
                if(cols_this >= 6){
                    svst1_s32(svwhilelt_b32(0, 2), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 2), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 2), C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 2), C3 + 4, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0, 2), C4 + 4, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(svwhilelt_b32(0, 2), C5 + 4, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(svwhilelt_b32(0, 2), C6 + 4, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(svwhilelt_b32(0, 2), C7 + 4, svreinterpret_s32_u32(acc47_c7));
                }
                else{
                    svst1_s32(svwhilelt_b32(0, 1), C0 + 4, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 1), C1 + 4, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 1), C2 + 4, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 1), C3 + 4, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0, 1), C4 + 4, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(svwhilelt_b32(0, 1), C5 + 4, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(svwhilelt_b32(0, 1), C6 + 4, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(svwhilelt_b32(0, 1), C7 + 4, svreinterpret_s32_u32(acc47_c7));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(pg_b32_4, C0);
                svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(pg_b32_4, C0, sum);
                prev = svld1_s32(pg_b32_4, C1);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(pg_b32_4, C1, sum);
                prev = svld1_s32(pg_b32_4, C2);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(pg_b32_4, C2, sum);
                prev = svld1_s32(pg_b32_4, C3);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(pg_b32_4, C3, sum);
                prev = svld1_s32(pg_b32_4, C4);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c4));
                svst1_s32(pg_b32_4, C4, sum);
                prev = svld1_s32(pg_b32_4, C5);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c5));
                svst1_s32(pg_b32_4, C5, sum);
                prev = svld1_s32(pg_b32_4, C6);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c6));
                svst1_s32(pg_b32_4, C6, sum);
                prev = svld1_s32(pg_b32_4, C7);
                sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c7));
                svst1_s32(pg_b32_4, C7, sum);
                // Row 1 (C1)
                if(cols_this >= 6){
                    prev = svld1_s32(svwhilelt_b32(0, 2), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 2), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 2), C1 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C2 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 2), C2 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C3 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0, 2), C3 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C4 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(svwhilelt_b32(0, 2), C4 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C5 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(svwhilelt_b32(0, 2), C5 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C6 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(svwhilelt_b32(0, 2), C6 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 2), C7 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 2), prev, svreinterpret_s32_u32(acc47_c7));
                    svst1_s32(svwhilelt_b32(0, 2), C7 + 4, sum);
                }
                else{
                    prev = svld1_s32(svwhilelt_b32(0, 1), C0 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c0));
                    svst1_s32(svwhilelt_b32(0, 1), C0 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C1 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c1));
                    svst1_s32(svwhilelt_b32(0, 1), C1 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C2 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c2));
                    svst1_s32(svwhilelt_b32(0, 1), C2 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C3 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c3));
                    svst1_s32(svwhilelt_b32(0, 1), C3 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C4 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c4));
                    svst1_s32(svwhilelt_b32(0, 1), C4 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C5 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c5));
                    svst1_s32(svwhilelt_b32(0, 1), C5 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C6 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c6));
                    svst1_s32(svwhilelt_b32(0, 1), C6 + 4, sum);
                    prev = svld1_s32(svwhilelt_b32(0, 1), C7 + 4);
                    sum  = svadd_s32_x(svwhilelt_b32(0, 1), prev, svreinterpret_s32_u32(acc47_c7));
                    svst1_s32(svwhilelt_b32(0, 1), C7 + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(isZeroPointB){
                if(cols_this >= 4){
                    svst1_s32(pg_b32_4, C0, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(pg_b32_4, C1, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(pg_b32_4, C2, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(pg_b32_4, C3, svreinterpret_s32_u32(acc03_c3));
                    svst1_s32(pg_b32_4, C4, svreinterpret_s32_u32(acc03_c4));
                    svst1_s32(pg_b32_4, C5, svreinterpret_s32_u32(acc03_c5));
                    svst1_s32(pg_b32_4, C6, svreinterpret_s32_u32(acc03_c6));
                    svst1_s32(pg_b32_4, C7, svreinterpret_s32_u32(acc03_c7));
                }
                else{
                    svst1_s32(svwhilelt_b32(0,3), C0, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(svwhilelt_b32(0,3), C1, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(svwhilelt_b32(0,3), C2, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(svwhilelt_b32(0,3), C3, svreinterpret_s32_u32(acc03_c3));
                    svst1_s32(svwhilelt_b32(0,3), C4, svreinterpret_s32_u32(acc03_c4));
                    svst1_s32(svwhilelt_b32(0,3), C5, svreinterpret_s32_u32(acc03_c5));
                    svst1_s32(svwhilelt_b32(0,3), C6, svreinterpret_s32_u32(acc03_c6));
                    svst1_s32(svwhilelt_b32(0,3), C7, svreinterpret_s32_u32(acc03_c7));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(pg_b32_4, C0);
                    svint32_t sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(pg_b32_4, C0, sum);
                    prev = svld1_s32(pg_b32_4, C1);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(pg_b32_4, C1, sum);
                    prev = svld1_s32(pg_b32_4, C2);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(pg_b32_4, C2, sum);
                    prev = svld1_s32(pg_b32_4, C3);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c3));
                    svst1_s32(pg_b32_4, C3, sum);
                    prev = svld1_s32(pg_b32_4, C4);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c4));
                    svst1_s32(pg_b32_4, C4, sum);
                    prev = svld1_s32(pg_b32_4, C5);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c5));
                    svst1_s32(pg_b32_4, C5, sum);
                    prev = svld1_s32(pg_b32_4, C6);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c6));
                    svst1_s32(pg_b32_4, C6, sum);
                    prev = svld1_s32(pg_b32_4, C7);
                    sum = svadd_s32_x(pg_b32_4, prev, svreinterpret_s32_u32(acc03_c7));
                    svst1_s32(pg_b32_4, C7, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = svld1_s32(svwhilelt_b32(0,3), C0);
                    svint32_t sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c0));
                    svst1_s32(svwhilelt_b32(0,3), C0, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C1);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c1));
                    svst1_s32(svwhilelt_b32(0,3), C1, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C2);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c2));
                    svst1_s32(svwhilelt_b32(0,3), C2, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C3);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c3));
                    svst1_s32(svwhilelt_b32(0,3), C3, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C4);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c4));
                    svst1_s32(svwhilelt_b32(0,3), C4, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C5);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c5));
                    svst1_s32(svwhilelt_b32(0,3), C5, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C6);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c6));
                    svst1_s32(svwhilelt_b32(0,3), C6, sum);
                    prev = svld1_s32(svwhilelt_b32(0,3), C7);
                    sum = svadd_s32_x(svwhilelt_b32(0,3), prev, svreinterpret_s32_u32(acc03_c7));
                    svst1_s32(svwhilelt_b32(0,3), C7, sum);
                }   
            }
        }
        else{
            if(isZeroPointB){
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C2, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C3, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C4, svreinterpret_s32_u32(acc03_c4));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C5, svreinterpret_s32_u32(acc03_c5));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C6, svreinterpret_s32_u32(acc03_c6));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C7, svreinterpret_s32_u32(acc03_c7));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0);
                svint32_t sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c0));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C0, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c1));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C1, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C2);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c2));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C2, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C3);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c3));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C3, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C4);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c4));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C4, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C5);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c5));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C5, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C6);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c6));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C6, sum);
                prev = svld1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C7);
                sum = svadd_s32_x(svwhilelt_b32(0, std::min(int(cols_this), 2)), prev, svreinterpret_s32_u32(acc03_c7));
                svst1_s32(svwhilelt_b32(0, std::min(int(cols_this), 2)), C7, sum);
            }
        }
        // Last 4 columns

        //------------------------------------------------------------------
        // Advance
        //------------------------------------------------------------------
        C0            += cols_this;
        C1            += cols_this;
        C2            += cols_this;
        C3            += cols_this;
        C4            += cols_this;
        C5            += cols_this;
        C6            += cols_this;
        C7            += cols_this;
        ColumnSumBuffer  += cols_this;
        col              += cols_this;
        if (ZeroPointB) {
            ZeroPointB += cols_this;
        }
    }
    return 8;
}

size_t MlasSveQgemmU8X8KernelUmmlaAdd(const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* A,
                                const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* B,
                                int32_t* C,
                                size_t PackedCountK,
                                size_t CountM,
                                size_t CountN,
                                size_t ldc,
                                const int32_t* RowSumBuffer,
                                const int32_t* ColumnSumBuffer,
                                const int32_t* ZeroPointB
                              )
{
    bool isZeroPointB = false;
     if (svcntb() >= 32) {
        if (CountM >= 8) {
                return Process8RowsTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                        RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
        } 
        else if (CountM >= 4) {
            return Process4RowsTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                    RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
        } else if (CountM >= 2) {
            return Process2RowsTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                    RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
        } else if (CountM >= 1) {
            return Process1RowTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
        }
     }
     else{
        MlasGemmU8X8KernelUmmlaAdd(A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer, ZeroPointB);
     };
    return 0; // Unsupported CountM
}

size_t MlasSveQgemmU8X8KernelUmmlaZero(const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* A,
                                const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* B,
                                int32_t* C,
                                size_t PackedCountK,
                                size_t CountM,
                                size_t CountN,
                                size_t ldc,
                                const int32_t* RowSumBuffer,
                                const int32_t* ColumnSumBuffer,
                                const int32_t* ZeroPointB
                              )
{
    bool isZeroPointB = true;
    if(svcntb() >= 32){
        if (CountM >= 8) {
            return Process8RowsTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                    RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
        }
        else if (CountM >= 4) {
            return Process4RowsTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                    RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
            
        } else if (CountM >= 2) {
            return Process2RowsTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                    RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);

        } else if (CountM >= 1) {
            return Process1RowTest256(A, B, C, PackedCountK, CountM, CountN, ldc,
                                RowSumBuffer, ColumnSumBuffer, ZeroPointB, isZeroPointB);
        }
    }
    else{
        MlasGemmU8X8KernelUmmlaZero(A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer, ZeroPointB);
    }
    return 0; // Unsupported CountM
}
#pragma GCC diagnostic pop