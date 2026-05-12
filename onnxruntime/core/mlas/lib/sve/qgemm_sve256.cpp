#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mlasi_sve.h"
#include "mlasi_sve_i8.h"

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

template<bool HasZeroPointB>
MLAS_FORCEINLINE size_t Process1Row(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    (void)CountM;
    (void)ldc;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = MlasSveWhileLtB32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = MlasSvePtrueB32();  // full accumulator lanes
    svbool_t pg_b8_32 = MlasSvePtrueB8(); // loads 32 bytes from A/B
    svuint32_t acc03, acc47;

    const int32_t rowsum = RowSumBuffer[0];
    const svint32_t rowSumVec = MlasSveBroadcastInt32(rowsum);

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
        // svint32_t acc00 = MlasSveBroadcastInt32(0); // for columns 0–3
        // svint32_t acc01 = MlasSveBroadcastInt32(0); // for columns 4–7

        svint32_t zpb_0_3, col_0_3, zpb_4_7, col_4_7;
        svint64_t zpb64_0_3, col64_0_3, zpb64_4_7, col64_4_7;
        // --- Columns 0–3 ---
        if (ZeroPointB){
            zpb_0_3 = MlasSveLoadInt32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = MlasSveReinterpretS64FromS32(zpb_0_3);
            zpb_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = MlasSveBroadcastInt32(1);
        col_0_3 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = MlasSveReinterpretS64FromS32(col_0_3);
        col_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_0_3, col64_0_3));
        acc03 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec, zpb_0_3), col_0_3));
        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = MlasSveLoadInt32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = MlasSveReinterpretS64FromS32(zpb_4_7);
                zpb_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = MlasSveBroadcastInt32(1);
            }
            col_4_7 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = MlasSveReinterpretS64FromS32(col_4_7);
            col_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_4_7, col64_4_7));
            // acc47 = MlasSveReinterpretU32FromS32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t b0 = MlasSveLoadU8(pg_b8_32, B);
            svuint8_t b1 = MlasSveLoadU8(pg_b8_32, B + 32);

            acc03 = MlasSveMatMulAddU32(acc03, a, b0);
            if (cols_this > 4)
                acc47 = MlasSveMatMulAddU32(acc47, a, b1);

            A_ptr += 8;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx = MlasSveDupqU32(0, 1, 4, 5);
        acc03 = MlasSveTblU32(acc03, idx);
        if (cols_this > 4)
            acc47 = MlasSveTblU32(acc47, idx);
        // First 4 columns
        // 
        if(cols_this > 6){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C_ptr, MlasSveReinterpretS32FromU32(acc03));
                if(cols_this >= 8){
                    MlasSveStoreInt32(pg_b32_4, C_ptr + 4, MlasSveReinterpretS32FromU32(acc47));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C_ptr + 4, MlasSveReinterpretS32FromU32(acc47));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C_ptr);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03));
                MlasSveStoreInt32(pg_b32_4, C_ptr, sum);

                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = MlasSveLoadInt32(pg_b32_4, C_ptr + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47));
                    MlasSveStoreInt32(pg_b32_4, C_ptr + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C_ptr + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C_ptr + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C_ptr, MlasSveReinterpretS32FromU32(acc03));
                if(cols_this >= 6){
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C_ptr + 4, MlasSveReinterpretS32FromU32(acc47));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C_ptr + 4, MlasSveReinterpretS32FromU32(acc47));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C_ptr);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03));
                MlasSveStoreInt32(pg_b32_4, C_ptr, sum);
                if(cols_this >= 6){
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C_ptr + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C_ptr + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C_ptr + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C_ptr + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(HasZeroPointB){
                if(cols_this >= 4){
                    MlasSveStoreInt32(pg_b32_4, C_ptr, MlasSveReinterpretS32FromU32(acc03));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C_ptr, MlasSveReinterpretS32FromU32(acc03));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(pg_b32_4, C_ptr);
                    svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03));
                    MlasSveStoreInt32(pg_b32_4, C_ptr, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C_ptr);
                    svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C_ptr, sum);
                }   
            }
        }
        else{
            if(HasZeroPointB){
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C_ptr, MlasSveReinterpretS32FromU32(acc03));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C_ptr);
                svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C_ptr, sum);
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

template<bool HasZeroPointB>
MLAS_FORCEINLINE size_t Process2Rows(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    (void)CountM;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = MlasSveWhileLtB32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = MlasSvePtrueB32();  // full accumulator lanes
    svbool_t pg_b8_32 = MlasSvePtrueB8(); // loads 32 bytes from A/B
    svuint32_t acc03, acc47;

    // const int32_t rowsum = RowSumBuffer[0];
    svint32_t rowSumVec, r0, r1;
    r0 = MlasSveBroadcastInt32(RowSumBuffer[0]);
    r1 = MlasSveBroadcastInt32(RowSumBuffer[1]);
    rowSumVec = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r0), MlasSveReinterpretS64FromS32(r1)));
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
            zpb_0_3 = MlasSveLoadInt32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = MlasSveReinterpretS64FromS32(zpb_0_3);
            zpb_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = MlasSveBroadcastInt32(1);

        col_0_3 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = MlasSveReinterpretS64FromS32(col_0_3);
        col_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_0_3, col64_0_3));
        acc03 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec, zpb_0_3), col_0_3));
        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = MlasSveLoadInt32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = MlasSveReinterpretS64FromS32(zpb_4_7);
                zpb_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = MlasSveBroadcastInt32(1);
            }
            col_4_7 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = MlasSveReinterpretS64FromS32(col_4_7);
            col_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_4_7, col64_4_7));
            // acc47 = MlasSveReinterpretU32FromS32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t b0 = MlasSveLoadU8(pg_b8_32, B);
            svuint8_t b1 = MlasSveLoadU8(pg_b8_32, B + 32);

            acc03 = MlasSveMatMulAddU32(acc03, a, b0);
            if (cols_this > 4)
                acc47 = MlasSveMatMulAddU32(acc47, a, b1); 

            A_ptr += 16;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx1 = MlasSveDupqU32(0, 1, 4, 5);
        const svuint32_t idx2 = MlasSveDupqU32(2, 3, 6, 7);
        svuint32_t acc03_c0, acc03_c1, acc47_c0, acc47_c1;
        acc03_c0 = MlasSveTblU32(acc03, idx1);
        acc03_c1 = MlasSveTblU32(acc03, idx2);
        if (cols_this > 4){
            acc47_c0 = MlasSveTblU32(acc47, idx1);
            acc47_c1 = MlasSveTblU32(acc47, idx2);
        }
        // First 4 columns
        // 
        if(cols_this > 6){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                if(cols_this >= 8){
                    MlasSveStoreInt32(pg_b32_4, C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(pg_b32_4, C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C0, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C1);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C1, sum);
                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = MlasSveLoadInt32(pg_b32_4, C0 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(pg_b32_4, C0 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C1 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(pg_b32_4, C1 + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1 + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                if(cols_this >= 6){
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C0, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C1);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C1, sum);
                if(cols_this >= 6){
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C1 + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C1 + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(HasZeroPointB){
                if(cols_this >= 4){
                    MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1, MlasSveReinterpretS32FromU32(acc03_c1));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                    svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(pg_b32_4, C0, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C1);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(pg_b32_4, C1, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C0);
                    svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C1);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1, sum);
                }   
            }
        }
        else{
            if(HasZeroPointB){
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1, MlasSveReinterpretS32FromU32(acc03_c1));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0);
                svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1, sum);
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

template<bool HasZeroPointB>
MLAS_FORCEINLINE size_t Process4Rows(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    (void)CountM;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = MlasSveWhileLtB32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = MlasSvePtrueB32();  // full accumulator lanes
    svbool_t pg_b8_32 = MlasSvePtrueB8(); // loads 32 bytes from A/B
    svuint32_t acc03_0, acc03_1, acc47_0, acc47_1;

    // const int32_t rowsum = RowSumBuffer[0];
    svint32_t rowSumVec01, rowSumVec23, r0, r1, r2, r3;
    r0 = MlasSveBroadcastInt32(RowSumBuffer[0]);
    r1 = MlasSveBroadcastInt32(RowSumBuffer[1]);
    rowSumVec01 = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r0), MlasSveReinterpretS64FromS32(r1)));
    r2 = MlasSveBroadcastInt32(RowSumBuffer[2]);
    r3 = MlasSveBroadcastInt32(RowSumBuffer[3]);
    rowSumVec23 = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r2), MlasSveReinterpretS64FromS32(r3)));
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
            zpb_0_3 = MlasSveLoadInt32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = MlasSveReinterpretS64FromS32(zpb_0_3);
            zpb_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = MlasSveBroadcastInt32(1);
        col_0_3 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = MlasSveReinterpretS64FromS32(col_0_3);
        col_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_0_3, col64_0_3));

        acc03_0 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec01, zpb_0_3), col_0_3));
        acc03_1 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec23, zpb_0_3), col_0_3));

        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = MlasSveLoadInt32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = MlasSveReinterpretS64FromS32(zpb_4_7);
                zpb_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = MlasSveBroadcastInt32(1);
            }
            col_4_7 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = MlasSveReinterpretS64FromS32(col_4_7);
            col_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_4_7, col64_4_7));
            // acc47 = MlasSveReinterpretU32FromS32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47_0 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec01, zpb_4_7), col_4_7));
            acc47_1 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec23, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a0  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t a1  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr + 16); //load and replicate
            svuint8_t b0 = MlasSveLoadU8(pg_b8_32, B);
            svuint8_t b1 = MlasSveLoadU8(pg_b8_32, B + 32);

            acc03_0 = MlasSveMatMulAddU32(acc03_0, a0, b0);
            acc03_1 = MlasSveMatMulAddU32(acc03_1, a1, b0);
            if (cols_this > 4){
                acc47_0 = MlasSveMatMulAddU32(acc47_0, a0, b1);
                acc47_1 = MlasSveMatMulAddU32(acc47_1, a1, b1);
            }

            A_ptr += 32;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx1 = MlasSveDupqU32(0, 1, 4, 5);
        const svuint32_t idx2 = MlasSveDupqU32(2, 3, 6, 7);
        svuint32_t acc03_c0, acc03_c1, acc47_c0, acc47_c1, acc03_c2, acc03_c3, acc47_c2, acc47_c3;
        acc03_c0 = MlasSveTblU32(acc03_0, idx1);
        acc03_c1 = MlasSveTblU32(acc03_0, idx2);
        acc03_c2 = MlasSveTblU32(acc03_1, idx1);
        acc03_c3 = MlasSveTblU32(acc03_1, idx2);
        if (cols_this > 4){
            acc47_c0 = MlasSveTblU32(acc47_0, idx1);
            acc47_c1 = MlasSveTblU32(acc47_0, idx2);
            acc47_c2 = MlasSveTblU32(acc47_1, idx1);
            acc47_c3 = MlasSveTblU32(acc47_1, idx2);
        }
        // First 4 columns
        // 
        if(cols_this > 6){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C2, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C3, MlasSveReinterpretS32FromU32(acc03_c3));
                if(cols_this >= 8){
                    MlasSveStoreInt32(pg_b32_4, C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(pg_b32_4, C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(pg_b32_4, C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(pg_b32_4, C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C0, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C1);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C1, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C2);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C2, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C3);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(pg_b32_4, C3, sum);
                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = MlasSveLoadInt32(pg_b32_4, C0 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(pg_b32_4, C0 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C1 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(pg_b32_4, C1 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C2 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(pg_b32_4, C2 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C3 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(pg_b32_4, C3 + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C2 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C3 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3 + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C2, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C3, MlasSveReinterpretS32FromU32(acc03_c3));
                if(cols_this >= 6){
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C0, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C1);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C1, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C2);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C2, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C3);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(pg_b32_4, C3, sum);
                if(cols_this >= 6){
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C1 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C2 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C2 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C3 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C3 + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C1 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C2 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C2 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C3 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C3 + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(HasZeroPointB){
                if(cols_this >= 4){
                    MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(pg_b32_4, C2, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(pg_b32_4, C3, MlasSveReinterpretS32FromU32(acc03_c3));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3, MlasSveReinterpretS32FromU32(acc03_c3));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                    svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(pg_b32_4, C0, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C1);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(pg_b32_4, C1, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C2);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(pg_b32_4, C2, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C3);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c3));
                    MlasSveStoreInt32(pg_b32_4, C3, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C0);
                    svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C1);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C2);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C3);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3, sum);
                }   
            }
        }
        else{
            if(HasZeroPointB){
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C2, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C3, MlasSveReinterpretS32FromU32(acc03_c3));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0);
                svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C2);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C2, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C3);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C3, sum);
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

template<bool HasZeroPointB>
MLAS_FORCEINLINE size_t Process8Rows(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    (void)CountM;

    //------------------------------------------------------------------
    // VL = 256 → int32 lanes = 8 → process columns in groups of 4
    //------------------------------------------------------------------
    svbool_t pg_b32_4 = MlasSveWhileLtB32(0, 4);  // stores 4 output columns
    svbool_t pg_b32_8 = MlasSvePtrueB32();  // full accumulator lanes
    svbool_t pg_b8_32 = MlasSvePtrueB8(); // loads 32 bytes from A/B
    svuint32_t acc03_0, acc03_1, acc03_2, acc03_3, acc47_0, acc47_1, acc47_2, acc47_3;

    // const int32_t rowsum = RowSumBuffer[0];
    svint32_t rowSumVec01, rowSumVec23, rowSumVec45, rowSumVec67, r0, r1, r2, r3, r4, r5, r6, r7;
    r0 = MlasSveBroadcastInt32(RowSumBuffer[0]);
    r1 = MlasSveBroadcastInt32(RowSumBuffer[1]);
    rowSumVec01 = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r0), MlasSveReinterpretS64FromS32(r1)));
    r2 = MlasSveBroadcastInt32(RowSumBuffer[2]);
    r3 = MlasSveBroadcastInt32(RowSumBuffer[3]);
    rowSumVec23 = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r2), MlasSveReinterpretS64FromS32(r3)));
    r4 = MlasSveBroadcastInt32(RowSumBuffer[4]);
    r5 = MlasSveBroadcastInt32(RowSumBuffer[5]);
    rowSumVec45 = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r4), MlasSveReinterpretS64FromS32(r5)));
    r6 = MlasSveBroadcastInt32(RowSumBuffer[6]);
    r7 = MlasSveBroadcastInt32(RowSumBuffer[7]);
    rowSumVec67 = MlasSveReinterpretS32FromS64(MlasSveTrn1S64(MlasSveReinterpretS64FromS32(r6), MlasSveReinterpretS64FromS32(r7)));
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
            zpb_0_3 = MlasSveLoadInt32(pg_b32_4, ZeroPointB); //pg_b32_4
            zpb64_0_3 = MlasSveReinterpretS64FromS32(zpb_0_3);
            zpb_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_0_3, zpb64_0_3));
        }
        else
            zpb_0_3 = MlasSveBroadcastInt32(1);
        col_0_3 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer);
        col64_0_3 = MlasSveReinterpretS64FromS32(col_0_3);
        col_0_3 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_0_3, col64_0_3));

        acc03_0 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec01, zpb_0_3), col_0_3));
        acc03_1 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec23, zpb_0_3), col_0_3));
        acc03_2 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec45, zpb_0_3), col_0_3));
        acc03_3 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec67, zpb_0_3), col_0_3));
        // --- Columns 4–7 ---
        if (cols_this > 4) {
            if (ZeroPointB){
                zpb_4_7 = MlasSveLoadInt32(pg_b32_4, ZeroPointB + 4);
                zpb64_4_7 = MlasSveReinterpretS64FromS32(zpb_4_7);
                zpb_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(zpb64_4_7, zpb64_4_7));
            }
            else{
                zpb_4_7 = MlasSveBroadcastInt32(1);
            }
            col_4_7 = MlasSveLoadInt32(pg_b32_4, ColumnSumBuffer + 4);
            col64_4_7 = MlasSveReinterpretS64FromS32(col_4_7);
            col_4_7 = MlasSveReinterpretS32FromS64(MlasSveZip1S64(col64_4_7, col64_4_7));
            // acc47 = MlasSveReinterpretU32FromS32(svmad_s32_x(pg_b32_4, rowSumVec, zpb_4_7, col_4_7));
            acc47_0 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec01, zpb_4_7), col_4_7));
            acc47_1 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec23, zpb_4_7), col_4_7));
            acc47_2 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec45, zpb_4_7), col_4_7));
            acc47_3 = MlasSveReinterpretU32FromS32(MlasSveAddInt32X(pg_b32_8, MlasSveMulInt32(pg_b32_8, rowSumVec67, zpb_4_7), col_4_7));
        }
        //------------------------------------------------------------------
        // K loop
        //------------------------------------------------------------------
        // const uint8_t* B_ptr = B + col * 32; // 32 bytes per slice for VL=256

        for (size_t k = 0; k < PackedCountK; k++)
        {
            // A: 32 bytes, B: 32 bytes per 4 columns
            svuint8_t a0  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr); //load and replicate
            svuint8_t a1  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr + 16); //load and replicate
            svuint8_t a2  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr + 32); //load and replicate
            svuint8_t a3  = MlasSveLoadReplicateU8(pg_b8_32, A_ptr + 48); //load and replicate
            svuint8_t b0 = MlasSveLoadU8(pg_b8_32, B);
            svuint8_t b1 = MlasSveLoadU8(pg_b8_32, B + 32);

            acc03_0 = MlasSveMatMulAddU32(acc03_0, a0, b0);
            acc03_1 = MlasSveMatMulAddU32(acc03_1, a1, b0);
            acc03_2 = MlasSveMatMulAddU32(acc03_2, a2, b0);
            acc03_3 = MlasSveMatMulAddU32(acc03_3, a3, b0);
            if (cols_this > 4){
                acc47_0 = MlasSveMatMulAddU32(acc47_0, a0, b1);
                acc47_1 = MlasSveMatMulAddU32(acc47_1, a1, b1);
                acc47_2 = MlasSveMatMulAddU32(acc47_2, a2, b1);
                acc47_3 = MlasSveMatMulAddU32(acc47_3, a3, b1);
            }

            A_ptr += 64;   // Each tile is 32 bytes for 256-bit
            B += 64;     // Two × 32-byte blocks per 8 columns
        }

        //------------------------------------------------------------------
        // Store results into C
        //------------------------------------------------------------------
        const svuint32_t idx1 = MlasSveDupqU32(0, 1, 4, 5);
        const svuint32_t idx2 = MlasSveDupqU32(2, 3, 6, 7);
        svuint32_t acc03_c0, acc03_c1, acc47_c0, acc47_c1, acc03_c2, acc03_c3, acc47_c2, acc47_c3, acc03_c4, acc03_c5, acc47_c4, acc47_c5, acc03_c6, acc03_c7, acc47_c6, acc47_c7;
        acc03_c0 = MlasSveTblU32(acc03_0, idx1);
        acc03_c1 = MlasSveTblU32(acc03_0, idx2);
        acc03_c2 = MlasSveTblU32(acc03_1, idx1);
        acc03_c3 = MlasSveTblU32(acc03_1, idx2);
        acc03_c4 = MlasSveTblU32(acc03_2, idx1);
        acc03_c5 = MlasSveTblU32(acc03_2, idx2);
        acc03_c6 = MlasSveTblU32(acc03_3, idx1);
        acc03_c7 = MlasSveTblU32(acc03_3, idx2);
        if (cols_this > 4){
            acc47_c0 = MlasSveTblU32(acc47_0, idx1);
            acc47_c1 = MlasSveTblU32(acc47_0, idx2);
            acc47_c2 = MlasSveTblU32(acc47_1, idx1);
            acc47_c3 = MlasSveTblU32(acc47_1, idx2);
            acc47_c4 = MlasSveTblU32(acc47_2, idx1);
            acc47_c5 = MlasSveTblU32(acc47_2, idx2);
            acc47_c6 = MlasSveTblU32(acc47_3, idx1);
            acc47_c7 = MlasSveTblU32(acc47_3, idx2);
        }
        // First 4 columns
        // 
        if(cols_this > 6){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C2, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C3, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(pg_b32_4, C4, MlasSveReinterpretS32FromU32(acc03_c4));
                MlasSveStoreInt32(pg_b32_4, C5, MlasSveReinterpretS32FromU32(acc03_c5));
                MlasSveStoreInt32(pg_b32_4, C6, MlasSveReinterpretS32FromU32(acc03_c6));
                MlasSveStoreInt32(pg_b32_4, C7, MlasSveReinterpretS32FromU32(acc03_c7));
                if(cols_this >= 8){
                    MlasSveStoreInt32(pg_b32_4, C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(pg_b32_4, C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(pg_b32_4, C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(pg_b32_4, C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(pg_b32_4, C4 + 4, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(pg_b32_4, C5 + 4, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(pg_b32_4, C6 + 4, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(pg_b32_4, C7 + 4, MlasSveReinterpretS32FromU32(acc47_c7));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C4 + 4, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C5 + 4, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C6 + 4, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C7 + 4, MlasSveReinterpretS32FromU32(acc47_c7));
                }            
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C0, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C1);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C1, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C2);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C2, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C3);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(pg_b32_4, C3, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C4);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c4));
                MlasSveStoreInt32(pg_b32_4, C4, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C5);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c5));
                MlasSveStoreInt32(pg_b32_4, C5, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C6);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c6));
                MlasSveStoreInt32(pg_b32_4, C6, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C7);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c7));
                MlasSveStoreInt32(pg_b32_4, C7, sum);
                // Row 1 (C1)
                if(cols_this >= 8){
                    prev = MlasSveLoadInt32(pg_b32_4, C0 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(pg_b32_4, C0 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C1 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(pg_b32_4, C1 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C2 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(pg_b32_4, C2 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C3 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(pg_b32_4, C3 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C4 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(pg_b32_4, C4 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C5 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(pg_b32_4, C5 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C6 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(pg_b32_4, C6 + 4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C7 + 4);
                    sum  = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc47_c7));
                    MlasSveStoreInt32(pg_b32_4, C7 + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C2 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C3 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C4 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C4 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C5 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C5 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C6 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C6 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C7 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc47_c7));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C7 + 4, sum);
                }
            }
        }
        else if(cols_this > 4){
            if(HasZeroPointB){
                MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C2, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C3, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(pg_b32_4, C4, MlasSveReinterpretS32FromU32(acc03_c4));
                MlasSveStoreInt32(pg_b32_4, C5, MlasSveReinterpretS32FromU32(acc03_c5));
                MlasSveStoreInt32(pg_b32_4, C6, MlasSveReinterpretS32FromU32(acc03_c6));
                MlasSveStoreInt32(pg_b32_4, C7, MlasSveReinterpretS32FromU32(acc03_c7));
                if(cols_this >= 6){
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C4 + 4, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C5 + 4, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C6 + 4, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C7 + 4, MlasSveReinterpretS32FromU32(acc47_c7));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C0 + 4, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C1 + 4, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C2 + 4, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C3 + 4, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C4 + 4, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C5 + 4, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C6 + 4, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C7 + 4, MlasSveReinterpretS32FromU32(acc47_c7));
                }
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(pg_b32_4, C0, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C1);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(pg_b32_4, C1, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C2);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(pg_b32_4, C2, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C3);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(pg_b32_4, C3, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C4);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c4));
                MlasSveStoreInt32(pg_b32_4, C4, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C5);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c5));
                MlasSveStoreInt32(pg_b32_4, C5, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C6);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c6));
                MlasSveStoreInt32(pg_b32_4, C6, sum);
                prev = MlasSveLoadInt32(pg_b32_4, C7);
                sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c7));
                MlasSveStoreInt32(pg_b32_4, C7, sum);
                // Row 1 (C1)
                if(cols_this >= 6){
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C1 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C2 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C2 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C3 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C3 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C4 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C4 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C5 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C5 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C6 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C6 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 2), C7 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 2), prev, MlasSveReinterpretS32FromU32(acc47_c7));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 2), C7 + 4, sum);
                }
                else{
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C0 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C0 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C1 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C1 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C2 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C2 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C3 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C3 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C4 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C4 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C5 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C5 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C6 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C6 + 4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 1), C7 + 4);
                    sum  = MlasSveAddInt32X(MlasSveWhileLtB32(0, 1), prev, MlasSveReinterpretS32FromU32(acc47_c7));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 1), C7 + 4, sum);
                }
            }
        }
        else if(cols_this > 2){
            if(HasZeroPointB){
                if(cols_this >= 4){
                    MlasSveStoreInt32(pg_b32_4, C0, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(pg_b32_4, C1, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(pg_b32_4, C2, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(pg_b32_4, C3, MlasSveReinterpretS32FromU32(acc03_c3));
                    MlasSveStoreInt32(pg_b32_4, C4, MlasSveReinterpretS32FromU32(acc03_c4));
                    MlasSveStoreInt32(pg_b32_4, C5, MlasSveReinterpretS32FromU32(acc03_c5));
                    MlasSveStoreInt32(pg_b32_4, C6, MlasSveReinterpretS32FromU32(acc03_c6));
                    MlasSveStoreInt32(pg_b32_4, C7, MlasSveReinterpretS32FromU32(acc03_c7));
                }
                else{
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3, MlasSveReinterpretS32FromU32(acc03_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C4, MlasSveReinterpretS32FromU32(acc03_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C5, MlasSveReinterpretS32FromU32(acc03_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C6, MlasSveReinterpretS32FromU32(acc03_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C7, MlasSveReinterpretS32FromU32(acc03_c7));
                }
            }
            else{
                if(cols_this >= 4){
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(pg_b32_4, C0);
                    svint32_t sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(pg_b32_4, C0, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C1);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(pg_b32_4, C1, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C2);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(pg_b32_4, C2, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C3);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c3));
                    MlasSveStoreInt32(pg_b32_4, C3, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C4);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c4));
                    MlasSveStoreInt32(pg_b32_4, C4, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C5);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c5));
                    MlasSveStoreInt32(pg_b32_4, C5, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C6);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c6));
                    MlasSveStoreInt32(pg_b32_4, C6, sum);
                    prev = MlasSveLoadInt32(pg_b32_4, C7);
                    sum = MlasSveAddInt32X(pg_b32_4, prev, MlasSveReinterpretS32FromU32(acc03_c7));
                    MlasSveStoreInt32(pg_b32_4, C7, sum);
                }
                else{
                    // Row 0 (C0)
                    svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C0);
                    svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c0));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C0, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C1);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c1));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C1, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C2);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c2));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C2, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C3);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c3));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C3, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C4);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c4));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C4, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C5);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c5));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C5, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C6);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c6));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C6, sum);
                    prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, 3), C7);
                    sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, 3), prev, MlasSveReinterpretS32FromU32(acc03_c7));
                    MlasSveStoreInt32(MlasSveWhileLtB32(0, 3), C7, sum);
                }   
            }
        }
        else{
            if(HasZeroPointB){
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C2, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C3, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C4, MlasSveReinterpretS32FromU32(acc03_c4));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C5, MlasSveReinterpretS32FromU32(acc03_c5));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C6, MlasSveReinterpretS32FromU32(acc03_c6));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C7, MlasSveReinterpretS32FromU32(acc03_c7));
            }
            else{
                // Row 0 (C0)
                svint32_t prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0);
                svint32_t sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c0));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C0, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c1));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C1, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C2);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c2));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C2, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C3);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c3));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C3, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C4);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c4));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C4, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C5);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c5));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C5, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C6);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c6));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C6, sum);
                prev = MlasSveLoadInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C7);
                sum = MlasSveAddInt32X(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), prev, MlasSveReinterpretS32FromU32(acc03_c7));
                MlasSveStoreInt32(MlasSveWhileLtB32(0, std::min(int(cols_this), 2)), C7, sum);
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

template<bool HasZeroPointB>
MLAS_FORCEINLINE size_t
MlasSveQgemmU8X8KernelUmmlaImpl(
    const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    if (svcntb() == 32) {

        if (CountM >= 8) {
            return Process8Rows<HasZeroPointB>(
                A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer,
                ZeroPointB);
        }
        else if (CountM >= 4) {
            return Process4Rows<HasZeroPointB>(
                A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer,
                ZeroPointB);
        }
        else if (CountM >= 2) {
            return Process2Rows<HasZeroPointB>(
                A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer,
                ZeroPointB);
        }
        else if (CountM >= 1) {
            return Process1Row<HasZeroPointB>(
                A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer,
                ZeroPointB);
        }
    }
    else {
        if constexpr (HasZeroPointB) {
            return MlasGemmU8X8KernelUmmlaZero(
                A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer, ZeroPointB);

        } else {
            return MlasGemmU8X8KernelUmmlaAdd(
                A, B, C, PackedCountK, CountM, CountN, ldc,
                RowSumBuffer, ColumnSumBuffer, ZeroPointB);
        }
    }

    return 0;
}

size_t
MlasSveQgemmU8X8KernelUmmlaZero(
    const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    return MlasSveQgemmU8X8KernelUmmlaImpl<true>(
        A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB);
}

size_t
MlasSveQgemmU8X8KernelUmmlaAdd(
    const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB)
{
    return MlasSveQgemmU8X8KernelUmmlaImpl<false>(
        A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB);
}
#pragma GCC diagnostic pop