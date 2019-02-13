#include "BrainSliceOperator.h"

/* Firmware custom HEX definitions */
#define NIOS_HEX_CNN_AUTOGEN_NATIVE_DIM_MISMATCH 700
#define NIOS_HEX_CNN_AUTOGEN_MFUS_TOO_FEW 701
#define NIOS_HEX_CNN_AUTOGEN_INITIAL_VRF_TOO_SMALL 702
#define NIOS_HEX_CNN_AUTOGEN_MATRIX_RF_TOO_SMALL 703
#define NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL 704
#define NIOS_HEX_CNN_AUTOGEN_MULTIPLY_VRF_TOO_SMALL 705
#define NIOS_HEX_CNN_AUTOGEN_MISSING_NEEDED_DRAM 706
#define NIOS_HEX_CNN_AUTOGEN_VECTOR_MEM_TOO_SMALL 707
#define NIOS_HEX_CNN_AUTOGEN_MAX_TILE_ROWS_TOO_SMALL 708
#define NIOS_HEX_CNN_AUTOGEN_MAX_TILE_COLS_TOO_SMALL 709
#define NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR 710
#define NIOS_HEX_CNN_AUTOGEN_SUBSET_RANGE_INVALID 711
#define NIOS_HEX_CNN_AUTOGEN_PAYLOAD_SIZE_WRONG 712
#define NIOS_HEX_CNN_AUTOGEN_OPERATION_MODE_INVALID 713

/* Assert macro for assertions that are only compiled when the firmware is run with the emulator */
#ifdef USE_SIM_API
#define Emulator_HEX_Assert(expr, hex_code) BSNL_HEX_Assert(expr, hex_code)
#else
#define Emulator_HEX_Assert(expr, hex_code)
#endif /* USE_SIM_API */

/* Parameter locations */
static const ISA_ExtAddress zeros = 0;
static const ISA_ExtAddress zeros_size = 226;
static const ISA_ExtAddress conv1_1_bias = 226;
static const ISA_ExtAddress conv1_1_bias_size = 1;
static const ISA_ExtAddress conv1_1_MRF = 0;
static const ISA_ExtAddress conv1_1_MRF_size = 2;
static const ISA_ExtAddress conv1_2_bias = 227;
static const ISA_ExtAddress conv1_2_bias_size = 1;
static const ISA_ExtAddress conv1_2_MRF = 2;
static const ISA_ExtAddress conv1_2_MRF_size = 9;
static const ISA_ExtAddress pool1_MRF = 11;
static const ISA_ExtAddress pool1_MRF_size = 1;
static const ISA_ExtAddress conv2_1_bias = 228;
static const ISA_ExtAddress conv2_1_bias_size = 1;
static const ISA_ExtAddress conv2_1_MRF = 12;
static const ISA_ExtAddress conv2_1_MRF_size = 9;
static const ISA_ExtAddress conv2_2_bias = 229;
static const ISA_ExtAddress conv2_2_bias_size = 1;
static const ISA_ExtAddress conv2_2_MRF = 21;
static const ISA_ExtAddress conv2_2_MRF_size = 9;
static const ISA_ExtAddress pool2_MRF = 30;
static const ISA_ExtAddress pool2_MRF_size = 1;
static const ISA_ExtAddress conv3_1_bias = 230;
static const ISA_ExtAddress conv3_1_bias_size = 2;
static const ISA_ExtAddress conv3_1_MRF = 31;
static const ISA_ExtAddress conv3_1_MRF_size = 18;
static const ISA_ExtAddress conv3_2_bias = 232;
static const ISA_ExtAddress conv3_2_bias_size = 2;
static const ISA_ExtAddress conv3_2_MRF = 49;
static const ISA_ExtAddress conv3_2_MRF_size = 36;
static const ISA_ExtAddress conv3_3_bias = 234;
static const ISA_ExtAddress conv3_3_bias_size = 2;
static const ISA_ExtAddress conv3_3_MRF = 85;
static const ISA_ExtAddress conv3_3_MRF_size = 36;
static const ISA_ExtAddress pool3_MRF = 121;
static const ISA_ExtAddress pool3_MRF_size = 1;
static const ISA_ExtAddress conv4_1_bias = 236;
static const ISA_ExtAddress conv4_1_bias_size = 4;
static const ISA_ExtAddress conv4_1_MRF = 122;
static const ISA_ExtAddress conv4_1_MRF_size = 72;
static const ISA_ExtAddress conv4_2_bias = 240;
static const ISA_ExtAddress conv4_2_bias_size = 4;
static const ISA_ExtAddress conv4_2_MRF = 194;
static const ISA_ExtAddress conv4_2_MRF_size = 144;
static const ISA_ExtAddress conv4_3_bias = 244;
static const ISA_ExtAddress conv4_3_bias_size = 4;
static const ISA_ExtAddress conv4_3_MRF = 338;
static const ISA_ExtAddress conv4_3_MRF_size = 144;
static const ISA_ExtAddress pool4_MRF = 482;
static const ISA_ExtAddress pool4_MRF_size = 1;
static const ISA_ExtAddress conv5_1_bias = 248;
static const ISA_ExtAddress conv5_1_bias_size = 4;
static const ISA_ExtAddress conv5_1_MRF = 483;
static const ISA_ExtAddress conv5_1_MRF_size = 144;
static const ISA_ExtAddress conv5_2_bias = 252;
static const ISA_ExtAddress conv5_2_bias_size = 4;
static const ISA_ExtAddress conv5_2_MRF = 627;
static const ISA_ExtAddress conv5_2_MRF_size = 144;
static const ISA_ExtAddress conv5_3_bias = 256;
static const ISA_ExtAddress conv5_3_bias_size = 4;
static const ISA_ExtAddress conv5_3_MRF = 771;
static const ISA_ExtAddress conv5_3_MRF_size = 144;
static const ISA_ExtAddress block5_pool_MaxPool_MRF = 915;
static const ISA_ExtAddress block5_pool_MaxPool_MRF_size = 1;

/* Common variables */
ISA_ExtAddress ivrf_inIterator;
ISA_MrfAddress mrf_start=0, mrf_next=64, mrf_tmp;

/* Layer function prototypes */
void input1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv11(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv41(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv42(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv43(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void pool4(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv51(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv52(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv53(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void block5PoolMaxpool(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);

/* The function table for the layer functions */
typedef void (*layer_fn_t)(PBS_CONTEXT, bool, bool, bool);
static const layer_fn_t c_LayerFunctionTable[] = {
    input1,
    conv11,
    conv41,
    conv42,
    conv43,
    pool4,
    conv51,
    conv52,
    conv53,
    block5PoolMaxpool,
};

// Init service function for use with ONNX Runtime.
void init(PBS_CONTEXT bs, const BrainSliceOperator_RuntimeArguments* args)
{
    /* Sanity check for the BrainSlice SKU for this firmware. */
    BSNL_HEX_Assert(bs->m_bsParameters.NATIVE_DIM == 128, NIOS_HEX_CNN_AUTOGEN_NATIVE_DIM_MISMATCH);
    BSNL_HEX_Assert(bs->m_bsParameters.MFUS >= 2, NIOS_HEX_CNN_AUTOGEN_MFUS_TOO_FEW);
    BSNL_HEX_Assert(bs->m_bsParameters.INITIAL_VRF_SIZE >= 9204, NIOS_HEX_CNN_AUTOGEN_INITIAL_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.MVM_MATRIX_RF_SIZE >= 128, NIOS_HEX_CNN_AUTOGEN_MATRIX_RF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.ADDSUB_VRF_0_SIZE >= 4, NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.ADDSUB_VRF_1_SIZE >= 3808, NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.MULTIPLY_VRF_SIZE >= 0, NIOS_HEX_CNN_AUTOGEN_MULTIPLY_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.USE_DRAM  , NIOS_HEX_CNN_AUTOGEN_MISSING_NEEDED_DRAM);
    BSNL_HEX_Assert(bs->m_bsParameters.VECTOR_MEM_SIZE >= 260, NIOS_HEX_CNN_AUTOGEN_VECTOR_MEM_TOO_SMALL);

    BSNL_postResponseSubmessage(bs, 0);
}

#ifdef __GNUC__
__attribute__((noinline))
#endif
void genericConvolution(PBS_CONTEXT p_bs, ISA_NativeCount input_height, ISA_NativeCount input_width, ISA_NativeCount input_depth,
    ISA_ExtAddress input_address, ISA_NativeCount output_depth, ISA_NativeCount kernel_size, ISA_Pad pad, ISA_Increment stride, bool include_relu, bool is_dummy,
    ISA_ExtAddress mrf_offset, ISA_ExtAddress mulParam_address, ISA_ExtAddress addParam_address,
    ISA_ExtAddress output_IVRF_address, ISA_ExtAddress output_ASVRF1_address,
    ISA_TensorId mrf_fetch_address, ISA_TensorId mrf_fetch_size,
    ISA_TensorId mrf_prefetch_next_address, ISA_NativeCount mrf_prefetch_next_size, bool swap_mrf_buffers)
{
    if (mrf_fetch_address != ((ISA_ExtAddress)-1)) {
        moveFilterCount128(p_bs, ISA_Mem_Dram, mrf_fetch_address, ISA_Mem_MatrixRf, mrf_start+mrf_offset, 1, mrf_fetch_size);
    }
    if (mulParam_address != ((ISA_ExtAddress)-1)) {
        vRead1D(p_bs, ISA_Mem_Dram, mulParam_address, output_depth);
        v_wr(p_bs, ISA_Mem_MultiplyVrf, 0);
    }
    if (addParam_address != ((ISA_ExtAddress)-1)) {
        vRead1D(p_bs, ISA_Mem_Dram, addParam_address, output_depth);
        v_wr_inc(p_bs, ISA_Mem_AddSubVrf_0, 0, 0);
    }
    if (mrf_prefetch_next_address != ((ISA_ExtAddress)-1)) {
        moveFilterCount128(p_bs, ISA_Mem_Dram, mrf_prefetch_next_address, ISA_Mem_MatrixRf, mrf_next, 1, mrf_prefetch_next_size);
    }
    for(unsigned row=0; row<output_depth; row++) {
        if (!is_dummy) {
            vRead3D(p_bs, ISA_Mem_MvmInitialVrf, input_address, input_width, input_height, input_depth, kernel_size, pad, stride);
            mv_mul(p_bs, mrf_start+mrf_offset+(row*kernel_size*kernel_size*input_depth));
        } else {
            vRead2D(p_bs, ISA_Mem_MvmInitialVrf, input_address + row, 1, input_width * input_height, output_depth);
            mv_mul(p_bs, mrf_start+mrf_offset);
        }
        if (mulParam_address != ((ISA_ExtAddress)-1)) {
            vv_mul(p_bs, row);
        }
        if (addParam_address != ((ISA_ExtAddress)-1)) {
            vv_add_inc(p_bs, ISA_Mem_AddSubVrf_0, row, 0);
        }
        if (include_relu) {
            v_relu(p_bs);
        }
        if (output_IVRF_address != ((ISA_ExtAddress)-1)) {
            v_wr_inc(p_bs, ISA_Mem_MvmInitialVrf, output_IVRF_address+row, output_depth);
        }
        if (output_ASVRF1_address != ((ISA_ExtAddress)-1)) {
            v_wr_inc(p_bs, ISA_Mem_AddSubVrf_1, output_ASVRF1_address+row, output_depth);
        }
    }
    if (swap_mrf_buffers) {
        mrf_tmp=mrf_start;
        mrf_start=mrf_next;
        mrf_next=mrf_tmp;
    }
}


/**
 * The main function that runs evaluation on the VGG-16 model.
 *
 * This runs the input on the network through the specified subset of the network.
 **/
void execute(PBS_CONTEXT p_bs, const BrainSliceOperator_RuntimeArguments* args, int p_startLayerID, int p_endLayerID, bool p_debugMode)
{
    // By default, run all the VGG-16 layers
    int numLayers = sizeof(c_LayerFunctionTable) / sizeof(c_LayerFunctionTable[0]);
    if (p_startLayerID == -1 && p_endLayerID == -1)
    {
        p_startLayerID = 0;
        p_endLayerID = numLayers - 1;
    }

    // Verify that the specified subset of the model is a valid one
    static const bool debugFirmware = false;
    BSNL_HEX_Assert(0 <= p_startLayerID && p_startLayerID < numLayers, NIOS_HEX_CNN_AUTOGEN_SUBSET_RANGE_INVALID);
    BSNL_HEX_Assert(1 <= p_endLayerID && p_endLayerID < numLayers, NIOS_HEX_CNN_AUTOGEN_SUBSET_RANGE_INVALID);
    BSNL_HEX_Assert(p_startLayerID <= p_endLayerID, NIOS_HEX_CNN_AUTOGEN_SUBSET_RANGE_INVALID);
    BSNL_HEX_Assert(debugFirmware || (p_startLayerID == 0 && p_endLayerID == numLayers - 1),
            NIOS_HEX_CNN_AUTOGEN_SUBSET_RANGE_INVALID);
    BSNL_HEX_Assert(debugFirmware || !p_debugMode, NIOS_HEX_CNN_AUTOGEN_OPERATION_MODE_INVALID);

    /* Verify that the payload matches the size that we expect for the inputs. Only needed if the first layer is being
     * run, because that is the only layer that has its inputs sent over the network. */
    if (p_startLayerID == 0)
    {
        static const int inputNativeSize = 1240;
        BSNL_HEX_Assert(p_bs->m_activeRequestSubmessage.PayloadByteLength == inputNativeSize
                * p_bs->m_bsParameters.HWVEC_BYTES, NIOS_HEX_CNN_AUTOGEN_PAYLOAD_SIZE_WRONG);
    }

    // Set the size of the response. If we are running the last layer, then it is sent over the network in the response.
    int outputNativeSize;
    if (p_endLayerID == numLayers - 1)
    {
        // Compute the total native size of the output of the ending layer of the subset
        outputNativeSize = 196;
    }
    /* To prevent the response from returning before the computation is done, a dummy native vector is sent in the response. */
    else
    {
        outputNativeSize = 1;
    }

    BSNL_postResponseSubmessage(p_bs, outputNativeSize * p_bs->m_bsParameters.HWVEC_BYTES);

    // Iterate over each layer in the selected subset of the model.
    for (int i = p_startLayerID; i <= p_endLayerID; i++)
    {
        c_LayerFunctionTable[i](p_bs, p_debugMode, (i==p_startLayerID), (i==p_endLayerID));
    }

    /* Write the output vectors for only the final layer to the network. If the ending layer is anything other than the
     * final layer, then the results are retrieved by the client with the ReadVector service function. */
    if (p_endLayerID == numLayers - 1)
    {
        if (false)
        {
            vRead1D(p_bs, ISA_Mem_Dram, -1, 196);
            v_wr(p_bs, ISA_Mem_NetOutputQ, DONTCARE);
        }
    }
    // Otherwise, a small chain is performed to send dummy data back so the client does not return until computation is done
    else
    {
        // Perform the mv_mul chain so that the response isn't sent until all the operations are done
        vRead1D(p_bs, ISA_Mem_MvmInitialVrf, 0, 1);
        mv_mul(p_bs, 0);
        v_wr(p_bs, ISA_Mem_NetOutputQ, DONTCARE);
    }

    // Ensure that the function ends with an end_chain instruction
    end_chain(p_bs);
}

void input1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Input layer: input_1(d=147, h=224, d=224) = Input() 1240 registers */
    vRead1D(bs, ISA_Mem_NetInputQ, DONTCARE, 1240);
    v_wr(bs, ISA_Mem_Expander, 0+0);
    /* End input layer */
}

void conv11(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution conv1_1(d=64, h=224, d=224) = Convolution(input_1(d=147, h=224, w=224), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    absorbed ReLU1_1 */
    /*    includes sublayer conv1_2(d=64, h=224, d=224) = Convolution(conv1_1(d=64, h=224, w=224), k_h=3, k_w=3, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed ReLU1_2 */
    /*    includes sublayer pool1(d=64, h=112, d=112) = MaxPool(conv1_2(d=64, h=224, w=224), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    /*    includes sublayer conv2_1(d=128, h=112, d=112) = Convolution(pool1(d=64, h=112, w=112), k_h=3, k_w=3, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed ReLU2_1 */
    /*    includes sublayer conv2_2(d=128, h=112, d=112) = Convolution(conv2_1(d=128, h=112, w=112), k_h=3, k_w=3, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed ReLU2_2 */
    /*    includes sublayer pool2(d=128, h=56, d=56) = MaxPool(conv2_2(d=128, h=112, w=112), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    /*    includes sublayer conv3_1(d=256, h=56, d=56) = Convolution(pool2(d=128, h=56, w=56), k_h=3, k_w=3, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed ReLU3_1 */
    /*    includes sublayer conv3_2(d=256, h=56, d=56) = Convolution(conv3_1(d=256, h=56, w=56), k_h=3, k_w=3, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed ReLU3_2 */
    /*    includes sublayer conv3_3(d=256, h=56, d=56) = Convolution(conv3_2(d=256, h=56, w=56), k_h=3, k_w=3, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed ReLU3_3 */
    /*    includes sublayer pool3(d=256, h=28, d=28) = MaxPool(conv3_3(d=256, h=56, w=56), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress input_1_inIndex;
    input_1_inIndex=0;
    ISA_ExtAddress tmp_MVMIVRF=2246, tmp_MVMIVRF_next=2638;
    ISA_ExtAddress tmp_ASVRF1=0, tmp_ASVRF1_next=336, tmp_ASVRF1_swap;
    /* Split the tile in half for double buffering */
    /* Layer conv1_1 tile size 1*224 */
    /* Temp vars and parameters for input layer conv1_1 */
    /* SetIterations on reads from the input expander must be a multiple of bs->m_bsParameters.CHANNELS or this must be the last read*/
    ISA_NativeCount maxReadSize=(ISA_NativeCount)((112/bs->m_bsParameters.CHANNELS)*bs->m_bsParameters.CHANNELS);
    /* _in is the read pointer (not adjusted for padding because we read the whole row), _next is the write pointer (adjusted for padding) */
    ISA_ExtAddress g0_conv1_1_in=8851, g0_conv1_1_inIterator=8851;
    ISA_ExtAddress g0_conv1_1_next=9075, g0_conv1_1_available=maxReadSize, g0_conv1_1_next_available=maxReadSize, g0_conv1_1_tmp;
    /* Need to track the start and offset within the output row to handle the padding in conv1_2 */
    ISA_ExtAddress g0_conv1_1_outOffset=1;
    ISA_ExtAddress g0_conv1_1_outRowStart=226;
    unsigned g0_conv1_1_iterationsLeft=50176;
    unsigned g0_conv1_1_loadLeft=50176;
    moveFilterCount128(bs, ISA_Mem_Dram, conv1_1_MRF, ISA_Mem_MatrixRf, 0, 1, 2);
    vRead1D(bs, ISA_Mem_Dram, conv1_1_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    /* Layer conv1_2 tile size 3*226 */
    /* Temp vars and parameters for input layer conv1_2 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_conv1_2_in=1568,g1_conv1_2_inIterator=1795;
    ISA_ExtAddress g1_conv1_2_available = 230;
    ISA_ExtAddress g1_conv1_2_outOffset=0;
    unsigned g1_conv1_2_iterationsLeft=50176;
    moveFilterCount128(bs, ISA_Mem_Dram, conv1_2_MRF, ISA_Mem_MatrixRf, 2, 1, 9);
    vRead1D(bs, ISA_Mem_Dram, conv1_2_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 1);
    /* Layer pool1 tile size 7*224 */
    /* Temp vars and parameters for input layer pool1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g2_pool1_in=3030,g2_pool1_inIterator=3030;
    ISA_ExtAddress g2_pool1_available = 0;
    ISA_ExtAddress g2_pool1_accumulators=2246;
    ISA_ExtAddress g2_pool1_availableVerticalRows=0;
    ISA_ExtAddress g2_pool1_outOffset=115;
    unsigned g2_pool1_iterationsLeft=12544;
    moveFilterCount128(bs, ISA_Mem_Dram, pool1_MRF, ISA_Mem_MatrixRf, 11, 1, 1);
    /* Layer conv2_1 tile size 5*114 */
    /* Temp vars and parameters for input layer conv2_1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g3_conv2_1_in=4598,g3_conv2_1_inIterator=4713;
    ISA_ExtAddress g3_conv2_1_available = 122;
    ISA_ExtAddress g3_conv2_1_outOffset=115;
    unsigned g3_conv2_1_iterationsLeft=12544;
    moveFilterCount128(bs, ISA_Mem_Dram, conv2_1_MRF, ISA_Mem_MatrixRf, 12, 1, 9);
    vRead1D(bs, ISA_Mem_Dram, conv2_1_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 2);
    /* Layer conv2_2 tile size 5*114 */
    /* Temp vars and parameters for input layer conv2_2 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g4_conv2_2_in=5168,g4_conv2_2_inIterator=5283;
    ISA_ExtAddress g4_conv2_2_available = 122;
    ISA_ExtAddress g4_conv2_2_outOffset=0;
    unsigned g4_conv2_2_iterationsLeft=12544;
    moveFilterCount128(bs, ISA_Mem_Dram, conv2_2_MRF, ISA_Mem_MatrixRf, 21, 1, 9);
    vRead1D(bs, ISA_Mem_Dram, conv2_2_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 3);
    /* Layer pool2 tile size 7*112 */
    /* Temp vars and parameters for input layer pool2 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g5_pool2_in=5738,g5_pool2_inIterator=5738;
    ISA_ExtAddress g5_pool2_available = 0;
    ISA_ExtAddress g5_pool2_accumulators=2246;
    ISA_ExtAddress g5_pool2_availableVerticalRows=0;
    ISA_ExtAddress g5_pool2_outOffset=59;
    unsigned g5_pool2_iterationsLeft=3136;
    moveFilterCount128(bs, ISA_Mem_Dram, pool2_MRF, ISA_Mem_MatrixRf, 30, 1, 1);
    /* Layer conv3_1 tile size 5*58 */
    /* Temp vars and parameters for input layer conv3_1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g6_conv3_1_in=6522,g6_conv3_1_inIterator=6581;
    ISA_ExtAddress g6_conv3_1_available = 66;
    ISA_ExtAddress g6_conv3_1_outOffset=118;
    unsigned g6_conv3_1_iterationsLeft=3136;
    moveFilterCount128(bs, ISA_Mem_Dram, conv3_1_MRF, ISA_Mem_MatrixRf, 31, 1, 18);
    vRead1D(bs, ISA_Mem_Dram, conv3_1_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 4);
    /* Layer conv3_2 tile size 5*116 */
    /* Temp vars and parameters for input layer conv3_2 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g7_conv3_2_in=6812,g7_conv3_2_inIterator=6930;
    ISA_ExtAddress g7_conv3_2_available = 132;
    ISA_ExtAddress g7_conv3_2_outOffset=118;
    unsigned g7_conv3_2_iterationsLeft=3136;
    moveFilterCount128(bs, ISA_Mem_Dram, conv3_2_MRF, ISA_Mem_MatrixRf, 49, 1, 36);
    vRead1D(bs, ISA_Mem_Dram, conv3_2_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 6);
    /* Layer conv3_3 tile size 5*116 */
    /* Temp vars and parameters for input layer conv3_3 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g8_conv3_3_in=7392,g8_conv3_3_inIterator=7510;
    ISA_ExtAddress g8_conv3_3_available = 132;
    ISA_ExtAddress g8_conv3_3_outOffset=0;
    unsigned g8_conv3_3_iterationsLeft=3136;
    moveFilterCount128(bs, ISA_Mem_Dram, conv3_3_MRF, ISA_Mem_MatrixRf, 85, 1, 36);
    vRead1D(bs, ISA_Mem_Dram, conv3_3_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 8);
    /* Layer pool3 tile size 7*112 */
    /* Temp vars and parameters for input layer pool3 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g9_pool3_in=7972,g9_pool3_inIterator=7972;
    ISA_ExtAddress g9_pool3_available = 0;
    ISA_ExtAddress g9_pool3_accumulators=2246;
    ISA_ExtAddress g9_pool3_availableVerticalRows=0;
    ISA_ExtAddress g9_pool3_outOffset=0;
    unsigned g9_pool3_iterationsLeft=784;
    moveFilterCount128(bs, ISA_Mem_Dram, pool3_MRF, ISA_Mem_MatrixRf, 121, 1, 1);
    vRead2D(bs, ISA_Mem_Expander, input_1_inIndex, 2, maxReadSize, 2);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_1_in, 2);
    vRead2D(bs, ISA_Mem_Expander, input_1_inIndex, 2, maxReadSize, 2);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_1_next, 2);
    g0_conv1_1_loadLeft -= 2 * maxReadSize;
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 226, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_in, 1);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 2, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_in+226, 226);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 2, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_in+451, 226);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 114, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_in, 1);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_in+114, 114);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_in+227, 114);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 114, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_in, 1);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_in+114, 114);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_in+227, 114);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 58, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_in, 1);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_in+58, 58);
    vRead2D(bs, ISA_Mem_Dram, zeros, 1, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_in+115, 58);
    vRead2D(bs, ISA_Mem_Dram, zeros, 2, 58, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_in, 2);
    vRead2D(bs, ISA_Mem_Dram, zeros, 2, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_in+116, 116);
    vRead2D(bs, ISA_Mem_Dram, zeros, 2, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_in+230, 116);
    vRead2D(bs, ISA_Mem_Dram, zeros, 2, 58, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_in, 2);
    vRead2D(bs, ISA_Mem_Dram, zeros, 2, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_in+116, 116);
    vRead2D(bs, ISA_Mem_Dram, zeros, 2, 4, 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_in+230, 116);
    /* Loop until we've read all outputs */
    while (g9_pool3_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_conv1_1_iterationsLeft>0) {

            /* Prefetch activations for the next iteration of the loop to hide latency */
            if (g0_conv1_1_available==0) {
                /* This is complicated in order to ensure that iterations%channels = 0 or this is the last transfer */
                /* swap buffers, then fetch next */
                g0_conv1_1_tmp=g0_conv1_1_in; g0_conv1_1_in=g0_conv1_1_next; g0_conv1_1_next=g0_conv1_1_tmp;
                g0_conv1_1_inIterator = g0_conv1_1_in;
                g0_conv1_1_available = g0_conv1_1_next_available;
                if (g0_conv1_1_loadLeft > 0) {
                    if (g0_conv1_1_loadLeft > maxReadSize) {
                        g0_conv1_1_next_available = maxReadSize;
                    } else {
                        g0_conv1_1_next_available = g0_conv1_1_loadLeft;
                    }
                    vRead2D(bs, ISA_Mem_Expander, input_1_inIndex, 2, g0_conv1_1_next_available, 2);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_1_next, 2);
                    g0_conv1_1_loadLeft -= g0_conv1_1_next_available;
                }
            }

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_conv1_1_available <= 112, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            if (g0_conv1_1_available > 0) {

                /* Start of layer 0 in group 0 (conv1_1) */
                /* Tile size 1*224 dimPerStep 2 */
                ISA_NativeCount toCompute=g0_conv1_1_available;
                if ((g0_conv1_1_outOffset + toCompute) >= 225) {
                    toCompute = 225 - g0_conv1_1_outOffset;
                }
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_conv1_1_inIterator, 2, toCompute, 2);
                mv_mul(bs, mrf_start+0);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0, 0); /* includes: conv1_1: bias */
                v_relu(bs); /* includes: ReLU1_1: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 1568+g0_conv1_1_outOffset+g0_conv1_1_outRowStart+0, 1);
                /* Advance the write pointer */
                g0_conv1_1_outOffset += toCompute;
                if (g0_conv1_1_outOffset == 225) {
                    g0_conv1_1_outOffset = 1;
                    if (g0_conv1_1_outRowStart==452) {
                        g0_conv1_1_outRowStart=0;
                    } else {
                        g0_conv1_1_outRowStart+=226;
                    }
                }
                g1_conv1_2_available += toCompute*1;
                g0_conv1_1_inIterator += toCompute*2 /* LHS is in native vectors; RHS is in activations */;
                g0_conv1_1_available -= toCompute;
                g0_conv1_1_iterationsLeft-=toCompute;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g0_conv1_1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        } else {
            /* pad for the next group */
            g1_conv1_2_available+=224;
            vRead1D(bs, ISA_Mem_Dram, zeros, 224);
            v_wr(bs, ISA_Mem_MvmInitialVrf, 1568+g0_conv1_1_outRowStart+1);
            if (g0_conv1_1_outRowStart==452) {
                g0_conv1_1_outRowStart=0;
            } else {
                g0_conv1_1_outRowStart+=226;
            }
        }

        /* Start of group 1 */
        if (g1_conv1_2_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g1_conv1_2_available <= 678, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            while (g1_conv1_2_available >= 678) {

                /* Start of layer 0 in group 1 (conv1_2) */
                /* Tile size 3*226 dimPerStep 1 */
                ISA_ExtAddress tmpOffset;
                g1_conv1_2_inIterator = g1_conv1_2_in;
                /* Start of kernel row 0 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+0*1, 1, 224, 1);
                mv_mul(bs, mrf_start+2);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+1*1, 1, 224, 1);
                mv_mul(bs, mrf_start+3);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+2*1, 1, 224, 1);
                mv_mul(bs, mrf_start+4);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* advance to the next row */
                if (g1_conv1_2_inIterator>=2020) {
                    g1_conv1_2_inIterator -= 452;
                } else {
                    g1_conv1_2_inIterator += 226;
                }
                /* End of kernel row 0 */
                /* Start of kernel row 1 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+0*1, 1, 224, 1);
                mv_mul(bs, mrf_start+5);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+1*1, 1, 224, 1);
                mv_mul(bs, mrf_start+6);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+2*1, 1, 224, 1);
                mv_mul(bs, mrf_start+7);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* advance to the next row */
                if (g1_conv1_2_inIterator>=2020) {
                    g1_conv1_2_inIterator -= 452;
                } else {
                    g1_conv1_2_inIterator += 226;
                }
                /* End of kernel row 1 */
                /* Start of kernel row 2 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+0*1, 1, 224, 1);
                mv_mul(bs, mrf_start+8);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+1*1, 1, 224, 1);
                mv_mul(bs, mrf_start+9);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv1_2_inIterator+2*1, 1, 224, 1);
                mv_mul(bs, mrf_start+10);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 1, 0); /* includes: conv1_2: bias */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_relu(bs); /* includes: ReLU1_2: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 3030+g1_conv1_2_outOffset+0, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 10532+g1_conv1_2_outOffset+0, 1);
                /* End of kernel row 2 */
                /* Advance the read pointer for the next step */
                if (g1_conv1_2_in>=2020) {
                    g1_conv1_2_in -= 452;
                } else {
                    g1_conv1_2_in += 226;
                }
                /* Advance the write pointer */
                if (g1_conv1_2_outOffset == 1344) {
                    g1_conv1_2_outOffset = 0;
                } else {
                    g1_conv1_2_outOffset += 224;
                }
                g2_pool1_available+=224;
                g1_conv1_2_iterationsLeft-=224;
                g1_conv1_2_available-=224;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g1_conv1_2_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        }

        /* Start of group 2 */
        if (g2_pool1_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g2_pool1_available <= 1568, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            if ((g2_pool1_available >= 1568) || ((g1_conv1_2_iterationsLeft==0))) {

                /* Start of layer 0 in group 2 (pool1) */
                /* Tile size 7*224 dimPerStep 1 */
                /* Decompose the MAX-pool into sequences of 1 horizontal pool operations followed by vertical pool operations */
                /* At the moment, all intermediate data is saved to both MVM_IVRF and AddSubVrf1 */
                /* Perform 112 horizontal pool operations on up to  7 rows with 1 steps */
                /* g2_pool1_inIterator iterates horizontal pool operations (INPUTS) */
                /* g2_pool1_in iterates vertical pool operations (OUTPUTS) */
                /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
                unsigned horizontalRows=0;
                unsigned verticalRows=0;
                /* Count vertical and horizontal rows */
                for (unsigned items=g2_pool1_availableVerticalRows*224;(items<g2_pool1_available);items+=224) {
                    horizontalRows++;
                    g2_pool1_availableVerticalRows++;
                    if (g2_pool1_availableVerticalRows >= 2) {
                        verticalRows++;
                        g2_pool1_availableVerticalRows-=2;
                    }
                }
                g2_pool1_available -= verticalRows*448;
                ISA_ExtAddress curOffset;
                curOffset=g2_pool1_inIterator;
                for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 112, 2);
                    mv_mul(bs, mrf_start+11);
                    /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                    vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-3030+10532+1, 2);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g2_pool1_accumulators+rowIterator*112, 1);
                    if (curOffset>=4374) {
                        curOffset-=1344;
                    } else {
                        curOffset+=224;
                    }
                }
                /* Horizontal sweep must end up in g2_pool1_accumulators because we can't read-modify-write ASVRF in a single chain */
                curOffset=g2_pool1_inIterator;
                for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g2_pool1_accumulators+rowIterator*112, 1, 112, 1);
                    mv_mul(bs, mrf_start+11);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+10532-3030, 1);
                    if (curOffset>=4374) {
                        curOffset-=1344;
                    } else {
                        curOffset+=224;
                    }
                }
                /* Update horizontal pool iterator start */
                g2_pool1_inIterator = curOffset-0;
                curOffset=g2_pool1_in;
                ISA_ExtAddress nextOffset=curOffset;
                if (nextOffset>=4374) {
                    nextOffset-=1344;
                } else {
                    nextOffset+=224;
                }
                for(unsigned rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 112, 1);
                    mv_mul(bs, mrf_start+11);
                    /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                    vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-3030+10532, 1);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 4598 + g2_pool1_outOffset, 1);
                    if (curOffset>=4150) {
                        curOffset-=1120;
                    } else {
                        curOffset+=448;
                    }
                    if (nextOffset>=4150) {
                        nextOffset-=1120;
                    } else {
                        nextOffset+=448;
                    }
                    if (g2_pool1_outOffset == 457) {
                        g2_pool1_outOffset = 1;
                    } else {
                        g2_pool1_outOffset += 114;
                    }
                }
                g2_pool1_in = curOffset;
                g3_conv2_1_available+=verticalRows*112;
                g2_pool1_iterationsLeft-=verticalRows*112;
                /* Make sure we didn't loop too many times (emulator only) */
                Emulator_HEX_Assert(g2_pool1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        } else {
            /* pad for the next group */
            g3_conv2_1_available+=112;
            vRead1D(bs, ISA_Mem_Dram, zeros, 112);
            v_wr(bs, ISA_Mem_MvmInitialVrf, 4598+g2_pool1_outOffset);
            if (g2_pool1_outOffset == 457) {
                g2_pool1_outOffset = 1;
            } else {
                g2_pool1_outOffset += 114;
            }
        }

        /* Start of group 3 */
        if (g3_conv2_1_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g3_conv2_1_available <= 570, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            while (g3_conv2_1_available >= 342) {

                /* Start of layer 0 in group 3 (conv2_1) */
                /* Tile size 5*114 dimPerStep 1 */
                ISA_ExtAddress tmpOffset;
                g3_conv2_1_inIterator = g3_conv2_1_in;
                /* Start of kernel row 0 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+0*1, 1, 112, 1);
                mv_mul(bs, mrf_start+12);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+1*1, 1, 112, 1);
                mv_mul(bs, mrf_start+13);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+2*1, 1, 112, 1);
                mv_mul(bs, mrf_start+14);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* advance to the next row */
                if (g3_conv2_1_inIterator>=5054) {
                    g3_conv2_1_inIterator -= 456;
                } else {
                    g3_conv2_1_inIterator += 114;
                }
                /* End of kernel row 0 */
                /* Start of kernel row 1 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+0*1, 1, 112, 1);
                mv_mul(bs, mrf_start+15);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+1*1, 1, 112, 1);
                mv_mul(bs, mrf_start+16);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+2*1, 1, 112, 1);
                mv_mul(bs, mrf_start+17);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* advance to the next row */
                if (g3_conv2_1_inIterator>=5054) {
                    g3_conv2_1_inIterator -= 456;
                } else {
                    g3_conv2_1_inIterator += 114;
                }
                /* End of kernel row 1 */
                /* Start of kernel row 2 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+0*1, 1, 112, 1);
                mv_mul(bs, mrf_start+18);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+1*1, 1, 112, 1);
                mv_mul(bs, mrf_start+19);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g3_conv2_1_inIterator+2*1, 1, 112, 1);
                mv_mul(bs, mrf_start+20);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 2, 0); /* includes: conv2_1: bias */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_relu(bs); /* includes: ReLU2_1: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 5168+g3_conv2_1_outOffset+0, 1);
                /* End of kernel row 2 */
                /* Advance the read pointer for the next step */
                if (g3_conv2_1_in>=5054) {
                    g3_conv2_1_in -= 456;
                } else {
                    g3_conv2_1_in += 114;
                }
                /* Advance the write pointer */
                if (g3_conv2_1_outOffset == 457) {
                    g3_conv2_1_outOffset = 1;
                } else {
                    g3_conv2_1_outOffset += 114;
                }
                g4_conv2_2_available+=112;
                g3_conv2_1_iterationsLeft-=112;
                g3_conv2_1_available-=112;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g3_conv2_1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        } else {
            /* pad for the next group */
            g4_conv2_2_available+=112;
            vRead1D(bs, ISA_Mem_Dram, zeros, 112);
            v_wr(bs, ISA_Mem_MvmInitialVrf, 5168+g3_conv2_1_outOffset);
            if (g3_conv2_1_outOffset == 457) {
                g3_conv2_1_outOffset = 1;
            } else {
                g3_conv2_1_outOffset += 114;
            }
        }

        /* Start of group 4 */
        if (g4_conv2_2_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g4_conv2_2_available <= 570, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            while (g4_conv2_2_available >= 342) {

                /* Start of layer 0 in group 4 (conv2_2) */
                /* Tile size 5*114 dimPerStep 1 */
                ISA_ExtAddress tmpOffset;
                g4_conv2_2_inIterator = g4_conv2_2_in;
                /* Start of kernel row 0 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+0*1, 1, 112, 1);
                mv_mul(bs, mrf_start+21);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+1*1, 1, 112, 1);
                mv_mul(bs, mrf_start+22);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+2*1, 1, 112, 1);
                mv_mul(bs, mrf_start+23);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* advance to the next row */
                if (g4_conv2_2_inIterator>=5624) {
                    g4_conv2_2_inIterator -= 456;
                } else {
                    g4_conv2_2_inIterator += 114;
                }
                /* End of kernel row 0 */
                /* Start of kernel row 1 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+0*1, 1, 112, 1);
                mv_mul(bs, mrf_start+24);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+1*1, 1, 112, 1);
                mv_mul(bs, mrf_start+25);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+2*1, 1, 112, 1);
                mv_mul(bs, mrf_start+26);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* advance to the next row */
                if (g4_conv2_2_inIterator>=5624) {
                    g4_conv2_2_inIterator -= 456;
                } else {
                    g4_conv2_2_inIterator += 114;
                }
                /* End of kernel row 1 */
                /* Start of kernel row 2 */
                /* Start of kernel col 0 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+0*1, 1, 112, 1);
                mv_mul(bs, mrf_start+27);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 1 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+1*1, 1, 112, 1);
                mv_mul(bs, mrf_start+28);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                /* swap tmp variables */
                tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                /* Start of kernel col 2 */
                tmpOffset = 0;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g4_conv2_2_inIterator+2*1, 1, 112, 1);
                mv_mul(bs, mrf_start+29);
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 3, 0); /* includes: conv2_2: bias */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                v_relu(bs); /* includes: ReLU2_2: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 5738+g4_conv2_2_outOffset+0, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 9748+g4_conv2_2_outOffset+0, 1);
                /* End of kernel row 2 */
                /* Advance the read pointer for the next step */
                if (g4_conv2_2_in>=5624) {
                    g4_conv2_2_in -= 456;
                } else {
                    g4_conv2_2_in += 114;
                }
                /* Advance the write pointer */
                if (g4_conv2_2_outOffset == 672) {
                    g4_conv2_2_outOffset = 0;
                } else {
                    g4_conv2_2_outOffset += 112;
                }
                g5_pool2_available+=112;
                g4_conv2_2_iterationsLeft-=112;
                g4_conv2_2_available-=112;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g4_conv2_2_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        }

        /* Start of group 5 */
        if (g5_pool2_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g5_pool2_available <= 784, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            if ((g5_pool2_available >= 560) || ((g4_conv2_2_iterationsLeft==0))) {

                /* Start of layer 0 in group 5 (pool2) */
                /* Tile size 7*112 dimPerStep 1 */
                /* Decompose the MAX-pool into sequences of 1 horizontal pool operations followed by vertical pool operations */
                /* At the moment, all intermediate data is saved to both MVM_IVRF and AddSubVrf1 */
                /* Perform 56 horizontal pool operations on up to  7 rows with 1 steps */
                /* g5_pool2_inIterator iterates horizontal pool operations (INPUTS) */
                /* g5_pool2_in iterates vertical pool operations (OUTPUTS) */
                /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
                unsigned horizontalRows=0;
                unsigned verticalRows=0;
                /* Count vertical and horizontal rows */
                for (unsigned items=g5_pool2_availableVerticalRows*112;(items<g5_pool2_available);items+=112) {
                    horizontalRows++;
                    g5_pool2_availableVerticalRows++;
                    if (g5_pool2_availableVerticalRows >= 2) {
                        verticalRows++;
                        g5_pool2_availableVerticalRows-=2;
                    }
                }
                g5_pool2_available -= verticalRows*224;
                ISA_ExtAddress curOffset;
                curOffset=g5_pool2_inIterator;
                for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 56, 2);
                    mv_mul(bs, mrf_start+30);
                    /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                    vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-5738+9748+1, 2);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g5_pool2_accumulators+rowIterator*56, 1);
                    if (curOffset>=6410) {
                        curOffset-=672;
                    } else {
                        curOffset+=112;
                    }
                }
                /* Horizontal sweep must end up in g5_pool2_accumulators because we can't read-modify-write ASVRF in a single chain */
                curOffset=g5_pool2_inIterator;
                for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g5_pool2_accumulators+rowIterator*56, 1, 56, 1);
                    mv_mul(bs, mrf_start+30);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+9748-5738, 1);
                    if (curOffset>=6410) {
                        curOffset-=672;
                    } else {
                        curOffset+=112;
                    }
                }
                /* Update horizontal pool iterator start */
                g5_pool2_inIterator = curOffset-0;
                curOffset=g5_pool2_in;
                ISA_ExtAddress nextOffset=curOffset;
                if (nextOffset>=6410) {
                    nextOffset-=672;
                } else {
                    nextOffset+=112;
                }
                for(unsigned rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 56, 1);
                    mv_mul(bs, mrf_start+30);
                    /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                    vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-5738+9748, 1);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6522 + g5_pool2_outOffset, 1);
                    if (curOffset>=6298) {
                        curOffset-=560;
                    } else {
                        curOffset+=224;
                    }
                    if (nextOffset>=6298) {
                        nextOffset-=560;
                    } else {
                        nextOffset+=224;
                    }
                    if (g5_pool2_outOffset == 233) {
                        g5_pool2_outOffset = 1;
                    } else {
                        g5_pool2_outOffset += 58;
                    }
                }
                g5_pool2_in = curOffset;
                g6_conv3_1_available+=verticalRows*56;
                g5_pool2_iterationsLeft-=verticalRows*56;
                /* Make sure we didn't loop too many times (emulator only) */
                Emulator_HEX_Assert(g5_pool2_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        } else {
            /* pad for the next group */
            g6_conv3_1_available+=56;
            vRead1D(bs, ISA_Mem_Dram, zeros, 56);
            v_wr(bs, ISA_Mem_MvmInitialVrf, 6522+g5_pool2_outOffset);
            if (g5_pool2_outOffset == 233) {
                g5_pool2_outOffset = 1;
            } else {
                g5_pool2_outOffset += 58;
            }
        }

        /* Start of group 6 */
        if (g6_conv3_1_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g6_conv3_1_available <= 290, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            while (g6_conv3_1_available >= 174) {

                /* Start of layer 0 in group 6 (conv3_1) */
                /* Tile size 5*58 dimPerStep 1 */
                for(unsigned outRow=0;outRow<2;outRow++) {
                    ISA_ExtAddress tmpOffset;
                    g6_conv3_1_inIterator = g6_conv3_1_in;
                    /* Start of kernel row 0 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+0*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+31+outRow*9);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+1*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+32+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+2*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+33+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* advance to the next row */
                    if (g6_conv3_1_inIterator>=6754) {
                        g6_conv3_1_inIterator -= 232;
                    } else {
                        g6_conv3_1_inIterator += 58;
                    }
                    /* End of kernel row 0 */
                    /* Start of kernel row 1 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+0*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+34+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+1*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+35+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+2*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+36+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* advance to the next row */
                    if (g6_conv3_1_inIterator>=6754) {
                        g6_conv3_1_inIterator -= 232;
                    } else {
                        g6_conv3_1_inIterator += 58;
                    }
                    /* End of kernel row 1 */
                    /* Start of kernel row 2 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+0*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+37+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+1*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+38+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 1);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g6_conv3_1_inIterator+2*1, 1, 56, 1);
                    mv_mul(bs, mrf_start+39+outRow*9);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 4+outRow, 0); /* includes: conv3_1: bias */
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 1);
                    v_relu(bs); /* includes: ReLU3_1: v_relu */
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6812+g6_conv3_1_outOffset+outRow+0, 2);
                    /* End of kernel row 2 */
                }
                /* Advance the read pointer for the next step */
                if (g6_conv3_1_in>=6754) {
                    g6_conv3_1_in -= 232;
                } else {
                    g6_conv3_1_in += 58;
                }
                /* Advance the write pointer */
                if (g6_conv3_1_outOffset == 466) {
                    g6_conv3_1_outOffset = 2;
                } else {
                    g6_conv3_1_outOffset += 116;
                }
                g7_conv3_2_available+=112;
                g6_conv3_1_iterationsLeft-=56;
                g6_conv3_1_available-=56;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g6_conv3_1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        } else {
            /* pad for the next group */
            g7_conv3_2_available+=112;
            vRead1D(bs, ISA_Mem_Dram, zeros, 112);
            v_wr(bs, ISA_Mem_MvmInitialVrf, 6812+g6_conv3_1_outOffset);
            if (g6_conv3_1_outOffset == 466) {
                g6_conv3_1_outOffset = 2;
            } else {
                g6_conv3_1_outOffset += 116;
            }
        }

        /* Start of group 7 */
        if (g7_conv3_2_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g7_conv3_2_available <= 580, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            while (g7_conv3_2_available >= 348) {

                /* Start of layer 0 in group 7 (conv3_2) */
                /* Tile size 5*116 dimPerStep 2 */
                for(unsigned outRow=0;outRow<2;outRow++) {
                    ISA_ExtAddress tmpOffset;
                    g7_conv3_2_inIterator = g7_conv3_2_in;
                    /* Start of kernel row 0 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+0*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+49+outRow*18);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+1*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+51+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+2*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+53+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* advance to the next row */
                    if (g7_conv3_2_inIterator>=7276) {
                        g7_conv3_2_inIterator -= 464;
                    } else {
                        g7_conv3_2_inIterator += 116;
                    }
                    /* End of kernel row 0 */
                    /* Start of kernel row 1 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+0*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+55+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+1*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+57+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+2*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+59+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* advance to the next row */
                    if (g7_conv3_2_inIterator>=7276) {
                        g7_conv3_2_inIterator -= 464;
                    } else {
                        g7_conv3_2_inIterator += 116;
                    }
                    /* End of kernel row 1 */
                    /* Start of kernel row 2 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+0*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+61+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+1*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+63+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g7_conv3_2_inIterator+2*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+65+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 6+outRow, 0); /* includes: conv3_2: bias */
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_relu(bs); /* includes: ReLU3_2: v_relu */
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7392+g7_conv3_2_outOffset+outRow+0, 2);
                    /* End of kernel row 2 */
                }
                /* Advance the read pointer for the next step */
                if (g7_conv3_2_in>=7276) {
                    g7_conv3_2_in -= 464;
                } else {
                    g7_conv3_2_in += 116;
                }
                /* Advance the write pointer */
                if (g7_conv3_2_outOffset == 466) {
                    g7_conv3_2_outOffset = 2;
                } else {
                    g7_conv3_2_outOffset += 116;
                }
                g8_conv3_3_available+=112;
                g7_conv3_2_iterationsLeft-=56;
                g7_conv3_2_available-=112;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g7_conv3_2_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        } else {
            /* pad for the next group */
            g8_conv3_3_available+=112;
            vRead1D(bs, ISA_Mem_Dram, zeros, 112);
            v_wr(bs, ISA_Mem_MvmInitialVrf, 7392+g7_conv3_2_outOffset);
            if (g7_conv3_2_outOffset == 466) {
                g7_conv3_2_outOffset = 2;
            } else {
                g7_conv3_2_outOffset += 116;
            }
        }

        /* Start of group 8 */
        if (g8_conv3_3_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g8_conv3_3_available <= 580, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            while (g8_conv3_3_available >= 348) {

                /* Start of layer 0 in group 8 (conv3_3) */
                /* Tile size 5*116 dimPerStep 2 */
                for(unsigned outRow=0;outRow<2;outRow++) {
                    ISA_ExtAddress tmpOffset;
                    g8_conv3_3_inIterator = g8_conv3_3_in;
                    /* Start of kernel row 0 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+0*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+85+outRow*18);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+1*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+87+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+2*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+89+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* advance to the next row */
                    if (g8_conv3_3_inIterator>=7856) {
                        g8_conv3_3_inIterator -= 464;
                    } else {
                        g8_conv3_3_inIterator += 116;
                    }
                    /* End of kernel row 0 */
                    /* Start of kernel row 1 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+0*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+91+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+1*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+93+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+2*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+95+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* advance to the next row */
                    if (g8_conv3_3_inIterator>=7856) {
                        g8_conv3_3_inIterator -= 464;
                    } else {
                        g8_conv3_3_inIterator += 116;
                    }
                    /* End of kernel row 1 */
                    /* Start of kernel row 2 */
                    /* Start of kernel col 0 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+0*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+97+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 1 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+1*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+99+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1_next+tmpOffset, 2);
                    /* swap tmp variables */
                    tmp_ASVRF1_swap=tmp_ASVRF1; tmp_ASVRF1=tmp_ASVRF1_next; tmp_ASVRF1_next=tmp_ASVRF1_swap;
                    /* Start of kernel col 2 */
                    tmpOffset = 0;
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g8_conv3_3_inIterator+2*2, 2, 56, 2);
                    mv_mul(bs, mrf_start+101+outRow*18);
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 8+outRow, 0); /* includes: conv3_3: bias */
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, tmp_ASVRF1+tmpOffset, 2);
                    v_relu(bs); /* includes: ReLU3_3: v_relu */
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7972+g8_conv3_3_outOffset+outRow+0, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 8964+g8_conv3_3_outOffset+outRow+0, 2);
                    /* End of kernel row 2 */
                }
                /* Advance the read pointer for the next step */
                if (g8_conv3_3_in>=7856) {
                    g8_conv3_3_in -= 464;
                } else {
                    g8_conv3_3_in += 116;
                }
                /* Advance the write pointer */
                if (g8_conv3_3_outOffset == 672) {
                    g8_conv3_3_outOffset = 0;
                } else {
                    g8_conv3_3_outOffset += 112;
                }
                g9_pool3_available+=112;
                g8_conv3_3_iterationsLeft-=56;
                g8_conv3_3_available-=112;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g8_conv3_3_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        }

        /* Start of group 9 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g9_pool3_available <= 784, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if ((g9_pool3_available >= 560) || ((g8_conv3_3_iterationsLeft==0))) {

            /* Start of layer 0 in group 9 (pool3) */
            /* Tile size 7*112 dimPerStep 2 */
            /* Decompose the MAX-pool into sequences of 1 horizontal pool operations followed by vertical pool operations */
            /* At the moment, all intermediate data is saved to both MVM_IVRF and AddSubVrf1 */
            /* Perform 28 horizontal pool operations on up to  7 rows with 1 steps */
            /* g9_pool3_inIterator iterates horizontal pool operations (INPUTS) */
            /* g9_pool3_in iterates vertical pool operations (OUTPUTS) */
            /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
            unsigned horizontalRows=0;
            unsigned verticalRows=0;
            /* Count vertical and horizontal rows */
            for (unsigned items=g9_pool3_availableVerticalRows*112;(items<g9_pool3_available);items+=112) {
                horizontalRows++;
                g9_pool3_availableVerticalRows++;
                if (g9_pool3_availableVerticalRows >= 2) {
                    verticalRows++;
                    g9_pool3_availableVerticalRows-=2;
                }
            }
            g9_pool3_available -= verticalRows*224;
            ISA_ExtAddress curOffset;
            for(unsigned vec=0; vec<2; vec++) {
                curOffset=g9_pool3_inIterator+vec;
                for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 28, 4);
                    mv_mul(bs, mrf_start+121);
                    /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                    vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-7972+8964+2, 4);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g9_pool3_accumulators+rowIterator*28, 1);
                    if (curOffset>=8644) {
                        curOffset-=672;
                    } else {
                        curOffset+=112;
                    }
                }
                /* Horizontal sweep must end up in g9_pool3_accumulators because we can't read-modify-write ASVRF in a single chain */
                curOffset=g9_pool3_inIterator+vec;
                for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g9_pool3_accumulators+rowIterator*28, 1, 28, 1);
                    mv_mul(bs, mrf_start+121);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+8964-7972, 2);
                    if (curOffset>=8644) {
                        curOffset-=672;
                    } else {
                        curOffset+=112;
                    }
                }
            }
            /* Update horizontal pool iterator start */
            g9_pool3_inIterator = curOffset-1;
            curOffset=g9_pool3_in;
            /* The horizontal sweep took care of the multiple input vectors */
            ISA_ExtAddress nextOffset=curOffset;
            if (nextOffset>=8644) {
                nextOffset-=672;
            } else {
                nextOffset+=112;
            }
            for(unsigned rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 56, 1);
                mv_mul(bs, mrf_start+121);
                /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-7972+8964, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + g9_pool3_outOffset, 1);
                g9_pool3_outOffset+=56;
                if (curOffset>=8532) {
                    curOffset-=560;
                } else {
                    curOffset+=224;
                }
                if (nextOffset>=8532) {
                    nextOffset-=560;
                } else {
                    nextOffset+=224;
                }
            }
            g9_pool3_in = curOffset;
            g9_pool3_iterationsLeft-=verticalRows*28;
            /* Make sure we didn't loop too many times (emulator only) */
            Emulator_HEX_Assert(g9_pool3_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void conv41(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_1(d=512, h=28, d=28) = Convolution(pool3(d=256, h=28, w=28), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /*      absorbed ReLU4_1 */

    ISA_ExtAddress pool3_inIndex,conv4_1_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    ISA_VrfAddress conv4_1_param_asvrf_0_cur=0, conv4_1_param_asvrf_0_next=2, conv4_1_param_asvrf_0_tmp;
    moveFilterCount128(bs, ISA_Mem_Dram, conv4_1_MRF+0*36, ISA_Mem_MatrixRf, mrf_start, 1, 36);
    vRead1D(bs, ISA_Mem_Dram, conv4_1_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, conv4_1_param_asvrf_0_cur);
    outChainOffset = 0;
    for(unsigned outRowBlock=0;outRowBlock<2;outRowBlock++) {
        if (outRowBlock!=1) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, conv4_1_MRF+(outRowBlock+1)*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
            vRead1D(bs, ISA_Mem_Dram, conv4_1_bias+(outRowBlock+1)*2, 2);
            v_wr(bs, ISA_Mem_AddSubVrf_0, conv4_1_param_asvrf_0_next);
        } else if (!p_last) {
            /* Prefetch the first part of conv4_2_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, conv4_2_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
        }
        for(unsigned outRow=0;outRow<2;outRow++) {
            pool3_inIndex=0;
            conv4_1_outOffset = 0;
            /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
            /* strided IVRF access mode on */
            vRead3D(bs, ISA_Mem_MvmInitialVrf, pool3_inIndex, 28, 28, 2, 3, 1, 1);
            mv_mul(bs, mrf_start+0+(outRow*18));
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, conv4_1_param_asvrf_0_cur+outRow, 0); /* includes: conv4_1: bias */
            v_relu(bs); /* includes: ReLU4_1: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6163 + conv4_1_outOffset + outChainOffset, 4);
            outChainOffset++;
        }
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        conv4_1_param_asvrf_0_tmp=conv4_1_param_asvrf_0_cur; conv4_1_param_asvrf_0_cur=conv4_1_param_asvrf_0_next; conv4_1_param_asvrf_0_next=conv4_1_param_asvrf_0_tmp;
    }
}

void conv42(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_2(d=512, h=28, d=28) = Convolution(conv4_1(d=512, h=28, w=28), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /*      absorbed ReLU4_2 */

    ISA_ExtAddress conv4_1_inIndex,conv4_2_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    ISA_VrfAddress conv4_2_param_asvrf_0_cur=0, conv4_2_param_asvrf_0_next=1, conv4_2_param_asvrf_0_tmp;
    /* conv4_2_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, conv4_2_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, conv4_2_param_asvrf_0_cur);
    outChainOffset = 0;
    for(unsigned outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, conv4_2_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
            vRead1D(bs, ISA_Mem_Dram, conv4_2_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, conv4_2_param_asvrf_0_next);
        } else if (!p_last) {
            /* Prefetch the first part of conv4_3_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, conv4_3_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
        }
        conv4_1_inIndex=6163;
        conv4_2_outOffset = 0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, conv4_1_inIndex, 28, 28, 4, 3, 1, 1);
        mv_mul(bs, mrf_start+0);
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, conv4_2_param_asvrf_0_cur, 0); /* includes: conv4_2: bias */
        v_relu(bs); /* includes: ReLU4_2: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + conv4_2_outOffset + outChainOffset, 4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        conv4_2_param_asvrf_0_tmp=conv4_2_param_asvrf_0_cur; conv4_2_param_asvrf_0_cur=conv4_2_param_asvrf_0_next; conv4_2_param_asvrf_0_next=conv4_2_param_asvrf_0_tmp;
    }
}

void conv43(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_3(d=512, h=28, d=28) = Convolution(conv4_2(d=512, h=28, w=28), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /*      absorbed ReLU4_3 */

    ISA_ExtAddress conv4_2_inIndex,conv4_3_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    ISA_VrfAddress conv4_3_param_asvrf_0_cur=0, conv4_3_param_asvrf_0_next=1, conv4_3_param_asvrf_0_tmp;
    /* conv4_3_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, conv4_3_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, conv4_3_param_asvrf_0_cur);
    outChainOffset = 0;
    for(unsigned outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, conv4_3_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
            vRead1D(bs, ISA_Mem_Dram, conv4_3_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, conv4_3_param_asvrf_0_next);
        } else if (!p_last) {
            /* Prefetch 16 entries starting at pool4 */
            moveFilterCount128(bs, ISA_Mem_Dram, pool4_MRF+0*1, ISA_Mem_MatrixRf, mrf_next, 1, 1);
        }
        conv4_2_inIndex=0;
        conv4_3_outOffset = 0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, conv4_2_inIndex, 28, 28, 4, 3, 1, 1);
        mv_mul(bs, mrf_start+0);
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, conv4_3_param_asvrf_0_cur, 0); /* includes: conv4_3: bias */
        v_relu(bs); /* includes: ReLU4_3: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6163 + conv4_3_outOffset + outChainOffset, 4);
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0 + conv4_3_outOffset + outChainOffset, 4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        conv4_3_param_asvrf_0_tmp=conv4_3_param_asvrf_0_cur; conv4_3_param_asvrf_0_cur=conv4_3_param_asvrf_0_next; conv4_3_param_asvrf_0_next=conv4_3_param_asvrf_0_tmp;
    }
}

void pool4(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Pool pool4(d=512, h=14, d=14) = MaxPool(conv4_3(d=512, h=28, w=28), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    /* Convolution pool4(d=512, h=14, d=14) = MaxPool(conv4_3(d=512, h=28, w=28), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress conv4_3_inIndex;
    conv4_3_inIndex=6163;
    moveFilterCount128(bs, ISA_Mem_Dram, pool4_MRF+0*1, ISA_Mem_MatrixRf, mrf_start, 1, 1);
    /* Prefetch the first part of conv5_1_MRF */
    moveFilterCount128(bs, ISA_Mem_Dram, conv5_1_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
    ISA_ExtAddress tmp_MVMIVRF=6065, tmp_MVMIVRF_next=6114;
    /* Layer pool4 tile size 28*112 */
    /* Temp vars and parameters for input layer pool4 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_pool4_in=6163,g0_pool4_inIterator=6163;
    ISA_ExtAddress g0_pool4_available = 784;
    ISA_ExtAddress g0_pool4_accumulators=6065;
    ISA_ExtAddress g0_pool4_availableVerticalRows=0;
    ISA_ExtAddress g0_pool4_outOffset=0;
    unsigned g0_pool4_iterationsLeft=196;
    /* Loop until we've read all outputs */
    while (g0_pool4_iterationsLeft>0) {

        /* Start of group 0 */


        /* Start of layer 0 in group 0 (pool4) */
        /* Tile size 28*112 dimPerStep 4 */
        /* Decompose the MAX-pool into 1 horizontal pool operations followed by and 1 vertical pool operations */
        /* At the moment, all intermediate data is saved to both MVM_IVRF and AddSubVrf1 */
        /* Perform 14 horizontal pool operations on 7 or 6 rows with 1 steps */
        /* After the first iteration we all skip horizontal pool operations for 0 rows that were computed by the previous iteration */
        /* The last iteration will perform 3 horizontal pool operations and 2 vertical operations */
        /* g0_pool4_inIterator iterates horizontal pool operations (INPUTS) */
        /* g0_pool4_in iterates vertical pool operations (OUTPUTS) */
        /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
        unsigned horizontalRows=6;
        unsigned verticalRows=3;
        if (g0_pool4_iterationsLeft==196) {
            horizontalRows=7;
        } else if (g0_pool4_iterationsLeft==28) {
            horizontalRows=3;
            verticalRows=2;
        }
        g0_pool4_available -= verticalRows*224;
        ISA_ExtAddress curOffset;
        for(unsigned vec=0; vec<4; vec++) {
            curOffset=g0_pool4_inIterator+vec;
            for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 14, 8);
                mv_mul(bs, mrf_start+0);
                /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-6163+0+4, 8);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_pool4_accumulators+rowIterator*14, 1);
                curOffset+=112;
            }
            /* Horizontal sweep must end up in g0_pool4_accumulators because we can't read-modify-write ASVRF in a single chain */
            curOffset=g0_pool4_inIterator+vec;
            for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_pool4_accumulators+rowIterator*14, 1, 14, 1);
                mv_mul(bs, mrf_start+0);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 4);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+0-6163, 4);
                curOffset+=112;
            }
        }
        /* Update horizontal pool iterator start */
        g0_pool4_inIterator = curOffset-3;
        curOffset=g0_pool4_in;
        /* The horizontal sweep took care of the multiple input vectors */
        ISA_ExtAddress nextOffset=curOffset;
        nextOffset+=112;
        for(unsigned rowIterator=0;rowIterator<verticalRows; rowIterator++) {
            vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 56, 1);
            mv_mul(bs, mrf_start+0);
            /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
            vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-6163+0, 1);
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + g0_pool4_outOffset, 1);
            g0_pool4_outOffset+=56;
            curOffset+=224;
            nextOffset+=224;
        }
        g0_pool4_in = curOffset;
        g0_pool4_iterationsLeft-=verticalRows*14;
        /* Make sure we didn't loop too many times (emulator only) */
        Emulator_HEX_Assert(g0_pool4_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void conv51(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_1(d=512, h=14, d=14) = Convolution(pool4(d=512, h=14, w=14), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /*      absorbed ReLU5_1 */

    ISA_ExtAddress pool4_inIndex,conv5_1_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    ISA_VrfAddress conv5_1_param_asvrf_0_cur=0, conv5_1_param_asvrf_0_next=1, conv5_1_param_asvrf_0_tmp;
    /* conv5_1_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, conv5_1_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, conv5_1_param_asvrf_0_cur);
    outChainOffset = 0;
    for(unsigned outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, conv5_1_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
            vRead1D(bs, ISA_Mem_Dram, conv5_1_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, conv5_1_param_asvrf_0_next);
        } else if (!p_last) {
            /* Prefetch the first part of conv5_2_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, conv5_2_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
        }
        pool4_inIndex=0;
        conv5_1_outOffset = 0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, pool4_inIndex, 14, 14, 4, 3, 1, 1);
        mv_mul(bs, mrf_start+0);
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, conv5_1_param_asvrf_0_cur, 0); /* includes: conv5_1: bias */
        v_relu(bs); /* includes: ReLU5_1: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515 + conv5_1_outOffset + outChainOffset, 4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        conv5_1_param_asvrf_0_tmp=conv5_1_param_asvrf_0_cur; conv5_1_param_asvrf_0_cur=conv5_1_param_asvrf_0_next; conv5_1_param_asvrf_0_next=conv5_1_param_asvrf_0_tmp;
    }
}

void conv52(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_2(d=512, h=14, d=14) = Convolution(conv5_1(d=512, h=14, w=14), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /*      absorbed ReLU5_2 */

    ISA_ExtAddress conv5_1_inIndex,conv5_2_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    ISA_VrfAddress conv5_2_param_asvrf_0_cur=0, conv5_2_param_asvrf_0_next=1, conv5_2_param_asvrf_0_tmp;
    /* conv5_2_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, conv5_2_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, conv5_2_param_asvrf_0_cur);
    outChainOffset = 0;
    for(unsigned outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, conv5_2_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
            vRead1D(bs, ISA_Mem_Dram, conv5_2_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, conv5_2_param_asvrf_0_next);
        } else if (!p_last) {
            /* Prefetch the first part of conv5_3_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, conv5_3_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
        }
        conv5_1_inIndex=8515;
        conv5_2_outOffset = 0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, conv5_1_inIndex, 14, 14, 4, 3, 1, 1);
        mv_mul(bs, mrf_start+0);
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, conv5_2_param_asvrf_0_cur, 0); /* includes: conv5_2: bias */
        v_relu(bs); /* includes: ReLU5_2: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + conv5_2_outOffset + outChainOffset, 4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        conv5_2_param_asvrf_0_tmp=conv5_2_param_asvrf_0_cur; conv5_2_param_asvrf_0_cur=conv5_2_param_asvrf_0_next; conv5_2_param_asvrf_0_next=conv5_2_param_asvrf_0_tmp;
    }
}

void conv53(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_3(d=512, h=14, d=14) = Convolution(conv5_2(d=512, h=14, w=14), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /*      absorbed ReLU5_3 */

    ISA_ExtAddress conv5_2_inIndex,conv5_3_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    ISA_VrfAddress conv5_3_param_asvrf_0_cur=0, conv5_3_param_asvrf_0_next=1, conv5_3_param_asvrf_0_tmp;
    /* conv5_3_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, conv5_3_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, conv5_3_param_asvrf_0_cur);
    outChainOffset = 0;
    for(unsigned outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, conv5_3_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 1, 36);
            vRead1D(bs, ISA_Mem_Dram, conv5_3_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, conv5_3_param_asvrf_0_next);
        } else if (!p_last) {
            /* Prefetch 16 entries starting at block5_pool_MaxPool */
            moveFilterCount128(bs, ISA_Mem_Dram, block5_pool_MaxPool_MRF+0*1, ISA_Mem_MatrixRf, mrf_next, 1, 1);
        }
        conv5_2_inIndex=0;
        conv5_3_outOffset = 0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, conv5_2_inIndex, 14, 14, 4, 3, 1, 1);
        mv_mul(bs, mrf_start+0);
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, conv5_3_param_asvrf_0_cur, 0); /* includes: conv5_3: bias */
        v_relu(bs); /* includes: ReLU5_3: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515 + conv5_3_outOffset + outChainOffset, 4);
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0 + conv5_3_outOffset + outChainOffset, 4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        conv5_3_param_asvrf_0_tmp=conv5_3_param_asvrf_0_cur; conv5_3_param_asvrf_0_cur=conv5_3_param_asvrf_0_next; conv5_3_param_asvrf_0_next=conv5_3_param_asvrf_0_tmp;
    }
}

void block5PoolMaxpool(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Pool block5_pool_MaxPool(d=512, h=7, d=7) = MaxPool(conv5_3(d=512, h=14, w=14), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    /* Convolution block5_pool_MaxPool(d=512, h=7, d=7) = MaxPool(conv5_3(d=512, h=14, w=14), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress conv5_3_inIndex;
    conv5_3_inIndex=8515;
    moveFilterCount128(bs, ISA_Mem_Dram, block5_pool_MaxPool_MRF+0*1, ISA_Mem_MatrixRf, mrf_start, 1, 1);
    ISA_ExtAddress tmp_MVMIVRF=0, tmp_MVMIVRF_next=24;
    /* Layer block5_pool_MaxPool tile size 14*56 */
    /* Temp vars and parameters for input layer block5_pool_MaxPool */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_block5_pool_MaxPool_in=8515,g0_block5_pool_MaxPool_inIterator=8515;
    ISA_ExtAddress g0_block5_pool_MaxPool_available = 196;
    ISA_ExtAddress g0_block5_pool_MaxPool_accumulators=0;
    ISA_ExtAddress g0_block5_pool_MaxPool_availableVerticalRows=0;
    ISA_ExtAddress g0_block5_pool_MaxPool_outOffset=0;
    unsigned g0_block5_pool_MaxPool_iterationsLeft=49;
    /* Loop until we've read all outputs */
    while (g0_block5_pool_MaxPool_iterationsLeft>0) {

        /* Start of group 0 */


        /* Start of layer 0 in group 0 (block5_pool_MaxPool) */
        /* Tile size 14*56 dimPerStep 4 */
        /* Decompose the MAX-pool into 1 horizontal pool operations followed by and 1 vertical pool operations */
        /* At the moment, all intermediate data is saved to both MVM_IVRF and AddSubVrf1 */
        /* Perform 7 horizontal pool operations on 7 or 6 rows with 1 steps */
        /* After the first iteration we all skip horizontal pool operations for 0 rows that were computed by the previous iteration */
        /* The last iteration will perform 1 horizontal pool operations and 1 vertical operations */
        /* g0_block5_pool_MaxPool_inIterator iterates horizontal pool operations (INPUTS) */
        /* g0_block5_pool_MaxPool_in iterates vertical pool operations (OUTPUTS) */
        /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
        unsigned horizontalRows=6;
        unsigned verticalRows=3;
        if (g0_block5_pool_MaxPool_iterationsLeft==49) {
            horizontalRows=7;
        } else if (g0_block5_pool_MaxPool_iterationsLeft==7) {
            horizontalRows=1;
            verticalRows=1;
        }
        g0_block5_pool_MaxPool_available -= verticalRows*112;
        ISA_ExtAddress curOffset;
        for(unsigned vec=0; vec<4; vec++) {
            curOffset=g0_block5_pool_MaxPool_inIterator+vec;
            for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 7, 8);
                mv_mul(bs, mrf_start+0);
                /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-8515+0+4, 8);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_block5_pool_MaxPool_accumulators+rowIterator*7, 1);
                curOffset+=56;
            }
            /* Horizontal sweep must end up in g0_block5_pool_MaxPool_accumulators because we can't read-modify-write ASVRF in a single chain */
            curOffset=g0_block5_pool_MaxPool_inIterator+vec;
            for(unsigned rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_block5_pool_MaxPool_accumulators+rowIterator*7, 1, 7, 1);
                mv_mul(bs, mrf_start+0);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 4);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+0-8515, 4);
                curOffset+=56;
            }
        }
        /* Update horizontal pool iterator start */
        g0_block5_pool_MaxPool_inIterator = curOffset-3;
        curOffset=g0_block5_pool_MaxPool_in;
        /* The horizontal sweep took care of the multiple input vectors */
        ISA_ExtAddress nextOffset=curOffset;
        nextOffset+=56;
        for(unsigned rowIterator=0;rowIterator<verticalRows; rowIterator++) {
            vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 28, 1);
            mv_mul(bs, mrf_start+0);
            /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
            vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-8515+0, 1);
            v_wr(bs, ISA_Mem_NetOutputQ, DONTCARE);
            g0_block5_pool_MaxPool_outOffset+=28;
            curOffset+=112;
            nextOffset+=112;
        }
        g0_block5_pool_MaxPool_in = curOffset;
        g0_block5_pool_MaxPool_iterationsLeft-=verticalRows*7;
        /* Make sure we didn't loop too many times (emulator only) */
        Emulator_HEX_Assert(g0_block5_pool_MaxPool_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
    }
}
