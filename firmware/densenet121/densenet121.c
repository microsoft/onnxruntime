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
static const ISA_ExtAddress zeros_size = 0;
static const ISA_ExtAddress conv1_bn_scale__vv_mul__conv1_scale_scale = 0;
static const ISA_ExtAddress conv1_bn_scale__vv_mul__conv1_scale_scale_size = 1;
static const ISA_ExtAddress conv1_bn_bias__vv_mul__conv1_scale_scale__vv_add__conv1_scale_bias = 1;
static const ISA_ExtAddress conv1_bn_bias__vv_mul__conv1_scale_scale__vv_add__conv1_scale_bias_size = 1;
static const ISA_ExtAddress conv1_MRF = 0;
static const ISA_ExtAddress conv1_MRF_size = 2;
static const ISA_ExtAddress pool1_MRF = 2;
static const ISA_ExtAddress pool1_MRF_size = 1;
static const ISA_ExtAddress conv2_1_x1_bn_scale__vv_mul__conv2_1_x1_scale_scale = 2;
static const ISA_ExtAddress conv2_1_x1_bn_scale__vv_mul__conv2_1_x1_scale_scale_size = 1;
static const ISA_ExtAddress conv2_1_x1_bn_bias__vv_mul__conv2_1_x1_scale_scale__vv_add__conv2_1_x1_scale_bias = 3;
static const ISA_ExtAddress conv2_1_x1_bn_bias__vv_mul__conv2_1_x1_scale_scale__vv_add__conv2_1_x1_scale_bias_size = 1;
static const ISA_ExtAddress dummy_conv_conv2_1_x1_bn_MRF = 3;
static const ISA_ExtAddress dummy_conv_conv2_1_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_1_x2_bn_scale__vv_mul__conv2_1_x2_scale_scale = 4;
static const ISA_ExtAddress conv2_1_x2_bn_scale__vv_mul__conv2_1_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv2_1_x2_bn_bias__vv_mul__conv2_1_x2_scale_scale__vv_add__conv2_1_x2_scale_bias = 5;
static const ISA_ExtAddress conv2_1_x2_bn_bias__vv_mul__conv2_1_x2_scale_scale__vv_add__conv2_1_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv2_1_x1_MRF = 4;
static const ISA_ExtAddress conv2_1_x1_MRF_size = 1;
static const ISA_ExtAddress conv2_1_x2_MRF = 5;
static const ISA_ExtAddress conv2_1_x2_MRF_size = 9;
static const ISA_ExtAddress concat_2_1_MRF = 14;
static const ISA_ExtAddress concat_2_1_MRF_size = 1;
static const ISA_ExtAddress conv2_2_x1_bn_scale__vv_mul__conv2_2_x1_scale_scale = 6;
static const ISA_ExtAddress conv2_2_x1_bn_scale__vv_mul__conv2_2_x1_scale_scale_size = 1;
static const ISA_ExtAddress conv2_2_x1_bn_bias__vv_mul__conv2_2_x1_scale_scale__vv_add__conv2_2_x1_scale_bias = 7;
static const ISA_ExtAddress conv2_2_x1_bn_bias__vv_mul__conv2_2_x1_scale_scale__vv_add__conv2_2_x1_scale_bias_size = 1;
static const ISA_ExtAddress dummy_conv_conv2_2_x1_bn_MRF = 15;
static const ISA_ExtAddress dummy_conv_conv2_2_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_2_x2_bn_scale__vv_mul__conv2_2_x2_scale_scale = 8;
static const ISA_ExtAddress conv2_2_x2_bn_scale__vv_mul__conv2_2_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv2_2_x2_bn_bias__vv_mul__conv2_2_x2_scale_scale__vv_add__conv2_2_x2_scale_bias = 9;
static const ISA_ExtAddress conv2_2_x2_bn_bias__vv_mul__conv2_2_x2_scale_scale__vv_add__conv2_2_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv2_2_x1_MRF = 16;
static const ISA_ExtAddress conv2_2_x1_MRF_size = 1;
static const ISA_ExtAddress conv2_2_x2_MRF = 17;
static const ISA_ExtAddress conv2_2_x2_MRF_size = 9;
static const ISA_ExtAddress concat_2_2_MRF = 26;
static const ISA_ExtAddress concat_2_2_MRF_size = 1;
static const ISA_ExtAddress conv2_3_x1_bn_scale__vv_mul__conv2_3_x1_scale_scale = 10;
static const ISA_ExtAddress conv2_3_x1_bn_scale__vv_mul__conv2_3_x1_scale_scale_size = 1;
static const ISA_ExtAddress conv2_3_x1_bn_bias__vv_mul__conv2_3_x1_scale_scale__vv_add__conv2_3_x1_scale_bias = 11;
static const ISA_ExtAddress conv2_3_x1_bn_bias__vv_mul__conv2_3_x1_scale_scale__vv_add__conv2_3_x1_scale_bias_size = 1;
static const ISA_ExtAddress dummy_conv_conv2_3_x1_bn_MRF = 27;
static const ISA_ExtAddress dummy_conv_conv2_3_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_3_x2_bn_scale__vv_mul__conv2_3_x2_scale_scale = 12;
static const ISA_ExtAddress conv2_3_x2_bn_scale__vv_mul__conv2_3_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv2_3_x2_bn_bias__vv_mul__conv2_3_x2_scale_scale__vv_add__conv2_3_x2_scale_bias = 13;
static const ISA_ExtAddress conv2_3_x2_bn_bias__vv_mul__conv2_3_x2_scale_scale__vv_add__conv2_3_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv2_3_x1_MRF = 28;
static const ISA_ExtAddress conv2_3_x1_MRF_size = 1;
static const ISA_ExtAddress conv2_3_x2_MRF = 29;
static const ISA_ExtAddress conv2_3_x2_MRF_size = 9;
static const ISA_ExtAddress concat_2_3_MRF = 38;
static const ISA_ExtAddress concat_2_3_MRF_size = 1;
static const ISA_ExtAddress conv2_4_x1_bn_scale__vv_mul__conv2_4_x1_scale_scale = 14;
static const ISA_ExtAddress conv2_4_x1_bn_scale__vv_mul__conv2_4_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv2_4_x1_bn_bias__vv_mul__conv2_4_x1_scale_scale__vv_add__conv2_4_x1_scale_bias = 16;
static const ISA_ExtAddress conv2_4_x1_bn_bias__vv_mul__conv2_4_x1_scale_scale__vv_add__conv2_4_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv2_4_x1_bn_MRF = 39;
static const ISA_ExtAddress dummy_conv_conv2_4_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_4_x2_bn_scale__vv_mul__conv2_4_x2_scale_scale = 18;
static const ISA_ExtAddress conv2_4_x2_bn_scale__vv_mul__conv2_4_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv2_4_x2_bn_bias__vv_mul__conv2_4_x2_scale_scale__vv_add__conv2_4_x2_scale_bias = 19;
static const ISA_ExtAddress conv2_4_x2_bn_bias__vv_mul__conv2_4_x2_scale_scale__vv_add__conv2_4_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv2_4_x1_MRF = 40;
static const ISA_ExtAddress conv2_4_x1_MRF_size = 2;
static const ISA_ExtAddress conv2_4_x2_MRF = 42;
static const ISA_ExtAddress conv2_4_x2_MRF_size = 9;
static const ISA_ExtAddress concat_2_4_MRF = 51;
static const ISA_ExtAddress concat_2_4_MRF_size = 1;
static const ISA_ExtAddress conv2_5_x1_bn_scale__vv_mul__conv2_5_x1_scale_scale = 20;
static const ISA_ExtAddress conv2_5_x1_bn_scale__vv_mul__conv2_5_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv2_5_x1_bn_bias__vv_mul__conv2_5_x1_scale_scale__vv_add__conv2_5_x1_scale_bias = 22;
static const ISA_ExtAddress conv2_5_x1_bn_bias__vv_mul__conv2_5_x1_scale_scale__vv_add__conv2_5_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv2_5_x1_bn_MRF = 52;
static const ISA_ExtAddress dummy_conv_conv2_5_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_5_x2_bn_scale__vv_mul__conv2_5_x2_scale_scale = 24;
static const ISA_ExtAddress conv2_5_x2_bn_scale__vv_mul__conv2_5_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv2_5_x2_bn_bias__vv_mul__conv2_5_x2_scale_scale__vv_add__conv2_5_x2_scale_bias = 25;
static const ISA_ExtAddress conv2_5_x2_bn_bias__vv_mul__conv2_5_x2_scale_scale__vv_add__conv2_5_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv2_5_x1_MRF = 53;
static const ISA_ExtAddress conv2_5_x1_MRF_size = 2;
static const ISA_ExtAddress conv2_5_x2_MRF = 55;
static const ISA_ExtAddress conv2_5_x2_MRF_size = 9;
static const ISA_ExtAddress concat_2_5_MRF = 64;
static const ISA_ExtAddress concat_2_5_MRF_size = 1;
static const ISA_ExtAddress conv2_6_x1_bn_scale__vv_mul__conv2_6_x1_scale_scale = 26;
static const ISA_ExtAddress conv2_6_x1_bn_scale__vv_mul__conv2_6_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv2_6_x1_bn_bias__vv_mul__conv2_6_x1_scale_scale__vv_add__conv2_6_x1_scale_bias = 28;
static const ISA_ExtAddress conv2_6_x1_bn_bias__vv_mul__conv2_6_x1_scale_scale__vv_add__conv2_6_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv2_6_x1_bn_MRF = 65;
static const ISA_ExtAddress dummy_conv_conv2_6_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_6_x2_bn_scale__vv_mul__conv2_6_x2_scale_scale = 30;
static const ISA_ExtAddress conv2_6_x2_bn_scale__vv_mul__conv2_6_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv2_6_x2_bn_bias__vv_mul__conv2_6_x2_scale_scale__vv_add__conv2_6_x2_scale_bias = 31;
static const ISA_ExtAddress conv2_6_x2_bn_bias__vv_mul__conv2_6_x2_scale_scale__vv_add__conv2_6_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv2_6_x1_MRF = 66;
static const ISA_ExtAddress conv2_6_x1_MRF_size = 2;
static const ISA_ExtAddress conv2_6_x2_MRF = 68;
static const ISA_ExtAddress conv2_6_x2_MRF_size = 9;
static const ISA_ExtAddress concat_2_6_MRF = 77;
static const ISA_ExtAddress concat_2_6_MRF_size = 1;
static const ISA_ExtAddress conv2_blk_bn_scale__vv_mul__conv2_blk_scale_scale = 32;
static const ISA_ExtAddress conv2_blk_bn_scale__vv_mul__conv2_blk_scale_scale_size = 2;
static const ISA_ExtAddress conv2_blk_bn_bias__vv_mul__conv2_blk_scale_scale__vv_add__conv2_blk_scale_bias = 34;
static const ISA_ExtAddress conv2_blk_bn_bias__vv_mul__conv2_blk_scale_scale__vv_add__conv2_blk_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv2_blk_bn_MRF = 78;
static const ISA_ExtAddress dummy_conv_conv2_blk_bn_MRF_size = 1;
static const ISA_ExtAddress conv2_blk_MRF = 79;
static const ISA_ExtAddress conv2_blk_MRF_size = 2;
static const ISA_ExtAddress pool2_scale = 36;
static const ISA_ExtAddress pool2_scale_size = 1;
static const ISA_ExtAddress pool2_MRF = 81;
static const ISA_ExtAddress pool2_MRF_size = 1;
static const ISA_ExtAddress conv3_1_x1_bn_scale__vv_mul__conv3_1_x1_scale_scale = 37;
static const ISA_ExtAddress conv3_1_x1_bn_scale__vv_mul__conv3_1_x1_scale_scale_size = 1;
static const ISA_ExtAddress conv3_1_x1_bn_bias__vv_mul__conv3_1_x1_scale_scale__vv_add__conv3_1_x1_scale_bias = 38;
static const ISA_ExtAddress conv3_1_x1_bn_bias__vv_mul__conv3_1_x1_scale_scale__vv_add__conv3_1_x1_scale_bias_size = 1;
static const ISA_ExtAddress dummy_conv_conv3_1_x1_bn_MRF = 82;
static const ISA_ExtAddress dummy_conv_conv3_1_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_1_x2_bn_scale__vv_mul__conv3_1_x2_scale_scale = 39;
static const ISA_ExtAddress conv3_1_x2_bn_scale__vv_mul__conv3_1_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_1_x2_bn_bias__vv_mul__conv3_1_x2_scale_scale__vv_add__conv3_1_x2_scale_bias = 40;
static const ISA_ExtAddress conv3_1_x2_bn_bias__vv_mul__conv3_1_x2_scale_scale__vv_add__conv3_1_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_1_x1_MRF = 83;
static const ISA_ExtAddress conv3_1_x1_MRF_size = 1;
static const ISA_ExtAddress conv3_1_x2_MRF = 84;
static const ISA_ExtAddress conv3_1_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_1_MRF = 93;
static const ISA_ExtAddress concat_3_1_MRF_size = 1;
static const ISA_ExtAddress conv3_2_x1_bn_scale__vv_mul__conv3_2_x1_scale_scale = 41;
static const ISA_ExtAddress conv3_2_x1_bn_scale__vv_mul__conv3_2_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv3_2_x1_bn_bias__vv_mul__conv3_2_x1_scale_scale__vv_add__conv3_2_x1_scale_bias = 43;
static const ISA_ExtAddress conv3_2_x1_bn_bias__vv_mul__conv3_2_x1_scale_scale__vv_add__conv3_2_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv3_2_x1_bn_MRF = 94;
static const ISA_ExtAddress dummy_conv_conv3_2_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_2_x2_bn_scale__vv_mul__conv3_2_x2_scale_scale = 45;
static const ISA_ExtAddress conv3_2_x2_bn_scale__vv_mul__conv3_2_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_2_x2_bn_bias__vv_mul__conv3_2_x2_scale_scale__vv_add__conv3_2_x2_scale_bias = 46;
static const ISA_ExtAddress conv3_2_x2_bn_bias__vv_mul__conv3_2_x2_scale_scale__vv_add__conv3_2_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_2_x1_MRF = 95;
static const ISA_ExtAddress conv3_2_x1_MRF_size = 2;
static const ISA_ExtAddress conv3_2_x2_MRF = 97;
static const ISA_ExtAddress conv3_2_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_2_MRF = 106;
static const ISA_ExtAddress concat_3_2_MRF_size = 1;
static const ISA_ExtAddress conv3_3_x1_bn_scale__vv_mul__conv3_3_x1_scale_scale = 47;
static const ISA_ExtAddress conv3_3_x1_bn_scale__vv_mul__conv3_3_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv3_3_x1_bn_bias__vv_mul__conv3_3_x1_scale_scale__vv_add__conv3_3_x1_scale_bias = 49;
static const ISA_ExtAddress conv3_3_x1_bn_bias__vv_mul__conv3_3_x1_scale_scale__vv_add__conv3_3_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv3_3_x1_bn_MRF = 107;
static const ISA_ExtAddress dummy_conv_conv3_3_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_3_x2_bn_scale__vv_mul__conv3_3_x2_scale_scale = 51;
static const ISA_ExtAddress conv3_3_x2_bn_scale__vv_mul__conv3_3_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_3_x2_bn_bias__vv_mul__conv3_3_x2_scale_scale__vv_add__conv3_3_x2_scale_bias = 52;
static const ISA_ExtAddress conv3_3_x2_bn_bias__vv_mul__conv3_3_x2_scale_scale__vv_add__conv3_3_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_3_x1_MRF = 108;
static const ISA_ExtAddress conv3_3_x1_MRF_size = 2;
static const ISA_ExtAddress conv3_3_x2_MRF = 110;
static const ISA_ExtAddress conv3_3_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_3_MRF = 119;
static const ISA_ExtAddress concat_3_3_MRF_size = 1;
static const ISA_ExtAddress conv3_4_x1_bn_scale__vv_mul__conv3_4_x1_scale_scale = 53;
static const ISA_ExtAddress conv3_4_x1_bn_scale__vv_mul__conv3_4_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv3_4_x1_bn_bias__vv_mul__conv3_4_x1_scale_scale__vv_add__conv3_4_x1_scale_bias = 55;
static const ISA_ExtAddress conv3_4_x1_bn_bias__vv_mul__conv3_4_x1_scale_scale__vv_add__conv3_4_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv3_4_x1_bn_MRF = 120;
static const ISA_ExtAddress dummy_conv_conv3_4_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_4_x2_bn_scale__vv_mul__conv3_4_x2_scale_scale = 57;
static const ISA_ExtAddress conv3_4_x2_bn_scale__vv_mul__conv3_4_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_4_x2_bn_bias__vv_mul__conv3_4_x2_scale_scale__vv_add__conv3_4_x2_scale_bias = 58;
static const ISA_ExtAddress conv3_4_x2_bn_bias__vv_mul__conv3_4_x2_scale_scale__vv_add__conv3_4_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_4_x1_MRF = 121;
static const ISA_ExtAddress conv3_4_x1_MRF_size = 2;
static const ISA_ExtAddress conv3_4_x2_MRF = 123;
static const ISA_ExtAddress conv3_4_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_4_MRF = 132;
static const ISA_ExtAddress concat_3_4_MRF_size = 1;
static const ISA_ExtAddress conv3_5_x1_bn_scale__vv_mul__conv3_5_x1_scale_scale = 59;
static const ISA_ExtAddress conv3_5_x1_bn_scale__vv_mul__conv3_5_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv3_5_x1_bn_bias__vv_mul__conv3_5_x1_scale_scale__vv_add__conv3_5_x1_scale_bias = 61;
static const ISA_ExtAddress conv3_5_x1_bn_bias__vv_mul__conv3_5_x1_scale_scale__vv_add__conv3_5_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv3_5_x1_bn_MRF = 133;
static const ISA_ExtAddress dummy_conv_conv3_5_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_5_x2_bn_scale__vv_mul__conv3_5_x2_scale_scale = 63;
static const ISA_ExtAddress conv3_5_x2_bn_scale__vv_mul__conv3_5_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_5_x2_bn_bias__vv_mul__conv3_5_x2_scale_scale__vv_add__conv3_5_x2_scale_bias = 64;
static const ISA_ExtAddress conv3_5_x2_bn_bias__vv_mul__conv3_5_x2_scale_scale__vv_add__conv3_5_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_5_x1_MRF = 134;
static const ISA_ExtAddress conv3_5_x1_MRF_size = 2;
static const ISA_ExtAddress conv3_5_x2_MRF = 136;
static const ISA_ExtAddress conv3_5_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_5_MRF = 145;
static const ISA_ExtAddress concat_3_5_MRF_size = 1;
static const ISA_ExtAddress conv3_6_x1_bn_scale__vv_mul__conv3_6_x1_scale_scale = 65;
static const ISA_ExtAddress conv3_6_x1_bn_scale__vv_mul__conv3_6_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv3_6_x1_bn_bias__vv_mul__conv3_6_x1_scale_scale__vv_add__conv3_6_x1_scale_bias = 68;
static const ISA_ExtAddress conv3_6_x1_bn_bias__vv_mul__conv3_6_x1_scale_scale__vv_add__conv3_6_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv3_6_x1_bn_MRF = 146;
static const ISA_ExtAddress dummy_conv_conv3_6_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_6_x2_bn_scale__vv_mul__conv3_6_x2_scale_scale = 71;
static const ISA_ExtAddress conv3_6_x2_bn_scale__vv_mul__conv3_6_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_6_x2_bn_bias__vv_mul__conv3_6_x2_scale_scale__vv_add__conv3_6_x2_scale_bias = 72;
static const ISA_ExtAddress conv3_6_x2_bn_bias__vv_mul__conv3_6_x2_scale_scale__vv_add__conv3_6_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_6_x1_MRF = 147;
static const ISA_ExtAddress conv3_6_x1_MRF_size = 3;
static const ISA_ExtAddress conv3_6_x2_MRF = 150;
static const ISA_ExtAddress conv3_6_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_6_MRF = 159;
static const ISA_ExtAddress concat_3_6_MRF_size = 1;
static const ISA_ExtAddress conv3_7_x1_bn_scale__vv_mul__conv3_7_x1_scale_scale = 73;
static const ISA_ExtAddress conv3_7_x1_bn_scale__vv_mul__conv3_7_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv3_7_x1_bn_bias__vv_mul__conv3_7_x1_scale_scale__vv_add__conv3_7_x1_scale_bias = 76;
static const ISA_ExtAddress conv3_7_x1_bn_bias__vv_mul__conv3_7_x1_scale_scale__vv_add__conv3_7_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv3_7_x1_bn_MRF = 160;
static const ISA_ExtAddress dummy_conv_conv3_7_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_7_x2_bn_scale__vv_mul__conv3_7_x2_scale_scale = 79;
static const ISA_ExtAddress conv3_7_x2_bn_scale__vv_mul__conv3_7_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_7_x2_bn_bias__vv_mul__conv3_7_x2_scale_scale__vv_add__conv3_7_x2_scale_bias = 80;
static const ISA_ExtAddress conv3_7_x2_bn_bias__vv_mul__conv3_7_x2_scale_scale__vv_add__conv3_7_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_7_x1_MRF = 161;
static const ISA_ExtAddress conv3_7_x1_MRF_size = 3;
static const ISA_ExtAddress conv3_7_x2_MRF = 164;
static const ISA_ExtAddress conv3_7_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_7_MRF = 173;
static const ISA_ExtAddress concat_3_7_MRF_size = 1;
static const ISA_ExtAddress conv3_8_x1_bn_scale__vv_mul__conv3_8_x1_scale_scale = 81;
static const ISA_ExtAddress conv3_8_x1_bn_scale__vv_mul__conv3_8_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv3_8_x1_bn_bias__vv_mul__conv3_8_x1_scale_scale__vv_add__conv3_8_x1_scale_bias = 84;
static const ISA_ExtAddress conv3_8_x1_bn_bias__vv_mul__conv3_8_x1_scale_scale__vv_add__conv3_8_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv3_8_x1_bn_MRF = 174;
static const ISA_ExtAddress dummy_conv_conv3_8_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_8_x2_bn_scale__vv_mul__conv3_8_x2_scale_scale = 87;
static const ISA_ExtAddress conv3_8_x2_bn_scale__vv_mul__conv3_8_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_8_x2_bn_bias__vv_mul__conv3_8_x2_scale_scale__vv_add__conv3_8_x2_scale_bias = 88;
static const ISA_ExtAddress conv3_8_x2_bn_bias__vv_mul__conv3_8_x2_scale_scale__vv_add__conv3_8_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_8_x1_MRF = 175;
static const ISA_ExtAddress conv3_8_x1_MRF_size = 3;
static const ISA_ExtAddress conv3_8_x2_MRF = 178;
static const ISA_ExtAddress conv3_8_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_8_MRF = 187;
static const ISA_ExtAddress concat_3_8_MRF_size = 1;
static const ISA_ExtAddress conv3_9_x1_bn_scale__vv_mul__conv3_9_x1_scale_scale = 89;
static const ISA_ExtAddress conv3_9_x1_bn_scale__vv_mul__conv3_9_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv3_9_x1_bn_bias__vv_mul__conv3_9_x1_scale_scale__vv_add__conv3_9_x1_scale_bias = 92;
static const ISA_ExtAddress conv3_9_x1_bn_bias__vv_mul__conv3_9_x1_scale_scale__vv_add__conv3_9_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv3_9_x1_bn_MRF = 188;
static const ISA_ExtAddress dummy_conv_conv3_9_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_9_x2_bn_scale__vv_mul__conv3_9_x2_scale_scale = 95;
static const ISA_ExtAddress conv3_9_x2_bn_scale__vv_mul__conv3_9_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_9_x2_bn_bias__vv_mul__conv3_9_x2_scale_scale__vv_add__conv3_9_x2_scale_bias = 96;
static const ISA_ExtAddress conv3_9_x2_bn_bias__vv_mul__conv3_9_x2_scale_scale__vv_add__conv3_9_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_9_x1_MRF = 189;
static const ISA_ExtAddress conv3_9_x1_MRF_size = 3;
static const ISA_ExtAddress conv3_9_x2_MRF = 192;
static const ISA_ExtAddress conv3_9_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_9_MRF = 201;
static const ISA_ExtAddress concat_3_9_MRF_size = 1;
static const ISA_ExtAddress conv3_10_x1_bn_scale__vv_mul__conv3_10_x1_scale_scale = 97;
static const ISA_ExtAddress conv3_10_x1_bn_scale__vv_mul__conv3_10_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv3_10_x1_bn_bias__vv_mul__conv3_10_x1_scale_scale__vv_add__conv3_10_x1_scale_bias = 101;
static const ISA_ExtAddress conv3_10_x1_bn_bias__vv_mul__conv3_10_x1_scale_scale__vv_add__conv3_10_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv3_10_x1_bn_MRF = 202;
static const ISA_ExtAddress dummy_conv_conv3_10_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_10_x2_bn_scale__vv_mul__conv3_10_x2_scale_scale = 105;
static const ISA_ExtAddress conv3_10_x2_bn_scale__vv_mul__conv3_10_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_10_x2_bn_bias__vv_mul__conv3_10_x2_scale_scale__vv_add__conv3_10_x2_scale_bias = 106;
static const ISA_ExtAddress conv3_10_x2_bn_bias__vv_mul__conv3_10_x2_scale_scale__vv_add__conv3_10_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_10_x1_MRF = 203;
static const ISA_ExtAddress conv3_10_x1_MRF_size = 4;
static const ISA_ExtAddress conv3_10_x2_MRF = 207;
static const ISA_ExtAddress conv3_10_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_10_MRF = 216;
static const ISA_ExtAddress concat_3_10_MRF_size = 1;
static const ISA_ExtAddress conv3_11_x1_bn_scale__vv_mul__conv3_11_x1_scale_scale = 107;
static const ISA_ExtAddress conv3_11_x1_bn_scale__vv_mul__conv3_11_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv3_11_x1_bn_bias__vv_mul__conv3_11_x1_scale_scale__vv_add__conv3_11_x1_scale_bias = 111;
static const ISA_ExtAddress conv3_11_x1_bn_bias__vv_mul__conv3_11_x1_scale_scale__vv_add__conv3_11_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv3_11_x1_bn_MRF = 217;
static const ISA_ExtAddress dummy_conv_conv3_11_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_11_x2_bn_scale__vv_mul__conv3_11_x2_scale_scale = 115;
static const ISA_ExtAddress conv3_11_x2_bn_scale__vv_mul__conv3_11_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_11_x2_bn_bias__vv_mul__conv3_11_x2_scale_scale__vv_add__conv3_11_x2_scale_bias = 116;
static const ISA_ExtAddress conv3_11_x2_bn_bias__vv_mul__conv3_11_x2_scale_scale__vv_add__conv3_11_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_11_x1_MRF = 218;
static const ISA_ExtAddress conv3_11_x1_MRF_size = 4;
static const ISA_ExtAddress conv3_11_x2_MRF = 222;
static const ISA_ExtAddress conv3_11_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_11_MRF = 231;
static const ISA_ExtAddress concat_3_11_MRF_size = 1;
static const ISA_ExtAddress conv3_12_x1_bn_scale__vv_mul__conv3_12_x1_scale_scale = 117;
static const ISA_ExtAddress conv3_12_x1_bn_scale__vv_mul__conv3_12_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv3_12_x1_bn_bias__vv_mul__conv3_12_x1_scale_scale__vv_add__conv3_12_x1_scale_bias = 121;
static const ISA_ExtAddress conv3_12_x1_bn_bias__vv_mul__conv3_12_x1_scale_scale__vv_add__conv3_12_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv3_12_x1_bn_MRF = 232;
static const ISA_ExtAddress dummy_conv_conv3_12_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_12_x2_bn_scale__vv_mul__conv3_12_x2_scale_scale = 125;
static const ISA_ExtAddress conv3_12_x2_bn_scale__vv_mul__conv3_12_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv3_12_x2_bn_bias__vv_mul__conv3_12_x2_scale_scale__vv_add__conv3_12_x2_scale_bias = 126;
static const ISA_ExtAddress conv3_12_x2_bn_bias__vv_mul__conv3_12_x2_scale_scale__vv_add__conv3_12_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv3_12_x1_MRF = 233;
static const ISA_ExtAddress conv3_12_x1_MRF_size = 4;
static const ISA_ExtAddress conv3_12_x2_MRF = 237;
static const ISA_ExtAddress conv3_12_x2_MRF_size = 9;
static const ISA_ExtAddress concat_3_12_MRF = 246;
static const ISA_ExtAddress concat_3_12_MRF_size = 1;
static const ISA_ExtAddress conv3_blk_bn_scale__vv_mul__conv3_blk_scale_scale = 127;
static const ISA_ExtAddress conv3_blk_bn_scale__vv_mul__conv3_blk_scale_scale_size = 4;
static const ISA_ExtAddress conv3_blk_bn_bias__vv_mul__conv3_blk_scale_scale__vv_add__conv3_blk_scale_bias = 131;
static const ISA_ExtAddress conv3_blk_bn_bias__vv_mul__conv3_blk_scale_scale__vv_add__conv3_blk_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv3_blk_bn_MRF = 247;
static const ISA_ExtAddress dummy_conv_conv3_blk_bn_MRF_size = 1;
static const ISA_ExtAddress conv3_blk_MRF = 248;
static const ISA_ExtAddress conv3_blk_MRF_size = 8;
static const ISA_ExtAddress pool3_scale = 135;
static const ISA_ExtAddress pool3_scale_size = 2;
static const ISA_ExtAddress pool3_MRF = 256;
static const ISA_ExtAddress pool3_MRF_size = 1;
static const ISA_ExtAddress conv4_1_x1_bn_scale__vv_mul__conv4_1_x1_scale_scale = 137;
static const ISA_ExtAddress conv4_1_x1_bn_scale__vv_mul__conv4_1_x1_scale_scale_size = 2;
static const ISA_ExtAddress conv4_1_x1_bn_bias__vv_mul__conv4_1_x1_scale_scale__vv_add__conv4_1_x1_scale_bias = 139;
static const ISA_ExtAddress conv4_1_x1_bn_bias__vv_mul__conv4_1_x1_scale_scale__vv_add__conv4_1_x1_scale_bias_size = 2;
static const ISA_ExtAddress dummy_conv_conv4_1_x1_bn_MRF = 257;
static const ISA_ExtAddress dummy_conv_conv4_1_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_1_x2_bn_scale__vv_mul__conv4_1_x2_scale_scale = 141;
static const ISA_ExtAddress conv4_1_x2_bn_scale__vv_mul__conv4_1_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_1_x2_bn_bias__vv_mul__conv4_1_x2_scale_scale__vv_add__conv4_1_x2_scale_bias = 142;
static const ISA_ExtAddress conv4_1_x2_bn_bias__vv_mul__conv4_1_x2_scale_scale__vv_add__conv4_1_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_1_x1_MRF = 258;
static const ISA_ExtAddress conv4_1_x1_MRF_size = 2;
static const ISA_ExtAddress conv4_1_x2_MRF = 260;
static const ISA_ExtAddress conv4_1_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_1_MRF = 269;
static const ISA_ExtAddress concat_4_1_MRF_size = 1;
static const ISA_ExtAddress conv4_2_x1_bn_scale__vv_mul__conv4_2_x1_scale_scale = 143;
static const ISA_ExtAddress conv4_2_x1_bn_scale__vv_mul__conv4_2_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv4_2_x1_bn_bias__vv_mul__conv4_2_x1_scale_scale__vv_add__conv4_2_x1_scale_bias = 146;
static const ISA_ExtAddress conv4_2_x1_bn_bias__vv_mul__conv4_2_x1_scale_scale__vv_add__conv4_2_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv4_2_x1_bn_MRF = 270;
static const ISA_ExtAddress dummy_conv_conv4_2_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_2_x2_bn_scale__vv_mul__conv4_2_x2_scale_scale = 149;
static const ISA_ExtAddress conv4_2_x2_bn_scale__vv_mul__conv4_2_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_2_x2_bn_bias__vv_mul__conv4_2_x2_scale_scale__vv_add__conv4_2_x2_scale_bias = 150;
static const ISA_ExtAddress conv4_2_x2_bn_bias__vv_mul__conv4_2_x2_scale_scale__vv_add__conv4_2_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_2_x1_MRF = 271;
static const ISA_ExtAddress conv4_2_x1_MRF_size = 3;
static const ISA_ExtAddress conv4_2_x2_MRF = 274;
static const ISA_ExtAddress conv4_2_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_2_MRF = 283;
static const ISA_ExtAddress concat_4_2_MRF_size = 1;
static const ISA_ExtAddress conv4_3_x1_bn_scale__vv_mul__conv4_3_x1_scale_scale = 151;
static const ISA_ExtAddress conv4_3_x1_bn_scale__vv_mul__conv4_3_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv4_3_x1_bn_bias__vv_mul__conv4_3_x1_scale_scale__vv_add__conv4_3_x1_scale_bias = 154;
static const ISA_ExtAddress conv4_3_x1_bn_bias__vv_mul__conv4_3_x1_scale_scale__vv_add__conv4_3_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv4_3_x1_bn_MRF = 284;
static const ISA_ExtAddress dummy_conv_conv4_3_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_3_x2_bn_scale__vv_mul__conv4_3_x2_scale_scale = 157;
static const ISA_ExtAddress conv4_3_x2_bn_scale__vv_mul__conv4_3_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_3_x2_bn_bias__vv_mul__conv4_3_x2_scale_scale__vv_add__conv4_3_x2_scale_bias = 158;
static const ISA_ExtAddress conv4_3_x2_bn_bias__vv_mul__conv4_3_x2_scale_scale__vv_add__conv4_3_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_3_x1_MRF = 285;
static const ISA_ExtAddress conv4_3_x1_MRF_size = 3;
static const ISA_ExtAddress conv4_3_x2_MRF = 288;
static const ISA_ExtAddress conv4_3_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_3_MRF = 297;
static const ISA_ExtAddress concat_4_3_MRF_size = 1;
static const ISA_ExtAddress conv4_4_x1_bn_scale__vv_mul__conv4_4_x1_scale_scale = 159;
static const ISA_ExtAddress conv4_4_x1_bn_scale__vv_mul__conv4_4_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv4_4_x1_bn_bias__vv_mul__conv4_4_x1_scale_scale__vv_add__conv4_4_x1_scale_bias = 162;
static const ISA_ExtAddress conv4_4_x1_bn_bias__vv_mul__conv4_4_x1_scale_scale__vv_add__conv4_4_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv4_4_x1_bn_MRF = 298;
static const ISA_ExtAddress dummy_conv_conv4_4_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_4_x2_bn_scale__vv_mul__conv4_4_x2_scale_scale = 165;
static const ISA_ExtAddress conv4_4_x2_bn_scale__vv_mul__conv4_4_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_4_x2_bn_bias__vv_mul__conv4_4_x2_scale_scale__vv_add__conv4_4_x2_scale_bias = 166;
static const ISA_ExtAddress conv4_4_x2_bn_bias__vv_mul__conv4_4_x2_scale_scale__vv_add__conv4_4_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_4_x1_MRF = 299;
static const ISA_ExtAddress conv4_4_x1_MRF_size = 3;
static const ISA_ExtAddress conv4_4_x2_MRF = 302;
static const ISA_ExtAddress conv4_4_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_4_MRF = 311;
static const ISA_ExtAddress concat_4_4_MRF_size = 1;
static const ISA_ExtAddress conv4_5_x1_bn_scale__vv_mul__conv4_5_x1_scale_scale = 167;
static const ISA_ExtAddress conv4_5_x1_bn_scale__vv_mul__conv4_5_x1_scale_scale_size = 3;
static const ISA_ExtAddress conv4_5_x1_bn_bias__vv_mul__conv4_5_x1_scale_scale__vv_add__conv4_5_x1_scale_bias = 170;
static const ISA_ExtAddress conv4_5_x1_bn_bias__vv_mul__conv4_5_x1_scale_scale__vv_add__conv4_5_x1_scale_bias_size = 3;
static const ISA_ExtAddress dummy_conv_conv4_5_x1_bn_MRF = 312;
static const ISA_ExtAddress dummy_conv_conv4_5_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_5_x2_bn_scale__vv_mul__conv4_5_x2_scale_scale = 173;
static const ISA_ExtAddress conv4_5_x2_bn_scale__vv_mul__conv4_5_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_5_x2_bn_bias__vv_mul__conv4_5_x2_scale_scale__vv_add__conv4_5_x2_scale_bias = 174;
static const ISA_ExtAddress conv4_5_x2_bn_bias__vv_mul__conv4_5_x2_scale_scale__vv_add__conv4_5_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_5_x1_MRF = 313;
static const ISA_ExtAddress conv4_5_x1_MRF_size = 3;
static const ISA_ExtAddress conv4_5_x2_MRF = 316;
static const ISA_ExtAddress conv4_5_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_5_MRF = 325;
static const ISA_ExtAddress concat_4_5_MRF_size = 1;
static const ISA_ExtAddress conv4_6_x1_bn_scale__vv_mul__conv4_6_x1_scale_scale = 175;
static const ISA_ExtAddress conv4_6_x1_bn_scale__vv_mul__conv4_6_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv4_6_x1_bn_bias__vv_mul__conv4_6_x1_scale_scale__vv_add__conv4_6_x1_scale_bias = 179;
static const ISA_ExtAddress conv4_6_x1_bn_bias__vv_mul__conv4_6_x1_scale_scale__vv_add__conv4_6_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv4_6_x1_bn_MRF = 326;
static const ISA_ExtAddress dummy_conv_conv4_6_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_6_x2_bn_scale__vv_mul__conv4_6_x2_scale_scale = 183;
static const ISA_ExtAddress conv4_6_x2_bn_scale__vv_mul__conv4_6_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_6_x2_bn_bias__vv_mul__conv4_6_x2_scale_scale__vv_add__conv4_6_x2_scale_bias = 184;
static const ISA_ExtAddress conv4_6_x2_bn_bias__vv_mul__conv4_6_x2_scale_scale__vv_add__conv4_6_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_6_x1_MRF = 327;
static const ISA_ExtAddress conv4_6_x1_MRF_size = 4;
static const ISA_ExtAddress conv4_6_x2_MRF = 331;
static const ISA_ExtAddress conv4_6_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_6_MRF = 340;
static const ISA_ExtAddress concat_4_6_MRF_size = 1;
static const ISA_ExtAddress conv4_7_x1_bn_scale__vv_mul__conv4_7_x1_scale_scale = 185;
static const ISA_ExtAddress conv4_7_x1_bn_scale__vv_mul__conv4_7_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv4_7_x1_bn_bias__vv_mul__conv4_7_x1_scale_scale__vv_add__conv4_7_x1_scale_bias = 189;
static const ISA_ExtAddress conv4_7_x1_bn_bias__vv_mul__conv4_7_x1_scale_scale__vv_add__conv4_7_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv4_7_x1_bn_MRF = 341;
static const ISA_ExtAddress dummy_conv_conv4_7_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_7_x2_bn_scale__vv_mul__conv4_7_x2_scale_scale = 193;
static const ISA_ExtAddress conv4_7_x2_bn_scale__vv_mul__conv4_7_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_7_x2_bn_bias__vv_mul__conv4_7_x2_scale_scale__vv_add__conv4_7_x2_scale_bias = 194;
static const ISA_ExtAddress conv4_7_x2_bn_bias__vv_mul__conv4_7_x2_scale_scale__vv_add__conv4_7_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_7_x1_MRF = 342;
static const ISA_ExtAddress conv4_7_x1_MRF_size = 4;
static const ISA_ExtAddress conv4_7_x2_MRF = 346;
static const ISA_ExtAddress conv4_7_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_7_MRF = 355;
static const ISA_ExtAddress concat_4_7_MRF_size = 1;
static const ISA_ExtAddress conv4_8_x1_bn_scale__vv_mul__conv4_8_x1_scale_scale = 195;
static const ISA_ExtAddress conv4_8_x1_bn_scale__vv_mul__conv4_8_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv4_8_x1_bn_bias__vv_mul__conv4_8_x1_scale_scale__vv_add__conv4_8_x1_scale_bias = 199;
static const ISA_ExtAddress conv4_8_x1_bn_bias__vv_mul__conv4_8_x1_scale_scale__vv_add__conv4_8_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv4_8_x1_bn_MRF = 356;
static const ISA_ExtAddress dummy_conv_conv4_8_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_8_x2_bn_scale__vv_mul__conv4_8_x2_scale_scale = 203;
static const ISA_ExtAddress conv4_8_x2_bn_scale__vv_mul__conv4_8_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_8_x2_bn_bias__vv_mul__conv4_8_x2_scale_scale__vv_add__conv4_8_x2_scale_bias = 204;
static const ISA_ExtAddress conv4_8_x2_bn_bias__vv_mul__conv4_8_x2_scale_scale__vv_add__conv4_8_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_8_x1_MRF = 357;
static const ISA_ExtAddress conv4_8_x1_MRF_size = 4;
static const ISA_ExtAddress conv4_8_x2_MRF = 361;
static const ISA_ExtAddress conv4_8_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_8_MRF = 370;
static const ISA_ExtAddress concat_4_8_MRF_size = 1;
static const ISA_ExtAddress conv4_9_x1_bn_scale__vv_mul__conv4_9_x1_scale_scale = 205;
static const ISA_ExtAddress conv4_9_x1_bn_scale__vv_mul__conv4_9_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv4_9_x1_bn_bias__vv_mul__conv4_9_x1_scale_scale__vv_add__conv4_9_x1_scale_bias = 209;
static const ISA_ExtAddress conv4_9_x1_bn_bias__vv_mul__conv4_9_x1_scale_scale__vv_add__conv4_9_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv4_9_x1_bn_MRF = 371;
static const ISA_ExtAddress dummy_conv_conv4_9_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_9_x2_bn_scale__vv_mul__conv4_9_x2_scale_scale = 213;
static const ISA_ExtAddress conv4_9_x2_bn_scale__vv_mul__conv4_9_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_9_x2_bn_bias__vv_mul__conv4_9_x2_scale_scale__vv_add__conv4_9_x2_scale_bias = 214;
static const ISA_ExtAddress conv4_9_x2_bn_bias__vv_mul__conv4_9_x2_scale_scale__vv_add__conv4_9_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_9_x1_MRF = 372;
static const ISA_ExtAddress conv4_9_x1_MRF_size = 4;
static const ISA_ExtAddress conv4_9_x2_MRF = 376;
static const ISA_ExtAddress conv4_9_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_9_MRF = 385;
static const ISA_ExtAddress concat_4_9_MRF_size = 1;
static const ISA_ExtAddress conv4_10_x1_bn_scale__vv_mul__conv4_10_x1_scale_scale = 215;
static const ISA_ExtAddress conv4_10_x1_bn_scale__vv_mul__conv4_10_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv4_10_x1_bn_bias__vv_mul__conv4_10_x1_scale_scale__vv_add__conv4_10_x1_scale_bias = 220;
static const ISA_ExtAddress conv4_10_x1_bn_bias__vv_mul__conv4_10_x1_scale_scale__vv_add__conv4_10_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv4_10_x1_bn_MRF = 386;
static const ISA_ExtAddress dummy_conv_conv4_10_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_10_x2_bn_scale__vv_mul__conv4_10_x2_scale_scale = 225;
static const ISA_ExtAddress conv4_10_x2_bn_scale__vv_mul__conv4_10_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_10_x2_bn_bias__vv_mul__conv4_10_x2_scale_scale__vv_add__conv4_10_x2_scale_bias = 226;
static const ISA_ExtAddress conv4_10_x2_bn_bias__vv_mul__conv4_10_x2_scale_scale__vv_add__conv4_10_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_10_x1_MRF = 387;
static const ISA_ExtAddress conv4_10_x1_MRF_size = 5;
static const ISA_ExtAddress conv4_10_x2_MRF = 392;
static const ISA_ExtAddress conv4_10_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_10_MRF = 401;
static const ISA_ExtAddress concat_4_10_MRF_size = 1;
static const ISA_ExtAddress conv4_11_x1_bn_scale__vv_mul__conv4_11_x1_scale_scale = 227;
static const ISA_ExtAddress conv4_11_x1_bn_scale__vv_mul__conv4_11_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv4_11_x1_bn_bias__vv_mul__conv4_11_x1_scale_scale__vv_add__conv4_11_x1_scale_bias = 232;
static const ISA_ExtAddress conv4_11_x1_bn_bias__vv_mul__conv4_11_x1_scale_scale__vv_add__conv4_11_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv4_11_x1_bn_MRF = 402;
static const ISA_ExtAddress dummy_conv_conv4_11_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_11_x2_bn_scale__vv_mul__conv4_11_x2_scale_scale = 237;
static const ISA_ExtAddress conv4_11_x2_bn_scale__vv_mul__conv4_11_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_11_x2_bn_bias__vv_mul__conv4_11_x2_scale_scale__vv_add__conv4_11_x2_scale_bias = 238;
static const ISA_ExtAddress conv4_11_x2_bn_bias__vv_mul__conv4_11_x2_scale_scale__vv_add__conv4_11_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_11_x1_MRF = 403;
static const ISA_ExtAddress conv4_11_x1_MRF_size = 5;
static const ISA_ExtAddress conv4_11_x2_MRF = 408;
static const ISA_ExtAddress conv4_11_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_11_MRF = 417;
static const ISA_ExtAddress concat_4_11_MRF_size = 1;
static const ISA_ExtAddress conv4_12_x1_bn_scale__vv_mul__conv4_12_x1_scale_scale = 239;
static const ISA_ExtAddress conv4_12_x1_bn_scale__vv_mul__conv4_12_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv4_12_x1_bn_bias__vv_mul__conv4_12_x1_scale_scale__vv_add__conv4_12_x1_scale_bias = 244;
static const ISA_ExtAddress conv4_12_x1_bn_bias__vv_mul__conv4_12_x1_scale_scale__vv_add__conv4_12_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv4_12_x1_bn_MRF = 418;
static const ISA_ExtAddress dummy_conv_conv4_12_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_12_x2_bn_scale__vv_mul__conv4_12_x2_scale_scale = 249;
static const ISA_ExtAddress conv4_12_x2_bn_scale__vv_mul__conv4_12_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_12_x2_bn_bias__vv_mul__conv4_12_x2_scale_scale__vv_add__conv4_12_x2_scale_bias = 250;
static const ISA_ExtAddress conv4_12_x2_bn_bias__vv_mul__conv4_12_x2_scale_scale__vv_add__conv4_12_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_12_x1_MRF = 419;
static const ISA_ExtAddress conv4_12_x1_MRF_size = 5;
static const ISA_ExtAddress conv4_12_x2_MRF = 424;
static const ISA_ExtAddress conv4_12_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_12_MRF = 433;
static const ISA_ExtAddress concat_4_12_MRF_size = 1;
static const ISA_ExtAddress conv4_13_x1_bn_scale__vv_mul__conv4_13_x1_scale_scale = 251;
static const ISA_ExtAddress conv4_13_x1_bn_scale__vv_mul__conv4_13_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv4_13_x1_bn_bias__vv_mul__conv4_13_x1_scale_scale__vv_add__conv4_13_x1_scale_bias = 256;
static const ISA_ExtAddress conv4_13_x1_bn_bias__vv_mul__conv4_13_x1_scale_scale__vv_add__conv4_13_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv4_13_x1_bn_MRF = 434;
static const ISA_ExtAddress dummy_conv_conv4_13_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_13_x2_bn_scale__vv_mul__conv4_13_x2_scale_scale = 261;
static const ISA_ExtAddress conv4_13_x2_bn_scale__vv_mul__conv4_13_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_13_x2_bn_bias__vv_mul__conv4_13_x2_scale_scale__vv_add__conv4_13_x2_scale_bias = 262;
static const ISA_ExtAddress conv4_13_x2_bn_bias__vv_mul__conv4_13_x2_scale_scale__vv_add__conv4_13_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_13_x1_MRF = 435;
static const ISA_ExtAddress conv4_13_x1_MRF_size = 5;
static const ISA_ExtAddress conv4_13_x2_MRF = 440;
static const ISA_ExtAddress conv4_13_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_13_MRF = 449;
static const ISA_ExtAddress concat_4_13_MRF_size = 1;
static const ISA_ExtAddress conv4_14_x1_bn_scale__vv_mul__conv4_14_x1_scale_scale = 263;
static const ISA_ExtAddress conv4_14_x1_bn_scale__vv_mul__conv4_14_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv4_14_x1_bn_bias__vv_mul__conv4_14_x1_scale_scale__vv_add__conv4_14_x1_scale_bias = 269;
static const ISA_ExtAddress conv4_14_x1_bn_bias__vv_mul__conv4_14_x1_scale_scale__vv_add__conv4_14_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv4_14_x1_bn_MRF = 450;
static const ISA_ExtAddress dummy_conv_conv4_14_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_14_x2_bn_scale__vv_mul__conv4_14_x2_scale_scale = 275;
static const ISA_ExtAddress conv4_14_x2_bn_scale__vv_mul__conv4_14_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_14_x2_bn_bias__vv_mul__conv4_14_x2_scale_scale__vv_add__conv4_14_x2_scale_bias = 276;
static const ISA_ExtAddress conv4_14_x2_bn_bias__vv_mul__conv4_14_x2_scale_scale__vv_add__conv4_14_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_14_x1_MRF = 451;
static const ISA_ExtAddress conv4_14_x1_MRF_size = 6;
static const ISA_ExtAddress conv4_14_x2_MRF = 457;
static const ISA_ExtAddress conv4_14_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_14_MRF = 466;
static const ISA_ExtAddress concat_4_14_MRF_size = 1;
static const ISA_ExtAddress conv4_15_x1_bn_scale__vv_mul__conv4_15_x1_scale_scale = 277;
static const ISA_ExtAddress conv4_15_x1_bn_scale__vv_mul__conv4_15_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv4_15_x1_bn_bias__vv_mul__conv4_15_x1_scale_scale__vv_add__conv4_15_x1_scale_bias = 283;
static const ISA_ExtAddress conv4_15_x1_bn_bias__vv_mul__conv4_15_x1_scale_scale__vv_add__conv4_15_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv4_15_x1_bn_MRF = 467;
static const ISA_ExtAddress dummy_conv_conv4_15_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_15_x2_bn_scale__vv_mul__conv4_15_x2_scale_scale = 289;
static const ISA_ExtAddress conv4_15_x2_bn_scale__vv_mul__conv4_15_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_15_x2_bn_bias__vv_mul__conv4_15_x2_scale_scale__vv_add__conv4_15_x2_scale_bias = 290;
static const ISA_ExtAddress conv4_15_x2_bn_bias__vv_mul__conv4_15_x2_scale_scale__vv_add__conv4_15_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_15_x1_MRF = 468;
static const ISA_ExtAddress conv4_15_x1_MRF_size = 6;
static const ISA_ExtAddress conv4_15_x2_MRF = 474;
static const ISA_ExtAddress conv4_15_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_15_MRF = 483;
static const ISA_ExtAddress concat_4_15_MRF_size = 1;
static const ISA_ExtAddress conv4_16_x1_bn_scale__vv_mul__conv4_16_x1_scale_scale = 291;
static const ISA_ExtAddress conv4_16_x1_bn_scale__vv_mul__conv4_16_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv4_16_x1_bn_bias__vv_mul__conv4_16_x1_scale_scale__vv_add__conv4_16_x1_scale_bias = 297;
static const ISA_ExtAddress conv4_16_x1_bn_bias__vv_mul__conv4_16_x1_scale_scale__vv_add__conv4_16_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv4_16_x1_bn_MRF = 484;
static const ISA_ExtAddress dummy_conv_conv4_16_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_16_x2_bn_scale__vv_mul__conv4_16_x2_scale_scale = 303;
static const ISA_ExtAddress conv4_16_x2_bn_scale__vv_mul__conv4_16_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_16_x2_bn_bias__vv_mul__conv4_16_x2_scale_scale__vv_add__conv4_16_x2_scale_bias = 304;
static const ISA_ExtAddress conv4_16_x2_bn_bias__vv_mul__conv4_16_x2_scale_scale__vv_add__conv4_16_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_16_x1_MRF = 485;
static const ISA_ExtAddress conv4_16_x1_MRF_size = 6;
static const ISA_ExtAddress conv4_16_x2_MRF = 491;
static const ISA_ExtAddress conv4_16_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_16_MRF = 500;
static const ISA_ExtAddress concat_4_16_MRF_size = 1;
static const ISA_ExtAddress conv4_17_x1_bn_scale__vv_mul__conv4_17_x1_scale_scale = 305;
static const ISA_ExtAddress conv4_17_x1_bn_scale__vv_mul__conv4_17_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv4_17_x1_bn_bias__vv_mul__conv4_17_x1_scale_scale__vv_add__conv4_17_x1_scale_bias = 311;
static const ISA_ExtAddress conv4_17_x1_bn_bias__vv_mul__conv4_17_x1_scale_scale__vv_add__conv4_17_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv4_17_x1_bn_MRF = 501;
static const ISA_ExtAddress dummy_conv_conv4_17_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_17_x2_bn_scale__vv_mul__conv4_17_x2_scale_scale = 317;
static const ISA_ExtAddress conv4_17_x2_bn_scale__vv_mul__conv4_17_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_17_x2_bn_bias__vv_mul__conv4_17_x2_scale_scale__vv_add__conv4_17_x2_scale_bias = 318;
static const ISA_ExtAddress conv4_17_x2_bn_bias__vv_mul__conv4_17_x2_scale_scale__vv_add__conv4_17_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_17_x1_MRF = 502;
static const ISA_ExtAddress conv4_17_x1_MRF_size = 6;
static const ISA_ExtAddress conv4_17_x2_MRF = 508;
static const ISA_ExtAddress conv4_17_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_17_MRF = 517;
static const ISA_ExtAddress concat_4_17_MRF_size = 1;
static const ISA_ExtAddress conv4_18_x1_bn_scale__vv_mul__conv4_18_x1_scale_scale = 319;
static const ISA_ExtAddress conv4_18_x1_bn_scale__vv_mul__conv4_18_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv4_18_x1_bn_bias__vv_mul__conv4_18_x1_scale_scale__vv_add__conv4_18_x1_scale_bias = 326;
static const ISA_ExtAddress conv4_18_x1_bn_bias__vv_mul__conv4_18_x1_scale_scale__vv_add__conv4_18_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv4_18_x1_bn_MRF = 518;
static const ISA_ExtAddress dummy_conv_conv4_18_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_18_x2_bn_scale__vv_mul__conv4_18_x2_scale_scale = 333;
static const ISA_ExtAddress conv4_18_x2_bn_scale__vv_mul__conv4_18_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_18_x2_bn_bias__vv_mul__conv4_18_x2_scale_scale__vv_add__conv4_18_x2_scale_bias = 334;
static const ISA_ExtAddress conv4_18_x2_bn_bias__vv_mul__conv4_18_x2_scale_scale__vv_add__conv4_18_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_18_x1_MRF = 519;
static const ISA_ExtAddress conv4_18_x1_MRF_size = 7;
static const ISA_ExtAddress conv4_18_x2_MRF = 526;
static const ISA_ExtAddress conv4_18_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_18_MRF = 535;
static const ISA_ExtAddress concat_4_18_MRF_size = 1;
static const ISA_ExtAddress conv4_19_x1_bn_scale__vv_mul__conv4_19_x1_scale_scale = 335;
static const ISA_ExtAddress conv4_19_x1_bn_scale__vv_mul__conv4_19_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv4_19_x1_bn_bias__vv_mul__conv4_19_x1_scale_scale__vv_add__conv4_19_x1_scale_bias = 342;
static const ISA_ExtAddress conv4_19_x1_bn_bias__vv_mul__conv4_19_x1_scale_scale__vv_add__conv4_19_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv4_19_x1_bn_MRF = 536;
static const ISA_ExtAddress dummy_conv_conv4_19_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_19_x2_bn_scale__vv_mul__conv4_19_x2_scale_scale = 349;
static const ISA_ExtAddress conv4_19_x2_bn_scale__vv_mul__conv4_19_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_19_x2_bn_bias__vv_mul__conv4_19_x2_scale_scale__vv_add__conv4_19_x2_scale_bias = 350;
static const ISA_ExtAddress conv4_19_x2_bn_bias__vv_mul__conv4_19_x2_scale_scale__vv_add__conv4_19_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_19_x1_MRF = 537;
static const ISA_ExtAddress conv4_19_x1_MRF_size = 7;
static const ISA_ExtAddress conv4_19_x2_MRF = 544;
static const ISA_ExtAddress conv4_19_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_19_MRF = 553;
static const ISA_ExtAddress concat_4_19_MRF_size = 1;
static const ISA_ExtAddress conv4_20_x1_bn_scale__vv_mul__conv4_20_x1_scale_scale = 351;
static const ISA_ExtAddress conv4_20_x1_bn_scale__vv_mul__conv4_20_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv4_20_x1_bn_bias__vv_mul__conv4_20_x1_scale_scale__vv_add__conv4_20_x1_scale_bias = 358;
static const ISA_ExtAddress conv4_20_x1_bn_bias__vv_mul__conv4_20_x1_scale_scale__vv_add__conv4_20_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv4_20_x1_bn_MRF = 554;
static const ISA_ExtAddress dummy_conv_conv4_20_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_20_x2_bn_scale__vv_mul__conv4_20_x2_scale_scale = 365;
static const ISA_ExtAddress conv4_20_x2_bn_scale__vv_mul__conv4_20_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_20_x2_bn_bias__vv_mul__conv4_20_x2_scale_scale__vv_add__conv4_20_x2_scale_bias = 366;
static const ISA_ExtAddress conv4_20_x2_bn_bias__vv_mul__conv4_20_x2_scale_scale__vv_add__conv4_20_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_20_x1_MRF = 555;
static const ISA_ExtAddress conv4_20_x1_MRF_size = 7;
static const ISA_ExtAddress conv4_20_x2_MRF = 562;
static const ISA_ExtAddress conv4_20_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_20_MRF = 571;
static const ISA_ExtAddress concat_4_20_MRF_size = 1;
static const ISA_ExtAddress conv4_21_x1_bn_scale__vv_mul__conv4_21_x1_scale_scale = 367;
static const ISA_ExtAddress conv4_21_x1_bn_scale__vv_mul__conv4_21_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv4_21_x1_bn_bias__vv_mul__conv4_21_x1_scale_scale__vv_add__conv4_21_x1_scale_bias = 374;
static const ISA_ExtAddress conv4_21_x1_bn_bias__vv_mul__conv4_21_x1_scale_scale__vv_add__conv4_21_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv4_21_x1_bn_MRF = 572;
static const ISA_ExtAddress dummy_conv_conv4_21_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_21_x2_bn_scale__vv_mul__conv4_21_x2_scale_scale = 381;
static const ISA_ExtAddress conv4_21_x2_bn_scale__vv_mul__conv4_21_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_21_x2_bn_bias__vv_mul__conv4_21_x2_scale_scale__vv_add__conv4_21_x2_scale_bias = 382;
static const ISA_ExtAddress conv4_21_x2_bn_bias__vv_mul__conv4_21_x2_scale_scale__vv_add__conv4_21_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_21_x1_MRF = 573;
static const ISA_ExtAddress conv4_21_x1_MRF_size = 7;
static const ISA_ExtAddress conv4_21_x2_MRF = 580;
static const ISA_ExtAddress conv4_21_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_21_MRF = 589;
static const ISA_ExtAddress concat_4_21_MRF_size = 1;
static const ISA_ExtAddress conv4_22_x1_bn_scale__vv_mul__conv4_22_x1_scale_scale = 383;
static const ISA_ExtAddress conv4_22_x1_bn_scale__vv_mul__conv4_22_x1_scale_scale_size = 8;
static const ISA_ExtAddress conv4_22_x1_bn_bias__vv_mul__conv4_22_x1_scale_scale__vv_add__conv4_22_x1_scale_bias = 391;
static const ISA_ExtAddress conv4_22_x1_bn_bias__vv_mul__conv4_22_x1_scale_scale__vv_add__conv4_22_x1_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv4_22_x1_bn_MRF = 590;
static const ISA_ExtAddress dummy_conv_conv4_22_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_22_x2_bn_scale__vv_mul__conv4_22_x2_scale_scale = 399;
static const ISA_ExtAddress conv4_22_x2_bn_scale__vv_mul__conv4_22_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_22_x2_bn_bias__vv_mul__conv4_22_x2_scale_scale__vv_add__conv4_22_x2_scale_bias = 400;
static const ISA_ExtAddress conv4_22_x2_bn_bias__vv_mul__conv4_22_x2_scale_scale__vv_add__conv4_22_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_22_x1_MRF = 591;
static const ISA_ExtAddress conv4_22_x1_MRF_size = 8;
static const ISA_ExtAddress conv4_22_x2_MRF = 599;
static const ISA_ExtAddress conv4_22_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_22_MRF = 608;
static const ISA_ExtAddress concat_4_22_MRF_size = 1;
static const ISA_ExtAddress conv4_23_x1_bn_scale__vv_mul__conv4_23_x1_scale_scale = 401;
static const ISA_ExtAddress conv4_23_x1_bn_scale__vv_mul__conv4_23_x1_scale_scale_size = 8;
static const ISA_ExtAddress conv4_23_x1_bn_bias__vv_mul__conv4_23_x1_scale_scale__vv_add__conv4_23_x1_scale_bias = 409;
static const ISA_ExtAddress conv4_23_x1_bn_bias__vv_mul__conv4_23_x1_scale_scale__vv_add__conv4_23_x1_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv4_23_x1_bn_MRF = 609;
static const ISA_ExtAddress dummy_conv_conv4_23_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_23_x2_bn_scale__vv_mul__conv4_23_x2_scale_scale = 417;
static const ISA_ExtAddress conv4_23_x2_bn_scale__vv_mul__conv4_23_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_23_x2_bn_bias__vv_mul__conv4_23_x2_scale_scale__vv_add__conv4_23_x2_scale_bias = 418;
static const ISA_ExtAddress conv4_23_x2_bn_bias__vv_mul__conv4_23_x2_scale_scale__vv_add__conv4_23_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_23_x1_MRF = 610;
static const ISA_ExtAddress conv4_23_x1_MRF_size = 8;
static const ISA_ExtAddress conv4_23_x2_MRF = 618;
static const ISA_ExtAddress conv4_23_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_23_MRF = 627;
static const ISA_ExtAddress concat_4_23_MRF_size = 1;
static const ISA_ExtAddress conv4_24_x1_bn_scale__vv_mul__conv4_24_x1_scale_scale = 419;
static const ISA_ExtAddress conv4_24_x1_bn_scale__vv_mul__conv4_24_x1_scale_scale_size = 8;
static const ISA_ExtAddress conv4_24_x1_bn_bias__vv_mul__conv4_24_x1_scale_scale__vv_add__conv4_24_x1_scale_bias = 427;
static const ISA_ExtAddress conv4_24_x1_bn_bias__vv_mul__conv4_24_x1_scale_scale__vv_add__conv4_24_x1_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv4_24_x1_bn_MRF = 628;
static const ISA_ExtAddress dummy_conv_conv4_24_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_24_x2_bn_scale__vv_mul__conv4_24_x2_scale_scale = 435;
static const ISA_ExtAddress conv4_24_x2_bn_scale__vv_mul__conv4_24_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv4_24_x2_bn_bias__vv_mul__conv4_24_x2_scale_scale__vv_add__conv4_24_x2_scale_bias = 436;
static const ISA_ExtAddress conv4_24_x2_bn_bias__vv_mul__conv4_24_x2_scale_scale__vv_add__conv4_24_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv4_24_x1_MRF = 629;
static const ISA_ExtAddress conv4_24_x1_MRF_size = 8;
static const ISA_ExtAddress conv4_24_x2_MRF = 637;
static const ISA_ExtAddress conv4_24_x2_MRF_size = 9;
static const ISA_ExtAddress concat_4_24_MRF = 646;
static const ISA_ExtAddress concat_4_24_MRF_size = 1;
static const ISA_ExtAddress conv4_blk_bn_scale__vv_mul__conv4_blk_scale_scale = 437;
static const ISA_ExtAddress conv4_blk_bn_scale__vv_mul__conv4_blk_scale_scale_size = 8;
static const ISA_ExtAddress conv4_blk_bn_bias__vv_mul__conv4_blk_scale_scale__vv_add__conv4_blk_scale_bias = 445;
static const ISA_ExtAddress conv4_blk_bn_bias__vv_mul__conv4_blk_scale_scale__vv_add__conv4_blk_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv4_blk_bn_MRF = 647;
static const ISA_ExtAddress dummy_conv_conv4_blk_bn_MRF_size = 1;
static const ISA_ExtAddress conv4_blk_MRF = 648;
static const ISA_ExtAddress conv4_blk_MRF_size = 32;
static const ISA_ExtAddress pool4_scale = 453;
static const ISA_ExtAddress pool4_scale_size = 4;
static const ISA_ExtAddress pool4_MRF = 680;
static const ISA_ExtAddress pool4_MRF_size = 1;
static const ISA_ExtAddress conv5_1_x1_bn_scale__vv_mul__conv5_1_x1_scale_scale = 457;
static const ISA_ExtAddress conv5_1_x1_bn_scale__vv_mul__conv5_1_x1_scale_scale_size = 4;
static const ISA_ExtAddress conv5_1_x1_bn_bias__vv_mul__conv5_1_x1_scale_scale__vv_add__conv5_1_x1_scale_bias = 461;
static const ISA_ExtAddress conv5_1_x1_bn_bias__vv_mul__conv5_1_x1_scale_scale__vv_add__conv5_1_x1_scale_bias_size = 4;
static const ISA_ExtAddress dummy_conv_conv5_1_x1_bn_MRF = 681;
static const ISA_ExtAddress dummy_conv_conv5_1_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_1_x2_bn_scale__vv_mul__conv5_1_x2_scale_scale = 465;
static const ISA_ExtAddress conv5_1_x2_bn_scale__vv_mul__conv5_1_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_1_x2_bn_bias__vv_mul__conv5_1_x2_scale_scale__vv_add__conv5_1_x2_scale_bias = 466;
static const ISA_ExtAddress conv5_1_x2_bn_bias__vv_mul__conv5_1_x2_scale_scale__vv_add__conv5_1_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_1_x1_MRF = 682;
static const ISA_ExtAddress conv5_1_x1_MRF_size = 4;
static const ISA_ExtAddress conv5_1_x2_MRF = 686;
static const ISA_ExtAddress conv5_1_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_1_MRF = 695;
static const ISA_ExtAddress concat_5_1_MRF_size = 1;
static const ISA_ExtAddress conv5_2_x1_bn_scale__vv_mul__conv5_2_x1_scale_scale = 467;
static const ISA_ExtAddress conv5_2_x1_bn_scale__vv_mul__conv5_2_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv5_2_x1_bn_bias__vv_mul__conv5_2_x1_scale_scale__vv_add__conv5_2_x1_scale_bias = 472;
static const ISA_ExtAddress conv5_2_x1_bn_bias__vv_mul__conv5_2_x1_scale_scale__vv_add__conv5_2_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv5_2_x1_bn_MRF = 696;
static const ISA_ExtAddress dummy_conv_conv5_2_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_2_x2_bn_scale__vv_mul__conv5_2_x2_scale_scale = 477;
static const ISA_ExtAddress conv5_2_x2_bn_scale__vv_mul__conv5_2_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_2_x2_bn_bias__vv_mul__conv5_2_x2_scale_scale__vv_add__conv5_2_x2_scale_bias = 478;
static const ISA_ExtAddress conv5_2_x2_bn_bias__vv_mul__conv5_2_x2_scale_scale__vv_add__conv5_2_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_2_x1_MRF = 697;
static const ISA_ExtAddress conv5_2_x1_MRF_size = 5;
static const ISA_ExtAddress conv5_2_x2_MRF = 702;
static const ISA_ExtAddress conv5_2_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_2_MRF = 711;
static const ISA_ExtAddress concat_5_2_MRF_size = 1;
static const ISA_ExtAddress conv5_3_x1_bn_scale__vv_mul__conv5_3_x1_scale_scale = 479;
static const ISA_ExtAddress conv5_3_x1_bn_scale__vv_mul__conv5_3_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv5_3_x1_bn_bias__vv_mul__conv5_3_x1_scale_scale__vv_add__conv5_3_x1_scale_bias = 484;
static const ISA_ExtAddress conv5_3_x1_bn_bias__vv_mul__conv5_3_x1_scale_scale__vv_add__conv5_3_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv5_3_x1_bn_MRF = 712;
static const ISA_ExtAddress dummy_conv_conv5_3_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_3_x2_bn_scale__vv_mul__conv5_3_x2_scale_scale = 489;
static const ISA_ExtAddress conv5_3_x2_bn_scale__vv_mul__conv5_3_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_3_x2_bn_bias__vv_mul__conv5_3_x2_scale_scale__vv_add__conv5_3_x2_scale_bias = 490;
static const ISA_ExtAddress conv5_3_x2_bn_bias__vv_mul__conv5_3_x2_scale_scale__vv_add__conv5_3_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_3_x1_MRF = 713;
static const ISA_ExtAddress conv5_3_x1_MRF_size = 5;
static const ISA_ExtAddress conv5_3_x2_MRF = 718;
static const ISA_ExtAddress conv5_3_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_3_MRF = 727;
static const ISA_ExtAddress concat_5_3_MRF_size = 1;
static const ISA_ExtAddress conv5_4_x1_bn_scale__vv_mul__conv5_4_x1_scale_scale = 491;
static const ISA_ExtAddress conv5_4_x1_bn_scale__vv_mul__conv5_4_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv5_4_x1_bn_bias__vv_mul__conv5_4_x1_scale_scale__vv_add__conv5_4_x1_scale_bias = 496;
static const ISA_ExtAddress conv5_4_x1_bn_bias__vv_mul__conv5_4_x1_scale_scale__vv_add__conv5_4_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv5_4_x1_bn_MRF = 728;
static const ISA_ExtAddress dummy_conv_conv5_4_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_4_x2_bn_scale__vv_mul__conv5_4_x2_scale_scale = 501;
static const ISA_ExtAddress conv5_4_x2_bn_scale__vv_mul__conv5_4_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_4_x2_bn_bias__vv_mul__conv5_4_x2_scale_scale__vv_add__conv5_4_x2_scale_bias = 502;
static const ISA_ExtAddress conv5_4_x2_bn_bias__vv_mul__conv5_4_x2_scale_scale__vv_add__conv5_4_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_4_x1_MRF = 729;
static const ISA_ExtAddress conv5_4_x1_MRF_size = 5;
static const ISA_ExtAddress conv5_4_x2_MRF = 734;
static const ISA_ExtAddress conv5_4_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_4_MRF = 743;
static const ISA_ExtAddress concat_5_4_MRF_size = 1;
static const ISA_ExtAddress conv5_5_x1_bn_scale__vv_mul__conv5_5_x1_scale_scale = 503;
static const ISA_ExtAddress conv5_5_x1_bn_scale__vv_mul__conv5_5_x1_scale_scale_size = 5;
static const ISA_ExtAddress conv5_5_x1_bn_bias__vv_mul__conv5_5_x1_scale_scale__vv_add__conv5_5_x1_scale_bias = 508;
static const ISA_ExtAddress conv5_5_x1_bn_bias__vv_mul__conv5_5_x1_scale_scale__vv_add__conv5_5_x1_scale_bias_size = 5;
static const ISA_ExtAddress dummy_conv_conv5_5_x1_bn_MRF = 744;
static const ISA_ExtAddress dummy_conv_conv5_5_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_5_x2_bn_scale__vv_mul__conv5_5_x2_scale_scale = 513;
static const ISA_ExtAddress conv5_5_x2_bn_scale__vv_mul__conv5_5_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_5_x2_bn_bias__vv_mul__conv5_5_x2_scale_scale__vv_add__conv5_5_x2_scale_bias = 514;
static const ISA_ExtAddress conv5_5_x2_bn_bias__vv_mul__conv5_5_x2_scale_scale__vv_add__conv5_5_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_5_x1_MRF = 745;
static const ISA_ExtAddress conv5_5_x1_MRF_size = 5;
static const ISA_ExtAddress conv5_5_x2_MRF = 750;
static const ISA_ExtAddress conv5_5_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_5_MRF = 759;
static const ISA_ExtAddress concat_5_5_MRF_size = 1;
static const ISA_ExtAddress conv5_6_x1_bn_scale__vv_mul__conv5_6_x1_scale_scale = 515;
static const ISA_ExtAddress conv5_6_x1_bn_scale__vv_mul__conv5_6_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv5_6_x1_bn_bias__vv_mul__conv5_6_x1_scale_scale__vv_add__conv5_6_x1_scale_bias = 521;
static const ISA_ExtAddress conv5_6_x1_bn_bias__vv_mul__conv5_6_x1_scale_scale__vv_add__conv5_6_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv5_6_x1_bn_MRF = 760;
static const ISA_ExtAddress dummy_conv_conv5_6_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_6_x2_bn_scale__vv_mul__conv5_6_x2_scale_scale = 527;
static const ISA_ExtAddress conv5_6_x2_bn_scale__vv_mul__conv5_6_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_6_x2_bn_bias__vv_mul__conv5_6_x2_scale_scale__vv_add__conv5_6_x2_scale_bias = 528;
static const ISA_ExtAddress conv5_6_x2_bn_bias__vv_mul__conv5_6_x2_scale_scale__vv_add__conv5_6_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_6_x1_MRF = 761;
static const ISA_ExtAddress conv5_6_x1_MRF_size = 6;
static const ISA_ExtAddress conv5_6_x2_MRF = 767;
static const ISA_ExtAddress conv5_6_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_6_MRF = 776;
static const ISA_ExtAddress concat_5_6_MRF_size = 1;
static const ISA_ExtAddress conv5_7_x1_bn_scale__vv_mul__conv5_7_x1_scale_scale = 529;
static const ISA_ExtAddress conv5_7_x1_bn_scale__vv_mul__conv5_7_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv5_7_x1_bn_bias__vv_mul__conv5_7_x1_scale_scale__vv_add__conv5_7_x1_scale_bias = 535;
static const ISA_ExtAddress conv5_7_x1_bn_bias__vv_mul__conv5_7_x1_scale_scale__vv_add__conv5_7_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv5_7_x1_bn_MRF = 777;
static const ISA_ExtAddress dummy_conv_conv5_7_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_7_x2_bn_scale__vv_mul__conv5_7_x2_scale_scale = 541;
static const ISA_ExtAddress conv5_7_x2_bn_scale__vv_mul__conv5_7_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_7_x2_bn_bias__vv_mul__conv5_7_x2_scale_scale__vv_add__conv5_7_x2_scale_bias = 542;
static const ISA_ExtAddress conv5_7_x2_bn_bias__vv_mul__conv5_7_x2_scale_scale__vv_add__conv5_7_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_7_x1_MRF = 778;
static const ISA_ExtAddress conv5_7_x1_MRF_size = 6;
static const ISA_ExtAddress conv5_7_x2_MRF = 784;
static const ISA_ExtAddress conv5_7_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_7_MRF = 793;
static const ISA_ExtAddress concat_5_7_MRF_size = 1;
static const ISA_ExtAddress conv5_8_x1_bn_scale__vv_mul__conv5_8_x1_scale_scale = 543;
static const ISA_ExtAddress conv5_8_x1_bn_scale__vv_mul__conv5_8_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv5_8_x1_bn_bias__vv_mul__conv5_8_x1_scale_scale__vv_add__conv5_8_x1_scale_bias = 549;
static const ISA_ExtAddress conv5_8_x1_bn_bias__vv_mul__conv5_8_x1_scale_scale__vv_add__conv5_8_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv5_8_x1_bn_MRF = 794;
static const ISA_ExtAddress dummy_conv_conv5_8_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_8_x2_bn_scale__vv_mul__conv5_8_x2_scale_scale = 555;
static const ISA_ExtAddress conv5_8_x2_bn_scale__vv_mul__conv5_8_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_8_x2_bn_bias__vv_mul__conv5_8_x2_scale_scale__vv_add__conv5_8_x2_scale_bias = 556;
static const ISA_ExtAddress conv5_8_x2_bn_bias__vv_mul__conv5_8_x2_scale_scale__vv_add__conv5_8_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_8_x1_MRF = 795;
static const ISA_ExtAddress conv5_8_x1_MRF_size = 6;
static const ISA_ExtAddress conv5_8_x2_MRF = 801;
static const ISA_ExtAddress conv5_8_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_8_MRF = 810;
static const ISA_ExtAddress concat_5_8_MRF_size = 1;
static const ISA_ExtAddress conv5_9_x1_bn_scale__vv_mul__conv5_9_x1_scale_scale = 557;
static const ISA_ExtAddress conv5_9_x1_bn_scale__vv_mul__conv5_9_x1_scale_scale_size = 6;
static const ISA_ExtAddress conv5_9_x1_bn_bias__vv_mul__conv5_9_x1_scale_scale__vv_add__conv5_9_x1_scale_bias = 563;
static const ISA_ExtAddress conv5_9_x1_bn_bias__vv_mul__conv5_9_x1_scale_scale__vv_add__conv5_9_x1_scale_bias_size = 6;
static const ISA_ExtAddress dummy_conv_conv5_9_x1_bn_MRF = 811;
static const ISA_ExtAddress dummy_conv_conv5_9_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_9_x2_bn_scale__vv_mul__conv5_9_x2_scale_scale = 569;
static const ISA_ExtAddress conv5_9_x2_bn_scale__vv_mul__conv5_9_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_9_x2_bn_bias__vv_mul__conv5_9_x2_scale_scale__vv_add__conv5_9_x2_scale_bias = 570;
static const ISA_ExtAddress conv5_9_x2_bn_bias__vv_mul__conv5_9_x2_scale_scale__vv_add__conv5_9_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_9_x1_MRF = 812;
static const ISA_ExtAddress conv5_9_x1_MRF_size = 6;
static const ISA_ExtAddress conv5_9_x2_MRF = 818;
static const ISA_ExtAddress conv5_9_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_9_MRF = 827;
static const ISA_ExtAddress concat_5_9_MRF_size = 1;
static const ISA_ExtAddress conv5_10_x1_bn_scale__vv_mul__conv5_10_x1_scale_scale = 571;
static const ISA_ExtAddress conv5_10_x1_bn_scale__vv_mul__conv5_10_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv5_10_x1_bn_bias__vv_mul__conv5_10_x1_scale_scale__vv_add__conv5_10_x1_scale_bias = 578;
static const ISA_ExtAddress conv5_10_x1_bn_bias__vv_mul__conv5_10_x1_scale_scale__vv_add__conv5_10_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv5_10_x1_bn_MRF = 828;
static const ISA_ExtAddress dummy_conv_conv5_10_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_10_x2_bn_scale__vv_mul__conv5_10_x2_scale_scale = 585;
static const ISA_ExtAddress conv5_10_x2_bn_scale__vv_mul__conv5_10_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_10_x2_bn_bias__vv_mul__conv5_10_x2_scale_scale__vv_add__conv5_10_x2_scale_bias = 586;
static const ISA_ExtAddress conv5_10_x2_bn_bias__vv_mul__conv5_10_x2_scale_scale__vv_add__conv5_10_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_10_x1_MRF = 829;
static const ISA_ExtAddress conv5_10_x1_MRF_size = 7;
static const ISA_ExtAddress conv5_10_x2_MRF = 836;
static const ISA_ExtAddress conv5_10_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_10_MRF = 845;
static const ISA_ExtAddress concat_5_10_MRF_size = 1;
static const ISA_ExtAddress conv5_11_x1_bn_scale__vv_mul__conv5_11_x1_scale_scale = 587;
static const ISA_ExtAddress conv5_11_x1_bn_scale__vv_mul__conv5_11_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv5_11_x1_bn_bias__vv_mul__conv5_11_x1_scale_scale__vv_add__conv5_11_x1_scale_bias = 594;
static const ISA_ExtAddress conv5_11_x1_bn_bias__vv_mul__conv5_11_x1_scale_scale__vv_add__conv5_11_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv5_11_x1_bn_MRF = 846;
static const ISA_ExtAddress dummy_conv_conv5_11_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_11_x2_bn_scale__vv_mul__conv5_11_x2_scale_scale = 601;
static const ISA_ExtAddress conv5_11_x2_bn_scale__vv_mul__conv5_11_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_11_x2_bn_bias__vv_mul__conv5_11_x2_scale_scale__vv_add__conv5_11_x2_scale_bias = 602;
static const ISA_ExtAddress conv5_11_x2_bn_bias__vv_mul__conv5_11_x2_scale_scale__vv_add__conv5_11_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_11_x1_MRF = 847;
static const ISA_ExtAddress conv5_11_x1_MRF_size = 7;
static const ISA_ExtAddress conv5_11_x2_MRF = 854;
static const ISA_ExtAddress conv5_11_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_11_MRF = 863;
static const ISA_ExtAddress concat_5_11_MRF_size = 1;
static const ISA_ExtAddress conv5_12_x1_bn_scale__vv_mul__conv5_12_x1_scale_scale = 603;
static const ISA_ExtAddress conv5_12_x1_bn_scale__vv_mul__conv5_12_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv5_12_x1_bn_bias__vv_mul__conv5_12_x1_scale_scale__vv_add__conv5_12_x1_scale_bias = 610;
static const ISA_ExtAddress conv5_12_x1_bn_bias__vv_mul__conv5_12_x1_scale_scale__vv_add__conv5_12_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv5_12_x1_bn_MRF = 864;
static const ISA_ExtAddress dummy_conv_conv5_12_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_12_x2_bn_scale__vv_mul__conv5_12_x2_scale_scale = 617;
static const ISA_ExtAddress conv5_12_x2_bn_scale__vv_mul__conv5_12_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_12_x2_bn_bias__vv_mul__conv5_12_x2_scale_scale__vv_add__conv5_12_x2_scale_bias = 618;
static const ISA_ExtAddress conv5_12_x2_bn_bias__vv_mul__conv5_12_x2_scale_scale__vv_add__conv5_12_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_12_x1_MRF = 865;
static const ISA_ExtAddress conv5_12_x1_MRF_size = 7;
static const ISA_ExtAddress conv5_12_x2_MRF = 872;
static const ISA_ExtAddress conv5_12_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_12_MRF = 881;
static const ISA_ExtAddress concat_5_12_MRF_size = 1;
static const ISA_ExtAddress conv5_13_x1_bn_scale__vv_mul__conv5_13_x1_scale_scale = 619;
static const ISA_ExtAddress conv5_13_x1_bn_scale__vv_mul__conv5_13_x1_scale_scale_size = 7;
static const ISA_ExtAddress conv5_13_x1_bn_bias__vv_mul__conv5_13_x1_scale_scale__vv_add__conv5_13_x1_scale_bias = 626;
static const ISA_ExtAddress conv5_13_x1_bn_bias__vv_mul__conv5_13_x1_scale_scale__vv_add__conv5_13_x1_scale_bias_size = 7;
static const ISA_ExtAddress dummy_conv_conv5_13_x1_bn_MRF = 882;
static const ISA_ExtAddress dummy_conv_conv5_13_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_13_x2_bn_scale__vv_mul__conv5_13_x2_scale_scale = 633;
static const ISA_ExtAddress conv5_13_x2_bn_scale__vv_mul__conv5_13_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_13_x2_bn_bias__vv_mul__conv5_13_x2_scale_scale__vv_add__conv5_13_x2_scale_bias = 634;
static const ISA_ExtAddress conv5_13_x2_bn_bias__vv_mul__conv5_13_x2_scale_scale__vv_add__conv5_13_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_13_x1_MRF = 883;
static const ISA_ExtAddress conv5_13_x1_MRF_size = 7;
static const ISA_ExtAddress conv5_13_x2_MRF = 890;
static const ISA_ExtAddress conv5_13_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_13_MRF = 899;
static const ISA_ExtAddress concat_5_13_MRF_size = 1;
static const ISA_ExtAddress conv5_14_x1_bn_scale__vv_mul__conv5_14_x1_scale_scale = 635;
static const ISA_ExtAddress conv5_14_x1_bn_scale__vv_mul__conv5_14_x1_scale_scale_size = 8;
static const ISA_ExtAddress conv5_14_x1_bn_bias__vv_mul__conv5_14_x1_scale_scale__vv_add__conv5_14_x1_scale_bias = 643;
static const ISA_ExtAddress conv5_14_x1_bn_bias__vv_mul__conv5_14_x1_scale_scale__vv_add__conv5_14_x1_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv5_14_x1_bn_MRF = 900;
static const ISA_ExtAddress dummy_conv_conv5_14_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_14_x2_bn_scale__vv_mul__conv5_14_x2_scale_scale = 651;
static const ISA_ExtAddress conv5_14_x2_bn_scale__vv_mul__conv5_14_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_14_x2_bn_bias__vv_mul__conv5_14_x2_scale_scale__vv_add__conv5_14_x2_scale_bias = 652;
static const ISA_ExtAddress conv5_14_x2_bn_bias__vv_mul__conv5_14_x2_scale_scale__vv_add__conv5_14_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_14_x1_MRF = 901;
static const ISA_ExtAddress conv5_14_x1_MRF_size = 8;
static const ISA_ExtAddress conv5_14_x2_MRF = 909;
static const ISA_ExtAddress conv5_14_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_14_MRF = 918;
static const ISA_ExtAddress concat_5_14_MRF_size = 1;
static const ISA_ExtAddress conv5_15_x1_bn_scale__vv_mul__conv5_15_x1_scale_scale = 653;
static const ISA_ExtAddress conv5_15_x1_bn_scale__vv_mul__conv5_15_x1_scale_scale_size = 8;
static const ISA_ExtAddress conv5_15_x1_bn_bias__vv_mul__conv5_15_x1_scale_scale__vv_add__conv5_15_x1_scale_bias = 661;
static const ISA_ExtAddress conv5_15_x1_bn_bias__vv_mul__conv5_15_x1_scale_scale__vv_add__conv5_15_x1_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv5_15_x1_bn_MRF = 919;
static const ISA_ExtAddress dummy_conv_conv5_15_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_15_x2_bn_scale__vv_mul__conv5_15_x2_scale_scale = 669;
static const ISA_ExtAddress conv5_15_x2_bn_scale__vv_mul__conv5_15_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_15_x2_bn_bias__vv_mul__conv5_15_x2_scale_scale__vv_add__conv5_15_x2_scale_bias = 670;
static const ISA_ExtAddress conv5_15_x2_bn_bias__vv_mul__conv5_15_x2_scale_scale__vv_add__conv5_15_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_15_x1_MRF = 920;
static const ISA_ExtAddress conv5_15_x1_MRF_size = 8;
static const ISA_ExtAddress conv5_15_x2_MRF = 928;
static const ISA_ExtAddress conv5_15_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_15_MRF = 937;
static const ISA_ExtAddress concat_5_15_MRF_size = 1;
static const ISA_ExtAddress conv5_16_x1_bn_scale__vv_mul__conv5_16_x1_scale_scale = 671;
static const ISA_ExtAddress conv5_16_x1_bn_scale__vv_mul__conv5_16_x1_scale_scale_size = 8;
static const ISA_ExtAddress conv5_16_x1_bn_bias__vv_mul__conv5_16_x1_scale_scale__vv_add__conv5_16_x1_scale_bias = 679;
static const ISA_ExtAddress conv5_16_x1_bn_bias__vv_mul__conv5_16_x1_scale_scale__vv_add__conv5_16_x1_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv5_16_x1_bn_MRF = 938;
static const ISA_ExtAddress dummy_conv_conv5_16_x1_bn_MRF_size = 1;
static const ISA_ExtAddress conv5_16_x2_bn_scale__vv_mul__conv5_16_x2_scale_scale = 687;
static const ISA_ExtAddress conv5_16_x2_bn_scale__vv_mul__conv5_16_x2_scale_scale_size = 1;
static const ISA_ExtAddress conv5_16_x2_bn_bias__vv_mul__conv5_16_x2_scale_scale__vv_add__conv5_16_x2_scale_bias = 688;
static const ISA_ExtAddress conv5_16_x2_bn_bias__vv_mul__conv5_16_x2_scale_scale__vv_add__conv5_16_x2_scale_bias_size = 1;
static const ISA_ExtAddress conv5_16_x1_MRF = 939;
static const ISA_ExtAddress conv5_16_x1_MRF_size = 8;
static const ISA_ExtAddress conv5_16_x2_MRF = 947;
static const ISA_ExtAddress conv5_16_x2_MRF_size = 9;
static const ISA_ExtAddress concat_5_16_MRF = 956;
static const ISA_ExtAddress concat_5_16_MRF_size = 1;
static const ISA_ExtAddress conv5_blk_bn_scale__vv_mul__conv5_blk_scale_scale = 689;
static const ISA_ExtAddress conv5_blk_bn_scale__vv_mul__conv5_blk_scale_scale_size = 8;
static const ISA_ExtAddress conv5_blk_bn_bias__vv_mul__conv5_blk_scale_scale__vv_add__conv5_blk_scale_bias = 697;
static const ISA_ExtAddress conv5_blk_bn_bias__vv_mul__conv5_blk_scale_scale__vv_add__conv5_blk_scale_bias_size = 8;
static const ISA_ExtAddress dummy_conv_conv5_blk_bn_MRF = 957;
static const ISA_ExtAddress dummy_conv_conv5_blk_bn_MRF_size = 1;

/* Common variables */
ISA_ExtAddress ivrf_inIterator;
ISA_MrfAddress mrf_start=0, mrf_next=64, mrf_tmp;

/* Layer function prototypes */
void data(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv21X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv21X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv21X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat21(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv22X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv22X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv22X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat22(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv23X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv23X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv23X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat23(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv24X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv24X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat24(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv25X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv25X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat25(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv26X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv26X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat26(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv2BlkBn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv2Blk(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv31X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv31X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv31X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat31(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv32X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv32X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv32X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat32(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv33X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv33X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv33X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat33(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv34X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv34X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv34X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat34(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv35X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv35X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv35X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat35(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv36X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv36X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv36X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat36(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv37X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv37X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv37X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat37(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv38X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv38X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv38X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat38(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv39X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv39X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv39X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat39(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv310X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv310X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv310X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat310(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv311X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv311X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv311X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat311(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv312X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv312X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv312X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat312(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv3BlkBn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv3Blk(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv41X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv41X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv41X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat41(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv42X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv42X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv42X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat42(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv43X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv43X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv43X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat43(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv44X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv44X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv44X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat44(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv45X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv45X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv45X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat45(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv46X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv46X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv46X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat46(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv47X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv47X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv47X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat47(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv48X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv48X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv48X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat48(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv49X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv49X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv49X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat49(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv410X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv410X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv410X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat410(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv411X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv411X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv411X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat411(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv412X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv412X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv412X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat412(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv413X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv413X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv413X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat413(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv414X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv414X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv414X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat414(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv415X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv415X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv415X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat415(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv416X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv416X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv416X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat416(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv417X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv417X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv417X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat417(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv418X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv418X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv418X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat418(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv419X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv419X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv419X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat419(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv420X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv420X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv420X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat420(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv421X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv421X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv421X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat421(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv422X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv422X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv422X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat422(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv423X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv423X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv423X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat423(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv424X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv424X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv424X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat424(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv4BlkBn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv4Blk(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv51X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv51X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv51X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat51(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv52X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv52X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv52X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat52(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv53X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv53X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv53X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat53(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv54X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv54X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv54X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat54(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv55X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv55X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv55X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat55(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv56X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv56X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv56X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat56(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv57X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv57X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv57X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat57(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv58X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv58X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv58X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat58(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv59X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv59X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv59X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat59(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv510X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv510X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv510X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat510(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv511X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv511X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv511X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat511(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv512X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv512X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv512X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat512(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv513X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv513X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv513X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat513(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv514X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv514X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv514X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat514(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv515X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv515X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv515X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat515(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv516X1Bn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv516X1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv516X2(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void concat516(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void dummyConvConv5BlkBn(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);

/* The function table for the layer functions */
typedef void (*layer_fn_t)(PBS_CONTEXT, bool, bool, bool);
static const layer_fn_t c_LayerFunctionTable[] = {
    data,
    conv1,
    dummyConvConv21X1Bn,
    conv21X1,
    conv21X2,
    concat21,
    dummyConvConv22X1Bn,
    conv22X1,
    conv22X2,
    concat22,
    dummyConvConv23X1Bn,
    conv23X1,
    conv23X2,
    concat23,
    dummyConvConv24X1Bn,
    conv24X2,
    concat24,
    dummyConvConv25X1Bn,
    conv25X2,
    concat25,
    dummyConvConv26X1Bn,
    conv26X2,
    concat26,
    dummyConvConv2BlkBn,
    conv2Blk,
    dummyConvConv31X1Bn,
    conv31X1,
    conv31X2,
    concat31,
    dummyConvConv32X1Bn,
    conv32X1,
    conv32X2,
    concat32,
    dummyConvConv33X1Bn,
    conv33X1,
    conv33X2,
    concat33,
    dummyConvConv34X1Bn,
    conv34X1,
    conv34X2,
    concat34,
    dummyConvConv35X1Bn,
    conv35X1,
    conv35X2,
    concat35,
    dummyConvConv36X1Bn,
    conv36X1,
    conv36X2,
    concat36,
    dummyConvConv37X1Bn,
    conv37X1,
    conv37X2,
    concat37,
    dummyConvConv38X1Bn,
    conv38X1,
    conv38X2,
    concat38,
    dummyConvConv39X1Bn,
    conv39X1,
    conv39X2,
    concat39,
    dummyConvConv310X1Bn,
    conv310X1,
    conv310X2,
    concat310,
    dummyConvConv311X1Bn,
    conv311X1,
    conv311X2,
    concat311,
    dummyConvConv312X1Bn,
    conv312X1,
    conv312X2,
    concat312,
    dummyConvConv3BlkBn,
    conv3Blk,
    dummyConvConv41X1Bn,
    conv41X1,
    conv41X2,
    concat41,
    dummyConvConv42X1Bn,
    conv42X1,
    conv42X2,
    concat42,
    dummyConvConv43X1Bn,
    conv43X1,
    conv43X2,
    concat43,
    dummyConvConv44X1Bn,
    conv44X1,
    conv44X2,
    concat44,
    dummyConvConv45X1Bn,
    conv45X1,
    conv45X2,
    concat45,
    dummyConvConv46X1Bn,
    conv46X1,
    conv46X2,
    concat46,
    dummyConvConv47X1Bn,
    conv47X1,
    conv47X2,
    concat47,
    dummyConvConv48X1Bn,
    conv48X1,
    conv48X2,
    concat48,
    dummyConvConv49X1Bn,
    conv49X1,
    conv49X2,
    concat49,
    dummyConvConv410X1Bn,
    conv410X1,
    conv410X2,
    concat410,
    dummyConvConv411X1Bn,
    conv411X1,
    conv411X2,
    concat411,
    dummyConvConv412X1Bn,
    conv412X1,
    conv412X2,
    concat412,
    dummyConvConv413X1Bn,
    conv413X1,
    conv413X2,
    concat413,
    dummyConvConv414X1Bn,
    conv414X1,
    conv414X2,
    concat414,
    dummyConvConv415X1Bn,
    conv415X1,
    conv415X2,
    concat415,
    dummyConvConv416X1Bn,
    conv416X1,
    conv416X2,
    concat416,
    dummyConvConv417X1Bn,
    conv417X1,
    conv417X2,
    concat417,
    dummyConvConv418X1Bn,
    conv418X1,
    conv418X2,
    concat418,
    dummyConvConv419X1Bn,
    conv419X1,
    conv419X2,
    concat419,
    dummyConvConv420X1Bn,
    conv420X1,
    conv420X2,
    concat420,
    dummyConvConv421X1Bn,
    conv421X1,
    conv421X2,
    concat421,
    dummyConvConv422X1Bn,
    conv422X1,
    conv422X2,
    concat422,
    dummyConvConv423X1Bn,
    conv423X1,
    conv423X2,
    concat423,
    dummyConvConv424X1Bn,
    conv424X1,
    conv424X2,
    concat424,
    dummyConvConv4BlkBn,
    conv4Blk,
    dummyConvConv51X1Bn,
    conv51X1,
    conv51X2,
    concat51,
    dummyConvConv52X1Bn,
    conv52X1,
    conv52X2,
    concat52,
    dummyConvConv53X1Bn,
    conv53X1,
    conv53X2,
    concat53,
    dummyConvConv54X1Bn,
    conv54X1,
    conv54X2,
    concat54,
    dummyConvConv55X1Bn,
    conv55X1,
    conv55X2,
    concat55,
    dummyConvConv56X1Bn,
    conv56X1,
    conv56X2,
    concat56,
    dummyConvConv57X1Bn,
    conv57X1,
    conv57X2,
    concat57,
    dummyConvConv58X1Bn,
    conv58X1,
    conv58X2,
    concat58,
    dummyConvConv59X1Bn,
    conv59X1,
    conv59X2,
    concat59,
    dummyConvConv510X1Bn,
    conv510X1,
    conv510X2,
    concat510,
    dummyConvConv511X1Bn,
    conv511X1,
    conv511X2,
    concat511,
    dummyConvConv512X1Bn,
    conv512X1,
    conv512X2,
    concat512,
    dummyConvConv513X1Bn,
    conv513X1,
    conv513X2,
    concat513,
    dummyConvConv514X1Bn,
    conv514X1,
    conv514X2,
    concat514,
    dummyConvConv515X1Bn,
    conv515X1,
    conv515X2,
    concat515,
    dummyConvConv516X1Bn,
    conv516X1,
    conv516X2,
    concat516,
    dummyConvConv5BlkBn,
};

// Init service function for use with ONNX Runtime.
void init(PBS_CONTEXT bs, const BrainSliceOperator_RuntimeArguments* args)
{
    /* Sanity check for the BrainSlice SKU for this firmware. */
    BSNL_HEX_Assert(bs->m_bsParameters.NATIVE_DIM == 128, NIOS_HEX_CNN_AUTOGEN_NATIVE_DIM_MISMATCH);
    BSNL_HEX_Assert(bs->m_bsParameters.MFUS >= 2, NIOS_HEX_CNN_AUTOGEN_MFUS_TOO_FEW);
    BSNL_HEX_Assert(bs->m_bsParameters.INITIAL_VRF_SIZE >= 12100, NIOS_HEX_CNN_AUTOGEN_INITIAL_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.MVM_MATRIX_RF_SIZE >= 128, NIOS_HEX_CNN_AUTOGEN_MATRIX_RF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.ADDSUB_VRF_0_SIZE >= 8, NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.ADDSUB_VRF_1_SIZE >= 3135, NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.MULTIPLY_VRF_SIZE >= 8, NIOS_HEX_CNN_AUTOGEN_MULTIPLY_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.USE_DRAM  , NIOS_HEX_CNN_AUTOGEN_MISSING_NEEDED_DRAM);
    BSNL_HEX_Assert(bs->m_bsParameters.VECTOR_MEM_SIZE >= 705, NIOS_HEX_CNN_AUTOGEN_VECTOR_MEM_TOO_SMALL);

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
    for(int row=0; row<output_depth; row++) {
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
 * The main function that runs evaluation on the DenseNet-121 model.
 *
 * This runs the input on the network through the specified subset of the network.
 **/
void execute(PBS_CONTEXT p_bs, const BrainSliceOperator_RuntimeArguments* args, int p_startLayerID, int p_endLayerID, bool p_debugMode)
{
    // By default, run all the DenseNet-121 layers
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
        outputNativeSize = 288;
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
            vRead1D(p_bs, ISA_Mem_Dram, -1, 288);
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

void data(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Input layer: data(d=147, h=112, d=112) = Input() 1240 registers */
    vRead1D(bs, ISA_Mem_NetInputQ, DONTCARE, 1240);
    v_wr(bs, ISA_Mem_Expander, 0+0);
    /* End input layer */
}

void conv1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution conv1(d=64, h=112, d=112) = Convolution(data(d=147, h=112, w=112), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    absorbed conv1_bn */
    /*    absorbed conv1_scale */
    /*    absorbed relu1 */
    /*    includes sublayer pool1(d=64, h=55, d=55) = MaxPool(conv1(d=64, h=112, w=112), k_h=3, k_w=3, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress data_inIndex;
    data_inIndex=0;
    moveFilterCount128(bs, ISA_Mem_Dram, conv1_MRF+0*64, ISA_Mem_MatrixRf, mrf_start, 1, 64);
    ISA_ExtAddress tmp_MVMIVRF=6886, tmp_MVMIVRF_next=8206;
    /* Split the tile in half for double buffering */
    /* Layer conv1 tile size 1*224 */
    /* Temp vars and parameters for input layer conv1 */
    /* SetIterations on reads from the input expander must be a multiple of bs->m_bsParameters.CHANNELS or this must be the last read*/
    ISA_NativeCount maxReadSize=(ISA_NativeCount)((112/bs->m_bsParameters.CHANNELS)*bs->m_bsParameters.CHANNELS);
    /* _in is the read pointer (not adjusted for padding because we read the whole row), _next is the write pointer (adjusted for padding) */
    ISA_ExtAddress g0_conv1_in=9526, g0_conv1_inIterator=9526;
    ISA_ExtAddress g0_conv1_next=9750, g0_conv1_available=maxReadSize, g0_conv1_next_available=maxReadSize, g0_conv1_tmp;
    /* Need to track the start and offset within the output row to handle the padding in pool1 */
    ISA_ExtAddress g0_conv1_outOffset=0;
    ISA_ExtAddress g0_conv1_outRowStart=0;
    int g0_conv1_iterationsLeft=12544;
    int g0_conv1_loadLeft=12544;
    vRead1D(bs, ISA_Mem_Dram, conv1_bn_scale__vv_mul__conv1_scale_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, conv1_bn_bias__vv_mul__conv1_scale_scale__vv_add__conv1_scale_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    /* Layer pool1 tile size 24*112 */
    /* Temp vars and parameters for input layer pool1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_pool1_in=4198,g1_pool1_inIterator=4198;
    ISA_ExtAddress g1_pool1_available = 0;
    ISA_ExtAddress g1_pool1_accumulators=6886;
    ISA_ExtAddress g1_pool1_availableVerticalRows=0;
    ISA_ExtAddress g1_pool1_outOffset=0;
    int g1_pool1_iterationsLeft=3025;
    vRead2D(bs, ISA_Mem_Expander, data_inIndex, 2, maxReadSize, 2);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_in, 2);
    vRead2D(bs, ISA_Mem_Expander, data_inIndex, 2, maxReadSize, 2);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_next, 2);
    g0_conv1_loadLeft -= 2 * maxReadSize;
    /* Loop until we've read all outputs */
    while (g1_pool1_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_conv1_iterationsLeft>0) {

            /* Prefetch activations for the next iteration of the loop to hide latency */
            if (g0_conv1_available==0) {
                /* This is complicated in order to ensure that iterations%channels = 0 or this is the last transfer */
                /* swap buffers, then fetch next */
                g0_conv1_tmp=g0_conv1_in; g0_conv1_in=g0_conv1_next; g0_conv1_next=g0_conv1_tmp;
                g0_conv1_inIterator = g0_conv1_in;
                g0_conv1_available = g0_conv1_next_available;
                if (g0_conv1_loadLeft > 0) {
                    if (g0_conv1_loadLeft > maxReadSize) {
                        g0_conv1_next_available = maxReadSize;
                    } else {
                        g0_conv1_next_available = g0_conv1_loadLeft;
                    }
                    vRead2D(bs, ISA_Mem_Expander, data_inIndex, 2, g0_conv1_next_available, 2);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_next, 2);
                    g0_conv1_loadLeft -= g0_conv1_next_available;
                }
            }

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_conv1_available <= 112, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            if (g0_conv1_available > 0) {

                /* Start of layer 0 in group 0 (conv1) */
                /* Tile size 1*224 dimPerStep 2 */
                ISA_NativeCount toCompute=g0_conv1_available;
                if ((g0_conv1_outOffset + toCompute) >= 112) {
                    toCompute = 112 - g0_conv1_outOffset;
                }
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_conv1_inIterator, 2, toCompute, 2);
                mv_mul(bs, mrf_start+0);
                vv_mul(bs, 0); /* includes: conv1_bn: scale, vv_mul, conv1_scale: scale */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0, 0); /* includes: conv1_bn: bias, vv_mul, conv1_scale: scale, vv_add, conv1_scale: bias */
                v_relu(bs); /* includes: relu1: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 4198+g0_conv1_outOffset+g0_conv1_outRowStart+0, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+g0_conv1_outOffset+g0_conv1_outRowStart+0, 1);
                /* Advance the write pointer */
                g0_conv1_outOffset += toCompute;
                if (g0_conv1_outOffset == 112) {
                    g0_conv1_outOffset = 0;
                    if (g0_conv1_outRowStart==2576) {
                        g0_conv1_outRowStart=0;
                    } else {
                        g0_conv1_outRowStart+=112;
                    }
                }
                g1_pool1_available += toCompute*1;
                g0_conv1_inIterator += toCompute*2 /* LHS is in native vectors; RHS is in activations */;
                g0_conv1_available -= toCompute;
                g0_conv1_iterationsLeft-=toCompute;
                /* Check there is enough data (emulator only) */
                Emulator_HEX_Assert(g0_conv1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
            }
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_pool1_available <= 2688, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if ((g1_pool1_available >= 2688) || ((g0_conv1_iterationsLeft==0))) {

            /* Start of layer 0 in group 1 (pool1) */
            /* Tile size 24*112 dimPerStep 1 */
            /* Decompose the MAX-pool into 2 horizontal pool operations followed by and 2 vertical pool operations */
            /* At the moment, all intermediate data is saved to both MVM_IVRF and AddSubVrf1 */
            /* Perform 55 horizontal pool operations on 24 or 22 rows with 2 steps */
            /* After the first iteration we all skip horizontal pool operations for 1 rows that were computed by the previous iteration */
            /* The last iteration will perform 22 horizontal pool operations and 11 vertical operations */
            /* g1_pool1_inIterator iterates horizontal pool operations (INPUTS) */
            /* g1_pool1_in iterates vertical pool operations (OUTPUTS) */
            /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
            int horizontalRows=22;
            int verticalRows=11;
            if (g1_pool1_iterationsLeft==3025) {
                horizontalRows=24;
            }
            g1_pool1_available -= verticalRows*224;
            ISA_ExtAddress curOffset;
            curOffset=g1_pool1_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 55, 2);
                mv_mul(bs, mrf_start+2);
                /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-4198+0+1, 2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1);
                if (curOffset>=6774) {
                    curOffset-=2576;
                } else {
                    curOffset+=112;
                }
            }
            curOffset=g1_pool1_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-4198+2+0, 2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, (g1_pool1_accumulators+1320)+rowIterator*55, 1);
                if (curOffset>=6774) {
                    curOffset-=2576;
                } else {
                    curOffset+=112;
                }
            }
            /* Horizontal sweep must end up in (g1_pool1_accumulators+1320) because we can't read-modify-write ASVRF in a single chain */
            curOffset=g1_pool1_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, (g1_pool1_accumulators+1320)+rowIterator*55, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+0-4198, 1);
                if (curOffset>=6774) {
                    curOffset-=2576;
                } else {
                    curOffset+=112;
                }
            }
            /* Update horizontal pool iterator start */
            g1_pool1_inIterator = curOffset-0;
            curOffset=g1_pool1_in;
            ISA_ExtAddress nextOffset=curOffset;
            if (nextOffset>=6774) {
                nextOffset-=2576;
            } else {
                nextOffset+=112;
            }
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-4198+0, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1);
                if (curOffset>=6662) {
                    curOffset-=2464;
                } else {
                    curOffset+=224;
                }
                if (nextOffset>=6662) {
                    nextOffset-=2464;
                } else {
                    nextOffset+=224;
                }
            }
            curOffset=g1_pool1_in;
            nextOffset=curOffset;
            if (nextOffset>=6662) {
                nextOffset-=2464;
            } else {
                nextOffset+=224;
            }
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-4198+0, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9974 + g1_pool1_outOffset, 1);
                g1_pool1_outOffset+=55;
                if (curOffset>=6662) {
                    curOffset-=2464;
                } else {
                    curOffset+=224;
                }
                if (nextOffset>=6662) {
                    nextOffset-=2464;
                } else {
                    nextOffset+=224;
                }
            }
            g1_pool1_in = curOffset;
            g1_pool1_iterationsLeft-=verticalRows*55;
            /* Make sure we didn't loop too many times (emulator only) */
            Emulator_HEX_Assert(g1_pool1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void dummyConvConv21X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv2_1_x1_bn(d=64, h=55, d=55) = Convolution(pool1(d=64, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_1_x1_bn */
    /*      absorbed conv2_1_x1_scale */
    /*      absorbed relu2_1_x1 */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 9974, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 3, /* mulParam_address */ conv2_1_x1_bn_scale__vv_mul__conv2_1_x1_scale_scale, /* addParam_address */ conv2_1_x1_bn_bias__vv_mul__conv2_1_x1_scale_scale__vv_add__conv2_1_x1_scale_bias,
                             /* output_IVRF_address */ 6949, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv21X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_1_x1(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_1_x1_bn(d=64, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_1_x2_bn */
    /*      absorbed conv2_1_x2_scale */
    /*      absorbed relu2_1_x2 */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 6949, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 4, /* mulParam_address */ conv2_1_x2_bn_scale__vv_mul__conv2_1_x2_scale_scale, /* addParam_address */ conv2_1_x2_bn_bias__vv_mul__conv2_1_x2_scale_scale__vv_add__conv2_1_x2_scale_bias,
                             /* output_IVRF_address */ 3924, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv21X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_1_x2(d=32, h=55, d=55) = Convolution(conv2_1_x1(d=128, h=55, w=55), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv2_1_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 3924, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 5, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat21(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_2_1(d=96, h=55, d=55) = Concat(Concat(pool1(d=64, h=55, w=55), conv2_1_x2(d=32, h=55, w=55))) */

    /* Input pool1 ISA_Mem_MvmInitialVrf memory: addresses [9974 - 12998] */
    /* Input conv2_1_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 3024] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [6949 - 9973] */

    /* This layer's matrix parameters were prefetched by layer conv1. */

    /* Concatenate layer pool1(d=0:64) with layer conv2_1_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 9974, 1, 55 * 55, 1);
    mv_mul(bs, mrf_start + 14);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv2_1_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6949, 1);
}

void dummyConvConv22X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv2_2_x1_bn(d=96, h=55, d=55) = Convolution(concat_2_1(d=96, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_2_x1_bn */
    /*      absorbed conv2_2_x1_scale */
    /*      absorbed relu2_2_x1 */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 6949, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 15, /* mulParam_address */ conv2_2_x1_bn_scale__vv_mul__conv2_2_x1_scale_scale, /* addParam_address */ conv2_2_x1_bn_bias__vv_mul__conv2_2_x1_scale_scale__vv_add__conv2_2_x1_scale_bias,
                             /* output_IVRF_address */ 9974, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv22X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_2_x1(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_2_x1_bn(d=96, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_2_x2_bn */
    /*      absorbed conv2_2_x2_scale */
    /*      absorbed relu2_2_x2 */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 9974, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 16, /* mulParam_address */ conv2_2_x2_bn_scale__vv_mul__conv2_2_x2_scale_scale, /* addParam_address */ conv2_2_x2_bn_bias__vv_mul__conv2_2_x2_scale_scale__vv_add__conv2_2_x2_scale_bias,
                             /* output_IVRF_address */ 3924, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv22X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_2_x2(d=32, h=55, d=55) = Convolution(conv2_2_x1(d=128, h=55, w=55), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv2_2_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 3924, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 17, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat22(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_2_2(d=128, h=55, d=55) = Concat(Concat(concat_2_1(d=96, h=55, w=55), conv2_2_x2(d=32, h=55, w=55))) */

    /* Input concat_2_1 ISA_Mem_MvmInitialVrf memory: addresses [6949 - 9973] */
    /* Input conv2_2_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 3024] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [9974 - 12998] */

    /* This layer's matrix parameters were prefetched by layer conv1. */

    /* Concatenate layer concat_2_1(d=0:96) with layer conv2_2_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 6949, 1, 55 * 55, 1);
    mv_mul(bs, mrf_start + 26);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv2_2_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9974, 1);
}

void dummyConvConv23X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv2_3_x1_bn(d=128, h=55, d=55) = Convolution(concat_2_2(d=128, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_3_x1_bn */
    /*      absorbed conv2_3_x1_scale */
    /*      absorbed relu2_3_x1 */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 9974, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 27, /* mulParam_address */ conv2_3_x1_bn_scale__vv_mul__conv2_3_x1_scale_scale, /* addParam_address */ conv2_3_x1_bn_bias__vv_mul__conv2_3_x1_scale_scale__vv_add__conv2_3_x1_scale_bias,
                             /* output_IVRF_address */ 6949, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv23X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_3_x1(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_3_x1_bn(d=128, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_3_x2_bn */
    /*      absorbed conv2_3_x2_scale */
    /*      absorbed relu2_3_x2 */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 6949, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 28, /* mulParam_address */ conv2_3_x2_bn_scale__vv_mul__conv2_3_x2_scale_scale, /* addParam_address */ conv2_3_x2_bn_bias__vv_mul__conv2_3_x2_scale_scale__vv_add__conv2_3_x2_scale_bias,
                             /* output_IVRF_address */ 3924, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv23X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_3_x2(d=32, h=55, d=55) = Convolution(conv2_3_x1(d=128, h=55, w=55), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 3924, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 29, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 6949, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat23(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_2_3(d=160, h=55, d=55) = Concat(Concat(concat_2_2(d=128, h=55, w=55), conv2_3_x2(d=32, h=55, w=55))) */

    /* Input concat_2_2 ISA_Mem_MvmInitialVrf memory: addresses [9974 - 12998] */
    /* Input conv2_3_x2 ISA_Mem_MvmInitialVrf memory: addresses [6949 - 9973] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 6049] */

    /* This layer's matrix parameters were prefetched by layer conv1. */

    /* Copy layer concat_2_2(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 9974, 1, 55 * 55, 1);
    mv_mul(bs, mrf_start + 38);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0, 2);

    /* Copy layer conv2_3_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 6949, 1, 55 * 55, 1);
    mv_mul(bs, mrf_start + 38);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 1, 2);
}

void dummyConvConv24X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution dummy_conv_conv2_4_x1_bn(d=160, h=55, d=55) = Convolution(concat_2_3(d=160, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    absorbed conv2_4_x1_bn */
    /*    absorbed conv2_4_x1_scale */
    /*    absorbed relu2_4_x1 */
    /*    includes sublayer conv2_4_x1(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_4_x1_bn(d=160, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed conv2_4_x2_bn */
    /*        absorbed conv2_4_x2_scale */
    /*        absorbed relu2_4_x2 */
    ISA_ExtAddress concat_2_3_inIndex;
    concat_2_3_inIndex=0;
    /* Layer dummy_conv_conv2_4_x1_bn tile size 1*22 */
    /* Temp vars and parameters for input layer dummy_conv_conv2_4_x1_bn */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_dummy_conv_conv2_4_x1_bn_in=0,g0_dummy_conv_conv2_4_x1_bn_inIterator=0;
    ISA_ExtAddress g0_dummy_conv_conv2_4_x1_bn_available = 3025;
    ISA_ExtAddress g0_dummy_conv_conv2_4_x1_bn_outOffset=0;
    int g0_dummy_conv_conv2_4_x1_bn_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, conv2_4_x1_bn_scale__vv_mul__conv2_4_x1_scale_scale, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, conv2_4_x1_bn_bias__vv_mul__conv2_4_x1_scale_scale__vv_add__conv2_4_x1_scale_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    /* Layer conv2_4_x1 tile size 1*22 */
    /* Temp vars and parameters for input layer conv2_4_x1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_conv2_4_x1_in=9075,g1_conv2_4_x1_inIterator=9075;
    ISA_ExtAddress g1_conv2_4_x1_available = 0;
    ISA_ExtAddress g1_conv2_4_x1_outOffset=0;
    int g1_conv2_4_x1_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, conv2_4_x2_bn_scale__vv_mul__conv2_4_x2_scale_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 2);
    vRead1D(bs, ISA_Mem_Dram, conv2_4_x2_bn_bias__vv_mul__conv2_4_x2_scale_scale__vv_add__conv2_4_x2_scale_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 2);
    /* Loop until we've read all outputs */
    while (g1_conv2_4_x1_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_dummy_conv_conv2_4_x1_bn_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_dummy_conv_conv2_4_x1_bn_available <= 11, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);

            /* Start of layer 0 in group 0 (dummy_conv_conv2_4_x1_bn) */
            /* Tile size 1*22 dimPerStep 2 */
            for(int outRow=0;outRow<2;outRow++) {
                g0_dummy_conv_conv2_4_x1_bn_inIterator = g0_dummy_conv_conv2_4_x1_bn_in;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_dummy_conv_conv2_4_x1_bn_inIterator+outRow, 1, 22, 2);
                mv_mul(bs, mrf_start+39);
                vv_mul(bs, 0+outRow); /* includes: conv2_4_x1_bn: scale, vv_mul, conv2_4_x1_scale: scale */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outRow, 0); /* includes: conv2_4_x1_bn: bias, vv_mul, conv2_4_x1_scale: scale, vv_add, conv2_4_x1_scale: bias */
                v_relu(bs); /* includes: relu2_4_x1: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9075+g0_dummy_conv_conv2_4_x1_bn_outOffset+outRow+0, 2);
            }
            /* Advance the read pointer for the next step */
            g0_dummy_conv_conv2_4_x1_bn_in += 22;
            /* Advance the write pointer */
            if (g0_dummy_conv_conv2_4_x1_bn_outOffset == 0) {
                g0_dummy_conv_conv2_4_x1_bn_outOffset = 0;
            } else {
                g0_dummy_conv_conv2_4_x1_bn_outOffset += 22;
            }
            g1_conv2_4_x1_available += 11;
            g0_dummy_conv_conv2_4_x1_bn_iterationsLeft -= 11;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_dummy_conv_conv2_4_x1_bn_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_conv2_4_x1_available <= 11, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        while (g1_conv2_4_x1_available >= 11) {

            /* Start of layer 0 in group 1 (conv2_4_x1) */
            /* Tile size 1*22 dimPerStep 2 */
            g1_conv2_4_x1_inIterator = g1_conv2_4_x1_in;
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv2_4_x1_inIterator, 2, 11, 2);
            mv_mul(bs, mrf_start+40);
            vv_mul(bs, 2); /* includes: conv2_4_x2_bn: scale, vv_mul, conv2_4_x2_scale: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 2, 0); /* includes: conv2_4_x2_bn: bias, vv_mul, conv2_4_x2_scale: scale, vv_add, conv2_4_x2_scale: bias */
            v_relu(bs); /* includes: relu2_4_x2: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050 + g1_conv2_4_x1_outOffset, 1);
            /* Advance the read pointer for the next step */
            if (g1_conv2_4_x1_in>=9075) {
                g1_conv2_4_x1_in -= 0;
            } else {
                g1_conv2_4_x1_in += 22;
            }
            /* Advance the write pointer */
            g1_conv2_4_x1_outOffset += 11;
            g1_conv2_4_x1_iterationsLeft -= 11;
            g1_conv2_4_x1_available=0;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g1_conv2_4_x1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void conv24X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_4_x2(d=32, h=55, d=55) = Convolution(conv2_4_x1(d=128, h=55, w=55), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv2_4_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 6050, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 42, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat24(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_2_4(d=192, h=55, d=55) = Concat(Concat(concat_2_3(d=160, h=55, w=55), conv2_4_x2(d=32, h=55, w=55))) */

    /* Input concat_2_3 ISA_Mem_MvmInitialVrf memory: addresses [0 - 6049] */
    /* Input conv2_4_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 3024] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [6050 - 12099] */

    /* This layer's matrix parameters were prefetched by layer conv1. */

    /* Copy layer concat_2_3(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 55 * 55, 2);
    mv_mul(bs, mrf_start + 51);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050, 2);

    /* Concatenate layer concat_2_3(d=128:160) with layer conv2_4_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (2 - 1), 1, 55 * 55, 2);
    mv_mul(bs, mrf_start + 51);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv2_4_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050 + (2 - 1), 2);
}

void dummyConvConv25X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution dummy_conv_conv2_5_x1_bn(d=192, h=55, d=55) = Convolution(concat_2_4(d=192, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    absorbed conv2_5_x1_bn */
    /*    absorbed conv2_5_x1_scale */
    /*    absorbed relu2_5_x1 */
    /*    includes sublayer conv2_5_x1(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_5_x1_bn(d=192, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed conv2_5_x2_bn */
    /*        absorbed conv2_5_x2_scale */
    /*        absorbed relu2_5_x2 */
    ISA_ExtAddress concat_2_4_inIndex;
    concat_2_4_inIndex=6050;
    /* Prefetch 59 entries starting at concat_2_5 */
    moveFilterCount128(bs, ISA_Mem_Dram, concat_2_5_MRF+0*59, ISA_Mem_MatrixRf, mrf_next, 1, 59);
    /* Layer dummy_conv_conv2_5_x1_bn tile size 1*22 */
    /* Temp vars and parameters for input layer dummy_conv_conv2_5_x1_bn */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_dummy_conv_conv2_5_x1_bn_in=6050,g0_dummy_conv_conv2_5_x1_bn_inIterator=6050;
    ISA_ExtAddress g0_dummy_conv_conv2_5_x1_bn_available = 3025;
    ISA_ExtAddress g0_dummy_conv_conv2_5_x1_bn_outOffset=0;
    int g0_dummy_conv_conv2_5_x1_bn_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, conv2_5_x1_bn_scale__vv_mul__conv2_5_x1_scale_scale, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, conv2_5_x1_bn_bias__vv_mul__conv2_5_x1_scale_scale__vv_add__conv2_5_x1_scale_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    /* Layer conv2_5_x1 tile size 1*22 */
    /* Temp vars and parameters for input layer conv2_5_x1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_conv2_5_x1_in=3025,g1_conv2_5_x1_inIterator=3025;
    ISA_ExtAddress g1_conv2_5_x1_available = 0;
    ISA_ExtAddress g1_conv2_5_x1_outOffset=0;
    int g1_conv2_5_x1_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, conv2_5_x2_bn_scale__vv_mul__conv2_5_x2_scale_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 2);
    vRead1D(bs, ISA_Mem_Dram, conv2_5_x2_bn_bias__vv_mul__conv2_5_x2_scale_scale__vv_add__conv2_5_x2_scale_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 2);
    /* Loop until we've read all outputs */
    while (g1_conv2_5_x1_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_dummy_conv_conv2_5_x1_bn_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_dummy_conv_conv2_5_x1_bn_available <= 11, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);

            /* Start of layer 0 in group 0 (dummy_conv_conv2_5_x1_bn) */
            /* Tile size 1*22 dimPerStep 2 */
            for(int outRow=0;outRow<2;outRow++) {
                g0_dummy_conv_conv2_5_x1_bn_inIterator = g0_dummy_conv_conv2_5_x1_bn_in;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_dummy_conv_conv2_5_x1_bn_inIterator+outRow, 1, 22, 2);
                mv_mul(bs, mrf_start+52);
                vv_mul(bs, 0+outRow); /* includes: conv2_5_x1_bn: scale, vv_mul, conv2_5_x1_scale: scale */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outRow, 0); /* includes: conv2_5_x1_bn: bias, vv_mul, conv2_5_x1_scale: scale, vv_add, conv2_5_x1_scale: bias */
                v_relu(bs); /* includes: relu2_5_x1: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 3025+g0_dummy_conv_conv2_5_x1_bn_outOffset+outRow+0, 2);
            }
            /* Advance the read pointer for the next step */
            g0_dummy_conv_conv2_5_x1_bn_in += 22;
            /* Advance the write pointer */
            if (g0_dummy_conv_conv2_5_x1_bn_outOffset == 0) {
                g0_dummy_conv_conv2_5_x1_bn_outOffset = 0;
            } else {
                g0_dummy_conv_conv2_5_x1_bn_outOffset += 22;
            }
            g1_conv2_5_x1_available += 11;
            g0_dummy_conv_conv2_5_x1_bn_iterationsLeft -= 11;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_dummy_conv_conv2_5_x1_bn_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_conv2_5_x1_available <= 11, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        while (g1_conv2_5_x1_available >= 11) {

            /* Start of layer 0 in group 1 (conv2_5_x1) */
            /* Tile size 1*22 dimPerStep 2 */
            g1_conv2_5_x1_inIterator = g1_conv2_5_x1_in;
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv2_5_x1_inIterator, 2, 11, 2);
            mv_mul(bs, mrf_start+53);
            vv_mul(bs, 2); /* includes: conv2_5_x2_bn: scale, vv_mul, conv2_5_x2_scale: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 2, 0); /* includes: conv2_5_x2_bn: bias, vv_mul, conv2_5_x2_scale: scale, vv_add, conv2_5_x2_scale: bias */
            v_relu(bs); /* includes: relu2_5_x2: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + g1_conv2_5_x1_outOffset, 1);
            /* Advance the read pointer for the next step */
            if (g1_conv2_5_x1_in>=3025) {
                g1_conv2_5_x1_in -= 0;
            } else {
                g1_conv2_5_x1_in += 22;
            }
            /* Advance the write pointer */
            g1_conv2_5_x1_outOffset += 11;
            g1_conv2_5_x1_iterationsLeft -= 11;
            g1_conv2_5_x1_available=0;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g1_conv2_5_x1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void conv25X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_5_x2(d=32, h=55, d=55) = Convolution(conv2_5_x1(d=128, h=55, w=55), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv2_5_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 55, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void concat25(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_2_5(d=224, h=55, d=55) = Concat(Concat(concat_2_4(d=192, h=55, w=55), conv2_5_x2(d=32, h=55, w=55))) */

    /* Input concat_2_4 ISA_Mem_MvmInitialVrf memory: addresses [6050 - 12099] */
    /* Input conv2_5_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 3024] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 6049] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv2_5_x1_bn. */

    /* Copy layer concat_2_4(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 6050, 1, 55 * 55, 2);
    mv_mul(bs, mrf_start + 0);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0, 2);

    /* Concatenate layer concat_2_4(d=128:192) with layer conv2_5_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 6050 + (2 - 1), 1, 55 * 55, 2);
    mv_mul(bs, mrf_start + 0);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv2_5_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (2 - 1), 2);
}

void dummyConvConv26X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution dummy_conv_conv2_6_x1_bn(d=224, h=55, d=55) = Convolution(concat_2_5(d=224, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    absorbed conv2_6_x1_bn */
    /*    absorbed conv2_6_x1_scale */
    /*    absorbed relu2_6_x1 */
    /*    includes sublayer conv2_6_x1(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_6_x1_bn(d=224, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*        absorbed conv2_6_x2_bn */
    /*        absorbed conv2_6_x2_scale */
    /*        absorbed relu2_6_x2 */
    ISA_ExtAddress concat_2_5_inIndex;
    concat_2_5_inIndex=0;
    /* Layer dummy_conv_conv2_6_x1_bn tile size 1*22 */
    /* Temp vars and parameters for input layer dummy_conv_conv2_6_x1_bn */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_dummy_conv_conv2_6_x1_bn_in=0,g0_dummy_conv_conv2_6_x1_bn_inIterator=0;
    ISA_ExtAddress g0_dummy_conv_conv2_6_x1_bn_available = 3025;
    ISA_ExtAddress g0_dummy_conv_conv2_6_x1_bn_outOffset=0;
    int g0_dummy_conv_conv2_6_x1_bn_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, conv2_6_x1_bn_scale__vv_mul__conv2_6_x1_scale_scale, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, conv2_6_x1_bn_bias__vv_mul__conv2_6_x1_scale_scale__vv_add__conv2_6_x1_scale_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    /* Layer conv2_6_x1 tile size 1*22 */
    /* Temp vars and parameters for input layer conv2_6_x1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_conv2_6_x1_in=9075,g1_conv2_6_x1_inIterator=9075;
    ISA_ExtAddress g1_conv2_6_x1_available = 0;
    ISA_ExtAddress g1_conv2_6_x1_outOffset=0;
    int g1_conv2_6_x1_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, conv2_6_x2_bn_scale__vv_mul__conv2_6_x2_scale_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 2);
    vRead1D(bs, ISA_Mem_Dram, conv2_6_x2_bn_bias__vv_mul__conv2_6_x2_scale_scale__vv_add__conv2_6_x2_scale_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 2);
    /* Loop until we've read all outputs */
    while (g1_conv2_6_x1_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_dummy_conv_conv2_6_x1_bn_iterationsLeft>0) {

            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_dummy_conv_conv2_6_x1_bn_available <= 11, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);

            /* Start of layer 0 in group 0 (dummy_conv_conv2_6_x1_bn) */
            /* Tile size 1*22 dimPerStep 2 */
            for(int outRow=0;outRow<2;outRow++) {
                g0_dummy_conv_conv2_6_x1_bn_inIterator = g0_dummy_conv_conv2_6_x1_bn_in;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_dummy_conv_conv2_6_x1_bn_inIterator+outRow, 1, 22, 2);
                mv_mul(bs, mrf_start+1);
                vv_mul(bs, 0+outRow); /* includes: conv2_6_x1_bn: scale, vv_mul, conv2_6_x1_scale: scale */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outRow, 0); /* includes: conv2_6_x1_bn: bias, vv_mul, conv2_6_x1_scale: scale, vv_add, conv2_6_x1_scale: bias */
                v_relu(bs); /* includes: relu2_6_x1: v_relu */
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9075+g0_dummy_conv_conv2_6_x1_bn_outOffset+outRow+0, 2);
            }
            /* Advance the read pointer for the next step */
            g0_dummy_conv_conv2_6_x1_bn_in += 22;
            /* Advance the write pointer */
            if (g0_dummy_conv_conv2_6_x1_bn_outOffset == 0) {
                g0_dummy_conv_conv2_6_x1_bn_outOffset = 0;
            } else {
                g0_dummy_conv_conv2_6_x1_bn_outOffset += 22;
            }
            g1_conv2_6_x1_available += 11;
            g0_dummy_conv_conv2_6_x1_bn_iterationsLeft -= 11;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_dummy_conv_conv2_6_x1_bn_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_conv2_6_x1_available <= 11, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        while (g1_conv2_6_x1_available >= 11) {

            /* Start of layer 0 in group 1 (conv2_6_x1) */
            /* Tile size 1*22 dimPerStep 2 */
            g1_conv2_6_x1_inIterator = g1_conv2_6_x1_in;
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_conv2_6_x1_inIterator, 2, 11, 2);
            mv_mul(bs, mrf_start+2);
            vv_mul(bs, 2); /* includes: conv2_6_x2_bn: scale, vv_mul, conv2_6_x2_scale: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 2, 0); /* includes: conv2_6_x2_bn: bias, vv_mul, conv2_6_x2_scale: scale, vv_add, conv2_6_x2_scale: bias */
            v_relu(bs); /* includes: relu2_6_x2: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050 + g1_conv2_6_x1_outOffset, 1);
            /* Advance the read pointer for the next step */
            if (g1_conv2_6_x1_in>=9075) {
                g1_conv2_6_x1_in -= 0;
            } else {
                g1_conv2_6_x1_in += 22;
            }
            /* Advance the write pointer */
            g1_conv2_6_x1_outOffset += 11;
            g1_conv2_6_x1_iterationsLeft -= 11;
            g1_conv2_6_x1_available=0;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g1_conv2_6_x1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void conv26X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv2_6_x2(d=32, h=55, d=55) = Convolution(conv2_6_x1(d=128, h=55, w=55), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv2_6_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 1,
                             /* input_address */ 6050, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 4, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat26(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_2_6(d=256, h=55, d=55) = Concat(Concat(concat_2_5(d=224, h=55, w=55), conv2_6_x2(d=32, h=55, w=55))) */

    /* Input concat_2_5 ISA_Mem_MvmInitialVrf memory: addresses [0 - 6049] */
    /* Input conv2_6_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 3024] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [6050 - 12099] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv2_5_x1_bn. */

    /* Copy layer concat_2_5(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 55 * 55, 2);
    mv_mul(bs, mrf_start + 13);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050, 2);

    /* Concatenate layer concat_2_5(d=128:224) with layer conv2_6_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (2 - 1), 1, 55 * 55, 2);
    mv_mul(bs, mrf_start + 13);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv2_6_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050 + (2 - 1), 2);
}

void dummyConvConv2BlkBn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv2_blk_bn(d=256, h=55, d=55) = Convolution(concat_2_6(d=256, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv2_blk_bn */
    /*      absorbed conv2_blk_scale */
    /*      absorbed relu2_blk */

    genericConvolution(bs, /* input_height */ 55, /* input_width */ 55, /* input_depth */ 2,
                             /* input_address */ 6050, /* output_depth */ 2, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 14, /* mulParam_address */ conv2_blk_bn_scale__vv_mul__conv2_blk_scale_scale, /* addParam_address */ conv2_blk_bn_bias__vv_mul__conv2_blk_scale_scale__vv_add__conv2_blk_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv2Blk(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution conv2_blk(d=128, h=55, d=55) = Convolution(dummy_conv_conv2_blk_bn(d=256, h=55, w=55), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    includes sublayer pool2(d=128, h=27, d=27) = AveragePool(conv2_blk(d=128, h=55, w=55), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress dummy_conv_conv2_blk_bn_inIndex;
    dummy_conv_conv2_blk_bn_inIndex=0;
    ISA_ExtAddress tmp_MVMIVRF=6779, tmp_MVMIVRF_next=7521;
    /* Layer conv2_blk tile size 55*110 */
    /* Temp vars and parameters for input layer conv2_blk */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_conv2_blk_in=0,g0_conv2_blk_inIterator=0;
    ISA_ExtAddress g0_conv2_blk_available = 3025;
    ISA_ExtAddress g0_conv2_blk_outOffset=0;
    int g0_conv2_blk_iterationsLeft=3025;
    /* Layer pool2 tile size 55*55 */
    /* Temp vars and parameters for input layer pool2 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_pool2_in=8264,g1_pool2_inIterator=8264;
    ISA_ExtAddress g1_pool2_available = 3025;
    ISA_ExtAddress g1_pool2_accumulators=6779;
    ISA_ExtAddress g1_pool2_availableVerticalRows=0;
    ISA_ExtAddress g1_pool2_outOffset=0;
    int g1_pool2_iterationsLeft=729;
    vRead1D(bs, ISA_Mem_Dram, pool2_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    /* Loop until we've read all outputs */
    while (g1_pool2_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_conv2_blk_iterationsLeft>0) {


            /* Start of layer 0 in group 0 (conv2_blk) */
            /* Tile size 55*110 dimPerStep 2 */
            g0_conv2_blk_inIterator = g0_conv2_blk_in;
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_conv2_blk_inIterator, 2, 55 * 55, 2);
            mv_mul(bs, mrf_start+15);
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8264+g0_conv2_blk_outOffset+0, 1);
            v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+g0_conv2_blk_outOffset+0, 1);
            g0_conv2_blk_iterationsLeft-=g0_conv2_blk_available;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_conv2_blk_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_pool2_available <= 3025, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if ((g1_pool2_available >= 55) || ((g0_conv2_blk_iterationsLeft==0))) {

            /* Start of layer 0 in group 1 (pool2) */
            /* Tile size 55*55 dimPerStep 1 */
            /* Decompose the AVE-pool into 1 horizontal pool operations followed by and 1 vertical pool operations */
            /* All of the data will be processed en-masse, since our input data doesn't need paged in. */
            /* Perform 27 horizontal pool operations on 55 or 54 rows with 1 steps */
            /* g1_pool2_inIterator iterates horizontal pool operations (INPUTS) */
            /* g1_pool2_in iterates vertical pool operations (OUTPUTS) */
            /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
            int horizontalRows=55;
            int verticalRows=27;
            g1_pool2_available -= verticalRows*110;
            ISA_ExtAddress curOffset;
            curOffset=g1_pool2_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 27, 2);
                mv_mul(bs, mrf_start+17);
                /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-8264+0+1, 2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool2_accumulators+rowIterator*27, 1);
                curOffset+=55;
            }
            /* Horizontal sweep must end up in g1_pool2_accumulators because we can't read-modify-write ASVRF in a single chain */
            curOffset=g1_pool2_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool2_accumulators+rowIterator*27, 1, 27, 1);
                mv_mul(bs, mrf_start+17);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 1);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+0-8264, 1);
                curOffset+=55;
            }
            /* Update horizontal pool iterator start */
            g1_pool2_inIterator = curOffset-0;
            curOffset=g1_pool2_in;
            ISA_ExtAddress nextOffset=curOffset;
            nextOffset+=55;
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 27, 1);
                mv_mul(bs, mrf_start+17);
                /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-8264+0, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool2_accumulators + rowIterator * 27 * 1, 1);

                curOffset+=110;
                nextOffset+=110;
            }
            g1_pool2_in = curOffset;

            /* Perform any additional operations in the chain after the pooling. */
            /* In particular this is needed for average pools, because the current hardware doesn't support a multiply after an AddSub VRF 1 add/max/sub operation. */
            /* Note that we forgo using a row loop here as it isn't needed */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool2_accumulators, 1, verticalRows * 27 * 1, 1);
            mv_mul(bs, mrf_start + 17);
            vv_mul(bs, 0); /* includes: pool2: scale */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6050 + g1_pool2_outOffset, 1);

            g1_pool2_outOffset += verticalRows * 27 * 1;
            g1_pool2_iterationsLeft-=verticalRows*27;
            /* Make sure we didn't loop too many times (emulator only) */
            Emulator_HEX_Assert(g1_pool2_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void dummyConvConv31X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_1_x1_bn(d=128, h=27, d=27) = Convolution(pool2(d=128, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_1_x1_bn */
    /*      absorbed conv3_1_x1_scale */
    /*      absorbed relu3_1_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 6050, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 18, /* mulParam_address */ conv3_1_x1_bn_scale__vv_mul__conv3_1_x1_scale_scale, /* addParam_address */ conv3_1_x1_bn_bias__vv_mul__conv3_1_x1_scale_scale__vv_add__conv3_1_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv31X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_1_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_1_x1_bn(d=128, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_1_x2_bn */
    /*      absorbed conv3_1_x2_scale */
    /*      absorbed relu3_1_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 19, /* mulParam_address */ conv3_1_x2_bn_scale__vv_mul__conv3_1_x2_scale_scale, /* addParam_address */ conv3_1_x2_bn_bias__vv_mul__conv3_1_x2_scale_scale__vv_add__conv3_1_x2_scale_bias,
                             /* output_IVRF_address */ 729, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv31X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_1_x2(d=32, h=27, d=27) = Convolution(conv3_1_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 729, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 20, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat31(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_1(d=160, h=27, d=27) = Concat(Concat(pool2(d=128, h=27, w=27), conv3_1_x2(d=32, h=27, w=27))) */

    /* Input pool2 ISA_Mem_MvmInitialVrf memory: addresses [6050 - 6778] */
    /* Input conv3_1_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11541 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv2_5_x1_bn. */

    /* Copy layer pool2(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 6050, 1, 27 * 27, 1);
    mv_mul(bs, mrf_start + 29);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11541, 2);

    /* Copy layer conv3_1_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 27 * 27, 1);
    mv_mul(bs, mrf_start + 29);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11541 + 1, 2);
}

void dummyConvConv32X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_2_x1_bn(d=160, h=27, d=27) = Convolution(concat_3_1(d=160, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_2_x1_bn */
    /*      absorbed conv3_2_x1_scale */
    /*      absorbed relu3_2_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 11541, /* output_depth */ 2, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 30, /* mulParam_address */ conv3_2_x1_bn_scale__vv_mul__conv3_2_x1_scale_scale, /* addParam_address */ conv3_2_x1_bn_bias__vv_mul__conv3_2_x1_scale_scale__vv_add__conv3_2_x1_scale_bias,
                             /* output_IVRF_address */ 10083, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv32X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_2_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_2_x1_bn(d=160, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_2_x2_bn */
    /*      absorbed conv3_2_x2_scale */
    /*      absorbed relu3_2_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 10083, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 31, /* mulParam_address */ conv3_2_x2_bn_scale__vv_mul__conv3_2_x2_scale_scale, /* addParam_address */ conv3_2_x2_bn_bias__vv_mul__conv3_2_x2_scale_scale__vv_add__conv3_2_x2_scale_bias,
                             /* output_IVRF_address */ 9354, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv32X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_2_x2(d=32, h=27, d=27) = Convolution(conv3_2_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_2_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 9354, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 33, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat32(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_2(d=192, h=27, d=27) = Concat(Concat(concat_3_1(d=160, h=27, w=27), conv3_2_x2(d=32, h=27, w=27))) */

    /* Input concat_3_1 ISA_Mem_MvmInitialVrf memory: addresses [11541 - 12998] */
    /* Input conv3_2_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10083 - 11540] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv2_5_x1_bn. */

    /* Copy layer concat_3_1(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11541, 1, 27 * 27, 2);
    mv_mul(bs, mrf_start + 42);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083, 2);

    /* Concatenate layer concat_3_1(d=128:160) with layer conv3_2_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11541 + (2 - 1), 1, 27 * 27, 2);
    mv_mul(bs, mrf_start + 42);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_2_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083 + (2 - 1), 2);
}

void dummyConvConv33X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_3_x1_bn(d=192, h=27, d=27) = Convolution(concat_3_2(d=192, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_3_x1_bn */
    /*      absorbed conv3_3_x1_scale */
    /*      absorbed relu3_3_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 10083, /* output_depth */ 2, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 43, /* mulParam_address */ conv3_3_x1_bn_scale__vv_mul__conv3_3_x1_scale_scale, /* addParam_address */ conv3_3_x1_bn_bias__vv_mul__conv3_3_x1_scale_scale__vv_add__conv3_3_x1_scale_bias,
                             /* output_IVRF_address */ 11541, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv33X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_3_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_3_x1_bn(d=192, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_3_x2_bn */
    /*      absorbed conv3_3_x2_scale */
    /*      absorbed relu3_3_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 11541, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 44, /* mulParam_address */ conv3_3_x2_bn_scale__vv_mul__conv3_3_x2_scale_scale, /* addParam_address */ conv3_3_x2_bn_bias__vv_mul__conv3_3_x2_scale_scale__vv_add__conv3_3_x2_scale_bias,
                             /* output_IVRF_address */ 9354, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv33X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_3_x2(d=32, h=27, d=27) = Convolution(conv3_3_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_3_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 9354, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 46, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat33(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_3(d=224, h=27, d=27) = Concat(Concat(concat_3_2(d=192, h=27, w=27), conv3_3_x2(d=32, h=27, w=27))) */

    /* Input concat_3_2 ISA_Mem_MvmInitialVrf memory: addresses [10083 - 11540] */
    /* Input conv3_3_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11541 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv2_5_x1_bn. */

    /* Copy layer concat_3_2(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083, 1, 27 * 27, 2);
    mv_mul(bs, mrf_start + 55);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11541, 2);

    /* Concatenate layer concat_3_2(d=128:192) with layer conv3_3_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083 + (2 - 1), 1, 27 * 27, 2);
    mv_mul(bs, mrf_start + 55);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_3_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11541 + (2 - 1), 2);
}

void dummyConvConv34X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_4_x1_bn(d=224, h=27, d=27) = Convolution(concat_3_3(d=224, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_4_x1_bn */
    /*      absorbed conv3_4_x1_scale */
    /*      absorbed relu3_4_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 11541, /* output_depth */ 2, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 56, /* mulParam_address */ conv3_4_x1_bn_scale__vv_mul__conv3_4_x1_scale_scale, /* addParam_address */ conv3_4_x1_bn_bias__vv_mul__conv3_4_x1_scale_scale__vv_add__conv3_4_x1_scale_bias,
                             /* output_IVRF_address */ 10083, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv3_4_x2_MRF, /* mrf_prefetch_next_size */ 64, /* swap_mrf_buffers */ false);
}

void conv34X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_4_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_4_x1_bn(d=224, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_4_x2_bn */
    /*      absorbed conv3_4_x2_scale */
    /*      absorbed relu3_4_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 10083, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 57, /* mulParam_address */ conv3_4_x2_bn_scale__vv_mul__conv3_4_x2_scale_scale, /* addParam_address */ conv3_4_x2_bn_bias__vv_mul__conv3_4_x2_scale_scale__vv_add__conv3_4_x2_scale_bias,
                             /* output_IVRF_address */ 9354, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv34X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_4_x2(d=32, h=27, d=27) = Convolution(conv3_4_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_4_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 9354, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat34(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_4(d=256, h=27, d=27) = Concat(Concat(concat_3_3(d=224, h=27, w=27), conv3_4_x2(d=32, h=27, w=27))) */

    /* Input concat_3_3 ISA_Mem_MvmInitialVrf memory: addresses [11541 - 12998] */
    /* Input conv3_4_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10083 - 11540] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv3_4_x1_bn. */

    /* Copy layer concat_3_3(d=0:128) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11541, 1, 27 * 27, 2);
    mv_mul(bs, mrf_start + 9);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083, 2);

    /* Concatenate layer concat_3_3(d=128:224) with layer conv3_4_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11541 + (2 - 1), 1, 27 * 27, 2);
    mv_mul(bs, mrf_start + 9);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_4_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083 + (2 - 1), 2);
}

void dummyConvConv35X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_5_x1_bn(d=256, h=27, d=27) = Convolution(concat_3_4(d=256, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_5_x1_bn */
    /*      absorbed conv3_5_x1_scale */
    /*      absorbed relu3_5_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 10083, /* output_depth */ 2, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv3_5_x1_bn_scale__vv_mul__conv3_5_x1_scale_scale, /* addParam_address */ conv3_5_x1_bn_bias__vv_mul__conv3_5_x1_scale_scale__vv_add__conv3_5_x1_scale_bias,
                             /* output_IVRF_address */ 11541, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv35X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_5_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_5_x1_bn(d=256, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_5_x2_bn */
    /*      absorbed conv3_5_x2_scale */
    /*      absorbed relu3_5_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 2,
                             /* input_address */ 11541, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv3_5_x2_bn_scale__vv_mul__conv3_5_x2_scale_scale, /* addParam_address */ conv3_5_x2_bn_bias__vv_mul__conv3_5_x2_scale_scale__vv_add__conv3_5_x2_scale_bias,
                             /* output_IVRF_address */ 9354, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv35X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_5_x2(d=32, h=27, d=27) = Convolution(conv3_5_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 9354, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 13, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 12270, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat35(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_5(d=288, h=27, d=27) = Concat(Concat(concat_3_4(d=256, h=27, w=27), conv3_5_x2(d=32, h=27, w=27))) */

    /* Input concat_3_4 ISA_Mem_MvmInitialVrf memory: addresses [10083 - 11540] */
    /* Input conv3_5_x2 ISA_Mem_MvmInitialVrf memory: addresses [12270 - 12998] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 2186] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv3_4_x1_bn. */

    /* Copy layer concat_3_4(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083 + inputDepth, 1, 27 * 27, 2);
        mv_mul(bs, mrf_start + 22);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 3);
    }

    /* Copy layer conv3_5_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12270, 1, 27 * 27, 1);
    mv_mul(bs, mrf_start + 22);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 2, 3);
}

void dummyConvConv36X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_6_x1_bn(d=288, h=27, d=27) = Convolution(concat_3_5(d=288, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_6_x1_bn */
    /*      absorbed conv3_6_x1_scale */
    /*      absorbed relu3_6_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 23, /* mulParam_address */ conv3_6_x1_bn_scale__vv_mul__conv3_6_x1_scale_scale, /* addParam_address */ conv3_6_x1_bn_bias__vv_mul__conv3_6_x1_scale_scale__vv_add__conv3_6_x1_scale_bias,
                             /* output_IVRF_address */ 2187, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv36X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_6_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_6_x1_bn(d=288, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_6_x2_bn */
    /*      absorbed conv3_6_x2_scale */
    /*      absorbed relu3_6_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 2187, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 24, /* mulParam_address */ conv3_6_x2_bn_scale__vv_mul__conv3_6_x2_scale_scale, /* addParam_address */ conv3_6_x2_bn_bias__vv_mul__conv3_6_x2_scale_scale__vv_add__conv3_6_x2_scale_bias,
                             /* output_IVRF_address */ 4374, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv36X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_6_x2(d=32, h=27, d=27) = Convolution(conv3_6_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_6_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 4374, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 27, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat36(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_6(d=320, h=27, d=27) = Concat(Concat(concat_3_5(d=288, h=27, w=27), conv3_6_x2(d=32, h=27, w=27))) */

    /* Input concat_3_5 ISA_Mem_MvmInitialVrf memory: addresses [0 - 2186] */
    /* Input conv3_6_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [2187 - 4373] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv3_4_x1_bn. */

    /* Copy layer concat_3_5(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 27 * 27, 3);
        mv_mul(bs, mrf_start + 36);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 2187 + inputDepth, 3);
    }

    /* Concatenate layer concat_3_5(d=256:288) with layer conv3_6_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (3 - 1), 1, 27 * 27, 3);
    mv_mul(bs, mrf_start + 36);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_6_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 2187 + (3 - 1), 3);
}

void dummyConvConv37X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_7_x1_bn(d=320, h=27, d=27) = Convolution(concat_3_6(d=320, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_7_x1_bn */
    /*      absorbed conv3_7_x1_scale */
    /*      absorbed relu3_7_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 2187, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 37, /* mulParam_address */ conv3_7_x1_bn_scale__vv_mul__conv3_7_x1_scale_scale, /* addParam_address */ conv3_7_x1_bn_bias__vv_mul__conv3_7_x1_scale_scale__vv_add__conv3_7_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv37X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_7_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_7_x1_bn(d=320, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_7_x2_bn */
    /*      absorbed conv3_7_x2_scale */
    /*      absorbed relu3_7_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 38, /* mulParam_address */ conv3_7_x2_bn_scale__vv_mul__conv3_7_x2_scale_scale, /* addParam_address */ conv3_7_x2_bn_bias__vv_mul__conv3_7_x2_scale_scale__vv_add__conv3_7_x2_scale_bias,
                             /* output_IVRF_address */ 4374, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv37X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_7_x2(d=32, h=27, d=27) = Convolution(conv3_7_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_7_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 4374, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 41, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat37(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_7(d=352, h=27, d=27) = Concat(Concat(concat_3_6(d=320, h=27, w=27), conv3_7_x2(d=32, h=27, w=27))) */

    /* Input concat_3_6 ISA_Mem_MvmInitialVrf memory: addresses [2187 - 4373] */
    /* Input conv3_7_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 2186] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv3_4_x1_bn. */

    /* Copy layer concat_3_6(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 2187 + inputDepth, 1, 27 * 27, 3);
        mv_mul(bs, mrf_start + 50);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 3);
    }

    /* Concatenate layer concat_3_6(d=256:320) with layer conv3_7_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 2187 + (3 - 1), 1, 27 * 27, 3);
    mv_mul(bs, mrf_start + 50);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_7_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (3 - 1), 3);
}

void dummyConvConv38X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_8_x1_bn(d=352, h=27, d=27) = Convolution(concat_3_7(d=352, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_8_x1_bn */
    /*      absorbed conv3_8_x1_scale */
    /*      absorbed relu3_8_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 51, /* mulParam_address */ conv3_8_x1_bn_scale__vv_mul__conv3_8_x1_scale_scale, /* addParam_address */ conv3_8_x1_bn_bias__vv_mul__conv3_8_x1_scale_scale__vv_add__conv3_8_x1_scale_bias,
                             /* output_IVRF_address */ 2187, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv38X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_8_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_8_x1_bn(d=352, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_8_x2_bn */
    /*      absorbed conv3_8_x2_scale */
    /*      absorbed relu3_8_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 2187, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 52, /* mulParam_address */ conv3_8_x2_bn_scale__vv_mul__conv3_8_x2_scale_scale, /* addParam_address */ conv3_8_x2_bn_bias__vv_mul__conv3_8_x2_scale_scale__vv_add__conv3_8_x2_scale_bias,
                             /* output_IVRF_address */ 4374, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ concat_3_8_MRF, /* mrf_prefetch_next_size */ 61, /* swap_mrf_buffers */ false);
}

void conv38X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_8_x2(d=32, h=27, d=27) = Convolution(conv3_8_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_8_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 4374, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 55, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void concat38(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_8(d=384, h=27, d=27) = Concat(Concat(concat_3_7(d=352, h=27, w=27), conv3_8_x2(d=32, h=27, w=27))) */

    /* Input concat_3_7 ISA_Mem_MvmInitialVrf memory: addresses [0 - 2186] */
    /* Input conv3_8_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [2187 - 4373] */

    /* This layer's matrix parameters were prefetched by layer conv3_8_x1. */

    /* Copy layer concat_3_7(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 27 * 27, 3);
        mv_mul(bs, mrf_start + 0);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 2187 + inputDepth, 3);
    }

    /* Concatenate layer concat_3_7(d=256:352) with layer conv3_8_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (3 - 1), 1, 27 * 27, 3);
    mv_mul(bs, mrf_start + 0);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_8_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 2187 + (3 - 1), 3);
}

void dummyConvConv39X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_9_x1_bn(d=384, h=27, d=27) = Convolution(concat_3_8(d=384, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_9_x1_bn */
    /*      absorbed conv3_9_x1_scale */
    /*      absorbed relu3_9_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 2187, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 1, /* mulParam_address */ conv3_9_x1_bn_scale__vv_mul__conv3_9_x1_scale_scale, /* addParam_address */ conv3_9_x1_bn_bias__vv_mul__conv3_9_x1_scale_scale__vv_add__conv3_9_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv39X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_9_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_9_x1_bn(d=384, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_9_x2_bn */
    /*      absorbed conv3_9_x2_scale */
    /*      absorbed relu3_9_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 2, /* mulParam_address */ conv3_9_x2_bn_scale__vv_mul__conv3_9_x2_scale_scale, /* addParam_address */ conv3_9_x2_bn_bias__vv_mul__conv3_9_x2_scale_scale__vv_add__conv3_9_x2_scale_bias,
                             /* output_IVRF_address */ 4374, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv39X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_9_x2(d=32, h=27, d=27) = Convolution(conv3_9_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 4374, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 5, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat39(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_9(d=416, h=27, d=27) = Concat(Concat(concat_3_8(d=384, h=27, w=27), conv3_9_x2(d=32, h=27, w=27))) */

    /* Input concat_3_8 ISA_Mem_MvmInitialVrf memory: addresses [2187 - 4373] */
    /* Input conv3_9_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10083 - 12998] */

    /* This layer's matrix parameters were prefetched by layer conv3_8_x1. */

    /* Copy layer concat_3_8(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 2187 + inputDepth, 1, 27 * 27, 3);
        mv_mul(bs, mrf_start + 14);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083 + inputDepth, 4);
    }

    /* Copy layer conv3_9_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 27 * 27, 1);
    mv_mul(bs, mrf_start + 14);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083 + 3, 4);
}

void dummyConvConv310X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_10_x1_bn(d=416, h=27, d=27) = Convolution(concat_3_9(d=416, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_10_x1_bn */
    /*      absorbed conv3_10_x1_scale */
    /*      absorbed relu3_10_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 10083, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 15, /* mulParam_address */ conv3_10_x1_bn_scale__vv_mul__conv3_10_x1_scale_scale, /* addParam_address */ conv3_10_x1_bn_bias__vv_mul__conv3_10_x1_scale_scale__vv_add__conv3_10_x1_scale_bias,
                             /* output_IVRF_address */ 7167, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv310X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_10_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_10_x1_bn(d=416, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_10_x2_bn */
    /*      absorbed conv3_10_x2_scale */
    /*      absorbed relu3_10_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 7167, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 16, /* mulParam_address */ conv3_10_x2_bn_scale__vv_mul__conv3_10_x2_scale_scale, /* addParam_address */ conv3_10_x2_bn_bias__vv_mul__conv3_10_x2_scale_scale__vv_add__conv3_10_x2_scale_bias,
                             /* output_IVRF_address */ 6438, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv310X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_10_x2(d=32, h=27, d=27) = Convolution(conv3_10_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_10_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 6438, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 20, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat310(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_10(d=448, h=27, d=27) = Concat(Concat(concat_3_9(d=416, h=27, w=27), conv3_10_x2(d=32, h=27, w=27))) */

    /* Input concat_3_9 ISA_Mem_MvmInitialVrf memory: addresses [10083 - 12998] */
    /* Input conv3_10_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [7167 - 10082] */

    /* This layer's matrix parameters were prefetched by layer conv3_8_x1. */

    /* Copy layer concat_3_9(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083 + inputDepth, 1, 27 * 27, 4);
        mv_mul(bs, mrf_start + 29);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7167 + inputDepth, 4);
    }

    /* Concatenate layer concat_3_9(d=384:416) with layer conv3_10_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083 + (4 - 1), 1, 27 * 27, 4);
    mv_mul(bs, mrf_start + 29);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_10_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7167 + (4 - 1), 4);
}

void dummyConvConv311X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_11_x1_bn(d=448, h=27, d=27) = Convolution(concat_3_10(d=448, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_11_x1_bn */
    /*      absorbed conv3_11_x1_scale */
    /*      absorbed relu3_11_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 7167, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 30, /* mulParam_address */ conv3_11_x1_bn_scale__vv_mul__conv3_11_x1_scale_scale, /* addParam_address */ conv3_11_x1_bn_bias__vv_mul__conv3_11_x1_scale_scale__vv_add__conv3_11_x1_scale_bias,
                             /* output_IVRF_address */ 10083, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv311X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_11_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_11_x1_bn(d=448, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_11_x2_bn */
    /*      absorbed conv3_11_x2_scale */
    /*      absorbed relu3_11_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 10083, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 31, /* mulParam_address */ conv3_11_x2_bn_scale__vv_mul__conv3_11_x2_scale_scale, /* addParam_address */ conv3_11_x2_bn_bias__vv_mul__conv3_11_x2_scale_scale__vv_add__conv3_11_x2_scale_bias,
                             /* output_IVRF_address */ 6438, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv311X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_11_x2(d=32, h=27, d=27) = Convolution(conv3_11_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_11_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 6438, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 35, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat311(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_11(d=480, h=27, d=27) = Concat(Concat(concat_3_10(d=448, h=27, w=27), conv3_11_x2(d=32, h=27, w=27))) */

    /* Input concat_3_10 ISA_Mem_MvmInitialVrf memory: addresses [7167 - 10082] */
    /* Input conv3_11_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10083 - 12998] */

    /* This layer's matrix parameters were prefetched by layer conv3_8_x1. */

    /* Copy layer concat_3_10(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 7167 + inputDepth, 1, 27 * 27, 4);
        mv_mul(bs, mrf_start + 44);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083 + inputDepth, 4);
    }

    /* Concatenate layer concat_3_10(d=384:448) with layer conv3_11_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 7167 + (4 - 1), 1, 27 * 27, 4);
    mv_mul(bs, mrf_start + 44);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_11_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10083 + (4 - 1), 4);
}

void dummyConvConv312X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_12_x1_bn(d=480, h=27, d=27) = Convolution(concat_3_11(d=480, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_12_x1_bn */
    /*      absorbed conv3_12_x1_scale */
    /*      absorbed relu3_12_x1 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 10083, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 45, /* mulParam_address */ conv3_12_x1_bn_scale__vv_mul__conv3_12_x1_scale_scale, /* addParam_address */ conv3_12_x1_bn_bias__vv_mul__conv3_12_x1_scale_scale__vv_add__conv3_12_x1_scale_bias,
                             /* output_IVRF_address */ 7167, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv312X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_12_x1(d=128, h=27, d=27) = Convolution(dummy_conv_conv3_12_x1_bn(d=480, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_12_x2_bn */
    /*      absorbed conv3_12_x2_scale */
    /*      absorbed relu3_12_x2 */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 7167, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 46, /* mulParam_address */ conv3_12_x2_bn_scale__vv_mul__conv3_12_x2_scale_scale, /* addParam_address */ conv3_12_x2_bn_bias__vv_mul__conv3_12_x2_scale_scale__vv_add__conv3_12_x2_scale_bias,
                             /* output_IVRF_address */ 6438, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv312X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv3_12_x2(d=32, h=27, d=27) = Convolution(conv3_12_x1(d=128, h=27, w=27), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv3_12_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 1,
                             /* input_address */ 6438, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 50, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat312(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_3_12(d=512, h=27, d=27) = Concat(Concat(concat_3_11(d=480, h=27, w=27), conv3_12_x2(d=32, h=27, w=27))) */

    /* Input concat_3_11 ISA_Mem_MvmInitialVrf memory: addresses [10083 - 12998] */
    /* Input conv3_12_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 728] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [7167 - 10082] */

    /* This layer's matrix parameters were prefetched by layer conv3_8_x1. */
    /* This layer prefetches the matrix parameters for layers conv3_blk, dummy_conv_conv4_1_x1_bn, conv4_1_x1, conv4_1_x2, concat_4_1, dummy_conv_conv4_2_x1_bn, conv4_2_x1, conv4_2_x2, concat_4_2, dummy_conv_conv4_3_x1_bn, conv4_3_x1, conv4_3_x2, concat_4_3, dummy_conv_conv4_4_x1_bn, conv4_4_x1, conv4_4_x2, concat_4_4. */

    /* Prefetch the matrix parameters for the next group of layers. */
    moveFilterCount128(bs, ISA_Mem_Dram, conv3_blk_MRF, ISA_Mem_MatrixRf, mrf_next, 1, 64);

    /* Copy layer concat_3_11(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083 + inputDepth, 1, 27 * 27, 4);
        mv_mul(bs, mrf_start + 59);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7167 + inputDepth, 4);
    }

    /* Concatenate layer concat_3_11(d=384:480) with layer conv3_12_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 10083 + (4 - 1), 1, 27 * 27, 4);
    mv_mul(bs, mrf_start + 59);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv3_12_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7167 + (4 - 1), 4);
}

void dummyConvConv3BlkBn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv3_blk_bn(d=512, h=27, d=27) = Convolution(concat_3_12(d=512, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv3_blk_bn */
    /*      absorbed conv3_blk_scale */
    /*      absorbed relu3_blk */

    genericConvolution(bs, /* input_height */ 27, /* input_width */ 27, /* input_depth */ 4,
                             /* input_address */ 7167, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 60, /* mulParam_address */ conv3_blk_bn_scale__vv_mul__conv3_blk_scale_scale, /* addParam_address */ conv3_blk_bn_bias__vv_mul__conv3_blk_scale_scale__vv_add__conv3_blk_scale_bias,
                             /* output_IVRF_address */ 10083, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv3Blk(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution conv3_blk(d=256, h=27, d=27) = Convolution(dummy_conv_conv3_blk_bn(d=512, h=27, w=27), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    includes sublayer pool3(d=256, h=13, d=13) = AveragePool(conv3_blk(d=256, h=27, w=27), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress dummy_conv_conv3_blk_bn_inIndex;
    dummy_conv_conv3_blk_bn_inIndex=10083;
    moveFilterCount128(bs, ISA_Mem_Dram, conv3_blk_MRF+0*64, ISA_Mem_MatrixRf, mrf_start, 1, 64);
    ISA_ExtAddress tmp_MVMIVRF=9394, tmp_MVMIVRF_next=9569;
    /* Layer conv3_blk tile size 27*108 */
    /* Temp vars and parameters for input layer conv3_blk */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_conv3_blk_in=10083,g0_conv3_blk_inIterator=10083;
    ISA_ExtAddress g0_conv3_blk_available = 729;
    ISA_ExtAddress g0_conv3_blk_outOffset=0;
    int g0_conv3_blk_iterationsLeft=729;
    /* Layer pool3 tile size 27*54 */
    /* Temp vars and parameters for input layer pool3 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_pool3_in=7936,g1_pool3_inIterator=7936;
    ISA_ExtAddress g1_pool3_available = 729;
    ISA_ExtAddress g1_pool3_accumulators=9394;
    ISA_ExtAddress g1_pool3_availableVerticalRows=0;
    ISA_ExtAddress g1_pool3_outOffset=0;
    int g1_pool3_iterationsLeft=169;
    vRead1D(bs, ISA_Mem_Dram, pool3_scale, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    /* Loop until we've read all outputs */
    while (g1_pool3_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_conv3_blk_iterationsLeft>0) {


            /* Start of layer 0 in group 0 (conv3_blk) */
            /* Tile size 27*108 dimPerStep 4 */
            for(int outRow=0;outRow<2;outRow++) {
                g0_conv3_blk_inIterator = g0_conv3_blk_in;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_conv3_blk_inIterator, 4, 27 * 27, 4);
                mv_mul(bs, mrf_start+0+outRow*4);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7936+g0_conv3_blk_outOffset+outRow+0, 2);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+g0_conv3_blk_outOffset+outRow+0, 2);
            }
            g0_conv3_blk_iterationsLeft-=g0_conv3_blk_available;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_conv3_blk_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_pool3_available <= 1458, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if ((g1_pool3_available >= 54) || ((g0_conv3_blk_iterationsLeft==0))) {

            /* Start of layer 0 in group 1 (pool3) */
            /* Tile size 27*54 dimPerStep 2 */
            /* Decompose the AVE-pool into 1 horizontal pool operations followed by and 1 vertical pool operations */
            /* All of the data will be processed en-masse, since our input data doesn't need paged in. */
            /* Perform 13 horizontal pool operations on 27 or 26 rows with 1 steps */
            /* g1_pool3_inIterator iterates horizontal pool operations (INPUTS) */
            /* g1_pool3_in iterates vertical pool operations (OUTPUTS) */
            /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
            int horizontalRows=27;
            int verticalRows=13;
            g1_pool3_available -= verticalRows*108;
            ISA_ExtAddress curOffset;
            for(unsigned vec=0; vec<2; vec++) {
                curOffset=g1_pool3_inIterator+vec;
                for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 13, 4);
                    mv_mul(bs, mrf_start+8);
                    /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-7936+0+2, 4);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool3_accumulators+rowIterator*13, 1);
                    curOffset+=54;
                }
                /* Horizontal sweep must end up in g1_pool3_accumulators because we can't read-modify-write ASVRF in a single chain */
                curOffset=g1_pool3_inIterator+vec;
                for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool3_accumulators+rowIterator*13, 1, 13, 1);
                    mv_mul(bs, mrf_start+8);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 2);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+0-7936, 2);
                    curOffset+=54;
                }
            }
            /* Update horizontal pool iterator start */
            g1_pool3_inIterator = curOffset-1;
            curOffset=g1_pool3_in;
            /* The horizontal sweep took care of the multiple input vectors */
            ISA_ExtAddress nextOffset=curOffset;
            nextOffset+=54;
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 26, 1);
                mv_mul(bs, mrf_start+8);
                /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-7936+0, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool3_accumulators + rowIterator * 13 * 2, 1);

                curOffset+=108;
                nextOffset+=108;
            }
            g1_pool3_in = curOffset;

            /* Perform any additional operations in the chain after the pooling. */
            /* In particular this is needed for average pools, because the current hardware doesn't support a multiply after an AddSub VRF 1 add/max/sub operation. */
            /* Note that we forgo using a row loop here as it isn't needed */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool3_accumulators, 1, verticalRows * 13 * 2, 1);
            mv_mul(bs, mrf_start + 8);
            vv_mul(bs, 0); /* includes: pool3: scale */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9745 + g1_pool3_outOffset, 1);

            g1_pool3_outOffset += verticalRows * 13 * 2;
            g1_pool3_iterationsLeft-=verticalRows*13;
            /* Make sure we didn't loop too many times (emulator only) */
            Emulator_HEX_Assert(g1_pool3_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void dummyConvConv41X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_1_x1_bn(d=256, h=13, d=13) = Convolution(pool3(d=256, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_1_x1_bn */
    /*      absorbed conv4_1_x1_scale */
    /*      absorbed relu4_1_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 2,
                             /* input_address */ 9745, /* output_depth */ 2, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 9, /* mulParam_address */ conv4_1_x1_bn_scale__vv_mul__conv4_1_x1_scale_scale, /* addParam_address */ conv4_1_x1_bn_bias__vv_mul__conv4_1_x1_scale_scale__vv_add__conv4_1_x1_scale_bias,
                             /* output_IVRF_address */ 12661, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv41X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_1_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_1_x1_bn(d=256, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_1_x2_bn */
    /*      absorbed conv4_1_x2_scale */
    /*      absorbed relu4_1_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 2,
                             /* input_address */ 12661, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 10, /* mulParam_address */ conv4_1_x2_bn_scale__vv_mul__conv4_1_x2_scale_scale, /* addParam_address */ conv4_1_x2_bn_bias__vv_mul__conv4_1_x2_scale_scale__vv_add__conv4_1_x2_scale_bias,
                             /* output_IVRF_address */ 12492, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv41X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_1_x2(d=32, h=13, d=13) = Convolution(conv4_1_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 12492, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 12, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 12830, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat41(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_1(d=288, h=13, d=13) = Concat(Concat(pool3(d=256, h=13, w=13), conv4_1_x2(d=32, h=13, w=13))) */

    /* Input pool3 ISA_Mem_MvmInitialVrf memory: addresses [9745 - 10082] */
    /* Input conv4_1_x2 ISA_Mem_MvmInitialVrf memory: addresses [12830 - 12998] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 506] */

    /* This layer's matrix parameters were prefetched by layer concat_3_12. */

    /* Copy layer pool3(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 9745 + inputDepth, 1, 13 * 13, 2);
        mv_mul(bs, mrf_start + 21);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 3);
    }

    /* Copy layer conv4_1_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12830, 1, 13 * 13, 1);
    mv_mul(bs, mrf_start + 21);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 2, 3);
}

void dummyConvConv42X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_2_x1_bn(d=288, h=13, d=13) = Convolution(concat_4_1(d=288, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_2_x1_bn */
    /*      absorbed conv4_2_x1_scale */
    /*      absorbed relu4_2_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 22, /* mulParam_address */ conv4_2_x1_bn_scale__vv_mul__conv4_2_x1_scale_scale, /* addParam_address */ conv4_2_x1_bn_bias__vv_mul__conv4_2_x1_scale_scale__vv_add__conv4_2_x1_scale_bias,
                             /* output_IVRF_address */ 507, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv42X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_2_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_2_x1_bn(d=288, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_2_x2_bn */
    /*      absorbed conv4_2_x2_scale */
    /*      absorbed relu4_2_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 507, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 23, /* mulParam_address */ conv4_2_x2_bn_scale__vv_mul__conv4_2_x2_scale_scale, /* addParam_address */ conv4_2_x2_bn_bias__vv_mul__conv4_2_x2_scale_scale__vv_add__conv4_2_x2_scale_bias,
                             /* output_IVRF_address */ 1014, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv42X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_2_x2(d=32, h=13, d=13) = Convolution(conv4_2_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_2_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1014, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 26, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat42(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_2(d=320, h=13, d=13) = Concat(Concat(concat_4_1(d=288, h=13, w=13), conv4_2_x2(d=32, h=13, w=13))) */

    /* Input concat_4_1 ISA_Mem_MvmInitialVrf memory: addresses [0 - 506] */
    /* Input conv4_2_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [507 - 1013] */

    /* This layer's matrix parameters were prefetched by layer concat_3_12. */

    /* Copy layer concat_4_1(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 13 * 13, 3);
        mv_mul(bs, mrf_start + 35);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 507 + inputDepth, 3);
    }

    /* Concatenate layer concat_4_1(d=256:288) with layer conv4_2_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (3 - 1), 1, 13 * 13, 3);
    mv_mul(bs, mrf_start + 35);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_2_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 507 + (3 - 1), 3);
}

void dummyConvConv43X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_3_x1_bn(d=320, h=13, d=13) = Convolution(concat_4_2(d=320, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_3_x1_bn */
    /*      absorbed conv4_3_x1_scale */
    /*      absorbed relu4_3_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 507, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 36, /* mulParam_address */ conv4_3_x1_bn_scale__vv_mul__conv4_3_x1_scale_scale, /* addParam_address */ conv4_3_x1_bn_bias__vv_mul__conv4_3_x1_scale_scale__vv_add__conv4_3_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv43X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_3_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_3_x1_bn(d=320, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_3_x2_bn */
    /*      absorbed conv4_3_x2_scale */
    /*      absorbed relu4_3_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 37, /* mulParam_address */ conv4_3_x2_bn_scale__vv_mul__conv4_3_x2_scale_scale, /* addParam_address */ conv4_3_x2_bn_bias__vv_mul__conv4_3_x2_scale_scale__vv_add__conv4_3_x2_scale_bias,
                             /* output_IVRF_address */ 1014, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv43X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_3_x2(d=32, h=13, d=13) = Convolution(conv4_3_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_3_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1014, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 40, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat43(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_3(d=352, h=13, d=13) = Concat(Concat(concat_4_2(d=320, h=13, w=13), conv4_3_x2(d=32, h=13, w=13))) */

    /* Input concat_4_2 ISA_Mem_MvmInitialVrf memory: addresses [507 - 1013] */
    /* Input conv4_3_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 506] */

    /* This layer's matrix parameters were prefetched by layer concat_3_12. */

    /* Copy layer concat_4_2(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 507 + inputDepth, 1, 13 * 13, 3);
        mv_mul(bs, mrf_start + 49);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 3);
    }

    /* Concatenate layer concat_4_2(d=256:320) with layer conv4_3_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 507 + (3 - 1), 1, 13 * 13, 3);
    mv_mul(bs, mrf_start + 49);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_3_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (3 - 1), 3);
}

void dummyConvConv44X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_4_x1_bn(d=352, h=13, d=13) = Convolution(concat_4_3(d=352, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_4_x1_bn */
    /*      absorbed conv4_4_x1_scale */
    /*      absorbed relu4_4_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 50, /* mulParam_address */ conv4_4_x1_bn_scale__vv_mul__conv4_4_x1_scale_scale, /* addParam_address */ conv4_4_x1_bn_bias__vv_mul__conv4_4_x1_scale_scale__vv_add__conv4_4_x1_scale_bias,
                             /* output_IVRF_address */ 507, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv44X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_4_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_4_x1_bn(d=352, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_4_x2_bn */
    /*      absorbed conv4_4_x2_scale */
    /*      absorbed relu4_4_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 507, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 51, /* mulParam_address */ conv4_4_x2_bn_scale__vv_mul__conv4_4_x2_scale_scale, /* addParam_address */ conv4_4_x2_bn_bias__vv_mul__conv4_4_x2_scale_scale__vv_add__conv4_4_x2_scale_bias,
                             /* output_IVRF_address */ 1014, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv44X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_4_x2(d=32, h=13, d=13) = Convolution(conv4_4_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_4_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1014, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 54, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ dummy_conv_conv4_5_x1_bn_MRF, /* mrf_prefetch_next_size */ 64, /* swap_mrf_buffers */ false);
}

void concat44(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_4(d=384, h=13, d=13) = Concat(Concat(concat_4_3(d=352, h=13, w=13), conv4_4_x2(d=32, h=13, w=13))) */

    /* Input concat_4_3 ISA_Mem_MvmInitialVrf memory: addresses [0 - 506] */
    /* Input conv4_4_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [507 - 1013] */

    /* This layer's matrix parameters were prefetched by layer concat_3_12. */

    /* Copy layer concat_4_3(d=0:256) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 2; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 13 * 13, 3);
        mv_mul(bs, mrf_start + 63);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 507 + inputDepth, 3);
    }

    /* Concatenate layer concat_4_3(d=256:352) with layer conv4_4_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (3 - 1), 1, 13 * 13, 3);
    mv_mul(bs, mrf_start + 63);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_4_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 507 + (3 - 1), 3);

    /* Swap the MRF buffers, as the next layer dummy_conv_conv4_5_x1_bn begins the start of a new MRF group. */
    mrf_tmp = mrf_start;
    mrf_start = mrf_next;
    mrf_next = mrf_tmp;
}

void dummyConvConv45X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_5_x1_bn(d=384, h=13, d=13) = Convolution(concat_4_4(d=384, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_5_x1_bn */
    /*      absorbed conv4_5_x1_scale */
    /*      absorbed relu4_5_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 507, /* output_depth */ 3, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 0, /* mulParam_address */ conv4_5_x1_bn_scale__vv_mul__conv4_5_x1_scale_scale, /* addParam_address */ conv4_5_x1_bn_bias__vv_mul__conv4_5_x1_scale_scale__vv_add__conv4_5_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv45X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_5_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_5_x1_bn(d=384, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_5_x2_bn */
    /*      absorbed conv4_5_x2_scale */
    /*      absorbed relu4_5_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 3,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 1, /* mulParam_address */ conv4_5_x2_bn_scale__vv_mul__conv4_5_x2_scale_scale, /* addParam_address */ conv4_5_x2_bn_bias__vv_mul__conv4_5_x2_scale_scale__vv_add__conv4_5_x2_scale_bias,
                             /* output_IVRF_address */ 1014, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv45X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_5_x2(d=32, h=13, d=13) = Convolution(conv4_5_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1014, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 4, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat45(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_5(d=416, h=13, d=13) = Concat(Concat(concat_4_4(d=384, h=13, w=13), conv4_5_x2(d=32, h=13, w=13))) */

    /* Input concat_4_4 ISA_Mem_MvmInitialVrf memory: addresses [507 - 1013] */
    /* Input conv4_5_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12323 - 12998] */

    /* This layer's matrix parameters were prefetched by layer conv4_4_x2. */

    /* Copy layer concat_4_4(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 507 + inputDepth, 1, 13 * 13, 3);
        mv_mul(bs, mrf_start + 13);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12323 + inputDepth, 4);
    }

    /* Copy layer conv4_5_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 13 * 13, 1);
    mv_mul(bs, mrf_start + 13);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12323 + 3, 4);
}

void dummyConvConv46X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_6_x1_bn(d=416, h=13, d=13) = Convolution(concat_4_5(d=416, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_6_x1_bn */
    /*      absorbed conv4_6_x1_scale */
    /*      absorbed relu4_6_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 12323, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 14, /* mulParam_address */ conv4_6_x1_bn_scale__vv_mul__conv4_6_x1_scale_scale, /* addParam_address */ conv4_6_x1_bn_bias__vv_mul__conv4_6_x1_scale_scale__vv_add__conv4_6_x1_scale_bias,
                             /* output_IVRF_address */ 11647, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv46X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_6_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_6_x1_bn(d=416, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_6_x2_bn */
    /*      absorbed conv4_6_x2_scale */
    /*      absorbed relu4_6_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 11647, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 15, /* mulParam_address */ conv4_6_x2_bn_scale__vv_mul__conv4_6_x2_scale_scale, /* addParam_address */ conv4_6_x2_bn_bias__vv_mul__conv4_6_x2_scale_scale__vv_add__conv4_6_x2_scale_bias,
                             /* output_IVRF_address */ 11478, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv46X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_6_x2(d=32, h=13, d=13) = Convolution(conv4_6_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_6_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 11478, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 19, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat46(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_6(d=448, h=13, d=13) = Concat(Concat(concat_4_5(d=416, h=13, w=13), conv4_6_x2(d=32, h=13, w=13))) */

    /* Input concat_4_5 ISA_Mem_MvmInitialVrf memory: addresses [12323 - 12998] */
    /* Input conv4_6_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12322] */

    /* This layer's matrix parameters were prefetched by layer conv4_4_x2. */

    /* Copy layer concat_4_5(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12323 + inputDepth, 1, 13 * 13, 4);
        mv_mul(bs, mrf_start + 28);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 4);
    }

    /* Concatenate layer concat_4_5(d=384:416) with layer conv4_6_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12323 + (4 - 1), 1, 13 * 13, 4);
    mv_mul(bs, mrf_start + 28);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_6_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + (4 - 1), 4);
}

void dummyConvConv47X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_7_x1_bn(d=448, h=13, d=13) = Convolution(concat_4_6(d=448, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_7_x1_bn */
    /*      absorbed conv4_7_x1_scale */
    /*      absorbed relu4_7_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 11647, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 29, /* mulParam_address */ conv4_7_x1_bn_scale__vv_mul__conv4_7_x1_scale_scale, /* addParam_address */ conv4_7_x1_bn_bias__vv_mul__conv4_7_x1_scale_scale__vv_add__conv4_7_x1_scale_bias,
                             /* output_IVRF_address */ 12323, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv47X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_7_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_7_x1_bn(d=448, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_7_x2_bn */
    /*      absorbed conv4_7_x2_scale */
    /*      absorbed relu4_7_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 12323, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 30, /* mulParam_address */ conv4_7_x2_bn_scale__vv_mul__conv4_7_x2_scale_scale, /* addParam_address */ conv4_7_x2_bn_bias__vv_mul__conv4_7_x2_scale_scale__vv_add__conv4_7_x2_scale_bias,
                             /* output_IVRF_address */ 11478, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv47X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_7_x2(d=32, h=13, d=13) = Convolution(conv4_7_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_7_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 11478, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 34, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat47(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_7(d=480, h=13, d=13) = Concat(Concat(concat_4_6(d=448, h=13, w=13), conv4_7_x2(d=32, h=13, w=13))) */

    /* Input concat_4_6 ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12322] */
    /* Input conv4_7_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12323 - 12998] */

    /* This layer's matrix parameters were prefetched by layer conv4_4_x2. */

    /* Copy layer concat_4_6(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 1, 13 * 13, 4);
        mv_mul(bs, mrf_start + 43);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12323 + inputDepth, 4);
    }

    /* Concatenate layer concat_4_6(d=384:448) with layer conv4_7_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + (4 - 1), 1, 13 * 13, 4);
    mv_mul(bs, mrf_start + 43);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_7_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12323 + (4 - 1), 4);
}

void dummyConvConv48X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_8_x1_bn(d=480, h=13, d=13) = Convolution(concat_4_7(d=480, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_8_x1_bn */
    /*      absorbed conv4_8_x1_scale */
    /*      absorbed relu4_8_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 12323, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 44, /* mulParam_address */ conv4_8_x1_bn_scale__vv_mul__conv4_8_x1_scale_scale, /* addParam_address */ conv4_8_x1_bn_bias__vv_mul__conv4_8_x1_scale_scale__vv_add__conv4_8_x1_scale_bias,
                             /* output_IVRF_address */ 11647, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv48X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_8_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_8_x1_bn(d=480, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_8_x2_bn */
    /*      absorbed conv4_8_x2_scale */
    /*      absorbed relu4_8_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 11647, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 45, /* mulParam_address */ conv4_8_x2_bn_scale__vv_mul__conv4_8_x2_scale_scale, /* addParam_address */ conv4_8_x2_bn_bias__vv_mul__conv4_8_x2_scale_scale__vv_add__conv4_8_x2_scale_bias,
                             /* output_IVRF_address */ 11478, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv48X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_8_x2(d=32, h=13, d=13) = Convolution(conv4_8_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_8_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 11478, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 49, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat48(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_8(d=512, h=13, d=13) = Concat(Concat(concat_4_7(d=480, h=13, w=13), conv4_8_x2(d=32, h=13, w=13))) */

    /* Input concat_4_7 ISA_Mem_MvmInitialVrf memory: addresses [12323 - 12998] */
    /* Input conv4_8_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12322] */

    /* This layer's matrix parameters were prefetched by layer conv4_4_x2. */

    /* Copy layer concat_4_7(d=0:384) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 3; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12323 + inputDepth, 1, 13 * 13, 4);
        mv_mul(bs, mrf_start + 58);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 4);
    }

    /* Concatenate layer concat_4_7(d=384:480) with layer conv4_8_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12323 + (4 - 1), 1, 13 * 13, 4);
    mv_mul(bs, mrf_start + 58);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_8_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + (4 - 1), 4);
}

void dummyConvConv49X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_9_x1_bn(d=512, h=13, d=13) = Convolution(concat_4_8(d=512, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_9_x1_bn */
    /*      absorbed conv4_9_x1_scale */
    /*      absorbed relu4_9_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 11647, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 59, /* mulParam_address */ conv4_9_x1_bn_scale__vv_mul__conv4_9_x1_scale_scale, /* addParam_address */ conv4_9_x1_bn_bias__vv_mul__conv4_9_x1_scale_scale__vv_add__conv4_9_x1_scale_bias,
                             /* output_IVRF_address */ 12323, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv4_9_x2_MRF, /* mrf_prefetch_next_size */ 64, /* swap_mrf_buffers */ false);
}

void conv49X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_9_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_9_x1_bn(d=512, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_9_x2_bn */
    /*      absorbed conv4_9_x2_scale */
    /*      absorbed relu4_9_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 4,
                             /* input_address */ 12323, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 60, /* mulParam_address */ conv4_9_x2_bn_scale__vv_mul__conv4_9_x2_scale_scale, /* addParam_address */ conv4_9_x2_bn_bias__vv_mul__conv4_9_x2_scale_scale__vv_add__conv4_9_x2_scale_bias,
                             /* output_IVRF_address */ 11478, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv49X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_9_x2(d=32, h=13, d=13) = Convolution(conv4_9_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 11478, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 12830, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat49(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_9(d=544, h=13, d=13) = Concat(Concat(concat_4_8(d=512, h=13, w=13), conv4_9_x2(d=32, h=13, w=13))) */

    /* Input concat_4_8 ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12322] */
    /* Input conv4_9_x2 ISA_Mem_MvmInitialVrf memory: addresses [12830 - 12998] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 844] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_9_x1_bn. */

    /* Copy layer concat_4_8(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 1, 13 * 13, 4);
        mv_mul(bs, mrf_start + 9);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 5);
    }

    /* Copy layer conv4_9_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12830, 1, 13 * 13, 1);
    mv_mul(bs, mrf_start + 9);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 4, 5);
}

void dummyConvConv410X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_10_x1_bn(d=544, h=13, d=13) = Convolution(concat_4_9(d=544, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_10_x1_bn */
    /*      absorbed conv4_10_x1_scale */
    /*      absorbed relu4_10_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv4_10_x1_bn_scale__vv_mul__conv4_10_x1_scale_scale, /* addParam_address */ conv4_10_x1_bn_bias__vv_mul__conv4_10_x1_scale_scale__vv_add__conv4_10_x1_scale_bias,
                             /* output_IVRF_address */ 845, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv410X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_10_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_10_x1_bn(d=544, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_10_x2_bn */
    /*      absorbed conv4_10_x2_scale */
    /*      absorbed relu4_10_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 845, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv4_10_x2_bn_scale__vv_mul__conv4_10_x2_scale_scale, /* addParam_address */ conv4_10_x2_bn_bias__vv_mul__conv4_10_x2_scale_scale__vv_add__conv4_10_x2_scale_bias,
                             /* output_IVRF_address */ 1690, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv410X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_10_x2(d=32, h=13, d=13) = Convolution(conv4_10_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_10_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1690, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 16, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat410(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_10(d=576, h=13, d=13) = Concat(Concat(concat_4_9(d=544, h=13, w=13), conv4_10_x2(d=32, h=13, w=13))) */

    /* Input concat_4_9 ISA_Mem_MvmInitialVrf memory: addresses [0 - 844] */
    /* Input conv4_10_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [845 - 1689] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_9_x1_bn. */

    /* Copy layer concat_4_9(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 13 * 13, 5);
        mv_mul(bs, mrf_start + 25);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 845 + inputDepth, 5);
    }

    /* Concatenate layer concat_4_9(d=512:544) with layer conv4_10_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (5 - 1), 1, 13 * 13, 5);
    mv_mul(bs, mrf_start + 25);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_10_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 845 + (5 - 1), 5);
}

void dummyConvConv411X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_11_x1_bn(d=576, h=13, d=13) = Convolution(concat_4_10(d=576, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_11_x1_bn */
    /*      absorbed conv4_11_x1_scale */
    /*      absorbed relu4_11_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 845, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 26, /* mulParam_address */ conv4_11_x1_bn_scale__vv_mul__conv4_11_x1_scale_scale, /* addParam_address */ conv4_11_x1_bn_bias__vv_mul__conv4_11_x1_scale_scale__vv_add__conv4_11_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv411X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_11_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_11_x1_bn(d=576, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_11_x2_bn */
    /*      absorbed conv4_11_x2_scale */
    /*      absorbed relu4_11_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 27, /* mulParam_address */ conv4_11_x2_bn_scale__vv_mul__conv4_11_x2_scale_scale, /* addParam_address */ conv4_11_x2_bn_bias__vv_mul__conv4_11_x2_scale_scale__vv_add__conv4_11_x2_scale_bias,
                             /* output_IVRF_address */ 1690, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv411X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_11_x2(d=32, h=13, d=13) = Convolution(conv4_11_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_11_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1690, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 32, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat411(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_11(d=608, h=13, d=13) = Concat(Concat(concat_4_10(d=576, h=13, w=13), conv4_11_x2(d=32, h=13, w=13))) */

    /* Input concat_4_10 ISA_Mem_MvmInitialVrf memory: addresses [845 - 1689] */
    /* Input conv4_11_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 844] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_9_x1_bn. */

    /* Copy layer concat_4_10(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 845 + inputDepth, 1, 13 * 13, 5);
        mv_mul(bs, mrf_start + 41);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 5);
    }

    /* Concatenate layer concat_4_10(d=512:576) with layer conv4_11_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 845 + (5 - 1), 1, 13 * 13, 5);
    mv_mul(bs, mrf_start + 41);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_11_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (5 - 1), 5);
}

void dummyConvConv412X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_12_x1_bn(d=608, h=13, d=13) = Convolution(concat_4_11(d=608, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_12_x1_bn */
    /*      absorbed conv4_12_x1_scale */
    /*      absorbed relu4_12_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 42, /* mulParam_address */ conv4_12_x1_bn_scale__vv_mul__conv4_12_x1_scale_scale, /* addParam_address */ conv4_12_x1_bn_bias__vv_mul__conv4_12_x1_scale_scale__vv_add__conv4_12_x1_scale_bias,
                             /* output_IVRF_address */ 845, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv412X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_12_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_12_x1_bn(d=608, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_12_x2_bn */
    /*      absorbed conv4_12_x2_scale */
    /*      absorbed relu4_12_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 845, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 43, /* mulParam_address */ conv4_12_x2_bn_scale__vv_mul__conv4_12_x2_scale_scale, /* addParam_address */ conv4_12_x2_bn_bias__vv_mul__conv4_12_x2_scale_scale__vv_add__conv4_12_x2_scale_bias,
                             /* output_IVRF_address */ 1690, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv412X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_12_x2(d=32, h=13, d=13) = Convolution(conv4_12_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_12_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1690, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 48, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat412(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_12(d=640, h=13, d=13) = Concat(Concat(concat_4_11(d=608, h=13, w=13), conv4_12_x2(d=32, h=13, w=13))) */

    /* Input concat_4_11 ISA_Mem_MvmInitialVrf memory: addresses [0 - 844] */
    /* Input conv4_12_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [845 - 1689] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_9_x1_bn. */

    /* Copy layer concat_4_11(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 13 * 13, 5);
        mv_mul(bs, mrf_start + 57);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 845 + inputDepth, 5);
    }

    /* Concatenate layer concat_4_11(d=512:608) with layer conv4_12_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (5 - 1), 1, 13 * 13, 5);
    mv_mul(bs, mrf_start + 57);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_12_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 845 + (5 - 1), 5);
}

void dummyConvConv413X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_13_x1_bn(d=640, h=13, d=13) = Convolution(concat_4_12(d=640, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_13_x1_bn */
    /*      absorbed conv4_13_x1_scale */
    /*      absorbed relu4_13_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 845, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 58, /* mulParam_address */ conv4_13_x1_bn_scale__vv_mul__conv4_13_x1_scale_scale, /* addParam_address */ conv4_13_x1_bn_bias__vv_mul__conv4_13_x1_scale_scale__vv_add__conv4_13_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv4_13_x2_MRF, /* mrf_prefetch_next_size */ 62, /* swap_mrf_buffers */ false);
}

void conv413X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_13_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_13_x1_bn(d=640, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_13_x2_bn */
    /*      absorbed conv4_13_x2_scale */
    /*      absorbed relu4_13_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 59, /* mulParam_address */ conv4_13_x2_bn_scale__vv_mul__conv4_13_x2_scale_scale, /* addParam_address */ conv4_13_x2_bn_bias__vv_mul__conv4_13_x2_scale_scale__vv_add__conv4_13_x2_scale_bias,
                             /* output_IVRF_address */ 1690, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv413X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_13_x2(d=32, h=13, d=13) = Convolution(conv4_13_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 1690, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat413(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_13(d=672, h=13, d=13) = Concat(Concat(concat_4_12(d=640, h=13, w=13), conv4_13_x2(d=32, h=13, w=13))) */

    /* Input concat_4_12 ISA_Mem_MvmInitialVrf memory: addresses [845 - 1689] */
    /* Input conv4_13_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11985 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_13_x1_bn. */

    /* Copy layer concat_4_12(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 845 + inputDepth, 1, 13 * 13, 5);
        mv_mul(bs, mrf_start + 9);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11985 + inputDepth, 6);
    }

    /* Copy layer conv4_13_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 13 * 13, 1);
    mv_mul(bs, mrf_start + 9);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11985 + 5, 6);
}

void dummyConvConv414X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_14_x1_bn(d=672, h=13, d=13) = Convolution(concat_4_13(d=672, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_14_x1_bn */
    /*      absorbed conv4_14_x1_scale */
    /*      absorbed relu4_14_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 11985, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv4_14_x1_bn_scale__vv_mul__conv4_14_x1_scale_scale, /* addParam_address */ conv4_14_x1_bn_bias__vv_mul__conv4_14_x1_scale_scale__vv_add__conv4_14_x1_scale_bias,
                             /* output_IVRF_address */ 10971, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv414X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_14_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_14_x1_bn(d=672, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_14_x2_bn */
    /*      absorbed conv4_14_x2_scale */
    /*      absorbed relu4_14_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 10971, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv4_14_x2_bn_scale__vv_mul__conv4_14_x2_scale_scale, /* addParam_address */ conv4_14_x2_bn_bias__vv_mul__conv4_14_x2_scale_scale__vv_add__conv4_14_x2_scale_bias,
                             /* output_IVRF_address */ 10802, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv414X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_14_x2(d=32, h=13, d=13) = Convolution(conv4_14_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_14_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10802, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 17, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat414(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_14(d=704, h=13, d=13) = Concat(Concat(concat_4_13(d=672, h=13, w=13), conv4_14_x2(d=32, h=13, w=13))) */

    /* Input concat_4_13 ISA_Mem_MvmInitialVrf memory: addresses [11985 - 12998] */
    /* Input conv4_14_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10971 - 11984] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_13_x1_bn. */

    /* Copy layer concat_4_13(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11985 + inputDepth, 1, 13 * 13, 6);
        mv_mul(bs, mrf_start + 26);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10971 + inputDepth, 6);
    }

    /* Concatenate layer concat_4_13(d=640:672) with layer conv4_14_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11985 + (6 - 1), 1, 13 * 13, 6);
    mv_mul(bs, mrf_start + 26);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_14_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10971 + (6 - 1), 6);
}

void dummyConvConv415X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_15_x1_bn(d=704, h=13, d=13) = Convolution(concat_4_14(d=704, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_15_x1_bn */
    /*      absorbed conv4_15_x1_scale */
    /*      absorbed relu4_15_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 10971, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 27, /* mulParam_address */ conv4_15_x1_bn_scale__vv_mul__conv4_15_x1_scale_scale, /* addParam_address */ conv4_15_x1_bn_bias__vv_mul__conv4_15_x1_scale_scale__vv_add__conv4_15_x1_scale_bias,
                             /* output_IVRF_address */ 11985, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv415X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_15_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_15_x1_bn(d=704, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_15_x2_bn */
    /*      absorbed conv4_15_x2_scale */
    /*      absorbed relu4_15_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 11985, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 28, /* mulParam_address */ conv4_15_x2_bn_scale__vv_mul__conv4_15_x2_scale_scale, /* addParam_address */ conv4_15_x2_bn_bias__vv_mul__conv4_15_x2_scale_scale__vv_add__conv4_15_x2_scale_bias,
                             /* output_IVRF_address */ 10802, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv415X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_15_x2(d=32, h=13, d=13) = Convolution(conv4_15_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_15_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10802, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 34, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat415(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_15(d=736, h=13, d=13) = Concat(Concat(concat_4_14(d=704, h=13, w=13), conv4_15_x2(d=32, h=13, w=13))) */

    /* Input concat_4_14 ISA_Mem_MvmInitialVrf memory: addresses [10971 - 11984] */
    /* Input conv4_15_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11985 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_13_x1_bn. */

    /* Copy layer concat_4_14(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 10971 + inputDepth, 1, 13 * 13, 6);
        mv_mul(bs, mrf_start + 43);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11985 + inputDepth, 6);
    }

    /* Concatenate layer concat_4_14(d=640:704) with layer conv4_15_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 10971 + (6 - 1), 1, 13 * 13, 6);
    mv_mul(bs, mrf_start + 43);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_15_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11985 + (6 - 1), 6);
}

void dummyConvConv416X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_16_x1_bn(d=736, h=13, d=13) = Convolution(concat_4_15(d=736, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_16_x1_bn */
    /*      absorbed conv4_16_x1_scale */
    /*      absorbed relu4_16_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 11985, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 44, /* mulParam_address */ conv4_16_x1_bn_scale__vv_mul__conv4_16_x1_scale_scale, /* addParam_address */ conv4_16_x1_bn_bias__vv_mul__conv4_16_x1_scale_scale__vv_add__conv4_16_x1_scale_bias,
                             /* output_IVRF_address */ 10971, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv416X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_16_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_16_x1_bn(d=736, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_16_x2_bn */
    /*      absorbed conv4_16_x2_scale */
    /*      absorbed relu4_16_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 10971, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 45, /* mulParam_address */ conv4_16_x2_bn_scale__vv_mul__conv4_16_x2_scale_scale, /* addParam_address */ conv4_16_x2_bn_bias__vv_mul__conv4_16_x2_scale_scale__vv_add__conv4_16_x2_scale_bias,
                             /* output_IVRF_address */ 10802, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv416X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_16_x2(d=32, h=13, d=13) = Convolution(conv4_16_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_16_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10802, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 51, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat416(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_16(d=768, h=13, d=13) = Concat(Concat(concat_4_15(d=736, h=13, w=13), conv4_16_x2(d=32, h=13, w=13))) */

    /* Input concat_4_15 ISA_Mem_MvmInitialVrf memory: addresses [11985 - 12998] */
    /* Input conv4_16_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10971 - 11984] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_13_x1_bn. */
    /* This layer prefetches the matrix parameters for layers conv4_17_x1, conv4_17_x2, concat_4_17, dummy_conv_conv4_18_x1_bn, conv4_18_x1, conv4_18_x2, concat_4_18, dummy_conv_conv4_19_x1_bn, conv4_19_x1, conv4_19_x2, concat_4_19, dummy_conv_conv4_20_x1_bn, conv4_20_x1. */

    /* Prefetch the matrix parameters for the next group of layers. */
    moveFilterCount128(bs, ISA_Mem_Dram, conv4_17_x1_MRF, ISA_Mem_MatrixRf, mrf_next, 1, 60);

    /* Copy layer concat_4_15(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11985 + inputDepth, 1, 13 * 13, 6);
        mv_mul(bs, mrf_start + 60);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10971 + inputDepth, 6);
    }

    /* Concatenate layer concat_4_15(d=640:736) with layer conv4_16_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11985 + (6 - 1), 1, 13 * 13, 6);
    mv_mul(bs, mrf_start + 60);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_16_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10971 + (6 - 1), 6);
}

void dummyConvConv417X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_17_x1_bn(d=768, h=13, d=13) = Convolution(concat_4_16(d=768, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_17_x1_bn */
    /*      absorbed conv4_17_x1_scale */
    /*      absorbed relu4_17_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 10971, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 61, /* mulParam_address */ conv4_17_x1_bn_scale__vv_mul__conv4_17_x1_scale_scale, /* addParam_address */ conv4_17_x1_bn_bias__vv_mul__conv4_17_x1_scale_scale__vv_add__conv4_17_x1_scale_bias,
                             /* output_IVRF_address */ 11985, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv417X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_17_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_17_x1_bn(d=768, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_17_x2_bn */
    /*      absorbed conv4_17_x2_scale */
    /*      absorbed relu4_17_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 6,
                             /* input_address */ 11985, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ conv4_17_x2_bn_scale__vv_mul__conv4_17_x2_scale_scale, /* addParam_address */ conv4_17_x2_bn_bias__vv_mul__conv4_17_x2_scale_scale__vv_add__conv4_17_x2_scale_bias,
                             /* output_IVRF_address */ 10802, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv417X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_17_x2(d=32, h=13, d=13) = Convolution(conv4_17_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10802, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 6, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 12830, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat417(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_17(d=800, h=13, d=13) = Concat(Concat(concat_4_16(d=768, h=13, w=13), conv4_17_x2(d=32, h=13, w=13))) */

    /* Input concat_4_16 ISA_Mem_MvmInitialVrf memory: addresses [10971 - 11984] */
    /* Input conv4_17_x2 ISA_Mem_MvmInitialVrf memory: addresses [12830 - 12998] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 1182] */

    /* This layer's matrix parameters were prefetched by layer concat_4_16. */

    /* Copy layer concat_4_16(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 10971 + inputDepth, 1, 13 * 13, 6);
        mv_mul(bs, mrf_start + 15);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 7);
    }

    /* Copy layer conv4_17_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12830, 1, 13 * 13, 1);
    mv_mul(bs, mrf_start + 15);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 6, 7);
}

void dummyConvConv418X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_18_x1_bn(d=800, h=13, d=13) = Convolution(concat_4_17(d=800, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_18_x1_bn */
    /*      absorbed conv4_18_x1_scale */
    /*      absorbed relu4_18_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 16, /* mulParam_address */ conv4_18_x1_bn_scale__vv_mul__conv4_18_x1_scale_scale, /* addParam_address */ conv4_18_x1_bn_bias__vv_mul__conv4_18_x1_scale_scale__vv_add__conv4_18_x1_scale_bias,
                             /* output_IVRF_address */ 1183, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv418X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_18_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_18_x1_bn(d=800, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_18_x2_bn */
    /*      absorbed conv4_18_x2_scale */
    /*      absorbed relu4_18_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 1183, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 17, /* mulParam_address */ conv4_18_x2_bn_scale__vv_mul__conv4_18_x2_scale_scale, /* addParam_address */ conv4_18_x2_bn_bias__vv_mul__conv4_18_x2_scale_scale__vv_add__conv4_18_x2_scale_bias,
                             /* output_IVRF_address */ 2366, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv418X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_18_x2(d=32, h=13, d=13) = Convolution(conv4_18_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_18_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 2366, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 24, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat418(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_18(d=832, h=13, d=13) = Concat(Concat(concat_4_17(d=800, h=13, w=13), conv4_18_x2(d=32, h=13, w=13))) */

    /* Input concat_4_17 ISA_Mem_MvmInitialVrf memory: addresses [0 - 1182] */
    /* Input conv4_18_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [1183 - 2365] */

    /* This layer's matrix parameters were prefetched by layer concat_4_16. */

    /* Copy layer concat_4_17(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 13 * 13, 7);
        mv_mul(bs, mrf_start + 33);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 1183 + inputDepth, 7);
    }

    /* Concatenate layer concat_4_17(d=768:800) with layer conv4_18_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (7 - 1), 1, 13 * 13, 7);
    mv_mul(bs, mrf_start + 33);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_18_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 1183 + (7 - 1), 7);
}

void dummyConvConv419X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_19_x1_bn(d=832, h=13, d=13) = Convolution(concat_4_18(d=832, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_19_x1_bn */
    /*      absorbed conv4_19_x1_scale */
    /*      absorbed relu4_19_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 1183, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 34, /* mulParam_address */ conv4_19_x1_bn_scale__vv_mul__conv4_19_x1_scale_scale, /* addParam_address */ conv4_19_x1_bn_bias__vv_mul__conv4_19_x1_scale_scale__vv_add__conv4_19_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv419X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_19_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_19_x1_bn(d=832, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_19_x2_bn */
    /*      absorbed conv4_19_x2_scale */
    /*      absorbed relu4_19_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 35, /* mulParam_address */ conv4_19_x2_bn_scale__vv_mul__conv4_19_x2_scale_scale, /* addParam_address */ conv4_19_x2_bn_bias__vv_mul__conv4_19_x2_scale_scale__vv_add__conv4_19_x2_scale_bias,
                             /* output_IVRF_address */ 2366, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv419X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_19_x2(d=32, h=13, d=13) = Convolution(conv4_19_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_19_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 2366, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 42, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat419(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_19(d=864, h=13, d=13) = Concat(Concat(concat_4_18(d=832, h=13, w=13), conv4_19_x2(d=32, h=13, w=13))) */

    /* Input concat_4_18 ISA_Mem_MvmInitialVrf memory: addresses [1183 - 2365] */
    /* Input conv4_19_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 1182] */

    /* This layer's matrix parameters were prefetched by layer concat_4_16. */

    /* Copy layer concat_4_18(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 1183 + inputDepth, 1, 13 * 13, 7);
        mv_mul(bs, mrf_start + 51);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 7);
    }

    /* Concatenate layer concat_4_18(d=768:832) with layer conv4_19_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 1183 + (7 - 1), 1, 13 * 13, 7);
    mv_mul(bs, mrf_start + 51);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_19_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (7 - 1), 7);
}

void dummyConvConv420X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_20_x1_bn(d=864, h=13, d=13) = Convolution(concat_4_19(d=864, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_20_x1_bn */
    /*      absorbed conv4_20_x1_scale */
    /*      absorbed relu4_20_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 52, /* mulParam_address */ conv4_20_x1_bn_scale__vv_mul__conv4_20_x1_scale_scale, /* addParam_address */ conv4_20_x1_bn_bias__vv_mul__conv4_20_x1_scale_scale__vv_add__conv4_20_x1_scale_bias,
                             /* output_IVRF_address */ 1183, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv4_20_x2_MRF, /* mrf_prefetch_next_size */ 56, /* swap_mrf_buffers */ false);
}

void conv420X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_20_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_20_x1_bn(d=864, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_20_x2_bn */
    /*      absorbed conv4_20_x2_scale */
    /*      absorbed relu4_20_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 1183, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 53, /* mulParam_address */ conv4_20_x2_bn_scale__vv_mul__conv4_20_x2_scale_scale, /* addParam_address */ conv4_20_x2_bn_bias__vv_mul__conv4_20_x2_scale_scale__vv_add__conv4_20_x2_scale_bias,
                             /* output_IVRF_address */ 2366, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv420X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_20_x2(d=32, h=13, d=13) = Convolution(conv4_20_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_20_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 2366, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat420(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_20(d=896, h=13, d=13) = Concat(Concat(concat_4_19(d=864, h=13, w=13), conv4_20_x2(d=32, h=13, w=13))) */

    /* Input concat_4_19 ISA_Mem_MvmInitialVrf memory: addresses [0 - 1182] */
    /* Input conv4_20_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [1183 - 2365] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_20_x1_bn. */

    /* Copy layer concat_4_19(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 13 * 13, 7);
        mv_mul(bs, mrf_start + 9);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 1183 + inputDepth, 7);
    }

    /* Concatenate layer concat_4_19(d=768:864) with layer conv4_20_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (7 - 1), 1, 13 * 13, 7);
    mv_mul(bs, mrf_start + 9);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_20_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 1183 + (7 - 1), 7);
}

void dummyConvConv421X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_21_x1_bn(d=896, h=13, d=13) = Convolution(concat_4_20(d=896, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_21_x1_bn */
    /*      absorbed conv4_21_x1_scale */
    /*      absorbed relu4_21_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 1183, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv4_21_x1_bn_scale__vv_mul__conv4_21_x1_scale_scale, /* addParam_address */ conv4_21_x1_bn_bias__vv_mul__conv4_21_x1_scale_scale__vv_add__conv4_21_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv421X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_21_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_21_x1_bn(d=896, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_21_x2_bn */
    /*      absorbed conv4_21_x2_scale */
    /*      absorbed relu4_21_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv4_21_x2_bn_scale__vv_mul__conv4_21_x2_scale_scale, /* addParam_address */ conv4_21_x2_bn_bias__vv_mul__conv4_21_x2_scale_scale__vv_add__conv4_21_x2_scale_bias,
                             /* output_IVRF_address */ 2366, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv421X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_21_x2(d=32, h=13, d=13) = Convolution(conv4_21_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 2366, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 18, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat421(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_21(d=928, h=13, d=13) = Concat(Concat(concat_4_20(d=896, h=13, w=13), conv4_21_x2(d=32, h=13, w=13))) */

    /* Input concat_4_20 ISA_Mem_MvmInitialVrf memory: addresses [1183 - 2365] */
    /* Input conv4_21_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_20_x1_bn. */

    /* Copy layer concat_4_20(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 1183 + inputDepth, 1, 13 * 13, 7);
        mv_mul(bs, mrf_start + 27);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 8);
    }

    /* Copy layer conv4_21_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 13 * 13, 1);
    mv_mul(bs, mrf_start + 27);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + 7, 8);
}

void dummyConvConv422X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_22_x1_bn(d=928, h=13, d=13) = Convolution(concat_4_21(d=928, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_22_x1_bn */
    /*      absorbed conv4_22_x1_scale */
    /*      absorbed relu4_22_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 11647, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 28, /* mulParam_address */ conv4_22_x1_bn_scale__vv_mul__conv4_22_x1_scale_scale, /* addParam_address */ conv4_22_x1_bn_bias__vv_mul__conv4_22_x1_scale_scale__vv_add__conv4_22_x1_scale_bias,
                             /* output_IVRF_address */ 10295, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv422X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_22_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_22_x1_bn(d=928, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_22_x2_bn */
    /*      absorbed conv4_22_x2_scale */
    /*      absorbed relu4_22_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 10295, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 29, /* mulParam_address */ conv4_22_x2_bn_scale__vv_mul__conv4_22_x2_scale_scale, /* addParam_address */ conv4_22_x2_bn_bias__vv_mul__conv4_22_x2_scale_scale__vv_add__conv4_22_x2_scale_bias,
                             /* output_IVRF_address */ 10126, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv422X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_22_x2(d=32, h=13, d=13) = Convolution(conv4_22_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_22_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10126, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 37, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat422(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_22(d=960, h=13, d=13) = Concat(Concat(concat_4_21(d=928, h=13, w=13), conv4_22_x2(d=32, h=13, w=13))) */

    /* Input concat_4_21 ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12998] */
    /* Input conv4_22_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10295 - 11646] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_20_x1_bn. */

    /* Copy layer concat_4_21(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 1, 13 * 13, 8);
        mv_mul(bs, mrf_start + 46);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10295 + inputDepth, 8);
    }

    /* Concatenate layer concat_4_21(d=896:928) with layer conv4_22_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + (8 - 1), 1, 13 * 13, 8);
    mv_mul(bs, mrf_start + 46);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_22_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10295 + (8 - 1), 8);
}

void dummyConvConv423X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_23_x1_bn(d=960, h=13, d=13) = Convolution(concat_4_22(d=960, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_23_x1_bn */
    /*      absorbed conv4_23_x1_scale */
    /*      absorbed relu4_23_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 10295, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 47, /* mulParam_address */ conv4_23_x1_bn_scale__vv_mul__conv4_23_x1_scale_scale, /* addParam_address */ conv4_23_x1_bn_bias__vv_mul__conv4_23_x1_scale_scale__vv_add__conv4_23_x1_scale_bias,
                             /* output_IVRF_address */ 11647, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv4_23_x2_MRF, /* mrf_prefetch_next_size */ 64, /* swap_mrf_buffers */ false);
}

void conv423X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_23_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_23_x1_bn(d=960, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_23_x2_bn */
    /*      absorbed conv4_23_x2_scale */
    /*      absorbed relu4_23_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 11647, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 48, /* mulParam_address */ conv4_23_x2_bn_scale__vv_mul__conv4_23_x2_scale_scale, /* addParam_address */ conv4_23_x2_bn_bias__vv_mul__conv4_23_x2_scale_scale__vv_add__conv4_23_x2_scale_bias,
                             /* output_IVRF_address */ 10126, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv423X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_23_x2(d=32, h=13, d=13) = Convolution(conv4_23_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_23_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10126, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat423(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_23(d=992, h=13, d=13) = Concat(Concat(concat_4_22(d=960, h=13, w=13), conv4_23_x2(d=32, h=13, w=13))) */

    /* Input concat_4_22 ISA_Mem_MvmInitialVrf memory: addresses [10295 - 11646] */
    /* Input conv4_23_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_23_x1_bn. */

    /* Copy layer concat_4_22(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 10295 + inputDepth, 1, 13 * 13, 8);
        mv_mul(bs, mrf_start + 9);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 8);
    }

    /* Concatenate layer concat_4_22(d=896:960) with layer conv4_23_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 10295 + (8 - 1), 1, 13 * 13, 8);
    mv_mul(bs, mrf_start + 9);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_23_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11647 + (8 - 1), 8);
}

void dummyConvConv424X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_24_x1_bn(d=992, h=13, d=13) = Convolution(concat_4_23(d=992, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_24_x1_bn */
    /*      absorbed conv4_24_x1_scale */
    /*      absorbed relu4_24_x1 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 11647, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv4_24_x1_bn_scale__vv_mul__conv4_24_x1_scale_scale, /* addParam_address */ conv4_24_x1_bn_bias__vv_mul__conv4_24_x1_scale_scale__vv_add__conv4_24_x1_scale_bias,
                             /* output_IVRF_address */ 10295, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv424X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_24_x1(d=128, h=13, d=13) = Convolution(dummy_conv_conv4_24_x1_bn(d=992, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_24_x2_bn */
    /*      absorbed conv4_24_x2_scale */
    /*      absorbed relu4_24_x2 */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 10295, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv4_24_x2_bn_scale__vv_mul__conv4_24_x2_scale_scale, /* addParam_address */ conv4_24_x2_bn_bias__vv_mul__conv4_24_x2_scale_scale__vv_add__conv4_24_x2_scale_bias,
                             /* output_IVRF_address */ 10126, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv424X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv4_24_x2(d=32, h=13, d=13) = Convolution(conv4_24_x1(d=128, h=13, w=13), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv4_24_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 1,
                             /* input_address */ 10126, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 19, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat424(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_4_24(d=1024, h=13, d=13) = Concat(Concat(concat_4_23(d=992, h=13, w=13), conv4_24_x2(d=32, h=13, w=13))) */

    /* Input concat_4_23 ISA_Mem_MvmInitialVrf memory: addresses [11647 - 12998] */
    /* Input conv4_24_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 168] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [10295 - 11646] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv4_23_x1_bn. */

    /* Copy layer concat_4_23(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + inputDepth, 1, 13 * 13, 8);
        mv_mul(bs, mrf_start + 28);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10295 + inputDepth, 8);
    }

    /* Concatenate layer concat_4_23(d=896:992) with layer conv4_24_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 11647 + (8 - 1), 1, 13 * 13, 8);
    mv_mul(bs, mrf_start + 28);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv4_24_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10295 + (8 - 1), 8);
}

void dummyConvConv4BlkBn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv4_blk_bn(d=1024, h=13, d=13) = Convolution(concat_4_24(d=1024, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv4_blk_bn */
    /*      absorbed conv4_blk_scale */
    /*      absorbed relu4_blk */

    genericConvolution(bs, /* input_height */ 13, /* input_width */ 13, /* input_depth */ 8,
                             /* input_address */ 10295, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 29, /* mulParam_address */ conv4_blk_bn_scale__vv_mul__conv4_blk_scale_scale, /* addParam_address */ conv4_blk_bn_bias__vv_mul__conv4_blk_scale_scale__vv_add__conv4_blk_scale_bias,
                             /* output_IVRF_address */ 11647, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv4Blk(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused block */
    /* Convolution conv4_blk(d=512, h=13, d=13) = Convolution(dummy_conv_conv4_blk_bn(d=1024, h=13, w=13), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*    includes sublayer pool4(d=512, h=6, d=6) = AveragePool(conv4_blk(d=512, h=13, w=13), k_h=2, k_w=2, s_h=2, s_w=2, p_h=0, p_w=0) */
    ISA_ExtAddress dummy_conv_conv4_blk_bn_inIndex;
    dummy_conv_conv4_blk_bn_inIndex=11647;
    /* Prefetch 63 entries starting at conv5_1_x1 */
    moveFilterCount128(bs, ISA_Mem_Dram, conv5_1_x1_MRF+0*63, ISA_Mem_MatrixRf, mrf_next, 1, 63);
    ISA_ExtAddress tmp_MVMIVRF=11425, tmp_MVMIVRF_next=11464;
    /* Layer conv4_blk tile size 13*104 */
    /* Temp vars and parameters for input layer conv4_blk */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g0_conv4_blk_in=11647,g0_conv4_blk_inIterator=11647;
    ISA_ExtAddress g0_conv4_blk_available = 169;
    ISA_ExtAddress g0_conv4_blk_outOffset=0;
    int g0_conv4_blk_iterationsLeft=169;
    /* Layer pool4 tile size 13*52 */
    /* Temp vars and parameters for input layer pool4 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row) */
    ISA_ExtAddress g1_pool4_in=10749,g1_pool4_inIterator=10749;
    ISA_ExtAddress g1_pool4_available = 169;
    ISA_ExtAddress g1_pool4_accumulators=11425;
    ISA_ExtAddress g1_pool4_availableVerticalRows=0;
    ISA_ExtAddress g1_pool4_outOffset=0;
    int g1_pool4_iterationsLeft=36;
    vRead1D(bs, ISA_Mem_Dram, pool4_scale, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    /* Loop until we've read all outputs */
    while (g1_pool4_iterationsLeft>0) {

        /* Start of group 0 */
        if (g0_conv4_blk_iterationsLeft>0) {


            /* Start of layer 0 in group 0 (conv4_blk) */
            /* Tile size 13*104 dimPerStep 8 */
            for(int outRow=0;outRow<4;outRow++) {
                g0_conv4_blk_inIterator = g0_conv4_blk_in;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_conv4_blk_inIterator, 8, 13 * 13, 8);
                mv_mul(bs, mrf_start+30+outRow*8);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 10749+g0_conv4_blk_outOffset+outRow+0, 4);
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+g0_conv4_blk_outOffset+outRow+0, 4);
            }
            g0_conv4_blk_iterationsLeft-=g0_conv4_blk_available;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_conv4_blk_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_pool4_available <= 676, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if ((g1_pool4_available >= 52) || ((g0_conv4_blk_iterationsLeft==0))) {

            /* Start of layer 0 in group 1 (pool4) */
            /* Tile size 13*52 dimPerStep 4 */
            /* Decompose the AVE-pool into 1 horizontal pool operations followed by and 1 vertical pool operations */
            /* All of the data will be processed en-masse, since our input data doesn't need paged in. */
            /* Perform 6 horizontal pool operations on 13 or 12 rows with 1 steps */
            /* g1_pool4_inIterator iterates horizontal pool operations (INPUTS) */
            /* g1_pool4_in iterates vertical pool operations (OUTPUTS) */
            /* Data is aligned to the original rather than reduced size (after stride) when written back to the main IVRF and ASVRF1 */
            int horizontalRows=13;
            int verticalRows=6;
            g1_pool4_available -= verticalRows*104;
            ISA_ExtAddress curOffset;
            for(unsigned vec=0; vec<4; vec++) {
                curOffset=g1_pool4_inIterator+vec;
                for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 6, 8);
                    mv_mul(bs, mrf_start+62);
                    /* The following line converts the IVRF-relative curOffset to a ASVRF-relative offset */
                    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, curOffset-10749+0+4, 8);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool4_accumulators+rowIterator*6, 1);
                    curOffset+=52;
                }
                /* Horizontal sweep must end up in g1_pool4_accumulators because we can't read-modify-write ASVRF in a single chain */
                curOffset=g1_pool4_inIterator+vec;
                for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                    vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool4_accumulators+rowIterator*6, 1, 6, 1);
                    mv_mul(bs, mrf_start+62);
                    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, curOffset, 4);
                    v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+0-10749, 4);
                    curOffset+=52;
                }
            }
            /* Update horizontal pool iterator start */
            g1_pool4_inIterator = curOffset-3;
            curOffset=g1_pool4_in;
            /* The horizontal sweep took care of the multiple input vectors */
            ISA_ExtAddress nextOffset=curOffset;
            nextOffset+=52;
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 24, 1);
                mv_mul(bs, mrf_start+62);
                /* The following line converts the IVRF-relative nextOffset to a ASVRF-relative offset */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset-10749+0, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool4_accumulators + rowIterator * 6 * 4, 1);

                curOffset+=104;
                nextOffset+=104;
            }
            g1_pool4_in = curOffset;

            /* Perform any additional operations in the chain after the pooling. */
            /* In particular this is needed for average pools, because the current hardware doesn't support a multiply after an AddSub VRF 1 add/max/sub operation. */
            /* Note that we forgo using a row loop here as it isn't needed */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool4_accumulators, 1, verticalRows * 6 * 4, 1);
            mv_mul(bs, mrf_start + 62);
            vv_mul(bs, 0); /* includes: pool4: scale */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 11503 + g1_pool4_outOffset, 1);

            g1_pool4_outOffset += verticalRows * 6 * 4;
            g1_pool4_iterationsLeft-=verticalRows*6;
            /* Make sure we didn't loop too many times (emulator only) */
            Emulator_HEX_Assert(g1_pool4_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void dummyConvConv51X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_1_x1_bn(d=512, h=6, d=6) = Convolution(pool4(d=512, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_1_x1_bn */
    /*      absorbed conv5_1_x1_scale */
    /*      absorbed relu5_1_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 4,
                             /* input_address */ 11503, /* output_depth */ 4, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 63, /* mulParam_address */ conv5_1_x1_bn_scale__vv_mul__conv5_1_x1_scale_scale, /* addParam_address */ conv5_1_x1_bn_bias__vv_mul__conv5_1_x1_scale_scale__vv_add__conv5_1_x1_scale_bias,
                             /* output_IVRF_address */ 12855, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv51X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_1_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_1_x1_bn(d=512, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_1_x2_bn */
    /*      absorbed conv5_1_x2_scale */
    /*      absorbed relu5_1_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 4,
                             /* input_address */ 12855, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ conv5_1_x2_bn_scale__vv_mul__conv5_1_x2_scale_scale, /* addParam_address */ conv5_1_x2_bn_bias__vv_mul__conv5_1_x2_scale_scale__vv_add__conv5_1_x2_scale_bias,
                             /* output_IVRF_address */ 12819, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv51X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_1_x2(d=32, h=6, d=6) = Convolution(conv5_1_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12819, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 4, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 12963, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat51(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_1(d=544, h=6, d=6) = Concat(Concat(pool4(d=512, h=6, w=6), conv5_1_x2(d=32, h=6, w=6))) */

    /* Input pool4 ISA_Mem_MvmInitialVrf memory: addresses [11503 - 11646] */
    /* Input conv5_1_x2 ISA_Mem_MvmInitialVrf memory: addresses [12963 - 12998] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 179] */

    /* This layer's matrix parameters were prefetched by layer conv4_blk. */

    /* Copy layer pool4(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 11503 + inputDepth, 1, 6 * 6, 4);
        mv_mul(bs, mrf_start + 13);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 5);
    }

    /* Copy layer conv5_1_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12963, 1, 6 * 6, 1);
    mv_mul(bs, mrf_start + 13);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 4, 5);
}

void dummyConvConv52X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_2_x1_bn(d=544, h=6, d=6) = Convolution(concat_5_1(d=544, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_2_x1_bn */
    /*      absorbed conv5_2_x1_scale */
    /*      absorbed relu5_2_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 14, /* mulParam_address */ conv5_2_x1_bn_scale__vv_mul__conv5_2_x1_scale_scale, /* addParam_address */ conv5_2_x1_bn_bias__vv_mul__conv5_2_x1_scale_scale__vv_add__conv5_2_x1_scale_bias,
                             /* output_IVRF_address */ 180, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv52X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_2_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_2_x1_bn(d=544, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_2_x2_bn */
    /*      absorbed conv5_2_x2_scale */
    /*      absorbed relu5_2_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 180, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 15, /* mulParam_address */ conv5_2_x2_bn_scale__vv_mul__conv5_2_x2_scale_scale, /* addParam_address */ conv5_2_x2_bn_bias__vv_mul__conv5_2_x2_scale_scale__vv_add__conv5_2_x2_scale_bias,
                             /* output_IVRF_address */ 360, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv52X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_2_x2(d=32, h=6, d=6) = Convolution(conv5_2_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_2_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 360, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 20, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat52(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_2(d=576, h=6, d=6) = Concat(Concat(concat_5_1(d=544, h=6, w=6), conv5_2_x2(d=32, h=6, w=6))) */

    /* Input concat_5_1 ISA_Mem_MvmInitialVrf memory: addresses [0 - 179] */
    /* Input conv5_2_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [180 - 359] */

    /* This layer's matrix parameters were prefetched by layer conv4_blk. */

    /* Copy layer concat_5_1(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 6 * 6, 5);
        mv_mul(bs, mrf_start + 29);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 180 + inputDepth, 5);
    }

    /* Concatenate layer concat_5_1(d=512:544) with layer conv5_2_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (5 - 1), 1, 6 * 6, 5);
    mv_mul(bs, mrf_start + 29);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_2_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 180 + (5 - 1), 5);
}

void dummyConvConv53X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_3_x1_bn(d=576, h=6, d=6) = Convolution(concat_5_2(d=576, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_3_x1_bn */
    /*      absorbed conv5_3_x1_scale */
    /*      absorbed relu5_3_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 180, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 30, /* mulParam_address */ conv5_3_x1_bn_scale__vv_mul__conv5_3_x1_scale_scale, /* addParam_address */ conv5_3_x1_bn_bias__vv_mul__conv5_3_x1_scale_scale__vv_add__conv5_3_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv53X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_3_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_3_x1_bn(d=576, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_3_x2_bn */
    /*      absorbed conv5_3_x2_scale */
    /*      absorbed relu5_3_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 31, /* mulParam_address */ conv5_3_x2_bn_scale__vv_mul__conv5_3_x2_scale_scale, /* addParam_address */ conv5_3_x2_bn_bias__vv_mul__conv5_3_x2_scale_scale__vv_add__conv5_3_x2_scale_bias,
                             /* output_IVRF_address */ 360, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv53X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_3_x2(d=32, h=6, d=6) = Convolution(conv5_3_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_3_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 360, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 36, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat53(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_3(d=608, h=6, d=6) = Concat(Concat(concat_5_2(d=576, h=6, w=6), conv5_3_x2(d=32, h=6, w=6))) */

    /* Input concat_5_2 ISA_Mem_MvmInitialVrf memory: addresses [180 - 359] */
    /* Input conv5_3_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 179] */

    /* This layer's matrix parameters were prefetched by layer conv4_blk. */

    /* Copy layer concat_5_2(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 180 + inputDepth, 1, 6 * 6, 5);
        mv_mul(bs, mrf_start + 45);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 5);
    }

    /* Concatenate layer concat_5_2(d=512:576) with layer conv5_3_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 180 + (5 - 1), 1, 6 * 6, 5);
    mv_mul(bs, mrf_start + 45);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_3_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (5 - 1), 5);
}

void dummyConvConv54X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_4_x1_bn(d=608, h=6, d=6) = Convolution(concat_5_3(d=608, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_4_x1_bn */
    /*      absorbed conv5_4_x1_scale */
    /*      absorbed relu5_4_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 46, /* mulParam_address */ conv5_4_x1_bn_scale__vv_mul__conv5_4_x1_scale_scale, /* addParam_address */ conv5_4_x1_bn_bias__vv_mul__conv5_4_x1_scale_scale__vv_add__conv5_4_x1_scale_bias,
                             /* output_IVRF_address */ 180, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv54X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_4_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_4_x1_bn(d=608, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_4_x2_bn */
    /*      absorbed conv5_4_x2_scale */
    /*      absorbed relu5_4_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 180, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 47, /* mulParam_address */ conv5_4_x2_bn_scale__vv_mul__conv5_4_x2_scale_scale, /* addParam_address */ conv5_4_x2_bn_bias__vv_mul__conv5_4_x2_scale_scale__vv_add__conv5_4_x2_scale_bias,
                             /* output_IVRF_address */ 360, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv54X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_4_x2(d=32, h=6, d=6) = Convolution(conv5_4_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_4_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 360, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 52, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat54(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_4(d=640, h=6, d=6) = Concat(Concat(concat_5_3(d=608, h=6, w=6), conv5_4_x2(d=32, h=6, w=6))) */

    /* Input concat_5_3 ISA_Mem_MvmInitialVrf memory: addresses [0 - 179] */
    /* Input conv5_4_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [180 - 359] */

    /* This layer's matrix parameters were prefetched by layer conv4_blk. */
    /* This layer prefetches the matrix parameters for layers conv5_5_x1, conv5_5_x2, concat_5_5, dummy_conv_conv5_6_x1_bn, conv5_6_x1, conv5_6_x2, concat_5_6, dummy_conv_conv5_7_x1_bn, conv5_7_x1, conv5_7_x2, concat_5_7, dummy_conv_conv5_8_x1_bn, conv5_8_x1. */

    /* Prefetch the matrix parameters for the next group of layers. */
    moveFilterCount128(bs, ISA_Mem_Dram, conv5_5_x1_MRF, ISA_Mem_MatrixRf, mrf_next, 1, 56);

    /* Copy layer concat_5_3(d=0:512) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 4; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 6 * 6, 5);
        mv_mul(bs, mrf_start + 61);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 180 + inputDepth, 5);
    }

    /* Concatenate layer concat_5_3(d=512:608) with layer conv5_4_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (5 - 1), 1, 6 * 6, 5);
    mv_mul(bs, mrf_start + 61);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_4_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 180 + (5 - 1), 5);
}

void dummyConvConv55X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_5_x1_bn(d=640, h=6, d=6) = Convolution(concat_5_4(d=640, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_5_x1_bn */
    /*      absorbed conv5_5_x1_scale */
    /*      absorbed relu5_5_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 180, /* output_depth */ 5, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 62, /* mulParam_address */ conv5_5_x1_bn_scale__vv_mul__conv5_5_x1_scale_scale, /* addParam_address */ conv5_5_x1_bn_bias__vv_mul__conv5_5_x1_scale_scale__vv_add__conv5_5_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv55X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_5_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_5_x1_bn(d=640, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_5_x2_bn */
    /*      absorbed conv5_5_x2_scale */
    /*      absorbed relu5_5_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 5,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ conv5_5_x2_bn_scale__vv_mul__conv5_5_x2_scale_scale, /* addParam_address */ conv5_5_x2_bn_bias__vv_mul__conv5_5_x2_scale_scale__vv_add__conv5_5_x2_scale_bias,
                             /* output_IVRF_address */ 360, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv55X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_5_x2(d=32, h=6, d=6) = Convolution(conv5_5_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 360, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 5, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat55(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_5(d=672, h=6, d=6) = Concat(Concat(concat_5_4(d=640, h=6, w=6), conv5_5_x2(d=32, h=6, w=6))) */

    /* Input concat_5_4 ISA_Mem_MvmInitialVrf memory: addresses [180 - 359] */
    /* Input conv5_5_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12783 - 12998] */

    /* This layer's matrix parameters were prefetched by layer concat_5_4. */

    /* Copy layer concat_5_4(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 180 + inputDepth, 1, 6 * 6, 5);
        mv_mul(bs, mrf_start + 14);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12783 + inputDepth, 6);
    }

    /* Copy layer conv5_5_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 6 * 6, 1);
    mv_mul(bs, mrf_start + 14);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12783 + 5, 6);
}

void dummyConvConv56X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_6_x1_bn(d=672, h=6, d=6) = Convolution(concat_5_5(d=672, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_6_x1_bn */
    /*      absorbed conv5_6_x1_scale */
    /*      absorbed relu5_6_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12783, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 15, /* mulParam_address */ conv5_6_x1_bn_scale__vv_mul__conv5_6_x1_scale_scale, /* addParam_address */ conv5_6_x1_bn_bias__vv_mul__conv5_6_x1_scale_scale__vv_add__conv5_6_x1_scale_bias,
                             /* output_IVRF_address */ 12567, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv56X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_6_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_6_x1_bn(d=672, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_6_x2_bn */
    /*      absorbed conv5_6_x2_scale */
    /*      absorbed relu5_6_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12567, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 16, /* mulParam_address */ conv5_6_x2_bn_scale__vv_mul__conv5_6_x2_scale_scale, /* addParam_address */ conv5_6_x2_bn_bias__vv_mul__conv5_6_x2_scale_scale__vv_add__conv5_6_x2_scale_bias,
                             /* output_IVRF_address */ 12531, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv56X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_6_x2(d=32, h=6, d=6) = Convolution(conv5_6_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_6_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12531, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 22, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat56(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_6(d=704, h=6, d=6) = Concat(Concat(concat_5_5(d=672, h=6, w=6), conv5_6_x2(d=32, h=6, w=6))) */

    /* Input concat_5_5 ISA_Mem_MvmInitialVrf memory: addresses [12783 - 12998] */
    /* Input conv5_6_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12567 - 12782] */

    /* This layer's matrix parameters were prefetched by layer concat_5_4. */

    /* Copy layer concat_5_5(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12783 + inputDepth, 1, 6 * 6, 6);
        mv_mul(bs, mrf_start + 31);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12567 + inputDepth, 6);
    }

    /* Concatenate layer concat_5_5(d=640:672) with layer conv5_6_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12783 + (6 - 1), 1, 6 * 6, 6);
    mv_mul(bs, mrf_start + 31);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_6_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12567 + (6 - 1), 6);
}

void dummyConvConv57X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_7_x1_bn(d=704, h=6, d=6) = Convolution(concat_5_6(d=704, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_7_x1_bn */
    /*      absorbed conv5_7_x1_scale */
    /*      absorbed relu5_7_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12567, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 32, /* mulParam_address */ conv5_7_x1_bn_scale__vv_mul__conv5_7_x1_scale_scale, /* addParam_address */ conv5_7_x1_bn_bias__vv_mul__conv5_7_x1_scale_scale__vv_add__conv5_7_x1_scale_bias,
                             /* output_IVRF_address */ 12783, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv57X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_7_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_7_x1_bn(d=704, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_7_x2_bn */
    /*      absorbed conv5_7_x2_scale */
    /*      absorbed relu5_7_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12783, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 33, /* mulParam_address */ conv5_7_x2_bn_scale__vv_mul__conv5_7_x2_scale_scale, /* addParam_address */ conv5_7_x2_bn_bias__vv_mul__conv5_7_x2_scale_scale__vv_add__conv5_7_x2_scale_bias,
                             /* output_IVRF_address */ 12531, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv57X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_7_x2(d=32, h=6, d=6) = Convolution(conv5_7_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_7_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12531, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 39, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat57(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_7(d=736, h=6, d=6) = Concat(Concat(concat_5_6(d=704, h=6, w=6), conv5_7_x2(d=32, h=6, w=6))) */

    /* Input concat_5_6 ISA_Mem_MvmInitialVrf memory: addresses [12567 - 12782] */
    /* Input conv5_7_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12783 - 12998] */

    /* This layer's matrix parameters were prefetched by layer concat_5_4. */

    /* Copy layer concat_5_6(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12567 + inputDepth, 1, 6 * 6, 6);
        mv_mul(bs, mrf_start + 48);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12783 + inputDepth, 6);
    }

    /* Concatenate layer concat_5_6(d=640:704) with layer conv5_7_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12567 + (6 - 1), 1, 6 * 6, 6);
    mv_mul(bs, mrf_start + 48);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_7_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12783 + (6 - 1), 6);
}

void dummyConvConv58X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_8_x1_bn(d=736, h=6, d=6) = Convolution(concat_5_7(d=736, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_8_x1_bn */
    /*      absorbed conv5_8_x1_scale */
    /*      absorbed relu5_8_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12783, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 49, /* mulParam_address */ conv5_8_x1_bn_scale__vv_mul__conv5_8_x1_scale_scale, /* addParam_address */ conv5_8_x1_bn_bias__vv_mul__conv5_8_x1_scale_scale__vv_add__conv5_8_x1_scale_bias,
                             /* output_IVRF_address */ 12567, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv5_8_x2_MRF, /* mrf_prefetch_next_size */ 64, /* swap_mrf_buffers */ false);
}

void conv58X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_8_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_8_x1_bn(d=736, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_8_x2_bn */
    /*      absorbed conv5_8_x2_scale */
    /*      absorbed relu5_8_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12567, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 50, /* mulParam_address */ conv5_8_x2_bn_scale__vv_mul__conv5_8_x2_scale_scale, /* addParam_address */ conv5_8_x2_bn_bias__vv_mul__conv5_8_x2_scale_scale__vv_add__conv5_8_x2_scale_bias,
                             /* output_IVRF_address */ 12531, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv58X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_8_x2(d=32, h=6, d=6) = Convolution(conv5_8_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_8_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12531, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat58(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_8(d=768, h=6, d=6) = Concat(Concat(concat_5_7(d=736, h=6, w=6), conv5_8_x2(d=32, h=6, w=6))) */

    /* Input concat_5_7 ISA_Mem_MvmInitialVrf memory: addresses [12783 - 12998] */
    /* Input conv5_8_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12567 - 12782] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv5_8_x1_bn. */

    /* Copy layer concat_5_7(d=0:640) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 5; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12783 + inputDepth, 1, 6 * 6, 6);
        mv_mul(bs, mrf_start + 9);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12567 + inputDepth, 6);
    }

    /* Concatenate layer concat_5_7(d=640:736) with layer conv5_8_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12783 + (6 - 1), 1, 6 * 6, 6);
    mv_mul(bs, mrf_start + 9);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_8_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12567 + (6 - 1), 6);
}

void dummyConvConv59X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_9_x1_bn(d=768, h=6, d=6) = Convolution(concat_5_8(d=768, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_9_x1_bn */
    /*      absorbed conv5_9_x1_scale */
    /*      absorbed relu5_9_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12567, /* output_depth */ 6, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv5_9_x1_bn_scale__vv_mul__conv5_9_x1_scale_scale, /* addParam_address */ conv5_9_x1_bn_bias__vv_mul__conv5_9_x1_scale_scale__vv_add__conv5_9_x1_scale_bias,
                             /* output_IVRF_address */ 12783, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv59X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_9_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_9_x1_bn(d=768, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_9_x2_bn */
    /*      absorbed conv5_9_x2_scale */
    /*      absorbed relu5_9_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 6,
                             /* input_address */ 12783, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv5_9_x2_bn_scale__vv_mul__conv5_9_x2_scale_scale, /* addParam_address */ conv5_9_x2_bn_bias__vv_mul__conv5_9_x2_scale_scale__vv_add__conv5_9_x2_scale_bias,
                             /* output_IVRF_address */ 12531, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv59X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_9_x2(d=32, h=6, d=6) = Convolution(conv5_9_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12531, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 17, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 12963, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat59(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_9(d=800, h=6, d=6) = Concat(Concat(concat_5_8(d=768, h=6, w=6), conv5_9_x2(d=32, h=6, w=6))) */

    /* Input concat_5_8 ISA_Mem_MvmInitialVrf memory: addresses [12567 - 12782] */
    /* Input conv5_9_x2 ISA_Mem_MvmInitialVrf memory: addresses [12963 - 12998] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 251] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv5_8_x1_bn. */

    /* Copy layer concat_5_8(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12567 + inputDepth, 1, 6 * 6, 6);
        mv_mul(bs, mrf_start + 26);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 7);
    }

    /* Copy layer conv5_9_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12963, 1, 6 * 6, 1);
    mv_mul(bs, mrf_start + 26);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + 6, 7);
}

void dummyConvConv510X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_10_x1_bn(d=800, h=6, d=6) = Convolution(concat_5_9(d=800, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_10_x1_bn */
    /*      absorbed conv5_10_x1_scale */
    /*      absorbed relu5_10_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 27, /* mulParam_address */ conv5_10_x1_bn_scale__vv_mul__conv5_10_x1_scale_scale, /* addParam_address */ conv5_10_x1_bn_bias__vv_mul__conv5_10_x1_scale_scale__vv_add__conv5_10_x1_scale_bias,
                             /* output_IVRF_address */ 252, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv510X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_10_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_10_x1_bn(d=800, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_10_x2_bn */
    /*      absorbed conv5_10_x2_scale */
    /*      absorbed relu5_10_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 252, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 28, /* mulParam_address */ conv5_10_x2_bn_scale__vv_mul__conv5_10_x2_scale_scale, /* addParam_address */ conv5_10_x2_bn_bias__vv_mul__conv5_10_x2_scale_scale__vv_add__conv5_10_x2_scale_bias,
                             /* output_IVRF_address */ 504, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv510X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_10_x2(d=32, h=6, d=6) = Convolution(conv5_10_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_10_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 504, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 35, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat510(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_10(d=832, h=6, d=6) = Concat(Concat(concat_5_9(d=800, h=6, w=6), conv5_10_x2(d=32, h=6, w=6))) */

    /* Input concat_5_9 ISA_Mem_MvmInitialVrf memory: addresses [0 - 251] */
    /* Input conv5_10_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [252 - 503] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv5_8_x1_bn. */

    /* Copy layer concat_5_9(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 6 * 6, 7);
        mv_mul(bs, mrf_start + 44);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 252 + inputDepth, 7);
    }

    /* Concatenate layer concat_5_9(d=768:800) with layer conv5_10_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (7 - 1), 1, 6 * 6, 7);
    mv_mul(bs, mrf_start + 44);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_10_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 252 + (7 - 1), 7);
}

void dummyConvConv511X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_11_x1_bn(d=832, h=6, d=6) = Convolution(concat_5_10(d=832, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_11_x1_bn */
    /*      absorbed conv5_11_x1_scale */
    /*      absorbed relu5_11_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 252, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 45, /* mulParam_address */ conv5_11_x1_bn_scale__vv_mul__conv5_11_x1_scale_scale, /* addParam_address */ conv5_11_x1_bn_bias__vv_mul__conv5_11_x1_scale_scale__vv_add__conv5_11_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv511X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_11_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_11_x1_bn(d=832, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_11_x2_bn */
    /*      absorbed conv5_11_x2_scale */
    /*      absorbed relu5_11_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 46, /* mulParam_address */ conv5_11_x2_bn_scale__vv_mul__conv5_11_x2_scale_scale, /* addParam_address */ conv5_11_x2_bn_bias__vv_mul__conv5_11_x2_scale_scale__vv_add__conv5_11_x2_scale_bias,
                             /* output_IVRF_address */ 504, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv511X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_11_x2(d=32, h=6, d=6) = Convolution(conv5_11_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_11_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 504, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 53, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat511(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_11(d=864, h=6, d=6) = Concat(Concat(concat_5_10(d=832, h=6, w=6), conv5_11_x2(d=32, h=6, w=6))) */

    /* Input concat_5_10 ISA_Mem_MvmInitialVrf memory: addresses [252 - 503] */
    /* Input conv5_11_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [0 - 251] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv5_8_x1_bn. */
    /* This layer prefetches the matrix parameters for layers conv5_12_x1, conv5_12_x2, concat_5_12, dummy_conv_conv5_13_x1_bn, conv5_13_x1, conv5_13_x2, concat_5_13, dummy_conv_conv5_14_x1_bn, conv5_14_x1, conv5_14_x2, concat_5_14, dummy_conv_conv5_15_x1_bn, conv5_15_x1. */

    /* Prefetch the matrix parameters for the next group of layers. */
    moveFilterCount128(bs, ISA_Mem_Dram, conv5_12_x1_MRF, ISA_Mem_MatrixRf, mrf_next, 1, 63);

    /* Copy layer concat_5_10(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 252 + inputDepth, 1, 6 * 6, 7);
        mv_mul(bs, mrf_start + 62);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 7);
    }

    /* Concatenate layer concat_5_10(d=768:832) with layer conv5_11_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 252 + (7 - 1), 1, 6 * 6, 7);
    mv_mul(bs, mrf_start + 62);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_11_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0 + (7 - 1), 7);
}

void dummyConvConv512X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_12_x1_bn(d=864, h=6, d=6) = Convolution(concat_5_11(d=864, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_12_x1_bn */
    /*      absorbed conv5_12_x1_scale */
    /*      absorbed relu5_12_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 63, /* mulParam_address */ conv5_12_x1_bn_scale__vv_mul__conv5_12_x1_scale_scale, /* addParam_address */ conv5_12_x1_bn_bias__vv_mul__conv5_12_x1_scale_scale__vv_add__conv5_12_x1_scale_bias,
                             /* output_IVRF_address */ 252, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv512X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_12_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_12_x1_bn(d=864, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_12_x2_bn */
    /*      absorbed conv5_12_x2_scale */
    /*      absorbed relu5_12_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 252, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ conv5_12_x2_bn_scale__vv_mul__conv5_12_x2_scale_scale, /* addParam_address */ conv5_12_x2_bn_bias__vv_mul__conv5_12_x2_scale_scale__vv_add__conv5_12_x2_scale_bias,
                             /* output_IVRF_address */ 504, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv512X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_12_x2(d=32, h=6, d=6) = Convolution(conv5_12_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_12_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 504, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 7, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat512(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_12(d=896, h=6, d=6) = Concat(Concat(concat_5_11(d=864, h=6, w=6), conv5_12_x2(d=32, h=6, w=6))) */

    /* Input concat_5_11 ISA_Mem_MvmInitialVrf memory: addresses [0 - 251] */
    /* Input conv5_12_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [252 - 503] */

    /* This layer's matrix parameters were prefetched by layer concat_5_11. */

    /* Copy layer concat_5_11(d=0:768) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 6; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + inputDepth, 1, 6 * 6, 7);
        mv_mul(bs, mrf_start + 16);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 252 + inputDepth, 7);
    }

    /* Concatenate layer concat_5_11(d=768:864) with layer conv5_12_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0 + (7 - 1), 1, 6 * 6, 7);
    mv_mul(bs, mrf_start + 16);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_12_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 252 + (7 - 1), 7);
}

void dummyConvConv513X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_13_x1_bn(d=896, h=6, d=6) = Convolution(concat_5_12(d=896, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_13_x1_bn */
    /*      absorbed conv5_13_x1_scale */
    /*      absorbed relu5_13_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 252, /* output_depth */ 7, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 17, /* mulParam_address */ conv5_13_x1_bn_scale__vv_mul__conv5_13_x1_scale_scale, /* addParam_address */ conv5_13_x1_bn_bias__vv_mul__conv5_13_x1_scale_scale__vv_add__conv5_13_x1_scale_bias,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv513X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_13_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_13_x1_bn(d=896, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_13_x2_bn */
    /*      absorbed conv5_13_x2_scale */
    /*      absorbed relu5_13_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 7,
                             /* input_address */ 0, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 18, /* mulParam_address */ conv5_13_x2_bn_scale__vv_mul__conv5_13_x2_scale_scale, /* addParam_address */ conv5_13_x2_bn_bias__vv_mul__conv5_13_x2_scale_scale__vv_add__conv5_13_x2_scale_bias,
                             /* output_IVRF_address */ 504, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv513X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_13_x2(d=32, h=6, d=6) = Convolution(conv5_13_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 504, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 25, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ 0, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat513(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_13(d=928, h=6, d=6) = Concat(Concat(concat_5_12(d=896, h=6, w=6), conv5_13_x2(d=32, h=6, w=6))) */

    /* Input concat_5_12 ISA_Mem_MvmInitialVrf memory: addresses [252 - 503] */
    /* Input conv5_13_x2 ISA_Mem_MvmInitialVrf memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12711 - 12998] */

    /* This layer's matrix parameters were prefetched by layer concat_5_11. */

    /* Copy layer concat_5_12(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 252 + inputDepth, 1, 6 * 6, 7);
        mv_mul(bs, mrf_start + 34);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12711 + inputDepth, 8);
    }

    /* Copy layer conv5_13_x2(d=0:32) to the output, as part of the concatenation operation */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 0, 1, 6 * 6, 1);
    mv_mul(bs, mrf_start + 34);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12711 + 7, 8);
}

void dummyConvConv514X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_14_x1_bn(d=928, h=6, d=6) = Convolution(concat_5_13(d=928, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_14_x1_bn */
    /*      absorbed conv5_14_x1_scale */
    /*      absorbed relu5_14_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 8,
                             /* input_address */ 12711, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 35, /* mulParam_address */ conv5_14_x1_bn_scale__vv_mul__conv5_14_x1_scale_scale, /* addParam_address */ conv5_14_x1_bn_bias__vv_mul__conv5_14_x1_scale_scale__vv_add__conv5_14_x1_scale_bias,
                             /* output_IVRF_address */ 12423, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv514X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_14_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_14_x1_bn(d=928, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_14_x2_bn */
    /*      absorbed conv5_14_x2_scale */
    /*      absorbed relu5_14_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 8,
                             /* input_address */ 12423, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 36, /* mulParam_address */ conv5_14_x2_bn_scale__vv_mul__conv5_14_x2_scale_scale, /* addParam_address */ conv5_14_x2_bn_bias__vv_mul__conv5_14_x2_scale_scale__vv_add__conv5_14_x2_scale_bias,
                             /* output_IVRF_address */ 12387, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv514X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_14_x2(d=32, h=6, d=6) = Convolution(conv5_14_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_14_x2 has shifted output. The depth shift is 32. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12387, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 44, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat514(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_14(d=960, h=6, d=6) = Concat(Concat(concat_5_13(d=928, h=6, w=6), conv5_14_x2(d=32, h=6, w=6))) */

    /* Input concat_5_13 ISA_Mem_MvmInitialVrf memory: addresses [12711 - 12998] */
    /* Input conv5_14_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12423 - 12710] */

    /* This layer's matrix parameters were prefetched by layer concat_5_11. */

    /* Copy layer concat_5_13(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12711 + inputDepth, 1, 6 * 6, 8);
        mv_mul(bs, mrf_start + 53);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12423 + inputDepth, 8);
    }

    /* Concatenate layer concat_5_13(d=896:928) with layer conv5_14_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12711 + (8 - 1), 1, 6 * 6, 8);
    mv_mul(bs, mrf_start + 53);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_14_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12423 + (8 - 1), 8);
}

void dummyConvConv515X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_15_x1_bn(d=960, h=6, d=6) = Convolution(concat_5_14(d=960, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_15_x1_bn */
    /*      absorbed conv5_15_x1_scale */
    /*      absorbed relu5_15_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 8,
                             /* input_address */ 12423, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 54, /* mulParam_address */ conv5_15_x1_bn_scale__vv_mul__conv5_15_x1_scale_scale, /* addParam_address */ conv5_15_x1_bn_bias__vv_mul__conv5_15_x1_scale_scale__vv_add__conv5_15_x1_scale_bias,
                             /* output_IVRF_address */ 12711, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ conv5_15_x2_MRF, /* mrf_prefetch_next_size */ 30, /* swap_mrf_buffers */ false);
}

void conv515X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_15_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_15_x1_bn(d=960, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_15_x2_bn */
    /*      absorbed conv5_15_x2_scale */
    /*      absorbed relu5_15_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 8,
                             /* input_address */ 12711, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 55, /* mulParam_address */ conv5_15_x2_bn_scale__vv_mul__conv5_15_x2_scale_scale, /* addParam_address */ conv5_15_x2_bn_bias__vv_mul__conv5_15_x2_scale_scale__vv_add__conv5_15_x2_scale_bias,
                             /* output_IVRF_address */ 12387, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ true);
}

void conv515X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_15_x2(d=32, h=6, d=6) = Convolution(conv5_15_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_15_x2 has shifted output. The depth shift is 64. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12387, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 0, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat515(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_15(d=992, h=6, d=6) = Concat(Concat(concat_5_14(d=960, h=6, w=6), conv5_15_x2(d=32, h=6, w=6))) */

    /* Input concat_5_14 ISA_Mem_MvmInitialVrf memory: addresses [12423 - 12710] */
    /* Input conv5_15_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12711 - 12998] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv5_15_x1_bn. */

    /* Copy layer concat_5_14(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12423 + inputDepth, 1, 6 * 6, 8);
        mv_mul(bs, mrf_start + 9);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12711 + inputDepth, 8);
    }

    /* Concatenate layer concat_5_14(d=896:960) with layer conv5_15_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12423 + (8 - 1), 1, 6 * 6, 8);
    mv_mul(bs, mrf_start + 9);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_15_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12711 + (8 - 1), 8);
}

void dummyConvConv516X1Bn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_16_x1_bn(d=992, h=6, d=6) = Convolution(concat_5_15(d=992, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_16_x1_bn */
    /*      absorbed conv5_16_x1_scale */
    /*      absorbed relu5_16_x1 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 8,
                             /* input_address */ 12711, /* output_depth */ 8, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ true,
                             /* mrfOffset */ 10, /* mulParam_address */ conv5_16_x1_bn_scale__vv_mul__conv5_16_x1_scale_scale, /* addParam_address */ conv5_16_x1_bn_bias__vv_mul__conv5_16_x1_scale_scale__vv_add__conv5_16_x1_scale_bias,
                             /* output_IVRF_address */ 12423, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv516X1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_16_x1(d=128, h=6, d=6) = Convolution(dummy_conv_conv5_16_x1_bn(d=992, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_16_x2_bn */
    /*      absorbed conv5_16_x2_scale */
    /*      absorbed relu5_16_x2 */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 8,
                             /* input_address */ 12423, /* output_depth */ 1, /* kernel_size */ 1, /* pad */ 0, /* stride */ 1, /* include_relu */ true, /* is_dummy */ false,
                             /* mrfOffset */ 11, /* mulParam_address */ conv5_16_x2_bn_scale__vv_mul__conv5_16_x2_scale_scale, /* addParam_address */ conv5_16_x2_bn_bias__vv_mul__conv5_16_x2_scale_scale__vv_add__conv5_16_x2_scale_bias,
                             /* output_IVRF_address */ 12387, /* output_ASVRF1_address */ -1,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void conv516X2(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution conv5_16_x2(d=32, h=6, d=6) = Convolution(conv5_16_x1(d=128, h=6, w=6), k_h=3, k_w=3, s_h=1, s_w=1, p_h=1, p_w=1) */
    /* Layer conv5_16_x2 has shifted output. The depth shift is 96. */

    genericConvolution(bs, /* input_height */ 6, /* input_width */ 6, /* input_depth */ 1,
                             /* input_address */ 12387, /* output_depth */ 1, /* kernel_size */ 3, /* pad */ 1, /* stride */ 1, /* include_relu */ false, /* is_dummy */ false,
                             /* mrfOffset */ 19, /* mulParam_address */ -1, /* addParam_address */ -1,
                             /* output_IVRF_address */ -1, /* output_ASVRF1_address */ 0,
                             /* mrf_fetch_address */ -1, /* mrf_fetch_size */ 0,
                             /* mrf_prefetch_next_address */ -1, /* mrf_prefetch_next_size */ 0, /* swap_mrf_buffers */ false);
}

void concat516(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone concatenation layer */
    /* Concatenation concat_5_16(d=1024, h=6, d=6) = Concat(Concat(concat_5_15(d=992, h=6, w=6), conv5_16_x2(d=32, h=6, w=6))) */

    /* Input concat_5_15 ISA_Mem_MvmInitialVrf memory: addresses [12711 - 12998] */
    /* Input conv5_16_x2 ISA_Mem_AddSubVrf_1 memory: addresses [0 - 35] */

    /* Output ISA_Mem_MvmInitialVrf memory: addresses [12423 - 12710] */

    /* This layer's matrix parameters were prefetched by layer dummy_conv_conv5_15_x1_bn. */

    /* Copy layer concat_5_15(d=0:896) to the output, as part of the concatenation operation */
    for (int inputDepth = 0; inputDepth < 7; inputDepth++) {
        vRead2D(bs, ISA_Mem_MvmInitialVrf, 12711 + inputDepth, 1, 6 * 6, 8);
        mv_mul(bs, mrf_start + 28);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12423 + inputDepth, 8);
    }

    /* Concatenate layer concat_5_15(d=896:992) with layer conv5_16_x2(d=0:32) */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, 12711 + (8 - 1), 1, 6 * 6, 8);
    mv_mul(bs, mrf_start + 28);
    vv_add_inc(bs, ISA_Mem_AddSubVrf_1, 0, 1); /* Concatenate with layer conv5_16_x2 */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 12423 + (8 - 1), 8);
}

void dummyConvConv5BlkBn(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution dummy_conv_conv5_blk_bn(d=1024, h=6, d=6) = Convolution(concat_5_16(d=1024, h=6, w=6), k_h=1, k_w=1, s_h=1, s_w=1, p_h=0, p_w=0) */
    /*      absorbed conv5_blk_bn */
    /*      absorbed conv5_blk_scale */
    /*      absorbed relu5_blk */

    ISA_ExtAddress concat_5_16_inIndex,dummy_conv_conv5_blk_bn_outOffset;
    ISA_ExtAddress outChainOffset = 0;
    /* dummy_conv_conv5_blk_bn_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, conv5_blk_bn_scale__vv_mul__conv5_blk_scale_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, conv5_blk_bn_bias__vv_mul__conv5_blk_scale_scale__vv_add__conv5_blk_scale_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    concat_5_16_inIndex=12423;
    dummy_conv_conv5_blk_bn_outOffset = 0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset = 0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, concat_5_16_inIndex+outRow, 1, 36, 8);
        mv_mul(bs, mrf_start+29);
        vv_mul(bs, 0+outChainOffset); /* includes: conv5_blk_bn: scale, vv_mul, conv5_blk_scale: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: conv5_blk_bn: bias, vv_mul, conv5_blk_scale: scale, vv_add, conv5_blk_scale: bias */
        v_relu(bs); /* includes: relu5_blk: v_relu */
        v_wr(bs, ISA_Mem_NetOutputQ, DONTCARE);
        outChainOffset++;
    }
}
