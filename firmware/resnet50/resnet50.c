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
static const ISA_ExtAddress bn_conv1_scale__vv_mul__scale_conv1_scale = 0;
static const ISA_ExtAddress bn_conv1_scale__vv_mul__scale_conv1_scale_size = 1;
static const ISA_ExtAddress conv1_bias__vv_add__bn_conv1_bias__vv_mul__scale_conv1_scale__vv_add__scale_conv1_bias = 1;
static const ISA_ExtAddress conv1_bias__vv_add__bn_conv1_bias__vv_mul__scale_conv1_scale__vv_add__scale_conv1_bias_size = 1;
static const ISA_ExtAddress conv1_MRF = 0;
static const ISA_ExtAddress conv1_MRF_size = 2;
static const ISA_ExtAddress pool1_MRF = 2;
static const ISA_ExtAddress pool1_MRF_size = 1;
static const ISA_ExtAddress bn2a_branch1_scale__vv_mul__scale2a_branch1_scale = 2;
static const ISA_ExtAddress bn2a_branch1_scale__vv_mul__scale2a_branch1_scale_size = 2;
static const ISA_ExtAddress bn2a_branch1_bias__vv_mul__scale2a_branch1_scale__vv_add__scale2a_branch1_bias = 4;
static const ISA_ExtAddress bn2a_branch1_bias__vv_mul__scale2a_branch1_scale__vv_add__scale2a_branch1_bias_size = 2;
static const ISA_ExtAddress res2a_branch1_MRF = 3;
static const ISA_ExtAddress res2a_branch1_MRF_size = 2;
static const ISA_ExtAddress bn2a_branch2a_scale__vv_mul__scale2a_branch2a_scale = 6;
static const ISA_ExtAddress bn2a_branch2a_scale__vv_mul__scale2a_branch2a_scale_size = 1;
static const ISA_ExtAddress bn2a_branch2a_bias__vv_mul__scale2a_branch2a_scale__vv_add__scale2a_branch2a_bias = 7;
static const ISA_ExtAddress bn2a_branch2a_bias__vv_mul__scale2a_branch2a_scale__vv_add__scale2a_branch2a_bias_size = 1;
static const ISA_ExtAddress res2a_branch2a_MRF = 5;
static const ISA_ExtAddress res2a_branch2a_MRF_size = 1;
static const ISA_ExtAddress bn2a_branch2b_scale__vv_mul__scale2a_branch2b_scale = 8;
static const ISA_ExtAddress bn2a_branch2b_scale__vv_mul__scale2a_branch2b_scale_size = 1;
static const ISA_ExtAddress bn2a_branch2b_bias__vv_mul__scale2a_branch2b_scale__vv_add__scale2a_branch2b_bias = 9;
static const ISA_ExtAddress bn2a_branch2b_bias__vv_mul__scale2a_branch2b_scale__vv_add__scale2a_branch2b_bias_size = 1;
static const ISA_ExtAddress res2a_branch2b_MRF = 6;
static const ISA_ExtAddress res2a_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn2a_branch2c_scale__vv_mul__scale2a_branch2c_scale = 10;
static const ISA_ExtAddress bn2a_branch2c_scale__vv_mul__scale2a_branch2c_scale_size = 2;
static const ISA_ExtAddress bn2a_branch2c_bias__vv_mul__scale2a_branch2c_scale__vv_add__scale2a_branch2c_bias = 12;
static const ISA_ExtAddress bn2a_branch2c_bias__vv_mul__scale2a_branch2c_scale__vv_add__scale2a_branch2c_bias_size = 2;
static const ISA_ExtAddress res2a_branch2c_MRF = 15;
static const ISA_ExtAddress res2a_branch2c_MRF_size = 2;
static const ISA_ExtAddress bn2b_branch2a_scale__vv_mul__scale2b_branch2a_scale = 14;
static const ISA_ExtAddress bn2b_branch2a_scale__vv_mul__scale2b_branch2a_scale_size = 1;
static const ISA_ExtAddress bn2b_branch2a_bias__vv_mul__scale2b_branch2a_scale__vv_add__scale2b_branch2a_bias = 15;
static const ISA_ExtAddress bn2b_branch2a_bias__vv_mul__scale2b_branch2a_scale__vv_add__scale2b_branch2a_bias_size = 1;
static const ISA_ExtAddress res2b_branch2a_MRF = 17;
static const ISA_ExtAddress res2b_branch2a_MRF_size = 2;
static const ISA_ExtAddress bn2b_branch2b_scale__vv_mul__scale2b_branch2b_scale = 16;
static const ISA_ExtAddress bn2b_branch2b_scale__vv_mul__scale2b_branch2b_scale_size = 1;
static const ISA_ExtAddress bn2b_branch2b_bias__vv_mul__scale2b_branch2b_scale__vv_add__scale2b_branch2b_bias = 17;
static const ISA_ExtAddress bn2b_branch2b_bias__vv_mul__scale2b_branch2b_scale__vv_add__scale2b_branch2b_bias_size = 1;
static const ISA_ExtAddress res2b_branch2b_MRF = 19;
static const ISA_ExtAddress res2b_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn2b_branch2c_scale__vv_mul__scale2b_branch2c_scale = 18;
static const ISA_ExtAddress bn2b_branch2c_scale__vv_mul__scale2b_branch2c_scale_size = 2;
static const ISA_ExtAddress bn2b_branch2c_bias__vv_mul__scale2b_branch2c_scale__vv_add__scale2b_branch2c_bias = 20;
static const ISA_ExtAddress bn2b_branch2c_bias__vv_mul__scale2b_branch2c_scale__vv_add__scale2b_branch2c_bias_size = 2;
static const ISA_ExtAddress res2b_branch2c_MRF = 28;
static const ISA_ExtAddress res2b_branch2c_MRF_size = 2;
static const ISA_ExtAddress bn2c_branch2a_scale__vv_mul__scale2c_branch2a_scale = 22;
static const ISA_ExtAddress bn2c_branch2a_scale__vv_mul__scale2c_branch2a_scale_size = 1;
static const ISA_ExtAddress bn2c_branch2a_bias__vv_mul__scale2c_branch2a_scale__vv_add__scale2c_branch2a_bias = 23;
static const ISA_ExtAddress bn2c_branch2a_bias__vv_mul__scale2c_branch2a_scale__vv_add__scale2c_branch2a_bias_size = 1;
static const ISA_ExtAddress res2c_branch2a_MRF = 30;
static const ISA_ExtAddress res2c_branch2a_MRF_size = 2;
static const ISA_ExtAddress bn2c_branch2b_scale__vv_mul__scale2c_branch2b_scale = 24;
static const ISA_ExtAddress bn2c_branch2b_scale__vv_mul__scale2c_branch2b_scale_size = 1;
static const ISA_ExtAddress bn2c_branch2b_bias__vv_mul__scale2c_branch2b_scale__vv_add__scale2c_branch2b_bias = 25;
static const ISA_ExtAddress bn2c_branch2b_bias__vv_mul__scale2c_branch2b_scale__vv_add__scale2c_branch2b_bias_size = 1;
static const ISA_ExtAddress res2c_branch2b_MRF = 32;
static const ISA_ExtAddress res2c_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn2c_branch2c_scale__vv_mul__scale2c_branch2c_scale = 26;
static const ISA_ExtAddress bn2c_branch2c_scale__vv_mul__scale2c_branch2c_scale_size = 2;
static const ISA_ExtAddress bn2c_branch2c_bias__vv_mul__scale2c_branch2c_scale__vv_add__scale2c_branch2c_bias = 28;
static const ISA_ExtAddress bn2c_branch2c_bias__vv_mul__scale2c_branch2c_scale__vv_add__scale2c_branch2c_bias_size = 2;
static const ISA_ExtAddress res2c_branch2c_MRF = 41;
static const ISA_ExtAddress res2c_branch2c_MRF_size = 2;
static const ISA_ExtAddress bn3a_branch1_scale__vv_mul__scale3a_branch1_scale = 30;
static const ISA_ExtAddress bn3a_branch1_scale__vv_mul__scale3a_branch1_scale_size = 4;
static const ISA_ExtAddress bn3a_branch1_bias__vv_mul__scale3a_branch1_scale__vv_add__scale3a_branch1_bias = 34;
static const ISA_ExtAddress bn3a_branch1_bias__vv_mul__scale3a_branch1_scale__vv_add__scale3a_branch1_bias_size = 4;
static const ISA_ExtAddress res3a_branch1_MRF = 43;
static const ISA_ExtAddress res3a_branch1_MRF_size = 8;
static const ISA_ExtAddress bn3a_branch2a_scale__vv_mul__scale3a_branch2a_scale = 38;
static const ISA_ExtAddress bn3a_branch2a_scale__vv_mul__scale3a_branch2a_scale_size = 1;
static const ISA_ExtAddress bn3a_branch2a_bias__vv_mul__scale3a_branch2a_scale__vv_add__scale3a_branch2a_bias = 39;
static const ISA_ExtAddress bn3a_branch2a_bias__vv_mul__scale3a_branch2a_scale__vv_add__scale3a_branch2a_bias_size = 1;
static const ISA_ExtAddress res3a_branch2a_MRF = 51;
static const ISA_ExtAddress res3a_branch2a_MRF_size = 2;
static const ISA_ExtAddress bn3a_branch2b_scale__vv_mul__scale3a_branch2b_scale = 40;
static const ISA_ExtAddress bn3a_branch2b_scale__vv_mul__scale3a_branch2b_scale_size = 1;
static const ISA_ExtAddress bn3a_branch2b_bias__vv_mul__scale3a_branch2b_scale__vv_add__scale3a_branch2b_bias = 41;
static const ISA_ExtAddress bn3a_branch2b_bias__vv_mul__scale3a_branch2b_scale__vv_add__scale3a_branch2b_bias_size = 1;
static const ISA_ExtAddress res3a_branch2b_MRF = 53;
static const ISA_ExtAddress res3a_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn3a_branch2c_scale__vv_mul__scale3a_branch2c_scale = 42;
static const ISA_ExtAddress bn3a_branch2c_scale__vv_mul__scale3a_branch2c_scale_size = 4;
static const ISA_ExtAddress bn3a_branch2c_bias__vv_mul__scale3a_branch2c_scale__vv_add__scale3a_branch2c_bias = 46;
static const ISA_ExtAddress bn3a_branch2c_bias__vv_mul__scale3a_branch2c_scale__vv_add__scale3a_branch2c_bias_size = 4;
static const ISA_ExtAddress res3a_branch2c_MRF = 62;
static const ISA_ExtAddress res3a_branch2c_MRF_size = 4;
static const ISA_ExtAddress bn3b_branch2a_scale__vv_mul__scale3b_branch2a_scale = 50;
static const ISA_ExtAddress bn3b_branch2a_scale__vv_mul__scale3b_branch2a_scale_size = 1;
static const ISA_ExtAddress bn3b_branch2a_bias__vv_mul__scale3b_branch2a_scale__vv_add__scale3b_branch2a_bias = 51;
static const ISA_ExtAddress bn3b_branch2a_bias__vv_mul__scale3b_branch2a_scale__vv_add__scale3b_branch2a_bias_size = 1;
static const ISA_ExtAddress res3b_branch2a_MRF = 66;
static const ISA_ExtAddress res3b_branch2a_MRF_size = 4;
static const ISA_ExtAddress bn3b_branch2b_scale__vv_mul__scale3b_branch2b_scale = 52;
static const ISA_ExtAddress bn3b_branch2b_scale__vv_mul__scale3b_branch2b_scale_size = 1;
static const ISA_ExtAddress bn3b_branch2b_bias__vv_mul__scale3b_branch2b_scale__vv_add__scale3b_branch2b_bias = 53;
static const ISA_ExtAddress bn3b_branch2b_bias__vv_mul__scale3b_branch2b_scale__vv_add__scale3b_branch2b_bias_size = 1;
static const ISA_ExtAddress res3b_branch2b_MRF = 70;
static const ISA_ExtAddress res3b_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn3b_branch2c_scale__vv_mul__scale3b_branch2c_scale = 54;
static const ISA_ExtAddress bn3b_branch2c_scale__vv_mul__scale3b_branch2c_scale_size = 4;
static const ISA_ExtAddress bn3b_branch2c_bias__vv_mul__scale3b_branch2c_scale__vv_add__scale3b_branch2c_bias = 58;
static const ISA_ExtAddress bn3b_branch2c_bias__vv_mul__scale3b_branch2c_scale__vv_add__scale3b_branch2c_bias_size = 4;
static const ISA_ExtAddress res3b_branch2c_MRF = 79;
static const ISA_ExtAddress res3b_branch2c_MRF_size = 4;
static const ISA_ExtAddress bn3c_branch2a_scale__vv_mul__scale3c_branch2a_scale = 62;
static const ISA_ExtAddress bn3c_branch2a_scale__vv_mul__scale3c_branch2a_scale_size = 1;
static const ISA_ExtAddress bn3c_branch2a_bias__vv_mul__scale3c_branch2a_scale__vv_add__scale3c_branch2a_bias = 63;
static const ISA_ExtAddress bn3c_branch2a_bias__vv_mul__scale3c_branch2a_scale__vv_add__scale3c_branch2a_bias_size = 1;
static const ISA_ExtAddress res3c_branch2a_MRF = 83;
static const ISA_ExtAddress res3c_branch2a_MRF_size = 4;
static const ISA_ExtAddress bn3c_branch2b_scale__vv_mul__scale3c_branch2b_scale = 64;
static const ISA_ExtAddress bn3c_branch2b_scale__vv_mul__scale3c_branch2b_scale_size = 1;
static const ISA_ExtAddress bn3c_branch2b_bias__vv_mul__scale3c_branch2b_scale__vv_add__scale3c_branch2b_bias = 65;
static const ISA_ExtAddress bn3c_branch2b_bias__vv_mul__scale3c_branch2b_scale__vv_add__scale3c_branch2b_bias_size = 1;
static const ISA_ExtAddress res3c_branch2b_MRF = 87;
static const ISA_ExtAddress res3c_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn3c_branch2c_scale__vv_mul__scale3c_branch2c_scale = 66;
static const ISA_ExtAddress bn3c_branch2c_scale__vv_mul__scale3c_branch2c_scale_size = 4;
static const ISA_ExtAddress bn3c_branch2c_bias__vv_mul__scale3c_branch2c_scale__vv_add__scale3c_branch2c_bias = 70;
static const ISA_ExtAddress bn3c_branch2c_bias__vv_mul__scale3c_branch2c_scale__vv_add__scale3c_branch2c_bias_size = 4;
static const ISA_ExtAddress res3c_branch2c_MRF = 96;
static const ISA_ExtAddress res3c_branch2c_MRF_size = 4;
static const ISA_ExtAddress bn3d_branch2a_scale__vv_mul__scale3d_branch2a_scale = 74;
static const ISA_ExtAddress bn3d_branch2a_scale__vv_mul__scale3d_branch2a_scale_size = 1;
static const ISA_ExtAddress bn3d_branch2a_bias__vv_mul__scale3d_branch2a_scale__vv_add__scale3d_branch2a_bias = 75;
static const ISA_ExtAddress bn3d_branch2a_bias__vv_mul__scale3d_branch2a_scale__vv_add__scale3d_branch2a_bias_size = 1;
static const ISA_ExtAddress res3d_branch2a_MRF = 100;
static const ISA_ExtAddress res3d_branch2a_MRF_size = 4;
static const ISA_ExtAddress bn3d_branch2b_scale__vv_mul__scale3d_branch2b_scale = 76;
static const ISA_ExtAddress bn3d_branch2b_scale__vv_mul__scale3d_branch2b_scale_size = 1;
static const ISA_ExtAddress bn3d_branch2b_bias__vv_mul__scale3d_branch2b_scale__vv_add__scale3d_branch2b_bias = 77;
static const ISA_ExtAddress bn3d_branch2b_bias__vv_mul__scale3d_branch2b_scale__vv_add__scale3d_branch2b_bias_size = 1;
static const ISA_ExtAddress res3d_branch2b_MRF = 104;
static const ISA_ExtAddress res3d_branch2b_MRF_size = 9;
static const ISA_ExtAddress bn3d_branch2c_scale__vv_mul__scale3d_branch2c_scale = 78;
static const ISA_ExtAddress bn3d_branch2c_scale__vv_mul__scale3d_branch2c_scale_size = 4;
static const ISA_ExtAddress bn3d_branch2c_bias__vv_mul__scale3d_branch2c_scale__vv_add__scale3d_branch2c_bias = 82;
static const ISA_ExtAddress bn3d_branch2c_bias__vv_mul__scale3d_branch2c_scale__vv_add__scale3d_branch2c_bias_size = 4;
static const ISA_ExtAddress res3d_branch2c_MRF = 113;
static const ISA_ExtAddress res3d_branch2c_MRF_size = 4;
static const ISA_ExtAddress bn4a_branch1_scale__vv_mul__scale4a_branch1_scale = 86;
static const ISA_ExtAddress bn4a_branch1_scale__vv_mul__scale4a_branch1_scale_size = 8;
static const ISA_ExtAddress bn4a_branch1_bias__vv_mul__scale4a_branch1_scale__vv_add__scale4a_branch1_bias = 94;
static const ISA_ExtAddress bn4a_branch1_bias__vv_mul__scale4a_branch1_scale__vv_add__scale4a_branch1_bias_size = 8;
static const ISA_ExtAddress res4a_branch1_MRF = 117;
static const ISA_ExtAddress res4a_branch1_MRF_size = 32;
static const ISA_ExtAddress bn4a_branch2a_scale__vv_mul__scale4a_branch2a_scale = 102;
static const ISA_ExtAddress bn4a_branch2a_scale__vv_mul__scale4a_branch2a_scale_size = 2;
static const ISA_ExtAddress bn4a_branch2a_bias__vv_mul__scale4a_branch2a_scale__vv_add__scale4a_branch2a_bias = 104;
static const ISA_ExtAddress bn4a_branch2a_bias__vv_mul__scale4a_branch2a_scale__vv_add__scale4a_branch2a_bias_size = 2;
static const ISA_ExtAddress res4a_branch2a_MRF = 149;
static const ISA_ExtAddress res4a_branch2a_MRF_size = 8;
static const ISA_ExtAddress bn4a_branch2b_scale__vv_mul__scale4a_branch2b_scale = 106;
static const ISA_ExtAddress bn4a_branch2b_scale__vv_mul__scale4a_branch2b_scale_size = 2;
static const ISA_ExtAddress bn4a_branch2b_bias__vv_mul__scale4a_branch2b_scale__vv_add__scale4a_branch2b_bias = 108;
static const ISA_ExtAddress bn4a_branch2b_bias__vv_mul__scale4a_branch2b_scale__vv_add__scale4a_branch2b_bias_size = 2;
static const ISA_ExtAddress res4a_branch2b_MRF = 157;
static const ISA_ExtAddress res4a_branch2b_MRF_size = 36;
static const ISA_ExtAddress bn4a_branch2c_scale__vv_mul__scale4a_branch2c_scale = 110;
static const ISA_ExtAddress bn4a_branch2c_scale__vv_mul__scale4a_branch2c_scale_size = 8;
static const ISA_ExtAddress bn4a_branch2c_bias__vv_mul__scale4a_branch2c_scale__vv_add__scale4a_branch2c_bias = 118;
static const ISA_ExtAddress bn4a_branch2c_bias__vv_mul__scale4a_branch2c_scale__vv_add__scale4a_branch2c_bias_size = 8;
static const ISA_ExtAddress res4a_branch2c_MRF = 193;
static const ISA_ExtAddress res4a_branch2c_MRF_size = 16;
static const ISA_ExtAddress bn4b_branch2a_scale__vv_mul__scale4b_branch2a_scale = 126;
static const ISA_ExtAddress bn4b_branch2a_scale__vv_mul__scale4b_branch2a_scale_size = 2;
static const ISA_ExtAddress bn4b_branch2a_bias__vv_mul__scale4b_branch2a_scale__vv_add__scale4b_branch2a_bias = 128;
static const ISA_ExtAddress bn4b_branch2a_bias__vv_mul__scale4b_branch2a_scale__vv_add__scale4b_branch2a_bias_size = 2;
static const ISA_ExtAddress res4b_branch2a_MRF = 209;
static const ISA_ExtAddress res4b_branch2a_MRF_size = 16;
static const ISA_ExtAddress bn4b_branch2b_scale__vv_mul__scale4b_branch2b_scale = 130;
static const ISA_ExtAddress bn4b_branch2b_scale__vv_mul__scale4b_branch2b_scale_size = 2;
static const ISA_ExtAddress bn4b_branch2b_bias__vv_mul__scale4b_branch2b_scale__vv_add__scale4b_branch2b_bias = 132;
static const ISA_ExtAddress bn4b_branch2b_bias__vv_mul__scale4b_branch2b_scale__vv_add__scale4b_branch2b_bias_size = 2;
static const ISA_ExtAddress res4b_branch2b_MRF = 225;
static const ISA_ExtAddress res4b_branch2b_MRF_size = 36;
static const ISA_ExtAddress bn4b_branch2c_scale__vv_mul__scale4b_branch2c_scale = 134;
static const ISA_ExtAddress bn4b_branch2c_scale__vv_mul__scale4b_branch2c_scale_size = 8;
static const ISA_ExtAddress bn4b_branch2c_bias__vv_mul__scale4b_branch2c_scale__vv_add__scale4b_branch2c_bias = 142;
static const ISA_ExtAddress bn4b_branch2c_bias__vv_mul__scale4b_branch2c_scale__vv_add__scale4b_branch2c_bias_size = 8;
static const ISA_ExtAddress res4b_branch2c_MRF = 261;
static const ISA_ExtAddress res4b_branch2c_MRF_size = 16;
static const ISA_ExtAddress bn4c_branch2a_scale__vv_mul__scale4c_branch2a_scale = 150;
static const ISA_ExtAddress bn4c_branch2a_scale__vv_mul__scale4c_branch2a_scale_size = 2;
static const ISA_ExtAddress bn4c_branch2a_bias__vv_mul__scale4c_branch2a_scale__vv_add__scale4c_branch2a_bias = 152;
static const ISA_ExtAddress bn4c_branch2a_bias__vv_mul__scale4c_branch2a_scale__vv_add__scale4c_branch2a_bias_size = 2;
static const ISA_ExtAddress res4c_branch2a_MRF = 277;
static const ISA_ExtAddress res4c_branch2a_MRF_size = 16;
static const ISA_ExtAddress bn4c_branch2b_scale__vv_mul__scale4c_branch2b_scale = 154;
static const ISA_ExtAddress bn4c_branch2b_scale__vv_mul__scale4c_branch2b_scale_size = 2;
static const ISA_ExtAddress bn4c_branch2b_bias__vv_mul__scale4c_branch2b_scale__vv_add__scale4c_branch2b_bias = 156;
static const ISA_ExtAddress bn4c_branch2b_bias__vv_mul__scale4c_branch2b_scale__vv_add__scale4c_branch2b_bias_size = 2;
static const ISA_ExtAddress res4c_branch2b_MRF = 293;
static const ISA_ExtAddress res4c_branch2b_MRF_size = 36;
static const ISA_ExtAddress bn4c_branch2c_scale__vv_mul__scale4c_branch2c_scale = 158;
static const ISA_ExtAddress bn4c_branch2c_scale__vv_mul__scale4c_branch2c_scale_size = 8;
static const ISA_ExtAddress bn4c_branch2c_bias__vv_mul__scale4c_branch2c_scale__vv_add__scale4c_branch2c_bias = 166;
static const ISA_ExtAddress bn4c_branch2c_bias__vv_mul__scale4c_branch2c_scale__vv_add__scale4c_branch2c_bias_size = 8;
static const ISA_ExtAddress res4c_branch2c_MRF = 329;
static const ISA_ExtAddress res4c_branch2c_MRF_size = 16;
static const ISA_ExtAddress bn4d_branch2a_scale__vv_mul__scale4d_branch2a_scale = 174;
static const ISA_ExtAddress bn4d_branch2a_scale__vv_mul__scale4d_branch2a_scale_size = 2;
static const ISA_ExtAddress bn4d_branch2a_bias__vv_mul__scale4d_branch2a_scale__vv_add__scale4d_branch2a_bias = 176;
static const ISA_ExtAddress bn4d_branch2a_bias__vv_mul__scale4d_branch2a_scale__vv_add__scale4d_branch2a_bias_size = 2;
static const ISA_ExtAddress res4d_branch2a_MRF = 345;
static const ISA_ExtAddress res4d_branch2a_MRF_size = 16;
static const ISA_ExtAddress bn4d_branch2b_scale__vv_mul__scale4d_branch2b_scale = 178;
static const ISA_ExtAddress bn4d_branch2b_scale__vv_mul__scale4d_branch2b_scale_size = 2;
static const ISA_ExtAddress bn4d_branch2b_bias__vv_mul__scale4d_branch2b_scale__vv_add__scale4d_branch2b_bias = 180;
static const ISA_ExtAddress bn4d_branch2b_bias__vv_mul__scale4d_branch2b_scale__vv_add__scale4d_branch2b_bias_size = 2;
static const ISA_ExtAddress res4d_branch2b_MRF = 361;
static const ISA_ExtAddress res4d_branch2b_MRF_size = 36;
static const ISA_ExtAddress bn4d_branch2c_scale__vv_mul__scale4d_branch2c_scale = 182;
static const ISA_ExtAddress bn4d_branch2c_scale__vv_mul__scale4d_branch2c_scale_size = 8;
static const ISA_ExtAddress bn4d_branch2c_bias__vv_mul__scale4d_branch2c_scale__vv_add__scale4d_branch2c_bias = 190;
static const ISA_ExtAddress bn4d_branch2c_bias__vv_mul__scale4d_branch2c_scale__vv_add__scale4d_branch2c_bias_size = 8;
static const ISA_ExtAddress res4d_branch2c_MRF = 397;
static const ISA_ExtAddress res4d_branch2c_MRF_size = 16;
static const ISA_ExtAddress bn4e_branch2a_scale__vv_mul__scale4e_branch2a_scale = 198;
static const ISA_ExtAddress bn4e_branch2a_scale__vv_mul__scale4e_branch2a_scale_size = 2;
static const ISA_ExtAddress bn4e_branch2a_bias__vv_mul__scale4e_branch2a_scale__vv_add__scale4e_branch2a_bias = 200;
static const ISA_ExtAddress bn4e_branch2a_bias__vv_mul__scale4e_branch2a_scale__vv_add__scale4e_branch2a_bias_size = 2;
static const ISA_ExtAddress res4e_branch2a_MRF = 413;
static const ISA_ExtAddress res4e_branch2a_MRF_size = 16;
static const ISA_ExtAddress bn4e_branch2b_scale__vv_mul__scale4e_branch2b_scale = 202;
static const ISA_ExtAddress bn4e_branch2b_scale__vv_mul__scale4e_branch2b_scale_size = 2;
static const ISA_ExtAddress bn4e_branch2b_bias__vv_mul__scale4e_branch2b_scale__vv_add__scale4e_branch2b_bias = 204;
static const ISA_ExtAddress bn4e_branch2b_bias__vv_mul__scale4e_branch2b_scale__vv_add__scale4e_branch2b_bias_size = 2;
static const ISA_ExtAddress res4e_branch2b_MRF = 429;
static const ISA_ExtAddress res4e_branch2b_MRF_size = 36;
static const ISA_ExtAddress bn4e_branch2c_scale__vv_mul__scale4e_branch2c_scale = 206;
static const ISA_ExtAddress bn4e_branch2c_scale__vv_mul__scale4e_branch2c_scale_size = 8;
static const ISA_ExtAddress bn4e_branch2c_bias__vv_mul__scale4e_branch2c_scale__vv_add__scale4e_branch2c_bias = 214;
static const ISA_ExtAddress bn4e_branch2c_bias__vv_mul__scale4e_branch2c_scale__vv_add__scale4e_branch2c_bias_size = 8;
static const ISA_ExtAddress res4e_branch2c_MRF = 465;
static const ISA_ExtAddress res4e_branch2c_MRF_size = 16;
static const ISA_ExtAddress bn4f_branch2a_scale__vv_mul__scale4f_branch2a_scale = 222;
static const ISA_ExtAddress bn4f_branch2a_scale__vv_mul__scale4f_branch2a_scale_size = 2;
static const ISA_ExtAddress bn4f_branch2a_bias__vv_mul__scale4f_branch2a_scale__vv_add__scale4f_branch2a_bias = 224;
static const ISA_ExtAddress bn4f_branch2a_bias__vv_mul__scale4f_branch2a_scale__vv_add__scale4f_branch2a_bias_size = 2;
static const ISA_ExtAddress res4f_branch2a_MRF = 481;
static const ISA_ExtAddress res4f_branch2a_MRF_size = 16;
static const ISA_ExtAddress bn4f_branch2b_scale__vv_mul__scale4f_branch2b_scale = 226;
static const ISA_ExtAddress bn4f_branch2b_scale__vv_mul__scale4f_branch2b_scale_size = 2;
static const ISA_ExtAddress bn4f_branch2b_bias__vv_mul__scale4f_branch2b_scale__vv_add__scale4f_branch2b_bias = 228;
static const ISA_ExtAddress bn4f_branch2b_bias__vv_mul__scale4f_branch2b_scale__vv_add__scale4f_branch2b_bias_size = 2;
static const ISA_ExtAddress res4f_branch2b_MRF = 497;
static const ISA_ExtAddress res4f_branch2b_MRF_size = 36;
static const ISA_ExtAddress bn4f_branch2c_scale__vv_mul__scale4f_branch2c_scale = 230;
static const ISA_ExtAddress bn4f_branch2c_scale__vv_mul__scale4f_branch2c_scale_size = 8;
static const ISA_ExtAddress bn4f_branch2c_bias__vv_mul__scale4f_branch2c_scale__vv_add__scale4f_branch2c_bias = 238;
static const ISA_ExtAddress bn4f_branch2c_bias__vv_mul__scale4f_branch2c_scale__vv_add__scale4f_branch2c_bias_size = 8;
static const ISA_ExtAddress res4f_branch2c_MRF = 533;
static const ISA_ExtAddress res4f_branch2c_MRF_size = 16;
static const ISA_ExtAddress bn5a_branch1_scale__vv_mul__scale5a_branch1_scale = 246;
static const ISA_ExtAddress bn5a_branch1_scale__vv_mul__scale5a_branch1_scale_size = 16;
static const ISA_ExtAddress bn5a_branch1_bias__vv_mul__scale5a_branch1_scale__vv_add__scale5a_branch1_bias = 262;
static const ISA_ExtAddress bn5a_branch1_bias__vv_mul__scale5a_branch1_scale__vv_add__scale5a_branch1_bias_size = 16;
static const ISA_ExtAddress res5a_branch1_MRF = 549;
static const ISA_ExtAddress res5a_branch1_MRF_size = 128;
static const ISA_ExtAddress bn5a_branch2a_scale__vv_mul__scale5a_branch2a_scale = 278;
static const ISA_ExtAddress bn5a_branch2a_scale__vv_mul__scale5a_branch2a_scale_size = 4;
static const ISA_ExtAddress bn5a_branch2a_bias__vv_mul__scale5a_branch2a_scale__vv_add__scale5a_branch2a_bias = 282;
static const ISA_ExtAddress bn5a_branch2a_bias__vv_mul__scale5a_branch2a_scale__vv_add__scale5a_branch2a_bias_size = 4;
static const ISA_ExtAddress res5a_branch2a_MRF = 677;
static const ISA_ExtAddress res5a_branch2a_MRF_size = 32;
static const ISA_ExtAddress bn5a_branch2b_scale__vv_mul__scale5a_branch2b_scale = 286;
static const ISA_ExtAddress bn5a_branch2b_scale__vv_mul__scale5a_branch2b_scale_size = 4;
static const ISA_ExtAddress bn5a_branch2b_bias__vv_mul__scale5a_branch2b_scale__vv_add__scale5a_branch2b_bias = 290;
static const ISA_ExtAddress bn5a_branch2b_bias__vv_mul__scale5a_branch2b_scale__vv_add__scale5a_branch2b_bias_size = 4;
static const ISA_ExtAddress res5a_branch2b_MRF = 709;
static const ISA_ExtAddress res5a_branch2b_MRF_size = 144;
static const ISA_ExtAddress bn5a_branch2c_scale__vv_mul__scale5a_branch2c_scale = 294;
static const ISA_ExtAddress bn5a_branch2c_scale__vv_mul__scale5a_branch2c_scale_size = 16;
static const ISA_ExtAddress bn5a_branch2c_bias__vv_mul__scale5a_branch2c_scale__vv_add__scale5a_branch2c_bias = 310;
static const ISA_ExtAddress bn5a_branch2c_bias__vv_mul__scale5a_branch2c_scale__vv_add__scale5a_branch2c_bias_size = 16;
static const ISA_ExtAddress res5a_branch2c_MRF = 853;
static const ISA_ExtAddress res5a_branch2c_MRF_size = 64;
static const ISA_ExtAddress bn5b_branch2a_scale__vv_mul__scale5b_branch2a_scale = 326;
static const ISA_ExtAddress bn5b_branch2a_scale__vv_mul__scale5b_branch2a_scale_size = 4;
static const ISA_ExtAddress bn5b_branch2a_bias__vv_mul__scale5b_branch2a_scale__vv_add__scale5b_branch2a_bias = 330;
static const ISA_ExtAddress bn5b_branch2a_bias__vv_mul__scale5b_branch2a_scale__vv_add__scale5b_branch2a_bias_size = 4;
static const ISA_ExtAddress res5b_branch2a_MRF = 917;
static const ISA_ExtAddress res5b_branch2a_MRF_size = 64;
static const ISA_ExtAddress bn5b_branch2b_scale__vv_mul__scale5b_branch2b_scale = 334;
static const ISA_ExtAddress bn5b_branch2b_scale__vv_mul__scale5b_branch2b_scale_size = 4;
static const ISA_ExtAddress bn5b_branch2b_bias__vv_mul__scale5b_branch2b_scale__vv_add__scale5b_branch2b_bias = 338;
static const ISA_ExtAddress bn5b_branch2b_bias__vv_mul__scale5b_branch2b_scale__vv_add__scale5b_branch2b_bias_size = 4;
static const ISA_ExtAddress res5b_branch2b_MRF = 981;
static const ISA_ExtAddress res5b_branch2b_MRF_size = 144;
static const ISA_ExtAddress bn5b_branch2c_scale__vv_mul__scale5b_branch2c_scale = 342;
static const ISA_ExtAddress bn5b_branch2c_scale__vv_mul__scale5b_branch2c_scale_size = 16;
static const ISA_ExtAddress bn5b_branch2c_bias__vv_mul__scale5b_branch2c_scale__vv_add__scale5b_branch2c_bias = 358;
static const ISA_ExtAddress bn5b_branch2c_bias__vv_mul__scale5b_branch2c_scale__vv_add__scale5b_branch2c_bias_size = 16;
static const ISA_ExtAddress res5b_branch2c_MRF = 1125;
static const ISA_ExtAddress res5b_branch2c_MRF_size = 64;
static const ISA_ExtAddress bn5c_branch2a_scale__vv_mul__scale5c_branch2a_scale = 374;
static const ISA_ExtAddress bn5c_branch2a_scale__vv_mul__scale5c_branch2a_scale_size = 4;
static const ISA_ExtAddress bn5c_branch2a_bias__vv_mul__scale5c_branch2a_scale__vv_add__scale5c_branch2a_bias = 378;
static const ISA_ExtAddress bn5c_branch2a_bias__vv_mul__scale5c_branch2a_scale__vv_add__scale5c_branch2a_bias_size = 4;
static const ISA_ExtAddress res5c_branch2a_MRF = 1189;
static const ISA_ExtAddress res5c_branch2a_MRF_size = 64;
static const ISA_ExtAddress bn5c_branch2b_scale__vv_mul__scale5c_branch2b_scale = 382;
static const ISA_ExtAddress bn5c_branch2b_scale__vv_mul__scale5c_branch2b_scale_size = 4;
static const ISA_ExtAddress bn5c_branch2b_bias__vv_mul__scale5c_branch2b_scale__vv_add__scale5c_branch2b_bias = 386;
static const ISA_ExtAddress bn5c_branch2b_bias__vv_mul__scale5c_branch2b_scale__vv_add__scale5c_branch2b_bias_size = 4;
static const ISA_ExtAddress res5c_branch2b_MRF = 1253;
static const ISA_ExtAddress res5c_branch2b_MRF_size = 144;
static const ISA_ExtAddress bn5c_branch2c_scale__vv_mul__scale5c_branch2c_scale = 390;
static const ISA_ExtAddress bn5c_branch2c_scale__vv_mul__scale5c_branch2c_scale_size = 16;
static const ISA_ExtAddress bn5c_branch2c_bias__vv_mul__scale5c_branch2c_scale__vv_add__scale5c_branch2c_bias = 406;
static const ISA_ExtAddress bn5c_branch2c_bias__vv_mul__scale5c_branch2c_scale__vv_add__scale5c_branch2c_bias_size = 16;
static const ISA_ExtAddress res5c_branch2c_MRF = 1397;
static const ISA_ExtAddress res5c_branch2c_MRF_size = 64;
static const ISA_ExtAddress zeros = 422;
static const ISA_ExtAddress zeros_size = 0;

/* Common variables */
ISA_ExtAddress ivrf_inIterator;
ISA_MrfAddress mrf_start=0, mrf_next=64, mrf_tmp;

/* Layer function prototypes */
void input(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void conv1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2aBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2aBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2bBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2bBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2bBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2cBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2cBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res2cBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3aBranch1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3aBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3aBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3aBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3bBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3bBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3bBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3cBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3cBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3cBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3dBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3dBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res3dBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4aBranch1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4aBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4aBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4aBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4bBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4bBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4bBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4cBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4cBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4cBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4dBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4dBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4dBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4eBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4eBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4eBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4fBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4fBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res4fBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5aBranch1(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5aBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5aBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5aBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5bBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5bBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5bBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5cBranch2a(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5cBranch2b(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);
void res5cBranch2c(PBS_CONTEXT p_bs, bool p_debugMode, bool p_first, bool p_last);

/* The function table for the layer functions */
typedef void (*layer_fn_t)(PBS_CONTEXT, bool, bool, bool);
static const layer_fn_t c_LayerFunctionTable[] = {
    input,
    conv1,
    res2aBranch2b,
    res2aBranch2c,
    res2bBranch2a,
    res2bBranch2b,
    res2bBranch2c,
    res2cBranch2a,
    res2cBranch2b,
    res2cBranch2c,
    res3aBranch1,
    res3aBranch2a,
    res3aBranch2b,
    res3aBranch2c,
    res3bBranch2a,
    res3bBranch2b,
    res3bBranch2c,
    res3cBranch2a,
    res3cBranch2b,
    res3cBranch2c,
    res3dBranch2a,
    res3dBranch2b,
    res3dBranch2c,
    res4aBranch1,
    res4aBranch2a,
    res4aBranch2b,
    res4aBranch2c,
    res4bBranch2a,
    res4bBranch2b,
    res4bBranch2c,
    res4cBranch2a,
    res4cBranch2b,
    res4cBranch2c,
    res4dBranch2a,
    res4dBranch2b,
    res4dBranch2c,
    res4eBranch2a,
    res4eBranch2b,
    res4eBranch2c,
    res4fBranch2a,
    res4fBranch2b,
    res4fBranch2c,
    res5aBranch1,
    res5aBranch2a,
    res5aBranch2b,
    res5aBranch2c,
    res5bBranch2a,
    res5bBranch2b,
    res5bBranch2c,
    res5cBranch2a,
    res5cBranch2b,
    res5cBranch2c,
};

// Init service function for use with ONNX Runtime.
void init(PBS_CONTEXT bs, const BrainSliceOperator_RuntimeArguments* args)
{
    /* Sanity check for the BrainSlice SKU for this firmware. */
    BSNL_HEX_Assert(bs->m_bsParameters.NATIVE_DIM == 128, NIOS_HEX_CNN_AUTOGEN_NATIVE_DIM_MISMATCH);
    BSNL_HEX_Assert(bs->m_bsParameters.MFUS >= 2, NIOS_HEX_CNN_AUTOGEN_MFUS_TOO_FEW);
    BSNL_HEX_Assert(bs->m_bsParameters.INITIAL_VRF_SIZE >= 9075, NIOS_HEX_CNN_AUTOGEN_INITIAL_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.MVM_MATRIX_RF_SIZE >= 128, NIOS_HEX_CNN_AUTOGEN_MATRIX_RF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.ADDSUB_VRF_0_SIZE >= 16, NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.ADDSUB_VRF_1_SIZE >= 12100, NIOS_HEX_CNN_AUTOGEN_ADDSUB_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.MULTIPLY_VRF_SIZE >= 16, NIOS_HEX_CNN_AUTOGEN_MULTIPLY_VRF_TOO_SMALL);
    BSNL_HEX_Assert(bs->m_bsParameters.USE_DRAM  , NIOS_HEX_CNN_AUTOGEN_MISSING_NEEDED_DRAM);
    BSNL_HEX_Assert(bs->m_bsParameters.VECTOR_MEM_SIZE >= 422, NIOS_HEX_CNN_AUTOGEN_VECTOR_MEM_TOO_SMALL);

    BSNL_postResponseSubmessage(bs, 0);
}

/**
 * The main function that runs evaluation on the ResNet-50 model.
 *
 * This runs the input on the network through the specified subset of the network.
 **/
void execute(PBS_CONTEXT p_bs, const BrainSliceOperator_RuntimeArguments* args, int p_startLayerID, int p_endLayerID, bool p_debugMode)
{
    // By default, run all the ResNet-50 layers
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
        outputNativeSize = 784;
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
            vRead1D(p_bs, ISA_Mem_Dram, -1, 784);
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

void input(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Input layer: input(d=147,h=112,w=112)=INPUT 1240 registers */
    vRead1D(bs, ISA_Mem_NetInputQ, DONTCARE, 1240);
    v_wr(bs, ISA_Mem_Expander, 0+0);
    /* End input layer */
}

void conv1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Fused convolution block */
    /* Convolution conv1(d=64,h=112,w=112)=Conv(input(d=147,h=112,w=112),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*  absorbed bn_conv1 */
    /*  absorbed scale_conv1 */
    /*  absorbed conv1_relu */
    /*  includes sublayer pool1(d=64,h=55,w=55)=Pool(conv1(d=64,h=112,w=112),Op=MAX,k_h=3,k_w=3,s_h=2,s_w=2,p_h=0,p_w=0) */
    /*  includes sublayer res2a_branch2a(d=64,h=55,w=55)=Conv(pool1(d=64,h=55,w=55),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*    absorbed bn2a_branch2a */
    /*    absorbed scale2a_branch2a */
    /*    absorbed res2a_branch2a_relu */
    /*  includes buddy res2a_branch1(d=256,h=55,w=55)=Conv(pool1(d=64,h=55,w=55),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*    absorbed bn2a_branch1 */
    /*    absorbed scale2a_branch1 */
    ISA_ExtAddress input_inIndex,res2a_branch1_outIterations;
    res2a_branch1_outIterations=0;
    input_inIndex=0;
    moveFilterCount128(bs, ISA_Mem_Dram, conv1_MRF+0*62, ISA_Mem_MatrixRf, mrf_start, 1, 62);
    /* Layer conv1 tile size 1*224 */
    /* Temp vars and parameters for input layer conv1 */
    /* SetIterations on reads from the input expander must be a multiple of bs->m_bsParameters.CHANNELS or this must be the last read*/
    ISA_NativeCount maxReadSize=(ISA_NativeCount)((112/bs->m_bsParameters.CHANNELS)*bs->m_bsParameters.CHANNELS);
    /* _in is the read pointer (not adjusted for padding because we read the whole row), _next is the write pointer (adjusted for padding) */
    ISA_ExtAddress g0_conv1_in=3025, g0_conv1_inIterator=3025;
    ISA_ExtAddress g0_conv1_next=3249, g0_conv1_available=maxReadSize, g0_conv1_next_available=maxReadSize, g0_conv1_tmp;
    int g0_conv1_iterationsLeft=12544;
    vRead1D(bs, ISA_Mem_Dram, bn_conv1_scale__vv_mul__scale_conv1_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, conv1_bias__vv_add__bn_conv1_bias__vv_mul__scale_conv1_scale__vv_add__scale_conv1_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    /* Layer pool1 tile size 24*112 */
    /* Temp vars and parameters for input layer pool1 */
    /* _in is the read pointer (not adjusted for padding because we read the whole row), _next is the write pointer (adjusted for padding) */
    ISA_ExtAddress g1_pool1_in=3473,g1_pool1_inIterator=3473;
    ISA_ExtAddress g1_pool1_next=3473, g1_pool1_tmp=3473;
    ISA_ExtAddress g1_pool1_available=0;
    ISA_ExtAddress g1_pool1_accumulators=6161;
    /* Output buffer for layer pool1 is 605 registers */
    /* Sharing part of the intermediate buffer g1_pool1_accumulators */
    /* Kernel size is odd. Using the second half. */
    ISA_ExtAddress g1_pool1_outBuffer=g1_pool1_accumulators+1320;
    int g1_pool1_iterationsLeft=3025;
    /* Layer res2a_branch1 gets its input from g1_pool1_in */
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch1_scale__vv_mul__scale2a_branch1_scale, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 1);
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch1_bias__vv_mul__scale2a_branch1_scale__vv_add__scale2a_branch1_bias, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 1);
    /* Layer res2a_branch2a gets its input from g2_res2a_branch2a_in */
    ISA_ExtAddress g2_res2a_branch2a_in=g1_pool1_outBuffer;
    ISA_ExtAddress g2_res2a_branch2a_available=0;
    ISA_ExtAddress g2_res2a_branch2a_inIterator;
    int g2_res2a_branch2a_iterationsLeft=3025;
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch2a_scale__vv_mul__scale2a_branch2a_scale, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 3);
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch2a_bias__vv_mul__scale2a_branch2a_scale__vv_add__scale2a_branch2a_bias, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 3);
    /* Page in the first group of input activations */
    vRead2D(bs, ISA_Mem_Expander, input_inIndex, 2, maxReadSize, 2);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_in, 2);
    vRead2D(bs, ISA_Mem_Expander, input_inIndex, 2, maxReadSize, 2);
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_next, 2);
    g0_conv1_iterationsLeft -= 2 * maxReadSize;
    /* Loop until we've read all outputs */
    while (g2_res2a_branch2a_iterationsLeft>0) {

        /* Start of group 0 */

        /* Prefetch activations for the next iteration of the loop to hide DRAM latency */
        if (g0_conv1_available==0) {
            /* This is complicated in order to ensure that iterations%channels = 0 or this is the last transfer */
            /* swap buffers, then fetch next */
            g0_conv1_tmp=g0_conv1_in; g0_conv1_in=g0_conv1_next; g0_conv1_next=g0_conv1_tmp;
            g0_conv1_inIterator = g0_conv1_in;
            g0_conv1_available = g0_conv1_next_available;
            if (g0_conv1_iterationsLeft > 0) {
                g0_conv1_next_available = g0_conv1_iterationsLeft;
                if (g0_conv1_next_available > maxReadSize) {
                    g0_conv1_next_available = maxReadSize;
                }
                vRead2D(bs, ISA_Mem_Expander, input_inIndex, 2, g0_conv1_next_available, 2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g0_conv1_next, 2);
                g0_conv1_iterationsLeft -= g0_conv1_next_available;
            }
        }

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g0_conv1_available <= 112, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if (g0_conv1_available > 0){

            /* Start of layer 0 in group 0 (conv1) */
            ISA_NativeCount toCompute=g0_conv1_available;
            if ((g1_pool1_tmp + toCompute) > 6161) {
                toCompute = 6161 - g1_pool1_tmp;
            }
            if ((g1_pool1_available + toCompute) > 2688) {
                toCompute = 2688 - g1_pool1_available;
            }
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g0_conv1_inIterator, 2, toCompute, 2);
            mv_mul(bs, mrf_start+0);
            vv_mul(bs, 0); /* includes: bn_conv1: scale, vv_mul, scale_conv1: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0, 0); /* includes: conv1: bias, vv_add, bn_conv1: bias, vv_mul, scale_conv1: scale, vv_add, scale_conv1: bias */
            v_relu(bs); /* includes: conv1_relu: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool1_tmp, 1);
            v_wr_inc(bs, ISA_Mem_AddSubVrf_1, g1_pool1_tmp+3025, 1);
            g1_pool1_next += toCompute;
            if (g1_pool1_next == 6161) {
                g1_pool1_next = 3473;
            }
            g1_pool1_available += toCompute;
            g0_conv1_inIterator += toCompute *2;
            g0_conv1_available -= toCompute;
            g1_pool1_tmp = g1_pool1_next;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g0_conv1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 1 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g1_pool1_available <= 2688, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        Emulator_HEX_Assert(g1_pool1_next <= 6161, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        Emulator_HEX_Assert(g1_pool1_next >= 3473, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if ((g1_pool1_available == 2688) || (g1_pool1_available == g1_pool1_iterationsLeft)) {

            /* Start of layer 0 in group 1 (pool1) */
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
            if (g1_pool1_inIterator==g1_pool1_in) {
                horizontalRows=24;
            }
            g1_pool1_available -= verticalRows*224;
            ISA_ExtAddress curOffset;
            curOffset=g1_pool1_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 55, 2);
                mv_mul(bs, mrf_start+2);
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+1+3025, 2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1);
                if (curOffset==6049) {
                    curOffset-=2576;
                } else {
                    curOffset+=112;
                }
            }
            curOffset=g1_pool1_inIterator;
            for(int rowIterator=0;rowIterator<horizontalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+2+3025, 2);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, (g1_pool1_accumulators+1320)+rowIterator*55, 1);
                if (curOffset==6049) {
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
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, curOffset+3025, 1);
                if (curOffset==6049) {
                    curOffset-=2576;
                } else {
                    curOffset+=112;
                }
            }
            /* Update horizontal pool iterator start */
            g1_pool1_inIterator = curOffset;
            curOffset=g1_pool1_in;
            ISA_ExtAddress nextOffset=curOffset;
            if (nextOffset==6049) {
                nextOffset-=2576;
            } else {
                nextOffset+=112;
            }
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, curOffset, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset+3025, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1);
                if (curOffset>=5937) {
                    curOffset-=2464;
                } else {
                    curOffset+=224;
                }
                if (nextOffset>=5937) {
                    nextOffset-=2464;
                } else {
                    nextOffset+=224;
                }
            }
            curOffset=g1_pool1_in;
            nextOffset=curOffset;
            if (nextOffset>=5937) {
                nextOffset-=2464;
            } else {
                nextOffset+=224;
            }
            for(int rowIterator=0;rowIterator<verticalRows; rowIterator++) {
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g1_pool1_accumulators+rowIterator*55, 1, 55, 1);
                mv_mul(bs, mrf_start+2);
                vv_max_inc(bs, ISA_Mem_AddSubVrf_1, nextOffset+3025, 1);
                v_wr_inc(bs, ISA_Mem_MvmInitialVrf, g1_pool1_outBuffer, 1);
                if (curOffset>=5937) {
                    curOffset-=2464;
                } else {
                    curOffset+=224;
                }
                if (nextOffset>=5937) {
                    nextOffset-=2464;
                } else {
                    nextOffset+=224;
                }
                g1_pool1_outBuffer += 55;
            }
            g1_pool1_in = curOffset;
            g2_res2a_branch2a_available+=verticalRows*55;
            g1_pool1_outBuffer=g1_pool1_accumulators+1320;
            g1_pool1_iterationsLeft-=verticalRows*55;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g1_pool1_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }

        /* Start of group 2 */

        /* Check there is enough data (emulator only) */
        Emulator_HEX_Assert(g2_res2a_branch2a_available <= 605, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        if (g2_res2a_branch2a_available > 0){

            /* Start of layer 1 in group 2 (res2a_branch1) */
            for(int outRow=0;outRow<2;outRow++) {
                g2_res2a_branch2a_inIterator = g2_res2a_branch2a_in;
                vRead2D(bs, ISA_Mem_MvmInitialVrf, g2_res2a_branch2a_inIterator, 1, g2_res2a_branch2a_available, 1);
                mv_mul(bs, mrf_start+3+outRow*1);
                vv_mul(bs, 1+outRow); /* includes: bn2a_branch1: scale, vv_mul, scale2a_branch1: scale */
                vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 1+outRow, 0); /* includes: bn2a_branch1: bias, vv_mul, scale2a_branch1: scale, vv_add, scale2a_branch1: bias */
                v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res2a_branch1_outIterations*2+outRow,2);
            }

            /* Start of layer 0 in group 2 (res2a_branch2a) */
            g2_res2a_branch2a_inIterator = g2_res2a_branch2a_in;
            vRead2D(bs, ISA_Mem_MvmInitialVrf, g2_res2a_branch2a_inIterator, 1, g2_res2a_branch2a_available, 1);
            mv_mul(bs, mrf_start+5);
            vv_mul(bs, 3); /* includes: bn2a_branch2a: scale, vv_mul, scale2a_branch2a: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 3, 0); /* includes: bn2a_branch2a: bias, vv_mul, scale2a_branch2a: scale, vv_add, scale2a_branch2a: bias */
            v_relu(bs); /* includes: res2a_branch2a_relu: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res2a_branch1_outIterations,1);
            res2a_branch1_outIterations += g2_res2a_branch2a_available;
            g2_res2a_branch2a_iterationsLeft-=g2_res2a_branch2a_available;
            g2_res2a_branch2a_available=0;
            /* Check there is enough data (emulator only) */
            Emulator_HEX_Assert(g2_res2a_branch2a_iterationsLeft >= 0, NIOS_HEX_CNN_AUTOGEN_LOOP_ITERATION_ERROR);
        }
    }
}

void res2aBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2a_branch2b(d=64,h=55,w=55)=Conv(res2a_branch2a(d=64,h=55,w=55),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn2a_branch2b */
    /*      absorbed scale2a_branch2b */
    /*      absorbed res2a_branch2b_relu */
    ISA_ExtAddress res2a_branch2a_inIndex,res2a_branch2b_outOffset;
    /* res2a_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch2b_scale__vv_mul__scale2a_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch2b_bias__vv_mul__scale2a_branch2b_scale__vv_add__scale2a_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res2a_branch2a_inIndex=0;
    res2a_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res2a_branch2a_inIndex, 55, 55, 1, 3, 1, 1);
    mv_mul(bs, mrf_start+6);
    vv_mul(bs, 0+outChainOffset); /* includes: bn2a_branch2b: scale, vv_mul, scale2a_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2a_branch2b: bias, vv_mul, scale2a_branch2b: scale, vv_add, scale2a_branch2b: bias */
    v_relu(bs); /* includes: res2a_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6274+res2a_branch2b_outOffset+outChainOffset,1);
}

void res2aBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2a_branch2c(d=256,h=55,w=55)=Conv(res2a_branch1(d=64,h=55,w=55),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn2a_branch2c */
    /*      absorbed scale2a_branch2c */
    /*      absorbed res2a */
    /*      absorbed res2a_relu */
    ISA_ExtAddress res2a_branch2b_inIndex,res2a_branch2c_outOffset;
    /* res2a_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch2c_scale__vv_mul__scale2a_branch2c_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2a_branch2c_bias__vv_mul__scale2a_branch2c_scale__vv_add__scale2a_branch2c_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res2a_branch1_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    res2a_branch2b_inIndex=6274;
    res2a_branch2c_outOffset=0;
    res2a_branch1_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res2a_branch2b_inIndex, 1, 3025, 1);
        mv_mul(bs, mrf_start+15+(outRow*1));
        vv_mul(bs, 0+outChainOffset); /* includes: bn2a_branch2c: scale, vv_mul, scale2a_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2a_branch2c: bias, vv_mul, scale2a_branch2c: scale, vv_add, scale2a_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res2a_branch1_iterator+outChainOffset, 2); /* includes: res2a_branch1 */
        v_relu(bs); /* includes: res2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 6050+res2a_branch2c_outOffset+outChainOffset,2);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res2a_branch2c_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res2bBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2b_branch2a(d=64,h=55,w=55)=Conv(res2a_branch2c(d=256,h=55,w=55),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn2b_branch2a */
    /*      absorbed scale2b_branch2a */
    /*      absorbed res2b_branch2a_relu */
    ISA_ExtAddress res2a_branch2c_inIndex,res2b_branch2a_outOffset;
    /* res2b_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2b_branch2a_scale__vv_mul__scale2b_branch2a_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2b_branch2a_bias__vv_mul__scale2b_branch2a_scale__vv_add__scale2b_branch2a_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res2a_branch2c_inIndex=0;
    res2b_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    /* strided IVRF access mode on */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, res2a_branch2c_inIndex, 2, 3025, 2);
    mv_mul(bs, mrf_start+17);
    vv_mul(bs, 0+outChainOffset); /* includes: bn2b_branch2a: scale, vv_mul, scale2b_branch2a: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2b_branch2a: bias, vv_mul, scale2b_branch2a: scale, vv_add, scale2b_branch2a: bias */
    v_relu(bs); /* includes: res2b_branch2a_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6274+res2b_branch2a_outOffset+outChainOffset,1);
}

void res2bBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2b_branch2b(d=64,h=55,w=55)=Conv(res2b_branch2a(d=64,h=55,w=55),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn2b_branch2b */
    /*      absorbed scale2b_branch2b */
    /*      absorbed res2b_branch2b_relu */
    ISA_ExtAddress res2b_branch2a_inIndex,res2b_branch2b_outOffset;
    /* res2b_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2b_branch2b_scale__vv_mul__scale2b_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2b_branch2b_bias__vv_mul__scale2b_branch2b_scale__vv_add__scale2b_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res2b_branch2a_inIndex=6274;
    res2b_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res2b_branch2a_inIndex, 55, 55, 1, 3, 1, 1);
    mv_mul(bs, mrf_start+19);
    vv_mul(bs, 0+outChainOffset); /* includes: bn2b_branch2b: scale, vv_mul, scale2b_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2b_branch2b: bias, vv_mul, scale2b_branch2b: scale, vv_add, scale2b_branch2b: bias */
    v_relu(bs); /* includes: res2b_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res2b_branch2b_outOffset+outChainOffset,1);
}

void res2bBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2b_branch2c(d=256,h=55,w=55)=Conv(res2a_branch2c(d=64,h=55,w=55),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn2b_branch2c */
    /*      absorbed scale2b_branch2c */
    /*      absorbed res2b */
    /*      absorbed res2b_relu */
    ISA_ExtAddress res2b_branch2b_inIndex,res2b_branch2c_outOffset;
    /* res2b_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2b_branch2c_scale__vv_mul__scale2b_branch2c_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2b_branch2c_bias__vv_mul__scale2b_branch2c_scale__vv_add__scale2b_branch2c_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res2a_branch2c_iterator = 6050;
    ISA_ExtAddress outChainOffset = 0;
    res2b_branch2b_inIndex=0;
    res2b_branch2c_outOffset=0;
    res2a_branch2c_iterator=6050;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res2b_branch2b_inIndex, 1, 3025, 1);
        mv_mul(bs, mrf_start+28+(outRow*1));
        vv_mul(bs, 0+outChainOffset); /* includes: bn2b_branch2c: scale, vv_mul, scale2b_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2b_branch2c: bias, vv_mul, scale2b_branch2c: scale, vv_add, scale2b_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res2a_branch2c_iterator+outChainOffset, 2); /* includes: res2a */
        v_relu(bs); /* includes: res2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res2b_branch2c_outOffset+outChainOffset,2);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 3249+res2b_branch2c_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res2cBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2c_branch2a(d=64,h=55,w=55)=Conv(res2b_branch2c(d=256,h=55,w=55),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn2c_branch2a */
    /*      absorbed scale2c_branch2a */
    /*      absorbed res2c_branch2a_relu */
    ISA_ExtAddress res2b_branch2c_inIndex,res2c_branch2a_outOffset;
    /* res2c_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2c_branch2a_scale__vv_mul__scale2c_branch2a_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2c_branch2a_bias__vv_mul__scale2c_branch2a_scale__vv_add__scale2c_branch2a_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res2b_branch2c_inIndex=3249;
    res2c_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    /* strided IVRF access mode on */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, res2b_branch2c_inIndex, 2, 3025, 2);
    mv_mul(bs, mrf_start+30);
    vv_mul(bs, 0+outChainOffset); /* includes: bn2c_branch2a: scale, vv_mul, scale2c_branch2a: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2c_branch2a: bias, vv_mul, scale2c_branch2a: scale, vv_add, scale2c_branch2a: bias */
    v_relu(bs); /* includes: res2c_branch2a_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res2c_branch2a_outOffset+outChainOffset,1);
}

void res2cBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2c_branch2b(d=64,h=28,w=28)=Conv(res2c_branch2a(d=64,h=55,w=55),k_h=3,k_w=3,s_h=2,s_w=2,p_h=1,p_w=1) */
    /*      absorbed bn2c_branch2b */
    /*      absorbed scale2c_branch2b */
    /*      absorbed res2c_branch2b_relu */
    ISA_ExtAddress res2c_branch2a_inIndex,res2c_branch2b_outOffset;
    /* res2c_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2c_branch2b_scale__vv_mul__scale2c_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2c_branch2b_bias__vv_mul__scale2c_branch2b_scale__vv_add__scale2c_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res2c_branch2a_inIndex=0;
    res2c_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 2 rows and (2 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res2c_branch2a_inIndex, 55, 55, 1, 3, 1, 2);
    mv_mul(bs, mrf_start+32);
    vv_mul(bs, 0+outChainOffset); /* includes: bn2c_branch2b: scale, vv_mul, scale2c_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2c_branch2b: bias, vv_mul, scale2c_branch2b: scale, vv_add, scale2c_branch2b: bias */
    v_relu(bs); /* includes: res2c_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515+res2c_branch2b_outOffset+outChainOffset,1);
    res2c_branch2a_inIndex += 54; /* skip 1 rows due to stride, adjusted for 55/56 discrepency */
}

void res2cBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res2c_branch2c(d=256,h=28,w=28)=Conv(res2b_branch2c(d=64,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn2c_branch2c */
    /*      absorbed scale2c_branch2c */
    /*      absorbed res2c */
    /*      absorbed res2c_relu */
    ISA_ExtAddress res2c_branch2b_inIndex,res2c_branch2c_outOffset;
    /* res2c_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn2c_branch2c_scale__vv_mul__scale2c_branch2c_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn2c_branch2c_bias__vv_mul__scale2c_branch2c_scale__vv_add__scale2c_branch2c_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res2b_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    res2c_branch2b_inIndex=8515;
    res2c_branch2c_outOffset=0;
    res2b_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    for (int rowIterator=0; rowIterator<28; rowIterator++) {
        outChainOffset=0;
        for(int outRow=0;outRow<2;outRow++) {
            /* strided IVRF access mode on */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, res2c_branch2b_inIndex, 1, 28, 1);
            mv_mul(bs, mrf_start+41+(outRow*1));
            vv_mul(bs, 0+outChainOffset); /* includes: bn2c_branch2c: scale, vv_mul, scale2c_branch2c: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn2c_branch2c: bias, vv_mul, scale2c_branch2c: scale, vv_add, scale2c_branch2c: bias */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res2b_branch2c_iterator+outChainOffset, 4); /* includes: res2b */
            v_relu(bs); /* includes: res2c_relu: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res2c_branch2c_outOffset+outChainOffset,2);
            outChainOffset++;
        }
        res2c_branch2b_inIndex += 28;
        res2c_branch2c_outOffset += 56;
        res2b_branch2c_iterator += 220;
    }
}

void res3aBranch1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3a_branch1(d=512,h=28,w=28)=Conv(res2c_branch2c(d=256,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3a_branch1 */
    /*      absorbed scale3a_branch1 */
    ISA_ExtAddress res2c_branch2c_inIndex,res3a_branch1_outOffset;
    /* res3a_branch1_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch1_scale__vv_mul__scale3a_branch1_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch1_bias__vv_mul__scale3a_branch1_scale__vv_add__scale3a_branch1_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res2c_branch2c_inIndex=0;
    res3a_branch1_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res2c_branch2c_inIndex, 2, 784, 2);
        mv_mul(bs, mrf_start+43+(outRow*2));
        vv_mul(bs, 0+outChainOffset); /* includes: bn3a_branch1: scale, vv_mul, scale3a_branch1: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3a_branch1: bias, vv_mul, scale3a_branch1: scale, vv_add, scale3a_branch1: bias */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 8964+res3a_branch1_outOffset+outChainOffset,4);
        outChainOffset++;
    }
}

void res3aBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3a_branch2a(d=128,h=28,w=28)=Conv(res2c_branch2c(d=256,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3a_branch2a */
    /*      absorbed scale3a_branch2a */
    /*      absorbed res3a_branch2a_relu */
    ISA_ExtAddress res2c_branch2c_inIndex,res3a_branch2a_outOffset;
    /* res3a_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch2a_scale__vv_mul__scale3a_branch2a_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch2a_bias__vv_mul__scale3a_branch2a_scale__vv_add__scale3a_branch2a_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 55 entries starting at res3a_branch2c */
        moveFilterCount128(bs, ISA_Mem_Dram, res3a_branch2c_MRF+0*55, ISA_Mem_MatrixRf, mrf_next, 1, 55);
    }
    res2c_branch2c_inIndex=0;
    res3a_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    /* strided IVRF access mode on */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, res2c_branch2c_inIndex, 2, 784, 2);
    mv_mul(bs, mrf_start+51);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3a_branch2a: scale, vv_mul, scale3a_branch2a: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3a_branch2a: bias, vv_mul, scale3a_branch2a: scale, vv_add, scale3a_branch2a: bias */
    v_relu(bs); /* includes: res3a_branch2a_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515+res3a_branch2a_outOffset+outChainOffset,1);
}

void res3aBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3a_branch2b(d=128,h=28,w=28)=Conv(res3a_branch2a(d=128,h=28,w=28),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn3a_branch2b */
    /*      absorbed scale3a_branch2b */
    /*      absorbed res3a_branch2b_relu */
    ISA_ExtAddress res3a_branch2a_inIndex,res3a_branch2b_outOffset;
    /* res3a_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch2b_scale__vv_mul__scale3a_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch2b_bias__vv_mul__scale3a_branch2b_scale__vv_add__scale3a_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3a_branch2a_inIndex=8515;
    res3a_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res3a_branch2a_inIndex, 28, 28, 1, 3, 1, 1);
    mv_mul(bs, mrf_start+53);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3a_branch2b: scale, vv_mul, scale3a_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3a_branch2b: bias, vv_mul, scale3a_branch2b: scale, vv_add, scale3a_branch2b: bias */
    v_relu(bs); /* includes: res3a_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res3a_branch2b_outOffset+outChainOffset,1);
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res3aBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3a_branch2c(d=512,h=28,w=28)=Conv(res3a_branch1(d=128,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3a_branch2c */
    /*      absorbed scale3a_branch2c */
    /*      absorbed res3a */
    /*      absorbed res3a_relu */
    ISA_ExtAddress res3a_branch2b_inIndex,res3a_branch2c_outOffset;
    /* res3a_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch2c_scale__vv_mul__scale3a_branch2c_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3a_branch2c_bias__vv_mul__scale3a_branch2c_scale__vv_add__scale3a_branch2c_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res3a_branch1_iterator = 8964;
    ISA_ExtAddress outChainOffset = 0;
    res3a_branch2b_inIndex=0;
    res3a_branch2c_outOffset=0;
    res3a_branch1_iterator=8964;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res3a_branch2b_inIndex, 1, 784, 1);
        mv_mul(bs, mrf_start+0+(outRow*1));
        vv_mul(bs, 0+outChainOffset); /* includes: bn3a_branch2c: scale, vv_mul, scale3a_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3a_branch2c: bias, vv_mul, scale3a_branch2c: scale, vv_add, scale3a_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res3a_branch1_iterator+outChainOffset, 4); /* includes: res3a_branch1 */
        v_relu(bs); /* includes: res3a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res3a_branch2c_outOffset+outChainOffset,4);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6163+res3a_branch2c_outOffset+outChainOffset,4);
        outChainOffset++;
    }
}

void res3bBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3b_branch2a(d=128,h=28,w=28)=Conv(res3a_branch2c(d=512,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3b_branch2a */
    /*      absorbed scale3b_branch2a */
    /*      absorbed res3b_branch2a_relu */
    ISA_ExtAddress res3a_branch2c_inIndex,res3b_branch2a_outOffset;
    /* res3b_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3b_branch2a_scale__vv_mul__scale3b_branch2a_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3b_branch2a_bias__vv_mul__scale3b_branch2a_scale__vv_add__scale3b_branch2a_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3a_branch2c_inIndex=6163;
    res3b_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    /* strided IVRF access mode on */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, res3a_branch2c_inIndex, 4, 784, 4);
    mv_mul(bs, mrf_start+4);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3b_branch2a: scale, vv_mul, scale3b_branch2a: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3b_branch2a: bias, vv_mul, scale3b_branch2a: scale, vv_add, scale3b_branch2a: bias */
    v_relu(bs); /* includes: res3b_branch2a_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res3b_branch2a_outOffset+outChainOffset,1);
}

void res3bBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3b_branch2b(d=128,h=28,w=28)=Conv(res3b_branch2a(d=128,h=28,w=28),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn3b_branch2b */
    /*      absorbed scale3b_branch2b */
    /*      absorbed res3b_branch2b_relu */
    ISA_ExtAddress res3b_branch2a_inIndex,res3b_branch2b_outOffset;
    /* res3b_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3b_branch2b_scale__vv_mul__scale3b_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3b_branch2b_bias__vv_mul__scale3b_branch2b_scale__vv_add__scale3b_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3b_branch2a_inIndex=0;
    res3b_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res3b_branch2a_inIndex, 28, 28, 1, 3, 1, 1);
    mv_mul(bs, mrf_start+8);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3b_branch2b: scale, vv_mul, scale3b_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3b_branch2b: bias, vv_mul, scale3b_branch2b: scale, vv_add, scale3b_branch2b: bias */
    v_relu(bs); /* includes: res3b_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515+res3b_branch2b_outOffset+outChainOffset,1);
}

void res3bBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3b_branch2c(d=512,h=28,w=28)=Conv(res3a_branch2c(d=128,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3b_branch2c */
    /*      absorbed scale3b_branch2c */
    /*      absorbed res3b */
    /*      absorbed res3b_relu */
    ISA_ExtAddress res3b_branch2b_inIndex,res3b_branch2c_outOffset;
    /* res3b_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3b_branch2c_scale__vv_mul__scale3b_branch2c_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3b_branch2c_bias__vv_mul__scale3b_branch2c_scale__vv_add__scale3b_branch2c_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res3a_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    res3b_branch2b_inIndex=8515;
    res3b_branch2c_outOffset=0;
    res3a_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res3b_branch2b_inIndex, 1, 784, 1);
        mv_mul(bs, mrf_start+17+(outRow*1));
        vv_mul(bs, 0+outChainOffset); /* includes: bn3b_branch2c: scale, vv_mul, scale3b_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3b_branch2c: bias, vv_mul, scale3b_branch2c: scale, vv_add, scale3b_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res3a_branch2c_iterator+outChainOffset, 4); /* includes: res3a */
        v_relu(bs); /* includes: res3b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 8964+res3b_branch2c_outOffset+outChainOffset,4);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res3b_branch2c_outOffset+outChainOffset,4);
        outChainOffset++;
    }
}

void res3cBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3c_branch2a(d=128,h=28,w=28)=Conv(res3b_branch2c(d=512,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3c_branch2a */
    /*      absorbed scale3c_branch2a */
    /*      absorbed res3c_branch2a_relu */
    ISA_ExtAddress res3b_branch2c_inIndex,res3c_branch2a_outOffset;
    /* res3c_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3c_branch2a_scale__vv_mul__scale3c_branch2a_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3c_branch2a_bias__vv_mul__scale3c_branch2a_scale__vv_add__scale3c_branch2a_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3b_branch2c_inIndex=0;
    res3c_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    /* strided IVRF access mode on */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, res3b_branch2c_inIndex, 4, 784, 4);
    mv_mul(bs, mrf_start+21);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3c_branch2a: scale, vv_mul, scale3c_branch2a: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3c_branch2a: bias, vv_mul, scale3c_branch2a: scale, vv_add, scale3c_branch2a: bias */
    v_relu(bs); /* includes: res3c_branch2a_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515+res3c_branch2a_outOffset+outChainOffset,1);
}

void res3cBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3c_branch2b(d=128,h=28,w=28)=Conv(res3c_branch2a(d=128,h=28,w=28),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn3c_branch2b */
    /*      absorbed scale3c_branch2b */
    /*      absorbed res3c_branch2b_relu */
    ISA_ExtAddress res3c_branch2a_inIndex,res3c_branch2b_outOffset;
    /* res3c_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3c_branch2b_scale__vv_mul__scale3c_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3c_branch2b_bias__vv_mul__scale3c_branch2b_scale__vv_add__scale3c_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3c_branch2a_inIndex=8515;
    res3c_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res3c_branch2a_inIndex, 28, 28, 1, 3, 1, 1);
    mv_mul(bs, mrf_start+25);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3c_branch2b: scale, vv_mul, scale3c_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3c_branch2b: bias, vv_mul, scale3c_branch2b: scale, vv_add, scale3c_branch2b: bias */
    v_relu(bs); /* includes: res3c_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res3c_branch2b_outOffset+outChainOffset,1);
}

void res3cBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3c_branch2c(d=512,h=28,w=28)=Conv(res3b_branch2c(d=128,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3c_branch2c */
    /*      absorbed scale3c_branch2c */
    /*      absorbed res3c */
    /*      absorbed res3c_relu */
    ISA_ExtAddress res3c_branch2b_inIndex,res3c_branch2c_outOffset;
    /* res3c_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3c_branch2c_scale__vv_mul__scale3c_branch2c_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3c_branch2c_bias__vv_mul__scale3c_branch2c_scale__vv_add__scale3c_branch2c_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res3b_branch2c_iterator = 8964;
    ISA_ExtAddress outChainOffset = 0;
    res3c_branch2b_inIndex=0;
    res3c_branch2c_outOffset=0;
    res3b_branch2c_iterator=8964;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res3c_branch2b_inIndex, 1, 784, 1);
        mv_mul(bs, mrf_start+34+(outRow*1));
        vv_mul(bs, 0+outChainOffset); /* includes: bn3c_branch2c: scale, vv_mul, scale3c_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3c_branch2c: bias, vv_mul, scale3c_branch2c: scale, vv_add, scale3c_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res3b_branch2c_iterator+outChainOffset, 4); /* includes: res3b */
        v_relu(bs); /* includes: res3c_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res3c_branch2c_outOffset+outChainOffset,4);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 6163+res3c_branch2c_outOffset+outChainOffset,4);
        outChainOffset++;
    }
}

void res3dBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3d_branch2a(d=128,h=28,w=28)=Conv(res3c_branch2c(d=512,h=28,w=28),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3d_branch2a */
    /*      absorbed scale3d_branch2a */
    /*      absorbed res3d_branch2a_relu */
    ISA_ExtAddress res3c_branch2c_inIndex,res3d_branch2a_outOffset;
    /* res3d_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3d_branch2a_scale__vv_mul__scale3d_branch2a_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3d_branch2a_bias__vv_mul__scale3d_branch2a_scale__vv_add__scale3d_branch2a_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3c_branch2c_inIndex=6163;
    res3d_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    /* strided IVRF access mode on */
    vRead2D(bs, ISA_Mem_MvmInitialVrf, res3c_branch2c_inIndex, 4, 784, 4);
    mv_mul(bs, mrf_start+38);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3d_branch2a: scale, vv_mul, scale3d_branch2a: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3d_branch2a: bias, vv_mul, scale3d_branch2a: scale, vv_add, scale3d_branch2a: bias */
    v_relu(bs); /* includes: res3d_branch2a_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res3d_branch2a_outOffset+outChainOffset,1);
}

void res3dBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3d_branch2b(d=128,h=14,w=14)=Conv(res3d_branch2a(d=128,h=28,w=28),k_h=3,k_w=3,s_h=2,s_w=2,p_h=1,p_w=1) */
    /*      absorbed bn3d_branch2b */
    /*      absorbed scale3d_branch2b */
    /*      absorbed res3d_branch2b_relu */
    ISA_ExtAddress res3d_branch2a_inIndex,res3d_branch2b_outOffset;
    /* res3d_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3d_branch2b_scale__vv_mul__scale3d_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3d_branch2b_bias__vv_mul__scale3d_branch2b_scale__vv_add__scale3d_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 40 entries starting at res4a_branch1 */
        moveFilterCount128(bs, ISA_Mem_Dram, res4a_branch1_MRF+0*40, ISA_Mem_MatrixRf, mrf_next, 1, 40);
    }
    res3d_branch2a_inIndex=0;
    res3d_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 2 rows and (2 columns * 1 registers/data element) */
    /* strided IVRF access mode on */
    vRead3D(bs, ISA_Mem_MvmInitialVrf, res3d_branch2a_inIndex, 28, 28, 1, 3, 1, 2);
    mv_mul(bs, mrf_start+42);
    vv_mul(bs, 0+outChainOffset); /* includes: bn3d_branch2b: scale, vv_mul, scale3d_branch2b: scale */
    vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3d_branch2b: bias, vv_mul, scale3d_branch2b: scale, vv_add, scale3d_branch2b: bias */
    v_relu(bs); /* includes: res3d_branch2b_relu: v_relu */
    v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9103+res3d_branch2b_outOffset+outChainOffset,1);
    res3d_branch2a_inIndex += 28; /* skip 1 rows due to stride, adjusted for 28/28 discrepency */
}

void res3dBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res3d_branch2c(d=512,h=14,w=14)=Conv(res3c_branch2c(d=128,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn3d_branch2c */
    /*      absorbed scale3d_branch2c */
    /*      absorbed res3d */
    /*      absorbed res3d_relu */
    ISA_ExtAddress res3d_branch2b_inIndex,res3d_branch2c_outOffset;
    /* res3d_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn3d_branch2c_scale__vv_mul__scale3d_branch2c_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn3d_branch2c_bias__vv_mul__scale3d_branch2c_scale__vv_add__scale3d_branch2c_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res3c_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    res3d_branch2b_inIndex=9103;
    res3d_branch2c_outOffset=0;
    res3c_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 1 registers/data element) */
    for (int rowIterator=0; rowIterator<14; rowIterator++) {
        outChainOffset=0;
        for(int outRow=0;outRow<4;outRow++) {
            /* strided IVRF access mode on */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, res3d_branch2b_inIndex, 1, 14, 1);
            mv_mul(bs, mrf_start+51+(outRow*1));
            vv_mul(bs, 0+outChainOffset); /* includes: bn3d_branch2c: scale, vv_mul, scale3d_branch2c: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn3d_branch2c: bias, vv_mul, scale3d_branch2c: scale, vv_add, scale3d_branch2c: bias */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res3c_branch2c_iterator+outChainOffset, 8); /* includes: res3c */
            v_relu(bs); /* includes: res3d_relu: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res3d_branch2c_outOffset+outChainOffset,4);
            outChainOffset++;
        }
        res3d_branch2b_inIndex += 14;
        res3d_branch2c_outOffset += 56;
        res3c_branch2c_iterator += 224;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4aBranch1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4a_branch1(d=1024,h=14,w=14)=Conv(res3d_branch2c(d=512,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4a_branch1 */
    /*      absorbed scale4a_branch1 */
    ISA_ExtAddress res3d_branch2c_inIndex,res4a_branch1_outOffset;
    /* res4a_branch1_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch1_scale__vv_mul__scale4a_branch1_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch1_bias__vv_mul__scale4a_branch1_scale__vv_add__scale4a_branch1_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 52 entries starting at res4a_branch2b */
        moveFilterCount128(bs, ISA_Mem_Dram, res4a_branch2b_MRF+0*52, ISA_Mem_MatrixRf, mrf_next, 1, 52);
    }
    res3d_branch2c_inIndex=0;
    res4a_branch1_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res3d_branch2c_inIndex, 4, 196, 4);
        mv_mul(bs, mrf_start+0+(outRow*4));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4a_branch1: scale, vv_mul, scale4a_branch1: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4a_branch1: bias, vv_mul, scale4a_branch1: scale, vv_add, scale4a_branch1: bias */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 10532+res4a_branch1_outOffset+outChainOffset,8);
        outChainOffset++;
    }
}

void res4aBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4a_branch2a(d=256,h=14,w=14)=Conv(res3d_branch2c(d=512,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4a_branch2a */
    /*      absorbed scale4a_branch2a */
    /*      absorbed res4a_branch2a_relu */
    ISA_ExtAddress res3d_branch2c_inIndex,res4a_branch2a_outOffset;
    /* res4a_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch2a_scale__vv_mul__scale4a_branch2a_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch2a_bias__vv_mul__scale4a_branch2a_scale__vv_add__scale4a_branch2a_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res3d_branch2c_inIndex=0;
    res4a_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res3d_branch2c_inIndex, 4, 196, 4);
        mv_mul(bs, mrf_start+32+(outRow*4));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4a_branch2a: scale, vv_mul, scale4a_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4a_branch2a: bias, vv_mul, scale4a_branch2a: scale, vv_add, scale4a_branch2a: bias */
        v_relu(bs); /* includes: res4a_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8907+res4a_branch2a_outOffset+outChainOffset,2);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4aBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4a_branch2b(d=256,h=14,w=14)=Conv(res4a_branch2a(d=256,h=14,w=14),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn4a_branch2b */
    /*      absorbed scale4a_branch2b */
    /*      absorbed res4a_branch2b_relu */
    ISA_ExtAddress res4a_branch2a_inIndex,res4a_branch2b_outOffset;
    /* res4a_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch2b_scale__vv_mul__scale4a_branch2b_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch2b_bias__vv_mul__scale4a_branch2b_scale__vv_add__scale4a_branch2b_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 52 entries starting at res4b_branch2a */
        moveFilterCount128(bs, ISA_Mem_Dram, res4b_branch2a_MRF+0*52, ISA_Mem_MatrixRf, mrf_next, 1, 52);
    }
    res4a_branch2a_inIndex=8907;
    res4a_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res4a_branch2a_inIndex, 14, 14, 2, 3, 1, 1);
        mv_mul(bs, mrf_start+0+(outRow*18));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4a_branch2b: scale, vv_mul, scale4a_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4a_branch2b: bias, vv_mul, scale4a_branch2b: scale, vv_add, scale4a_branch2b: bias */
        v_relu(bs); /* includes: res4a_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4a_branch2b_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res4aBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4a_branch2c(d=1024,h=14,w=14)=Conv(res4a_branch1(d=256,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4a_branch2c */
    /*      absorbed scale4a_branch2c */
    /*      absorbed res4a */
    /*      absorbed res4a_relu */
    ISA_ExtAddress res4a_branch2b_inIndex,res4a_branch2c_outOffset;
    /* res4a_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch2c_scale__vv_mul__scale4a_branch2c_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4a_branch2c_bias__vv_mul__scale4a_branch2c_scale__vv_add__scale4a_branch2c_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res4a_branch1_iterator = 10532;
    ISA_ExtAddress outChainOffset = 0;
    res4a_branch2b_inIndex=0;
    res4a_branch2c_outOffset=0;
    res4a_branch1_iterator=10532;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4a_branch2b_inIndex, 2, 196, 2);
        mv_mul(bs, mrf_start+36+(outRow*2));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4a_branch2c: scale, vv_mul, scale4a_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4a_branch2c: bias, vv_mul, scale4a_branch2c: scale, vv_add, scale4a_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res4a_branch1_iterator+outChainOffset, 8); /* includes: res4a_branch1 */
        v_relu(bs); /* includes: res4a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res4a_branch2c_outOffset+outChainOffset,8);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7731+res4a_branch2c_outOffset+outChainOffset,8);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4bBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4b_branch2a(d=256,h=14,w=14)=Conv(res4a_branch2c(d=1024,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4b_branch2a */
    /*      absorbed scale4b_branch2a */
    /*      absorbed res4b_branch2a_relu */
    ISA_ExtAddress res4a_branch2c_inIndex,res4b_branch2a_outOffset;
    /* res4b_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4b_branch2a_scale__vv_mul__scale4b_branch2a_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4b_branch2a_bias__vv_mul__scale4b_branch2a_scale__vv_add__scale4b_branch2a_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 32 entries starting at res4b_branch2c */
        moveFilterCount128(bs, ISA_Mem_Dram, res4b_branch2c_MRF+0*32, ISA_Mem_MatrixRf, mrf_next, 1, 32);
    }
    res4a_branch2c_inIndex=7731;
    res4b_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4a_branch2c_inIndex, 8, 196, 8);
        mv_mul(bs, mrf_start+0+(outRow*8));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4b_branch2a: scale, vv_mul, scale4b_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4b_branch2a: bias, vv_mul, scale4b_branch2a: scale, vv_add, scale4b_branch2a: bias */
        v_relu(bs); /* includes: res4b_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4b_branch2a_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res4bBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4b_branch2b(d=256,h=14,w=14)=Conv(res4b_branch2a(d=256,h=14,w=14),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn4b_branch2b */
    /*      absorbed scale4b_branch2b */
    /*      absorbed res4b_branch2b_relu */
    ISA_ExtAddress res4b_branch2a_inIndex,res4b_branch2b_outOffset;
    /* res4b_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4b_branch2b_scale__vv_mul__scale4b_branch2b_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4b_branch2b_bias__vv_mul__scale4b_branch2b_scale__vv_add__scale4b_branch2b_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res4b_branch2a_inIndex=0;
    res4b_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res4b_branch2a_inIndex, 14, 14, 2, 3, 1, 1);
        mv_mul(bs, mrf_start+16+(outRow*18));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4b_branch2b: scale, vv_mul, scale4b_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4b_branch2b: bias, vv_mul, scale4b_branch2b: scale, vv_add, scale4b_branch2b: bias */
        v_relu(bs); /* includes: res4b_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8907+res4b_branch2b_outOffset+outChainOffset,2);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4bBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4b_branch2c(d=1024,h=14,w=14)=Conv(res4a_branch2c(d=256,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4b_branch2c */
    /*      absorbed scale4b_branch2c */
    /*      absorbed res4b */
    /*      absorbed res4b_relu */
    ISA_ExtAddress res4b_branch2b_inIndex,res4b_branch2c_outOffset;
    /* res4b_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4b_branch2c_scale__vv_mul__scale4b_branch2c_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4b_branch2c_bias__vv_mul__scale4b_branch2c_scale__vv_add__scale4b_branch2c_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res4a_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 52 entries starting at res4c_branch2b */
        moveFilterCount128(bs, ISA_Mem_Dram, res4c_branch2b_MRF+0*52, ISA_Mem_MatrixRf, mrf_next, 1, 52);
    }
    res4b_branch2b_inIndex=8907;
    res4b_branch2c_outOffset=0;
    res4a_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4b_branch2b_inIndex, 2, 196, 2);
        mv_mul(bs, mrf_start+0+(outRow*2));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4b_branch2c: scale, vv_mul, scale4b_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4b_branch2c: bias, vv_mul, scale4b_branch2c: scale, vv_add, scale4b_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res4a_branch2c_iterator+outChainOffset, 8); /* includes: res4a */
        v_relu(bs); /* includes: res4b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 10532+res4b_branch2c_outOffset+outChainOffset,8);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4b_branch2c_outOffset+outChainOffset,8);
        outChainOffset++;
    }
}

void res4cBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4c_branch2a(d=256,h=14,w=14)=Conv(res4b_branch2c(d=1024,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4c_branch2a */
    /*      absorbed scale4c_branch2a */
    /*      absorbed res4c_branch2a_relu */
    ISA_ExtAddress res4b_branch2c_inIndex,res4c_branch2a_outOffset;
    /* res4c_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4c_branch2a_scale__vv_mul__scale4c_branch2a_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4c_branch2a_bias__vv_mul__scale4c_branch2a_scale__vv_add__scale4c_branch2a_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res4b_branch2c_inIndex=0;
    res4c_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4b_branch2c_inIndex, 8, 196, 8);
        mv_mul(bs, mrf_start+16+(outRow*8));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4c_branch2a: scale, vv_mul, scale4c_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4c_branch2a: bias, vv_mul, scale4c_branch2a: scale, vv_add, scale4c_branch2a: bias */
        v_relu(bs); /* includes: res4c_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8907+res4c_branch2a_outOffset+outChainOffset,2);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4cBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4c_branch2b(d=256,h=14,w=14)=Conv(res4c_branch2a(d=256,h=14,w=14),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn4c_branch2b */
    /*      absorbed scale4c_branch2b */
    /*      absorbed res4c_branch2b_relu */
    ISA_ExtAddress res4c_branch2a_inIndex,res4c_branch2b_outOffset;
    /* res4c_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4c_branch2b_scale__vv_mul__scale4c_branch2b_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4c_branch2b_bias__vv_mul__scale4c_branch2b_scale__vv_add__scale4c_branch2b_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 52 entries starting at res4d_branch2a */
        moveFilterCount128(bs, ISA_Mem_Dram, res4d_branch2a_MRF+0*52, ISA_Mem_MatrixRf, mrf_next, 1, 52);
    }
    res4c_branch2a_inIndex=8907;
    res4c_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res4c_branch2a_inIndex, 14, 14, 2, 3, 1, 1);
        mv_mul(bs, mrf_start+0+(outRow*18));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4c_branch2b: scale, vv_mul, scale4c_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4c_branch2b: bias, vv_mul, scale4c_branch2b: scale, vv_add, scale4c_branch2b: bias */
        v_relu(bs); /* includes: res4c_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4c_branch2b_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res4cBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4c_branch2c(d=1024,h=14,w=14)=Conv(res4b_branch2c(d=256,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4c_branch2c */
    /*      absorbed scale4c_branch2c */
    /*      absorbed res4c */
    /*      absorbed res4c_relu */
    ISA_ExtAddress res4c_branch2b_inIndex,res4c_branch2c_outOffset;
    /* res4c_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4c_branch2c_scale__vv_mul__scale4c_branch2c_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4c_branch2c_bias__vv_mul__scale4c_branch2c_scale__vv_add__scale4c_branch2c_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res4b_branch2c_iterator = 10532;
    ISA_ExtAddress outChainOffset = 0;
    res4c_branch2b_inIndex=0;
    res4c_branch2c_outOffset=0;
    res4b_branch2c_iterator=10532;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4c_branch2b_inIndex, 2, 196, 2);
        mv_mul(bs, mrf_start+36+(outRow*2));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4c_branch2c: scale, vv_mul, scale4c_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4c_branch2c: bias, vv_mul, scale4c_branch2c: scale, vv_add, scale4c_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res4b_branch2c_iterator+outChainOffset, 8); /* includes: res4b */
        v_relu(bs); /* includes: res4c_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res4c_branch2c_outOffset+outChainOffset,8);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7731+res4c_branch2c_outOffset+outChainOffset,8);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4dBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4d_branch2a(d=256,h=14,w=14)=Conv(res4c_branch2c(d=1024,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4d_branch2a */
    /*      absorbed scale4d_branch2a */
    /*      absorbed res4d_branch2a_relu */
    ISA_ExtAddress res4c_branch2c_inIndex,res4d_branch2a_outOffset;
    /* res4d_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4d_branch2a_scale__vv_mul__scale4d_branch2a_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4d_branch2a_bias__vv_mul__scale4d_branch2a_scale__vv_add__scale4d_branch2a_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 32 entries starting at res4d_branch2c */
        moveFilterCount128(bs, ISA_Mem_Dram, res4d_branch2c_MRF+0*32, ISA_Mem_MatrixRf, mrf_next, 1, 32);
    }
    res4c_branch2c_inIndex=7731;
    res4d_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4c_branch2c_inIndex, 8, 196, 8);
        mv_mul(bs, mrf_start+0+(outRow*8));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4d_branch2a: scale, vv_mul, scale4d_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4d_branch2a: bias, vv_mul, scale4d_branch2a: scale, vv_add, scale4d_branch2a: bias */
        v_relu(bs); /* includes: res4d_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4d_branch2a_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res4dBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4d_branch2b(d=256,h=14,w=14)=Conv(res4d_branch2a(d=256,h=14,w=14),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn4d_branch2b */
    /*      absorbed scale4d_branch2b */
    /*      absorbed res4d_branch2b_relu */
    ISA_ExtAddress res4d_branch2a_inIndex,res4d_branch2b_outOffset;
    /* res4d_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4d_branch2b_scale__vv_mul__scale4d_branch2b_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4d_branch2b_bias__vv_mul__scale4d_branch2b_scale__vv_add__scale4d_branch2b_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res4d_branch2a_inIndex=0;
    res4d_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res4d_branch2a_inIndex, 14, 14, 2, 3, 1, 1);
        mv_mul(bs, mrf_start+16+(outRow*18));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4d_branch2b: scale, vv_mul, scale4d_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4d_branch2b: bias, vv_mul, scale4d_branch2b: scale, vv_add, scale4d_branch2b: bias */
        v_relu(bs); /* includes: res4d_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8907+res4d_branch2b_outOffset+outChainOffset,2);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4dBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4d_branch2c(d=1024,h=14,w=14)=Conv(res4c_branch2c(d=256,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4d_branch2c */
    /*      absorbed scale4d_branch2c */
    /*      absorbed res4d */
    /*      absorbed res4d_relu */
    ISA_ExtAddress res4d_branch2b_inIndex,res4d_branch2c_outOffset;
    /* res4d_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4d_branch2c_scale__vv_mul__scale4d_branch2c_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4d_branch2c_bias__vv_mul__scale4d_branch2c_scale__vv_add__scale4d_branch2c_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res4c_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 52 entries starting at res4e_branch2b */
        moveFilterCount128(bs, ISA_Mem_Dram, res4e_branch2b_MRF+0*52, ISA_Mem_MatrixRf, mrf_next, 1, 52);
    }
    res4d_branch2b_inIndex=8907;
    res4d_branch2c_outOffset=0;
    res4c_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4d_branch2b_inIndex, 2, 196, 2);
        mv_mul(bs, mrf_start+0+(outRow*2));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4d_branch2c: scale, vv_mul, scale4d_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4d_branch2c: bias, vv_mul, scale4d_branch2c: scale, vv_add, scale4d_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res4c_branch2c_iterator+outChainOffset, 8); /* includes: res4c */
        v_relu(bs); /* includes: res4d_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 10532+res4d_branch2c_outOffset+outChainOffset,8);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4d_branch2c_outOffset+outChainOffset,8);
        outChainOffset++;
    }
}

void res4eBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4e_branch2a(d=256,h=14,w=14)=Conv(res4d_branch2c(d=1024,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4e_branch2a */
    /*      absorbed scale4e_branch2a */
    /*      absorbed res4e_branch2a_relu */
    ISA_ExtAddress res4d_branch2c_inIndex,res4e_branch2a_outOffset;
    /* res4e_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4e_branch2a_scale__vv_mul__scale4e_branch2a_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4e_branch2a_bias__vv_mul__scale4e_branch2a_scale__vv_add__scale4e_branch2a_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res4d_branch2c_inIndex=0;
    res4e_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4d_branch2c_inIndex, 8, 196, 8);
        mv_mul(bs, mrf_start+16+(outRow*8));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4e_branch2a: scale, vv_mul, scale4e_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4e_branch2a: bias, vv_mul, scale4e_branch2a: scale, vv_add, scale4e_branch2a: bias */
        v_relu(bs); /* includes: res4e_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8907+res4e_branch2a_outOffset+outChainOffset,2);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4eBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4e_branch2b(d=256,h=14,w=14)=Conv(res4e_branch2a(d=256,h=14,w=14),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn4e_branch2b */
    /*      absorbed scale4e_branch2b */
    /*      absorbed res4e_branch2b_relu */
    ISA_ExtAddress res4e_branch2a_inIndex,res4e_branch2b_outOffset;
    /* res4e_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4e_branch2b_scale__vv_mul__scale4e_branch2b_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4e_branch2b_bias__vv_mul__scale4e_branch2b_scale__vv_add__scale4e_branch2b_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 52 entries starting at res4f_branch2a */
        moveFilterCount128(bs, ISA_Mem_Dram, res4f_branch2a_MRF+0*52, ISA_Mem_MatrixRf, mrf_next, 1, 52);
    }
    res4e_branch2a_inIndex=8907;
    res4e_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res4e_branch2a_inIndex, 14, 14, 2, 3, 1, 1);
        mv_mul(bs, mrf_start+0+(outRow*18));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4e_branch2b: scale, vv_mul, scale4e_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4e_branch2b: bias, vv_mul, scale4e_branch2b: scale, vv_add, scale4e_branch2b: bias */
        v_relu(bs); /* includes: res4e_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4e_branch2b_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res4eBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4e_branch2c(d=1024,h=14,w=14)=Conv(res4d_branch2c(d=256,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4e_branch2c */
    /*      absorbed scale4e_branch2c */
    /*      absorbed res4e */
    /*      absorbed res4e_relu */
    ISA_ExtAddress res4e_branch2b_inIndex,res4e_branch2c_outOffset;
    /* res4e_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4e_branch2c_scale__vv_mul__scale4e_branch2c_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4e_branch2c_bias__vv_mul__scale4e_branch2c_scale__vv_add__scale4e_branch2c_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res4d_branch2c_iterator = 10532;
    ISA_ExtAddress outChainOffset = 0;
    res4e_branch2b_inIndex=0;
    res4e_branch2c_outOffset=0;
    res4d_branch2c_iterator=10532;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<8;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4e_branch2b_inIndex, 2, 196, 2);
        mv_mul(bs, mrf_start+36+(outRow*2));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4e_branch2c: scale, vv_mul, scale4e_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4e_branch2c: bias, vv_mul, scale4e_branch2c: scale, vv_add, scale4e_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res4d_branch2c_iterator+outChainOffset, 8); /* includes: res4d */
        v_relu(bs); /* includes: res4e_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res4e_branch2c_outOffset+outChainOffset,8);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 7731+res4e_branch2c_outOffset+outChainOffset,8);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4fBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4f_branch2a(d=256,h=14,w=14)=Conv(res4e_branch2c(d=1024,h=14,w=14),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4f_branch2a */
    /*      absorbed scale4f_branch2a */
    /*      absorbed res4f_branch2a_relu */
    ISA_ExtAddress res4e_branch2c_inIndex,res4f_branch2a_outOffset;
    /* res4f_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4f_branch2a_scale__vv_mul__scale4f_branch2a_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4f_branch2a_bias__vv_mul__scale4f_branch2a_scale__vv_add__scale4f_branch2a_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch 128 entries starting at res4f_branch2c */
        moveFilterCount128(bs, ISA_Mem_Dram, res4f_branch2c_MRF+0*16, ISA_Mem_MatrixRf, mrf_next, 1, 16);
    }
    res4e_branch2c_inIndex=7731;
    res4f_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4e_branch2c_inIndex, 8, 196, 8);
        mv_mul(bs, mrf_start+0+(outRow*8));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4f_branch2a: scale, vv_mul, scale4f_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4f_branch2a: bias, vv_mul, scale4f_branch2a: scale, vv_add, scale4f_branch2a: bias */
        v_relu(bs); /* includes: res4f_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4f_branch2a_outOffset+outChainOffset,2);
        outChainOffset++;
    }
}

void res4fBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4f_branch2b(d=256,h=7,w=7)=Conv(res4f_branch2a(d=256,h=14,w=14),k_h=3,k_w=3,s_h=2,s_w=2,p_h=1,p_w=1) */
    /*      absorbed bn4f_branch2b */
    /*      absorbed scale4f_branch2b */
    /*      absorbed res4f_branch2b_relu */
    ISA_ExtAddress res4f_branch2a_inIndex,res4f_branch2b_outOffset;
    /* res4f_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4f_branch2b_scale__vv_mul__scale4f_branch2b_scale+0, 2);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4f_branch2b_bias__vv_mul__scale4f_branch2b_scale__vv_add__scale4f_branch2b_bias+0, 2);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    res4f_branch2a_inIndex=0;
    res4f_branch2b_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 2 rows and (2 columns * 2 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<2;outRow++) {
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res4f_branch2a_inIndex, 14, 14, 2, 3, 1, 2);
        mv_mul(bs, mrf_start+16+(outRow*18));
        vv_mul(bs, 0+outChainOffset); /* includes: bn4f_branch2b: scale, vv_mul, scale4f_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4f_branch2b: bias, vv_mul, scale4f_branch2b: scale, vv_add, scale4f_branch2b: bias */
        v_relu(bs); /* includes: res4f_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9201+res4f_branch2b_outOffset+outChainOffset,2);
        outChainOffset++;
    }
    res4f_branch2a_inIndex += 28; /* skip 1 rows due to stride, adjusted for 14/14 discrepency */
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res4fBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res4f_branch2c(d=1024,h=7,w=7)=Conv(res4e_branch2c(d=256,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn4f_branch2c */
    /*      absorbed scale4f_branch2c */
    /*      absorbed res4f */
    /*      absorbed res4f_relu */
    ISA_ExtAddress res4f_branch2b_inIndex,res4f_branch2c_outOffset;
    /* res4f_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn4f_branch2c_scale__vv_mul__scale4f_branch2c_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn4f_branch2c_bias__vv_mul__scale4f_branch2c_scale__vv_add__scale4f_branch2c_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res4e_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch the first part of res5a_branch1_MRF */
        moveFilterCount128(bs, ISA_Mem_Dram, res5a_branch1_MRF+0*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
    }
    res4f_branch2b_inIndex=9201;
    res4f_branch2c_outOffset=0;
    res4e_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 2 registers/data element) */
    for (int rowIterator=0; rowIterator<7; rowIterator++) {
        outChainOffset=0;
        for(int outRow=0;outRow<8;outRow++) {
            /* strided IVRF access mode on */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, res4f_branch2b_inIndex, 2, 7, 2);
            mv_mul(bs, mrf_start+0+(outRow*2));
            vv_mul(bs, 0+outChainOffset); /* includes: bn4f_branch2c: scale, vv_mul, scale4f_branch2c: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn4f_branch2c: bias, vv_mul, scale4f_branch2c: scale, vv_add, scale4f_branch2c: bias */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res4e_branch2c_iterator+outChainOffset, 16); /* includes: res4e */
            v_relu(bs); /* includes: res4f_relu: v_relu */
            v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res4f_branch2c_outOffset+outChainOffset,8);
            outChainOffset++;
        }
        res4f_branch2b_inIndex += 14;
        res4f_branch2c_outOffset += 56;
        res4e_branch2c_iterator += 224;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res5aBranch1(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5a_branch1(d=2048,h=7,w=7)=Conv(res4f_branch2c(d=1024,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5a_branch1 */
    /*      absorbed scale5a_branch1 */
    ISA_ExtAddress res4f_branch2c_inIndex,res5a_branch1_outOffset;
    ISA_VrfAddress res5a_branch1_param_mulvrf_0_cur=0, res5a_branch1_param_mulvrf_0_next=8, res5a_branch1_param_mulvrf_0_tmp;
    ISA_VrfAddress res5a_branch1_param_asvrf_1_cur=0, res5a_branch1_param_asvrf_1_next=8, res5a_branch1_param_asvrf_1_tmp;
    /* res5a_branch1_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch1_scale__vv_mul__scale5a_branch1_scale+0, 8);
    v_wr(bs, ISA_Mem_MultiplyVrf, res5a_branch1_param_mulvrf_0_cur);
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch1_bias__vv_mul__scale5a_branch1_scale__vv_add__scale5a_branch1_bias+0, 8);
    v_wr(bs, ISA_Mem_AddSubVrf_0, res5a_branch1_param_asvrf_1_cur);
    ISA_ExtAddress outChainOffset = 0;
    outChainOffset=0;
    for(int outRowBlock=0;outRowBlock<2;outRowBlock++) {
        if (outRowBlock!=1) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, res5a_branch1_MRF+(outRowBlock+1)*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
            vRead1D(bs, ISA_Mem_Dram, bn5a_branch1_scale__vv_mul__scale5a_branch1_scale+(outRowBlock+1)*8, 8);
            v_wr(bs, ISA_Mem_MultiplyVrf, res5a_branch1_param_mulvrf_0_next);
            vRead1D(bs, ISA_Mem_Dram, bn5a_branch1_bias__vv_mul__scale5a_branch1_scale__vv_add__scale5a_branch1_bias+(outRowBlock+1)*8, 8);
            v_wr(bs, ISA_Mem_AddSubVrf_0, res5a_branch1_param_asvrf_1_next);
        } else if (!p_last) {
            /* Prefetch 128 entries starting at res5a_branch2a */
            moveFilterCount128(bs, ISA_Mem_Dram, res5a_branch2a_MRF+0*32, ISA_Mem_MatrixRf, mrf_next, 1, 32);
        }
        for(int outRow=0;outRow<8;outRow++) {
            res4f_branch2c_inIndex=0;
            res5a_branch1_outOffset=0;
            /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
            /* strided IVRF access mode on */
            vRead2D(bs, ISA_Mem_MvmInitialVrf, res4f_branch2c_inIndex, 8, 49, 8);
            mv_mul(bs, mrf_start+0+(outRow*8));
            vv_mul(bs, res5a_branch1_param_mulvrf_0_cur+outRow); /* includes: bn5a_branch1: scale, vv_mul, scale5a_branch1: scale */
            vv_add_inc(bs, ISA_Mem_AddSubVrf_0, res5a_branch1_param_asvrf_1_cur+outRow, 0); /* includes: bn5a_branch1: bias, vv_mul, scale5a_branch1: scale, vv_add, scale5a_branch1: bias */
            v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 11316+res5a_branch1_outOffset+outChainOffset,16);
            outChainOffset++;
        }
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        res5a_branch1_param_mulvrf_0_tmp=res5a_branch1_param_mulvrf_0_cur; res5a_branch1_param_mulvrf_0_cur=res5a_branch1_param_mulvrf_0_next; res5a_branch1_param_mulvrf_0_next=res5a_branch1_param_mulvrf_0_tmp;
        res5a_branch1_param_asvrf_1_tmp=res5a_branch1_param_asvrf_1_cur; res5a_branch1_param_asvrf_1_cur=res5a_branch1_param_asvrf_1_next; res5a_branch1_param_asvrf_1_next=res5a_branch1_param_asvrf_1_tmp;
    }
}

void res5aBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5a_branch2a(d=512,h=7,w=7)=Conv(res4f_branch2c(d=1024,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5a_branch2a */
    /*      absorbed scale5a_branch2a */
    /*      absorbed res5a_branch2a_relu */
    ISA_ExtAddress res4f_branch2c_inIndex,res5a_branch2a_outOffset;
    /* res5a_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch2a_scale__vv_mul__scale5a_branch2a_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch2a_bias__vv_mul__scale5a_branch2a_scale__vv_add__scale5a_branch2a_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch the first part of res5a_branch2b_MRF */
        moveFilterCount128(bs, ISA_Mem_Dram, res5a_branch2b_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 3, 4);
    }
    res4f_branch2c_inIndex=0;
    res5a_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 8 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res4f_branch2c_inIndex, 8, 49, 8);
        mv_mul(bs, mrf_start+0+(outRow*8));
        vv_mul(bs, 0+outChainOffset); /* includes: bn5a_branch2a: scale, vv_mul, scale5a_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn5a_branch2a: bias, vv_mul, scale5a_branch2a: scale, vv_add, scale5a_branch2a: bias */
        v_relu(bs); /* includes: res5a_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9103+res5a_branch2a_outOffset+outChainOffset,4);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res5aBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5a_branch2b(d=512,h=7,w=7)=Conv(res5a_branch2a(d=512,h=7,w=7),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn5a_branch2b */
    /*      absorbed scale5a_branch2b */
    /*      absorbed res5a_branch2b_relu */
    ISA_ExtAddress res5a_branch2a_inIndex,res5a_branch2b_outOffset;
    ISA_VrfAddress res5a_branch2b_param_mulvrf_0_cur=0, res5a_branch2b_param_mulvrf_0_next=1, res5a_branch2b_param_mulvrf_0_tmp;
    ISA_VrfAddress res5a_branch2b_param_asvrf_1_cur=0, res5a_branch2b_param_asvrf_1_next=1, res5a_branch2b_param_asvrf_1_tmp;
    /* res5a_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch2b_scale__vv_mul__scale5a_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, res5a_branch2b_param_mulvrf_0_cur);
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch2b_bias__vv_mul__scale5a_branch2b_scale__vv_add__scale5a_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, res5a_branch2b_param_asvrf_1_cur);
    ISA_ExtAddress outChainOffset = 0;
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, res5a_branch2b_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 3, 4);
            vRead1D(bs, ISA_Mem_Dram, bn5a_branch2b_scale__vv_mul__scale5a_branch2b_scale+(outRow+1), 1);
            v_wr(bs, ISA_Mem_MultiplyVrf, res5a_branch2b_param_mulvrf_0_next);
            vRead1D(bs, ISA_Mem_Dram, bn5a_branch2b_bias__vv_mul__scale5a_branch2b_scale__vv_add__scale5a_branch2b_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, res5a_branch2b_param_asvrf_1_next);
        } else if (!p_last) {
            /* Prefetch the first part of res5a_branch2c_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, res5a_branch2c_MRF+0*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
        }
        res5a_branch2a_inIndex=9103;
        res5a_branch2b_outOffset=0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res5a_branch2a_inIndex, 7, 7, 4, 3, 1, 1);
        mv_mul(bs, mrf_start);
        vv_mul(bs, res5a_branch2b_param_mulvrf_0_cur); /* includes: bn5a_branch2b: scale, vv_mul, scale5a_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, res5a_branch2b_param_asvrf_1_cur, 0); /* includes: bn5a_branch2b: bias, vv_mul, scale5a_branch2b: scale, vv_add, scale5a_branch2b: bias */
        v_relu(bs); /* includes: res5a_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res5a_branch2b_outOffset+outChainOffset,4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        res5a_branch2b_param_mulvrf_0_tmp=res5a_branch2b_param_mulvrf_0_cur; res5a_branch2b_param_mulvrf_0_cur=res5a_branch2b_param_mulvrf_0_next; res5a_branch2b_param_mulvrf_0_next=res5a_branch2b_param_mulvrf_0_tmp;
        res5a_branch2b_param_asvrf_1_tmp=res5a_branch2b_param_asvrf_1_cur; res5a_branch2b_param_asvrf_1_cur=res5a_branch2b_param_asvrf_1_next; res5a_branch2b_param_asvrf_1_next=res5a_branch2b_param_asvrf_1_tmp;
    }
}

void res5aBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5a_branch2c(d=2048,h=7,w=7)=Conv(res5a_branch1(d=512,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5a_branch2c */
    /*      absorbed scale5a_branch2c */
    /*      absorbed res5a */
    /*      absorbed res5a_relu */
    ISA_ExtAddress res5a_branch2b_inIndex,res5a_branch2c_outOffset;
    /* res5a_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch2c_scale__vv_mul__scale5a_branch2c_scale+0, 16);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn5a_branch2c_bias__vv_mul__scale5a_branch2c_scale__vv_add__scale5a_branch2c_bias+0, 16);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res5a_branch1_iterator = 11316;
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch the first part of res5b_branch2a_MRF */
        moveFilterCount128(bs, ISA_Mem_Dram, res5b_branch2a_MRF+0*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
    }
    res5a_branch2b_inIndex=0;
    res5a_branch2c_outOffset=0;
    res5a_branch1_iterator=11316;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<16;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res5a_branch2b_inIndex, 4, 49, 4);
        mv_mul(bs, mrf_start+0+(outRow*4));
        vv_mul(bs, 0+outChainOffset); /* includes: bn5a_branch2c: scale, vv_mul, scale5a_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn5a_branch2c: bias, vv_mul, scale5a_branch2c: scale, vv_add, scale5a_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res5a_branch1_iterator+outChainOffset, 16); /* includes: res5a_branch1 */
        v_relu(bs); /* includes: res5a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 0+res5a_branch2c_outOffset+outChainOffset,16);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 8515+res5a_branch2c_outOffset+outChainOffset,16);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res5bBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5b_branch2a(d=512,h=7,w=7)=Conv(res5a_branch2c(d=2048,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5b_branch2a */
    /*      absorbed scale5b_branch2a */
    /*      absorbed res5b_branch2a_relu */
    ISA_ExtAddress res5a_branch2c_inIndex,res5b_branch2a_outOffset;
    /* res5b_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5b_branch2a_scale__vv_mul__scale5b_branch2a_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn5b_branch2a_bias__vv_mul__scale5b_branch2a_scale__vv_add__scale5b_branch2a_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch the first part of res5b_branch2b_MRF */
        moveFilterCount128(bs, ISA_Mem_Dram, res5b_branch2b_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 3, 4);
    }
    res5a_branch2c_inIndex=8515;
    res5b_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 16 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res5a_branch2c_inIndex, 16, 49, 16);
        mv_mul(bs, mrf_start+0+(outRow*16));
        vv_mul(bs, 0+outChainOffset); /* includes: bn5b_branch2a: scale, vv_mul, scale5b_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn5b_branch2a: bias, vv_mul, scale5b_branch2a: scale, vv_add, scale5b_branch2a: bias */
        v_relu(bs); /* includes: res5b_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res5b_branch2a_outOffset+outChainOffset,4);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res5bBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5b_branch2b(d=512,h=7,w=7)=Conv(res5b_branch2a(d=512,h=7,w=7),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn5b_branch2b */
    /*      absorbed scale5b_branch2b */
    /*      absorbed res5b_branch2b_relu */
    ISA_ExtAddress res5b_branch2a_inIndex,res5b_branch2b_outOffset;
    ISA_VrfAddress res5b_branch2b_param_mulvrf_0_cur=0, res5b_branch2b_param_mulvrf_0_next=1, res5b_branch2b_param_mulvrf_0_tmp;
    ISA_VrfAddress res5b_branch2b_param_asvrf_1_cur=0, res5b_branch2b_param_asvrf_1_next=1, res5b_branch2b_param_asvrf_1_tmp;
    /* res5b_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5b_branch2b_scale__vv_mul__scale5b_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, res5b_branch2b_param_mulvrf_0_cur);
    vRead1D(bs, ISA_Mem_Dram, bn5b_branch2b_bias__vv_mul__scale5b_branch2b_scale__vv_add__scale5b_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, res5b_branch2b_param_asvrf_1_cur);
    ISA_ExtAddress outChainOffset = 0;
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, res5b_branch2b_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 3, 4);
            vRead1D(bs, ISA_Mem_Dram, bn5b_branch2b_scale__vv_mul__scale5b_branch2b_scale+(outRow+1), 1);
            v_wr(bs, ISA_Mem_MultiplyVrf, res5b_branch2b_param_mulvrf_0_next);
            vRead1D(bs, ISA_Mem_Dram, bn5b_branch2b_bias__vv_mul__scale5b_branch2b_scale__vv_add__scale5b_branch2b_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, res5b_branch2b_param_asvrf_1_next);
        } else if (!p_last) {
            /* Prefetch the first part of res5b_branch2c_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, res5b_branch2c_MRF+0*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
        }
        res5b_branch2a_inIndex=0;
        res5b_branch2b_outOffset=0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res5b_branch2a_inIndex, 7, 7, 4, 3, 1, 1);
        mv_mul(bs, mrf_start);
        vv_mul(bs, res5b_branch2b_param_mulvrf_0_cur); /* includes: bn5b_branch2b: scale, vv_mul, scale5b_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, res5b_branch2b_param_asvrf_1_cur, 0); /* includes: bn5b_branch2b: bias, vv_mul, scale5b_branch2b: scale, vv_add, scale5b_branch2b: bias */
        v_relu(bs); /* includes: res5b_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9103+res5b_branch2b_outOffset+outChainOffset,4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        res5b_branch2b_param_mulvrf_0_tmp=res5b_branch2b_param_mulvrf_0_cur; res5b_branch2b_param_mulvrf_0_cur=res5b_branch2b_param_mulvrf_0_next; res5b_branch2b_param_mulvrf_0_next=res5b_branch2b_param_mulvrf_0_tmp;
        res5b_branch2b_param_asvrf_1_tmp=res5b_branch2b_param_asvrf_1_cur; res5b_branch2b_param_asvrf_1_cur=res5b_branch2b_param_asvrf_1_next; res5b_branch2b_param_asvrf_1_next=res5b_branch2b_param_asvrf_1_tmp;
    }
}

void res5bBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5b_branch2c(d=2048,h=7,w=7)=Conv(res5a_branch2c(d=512,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5b_branch2c */
    /*      absorbed scale5b_branch2c */
    /*      absorbed res5b */
    /*      absorbed res5b_relu */
    ISA_ExtAddress res5b_branch2b_inIndex,res5b_branch2c_outOffset;
    /* res5b_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5b_branch2c_scale__vv_mul__scale5b_branch2c_scale+0, 16);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn5b_branch2c_bias__vv_mul__scale5b_branch2c_scale__vv_add__scale5b_branch2c_bias+0, 16);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res5a_branch2c_iterator = 0;
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch the first part of res5c_branch2a_MRF */
        moveFilterCount128(bs, ISA_Mem_Dram, res5c_branch2a_MRF+0*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
    }
    res5b_branch2b_inIndex=9103;
    res5b_branch2c_outOffset=0;
    res5a_branch2c_iterator=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<16;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res5b_branch2b_inIndex, 4, 49, 4);
        mv_mul(bs, mrf_start+0+(outRow*4));
        vv_mul(bs, 0+outChainOffset); /* includes: bn5b_branch2c: scale, vv_mul, scale5b_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn5b_branch2c: bias, vv_mul, scale5b_branch2c: scale, vv_add, scale5b_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res5a_branch2c_iterator+outChainOffset, 16); /* includes: res5a */
        v_relu(bs); /* includes: res5b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_AddSubVrf_1, 11316+res5b_branch2c_outOffset+outChainOffset,16);
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res5b_branch2c_outOffset+outChainOffset,16);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res5cBranch2a(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5c_branch2a(d=512,h=7,w=7)=Conv(res5b_branch2c(d=2048,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5c_branch2a */
    /*      absorbed scale5c_branch2a */
    /*      absorbed res5c_branch2a_relu */
    ISA_ExtAddress res5b_branch2c_inIndex,res5c_branch2a_outOffset;
    /* res5c_branch2a_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5c_branch2a_scale__vv_mul__scale5c_branch2a_scale+0, 4);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn5c_branch2a_bias__vv_mul__scale5c_branch2a_scale__vv_add__scale5c_branch2a_bias+0, 4);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress outChainOffset = 0;
    if (!p_last) {
        /* Prefetch the first part of res5c_branch2b_MRF */
        moveFilterCount128(bs, ISA_Mem_Dram, res5c_branch2b_MRF+0*36, ISA_Mem_MatrixRf, mrf_next, 3, 4);
    }
    res5b_branch2c_inIndex=0;
    res5c_branch2a_outOffset=0;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 16 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res5b_branch2c_inIndex, 16, 49, 16);
        mv_mul(bs, mrf_start+0+(outRow*16));
        vv_mul(bs, 0+outChainOffset); /* includes: bn5c_branch2a: scale, vv_mul, scale5c_branch2a: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn5c_branch2a: bias, vv_mul, scale5c_branch2a: scale, vv_add, scale5c_branch2a: bias */
        v_relu(bs); /* includes: res5c_branch2a_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 9103+res5c_branch2a_outOffset+outChainOffset,4);
        outChainOffset++;
    }
    mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
}

void res5cBranch2b(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5c_branch2b(d=512,h=7,w=7)=Conv(res5c_branch2a(d=512,h=7,w=7),k_h=3,k_w=3,s_h=1,s_w=1,p_h=1,p_w=1) */
    /*      absorbed bn5c_branch2b */
    /*      absorbed scale5c_branch2b */
    /*      absorbed res5c_branch2b_relu */
    ISA_ExtAddress res5c_branch2a_inIndex,res5c_branch2b_outOffset;
    ISA_VrfAddress res5c_branch2b_param_mulvrf_0_cur=0, res5c_branch2b_param_mulvrf_0_next=1, res5c_branch2b_param_mulvrf_0_tmp;
    ISA_VrfAddress res5c_branch2b_param_asvrf_1_cur=0, res5c_branch2b_param_asvrf_1_next=1, res5c_branch2b_param_asvrf_1_tmp;
    /* res5c_branch2b_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5c_branch2b_scale__vv_mul__scale5c_branch2b_scale+0, 1);
    v_wr(bs, ISA_Mem_MultiplyVrf, res5c_branch2b_param_mulvrf_0_cur);
    vRead1D(bs, ISA_Mem_Dram, bn5c_branch2b_bias__vv_mul__scale5c_branch2b_scale__vv_add__scale5c_branch2b_bias+0, 1);
    v_wr(bs, ISA_Mem_AddSubVrf_0, res5c_branch2b_param_asvrf_1_cur);
    ISA_ExtAddress outChainOffset = 0;
    outChainOffset=0;
    for(int outRow=0;outRow<4;outRow++) {
        if (outRow!=3) {
            // Fetch next set of parameters
            moveFilterCount128(bs, ISA_Mem_Dram, res5c_branch2b_MRF+(outRow+1)*36, ISA_Mem_MatrixRf, mrf_next, 3, 4);
            vRead1D(bs, ISA_Mem_Dram, bn5c_branch2b_scale__vv_mul__scale5c_branch2b_scale+(outRow+1), 1);
            v_wr(bs, ISA_Mem_MultiplyVrf, res5c_branch2b_param_mulvrf_0_next);
            vRead1D(bs, ISA_Mem_Dram, bn5c_branch2b_bias__vv_mul__scale5c_branch2b_scale__vv_add__scale5c_branch2b_bias+(outRow+1), 1);
            v_wr(bs, ISA_Mem_AddSubVrf_0, res5c_branch2b_param_asvrf_1_next);
        } else if (!p_last) {
            /* Prefetch the first part of res5c_branch2c_MRF */
            moveFilterCount128(bs, ISA_Mem_Dram, res5c_branch2c_MRF+0*64, ISA_Mem_MatrixRf, mrf_next, 1, 64);
        }
        res5c_branch2a_inIndex=9103;
        res5c_branch2b_outOffset=0;
        /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
        /* strided IVRF access mode on */
        vRead3D(bs, ISA_Mem_MvmInitialVrf, res5c_branch2a_inIndex, 7, 7, 4, 3, 1, 1);
        mv_mul(bs, mrf_start);
        vv_mul(bs, res5c_branch2b_param_mulvrf_0_cur); /* includes: bn5c_branch2b: scale, vv_mul, scale5c_branch2b: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, res5c_branch2b_param_asvrf_1_cur, 0); /* includes: bn5c_branch2b: bias, vv_mul, scale5c_branch2b: scale, vv_add, scale5c_branch2b: bias */
        v_relu(bs); /* includes: res5c_branch2b_relu: v_relu */
        v_wr_inc(bs, ISA_Mem_MvmInitialVrf, 0+res5c_branch2b_outOffset+outChainOffset,4);
        outChainOffset++;
        // Swap parameter buffers
        mrf_tmp=mrf_start; mrf_start=mrf_next; mrf_next=mrf_tmp;
        res5c_branch2b_param_mulvrf_0_tmp=res5c_branch2b_param_mulvrf_0_cur; res5c_branch2b_param_mulvrf_0_cur=res5c_branch2b_param_mulvrf_0_next; res5c_branch2b_param_mulvrf_0_next=res5c_branch2b_param_mulvrf_0_tmp;
        res5c_branch2b_param_asvrf_1_tmp=res5c_branch2b_param_asvrf_1_cur; res5c_branch2b_param_asvrf_1_cur=res5c_branch2b_param_asvrf_1_next; res5c_branch2b_param_asvrf_1_next=res5c_branch2b_param_asvrf_1_tmp;
    }
}

void res5cBranch2c(PBS_CONTEXT bs, bool p_debugMode, bool p_first, bool p_last)
{
    /* Standalone convolution block */
    /* Convolution res5c_branch2c(d=2048,h=7,w=7)=Conv(res5b_branch2c(d=512,h=7,w=7),k_h=1,k_w=1,s_h=1,s_w=1,p_h=0,p_w=0) */
    /*      absorbed bn5c_branch2c */
    /*      absorbed scale5c_branch2c */
    /*      absorbed res5c */
    /*      absorbed res5c_relu */
    ISA_ExtAddress res5c_branch2b_inIndex,res5c_branch2c_outOffset;
    /* res5c_branch2c_MRF was prefetched */
    vRead1D(bs, ISA_Mem_Dram, bn5c_branch2c_scale__vv_mul__scale5c_branch2c_scale+0, 16);
    v_wr(bs, ISA_Mem_MultiplyVrf, 0);
    vRead1D(bs, ISA_Mem_Dram, bn5c_branch2c_bias__vv_mul__scale5c_branch2c_scale__vv_add__scale5c_branch2c_bias+0, 16);
    v_wr(bs, ISA_Mem_AddSubVrf_0, 0);
    ISA_ExtAddress res5b_branch2c_iterator = 11316;
    ISA_ExtAddress outChainOffset = 0;
    res5c_branch2b_inIndex=0;
    res5c_branch2c_outOffset=0;
    res5b_branch2c_iterator=11316;
    /* Non-tiled iteration: Traverse the input feature map in steps of 1 rows and (1 columns * 4 registers/data element) */
    outChainOffset=0;
    for(int outRow=0;outRow<16;outRow++) {
        /* strided IVRF access mode on */
        vRead2D(bs, ISA_Mem_MvmInitialVrf, res5c_branch2b_inIndex, 4, 49, 4);
        mv_mul(bs, mrf_start+0+(outRow*4));
        vv_mul(bs, 0+outChainOffset); /* includes: bn5c_branch2c: scale, vv_mul, scale5c_branch2c: scale */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_0, 0+outChainOffset, 0); /* includes: bn5c_branch2c: bias, vv_mul, scale5c_branch2c: scale, vv_add, scale5c_branch2c: bias */
        vv_add_inc(bs, ISA_Mem_AddSubVrf_1, res5b_branch2c_iterator+outChainOffset, 16); /* includes: res5b */
        v_relu(bs); /* includes: res5c_relu: v_relu */
        v_wr(bs, ISA_Mem_NetOutputQ, DONTCARE);
        outChainOffset++;
    }
}
