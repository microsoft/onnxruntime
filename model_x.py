import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto
import numpy as np
import csv
import io
import ast

# ==== 1. Your full profiler CSV (as given) ====
PROFILER_CSV = r"""name	op_name	output_type_shape_NEON	output_type_shape_SME	input_type_shape_NEON	input_type_shape_SME	dur_NEON	dur_SME
1018_kernel_time	Conv	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [112, 384, 1, 1]}, {'float': [112]}]	[{'float': [1, 384, 16, 16]}, {'float': [112, 384, 1, 1]}, {'float': [112]}]	1618	361
1028_kernel_time	FusedConv	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 112, 16, 16]}, {'float': [672, 112, 1, 1]}, {'float': [672]}]	[{'float': [1, 112, 16, 16]}, {'float': [672, 112, 1, 1]}, {'float': [672]}]	2703	564
1050_kernel_time	FusedConv	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 672, 16, 16]}, {'float': [112, 672, 1, 1]}, {'float': [112]}, {'float': [1, 112, 16, 16]}]	[{'float': [1, 672, 16, 16]}, {'float': [112, 672, 1, 1]}, {'float': [112]}, {'float': [1, 112, 16, 16]}]	2842	500
1062_kernel_time	FusedConv	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 112, 16, 16]}, {'float': [672, 112, 1, 1]}, {'float': [672]}]	[{'float': [1, 112, 16, 16]}, {'float': [672, 112, 1, 1]}, {'float': [672]}]	2680	530
1084_kernel_time	FusedConv	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 672, 16, 16]}, {'float': [112, 672, 1, 1]}, {'float': [112]}, {'float': [1, 112, 16, 16]}]	[{'float': [1, 672, 16, 16]}, {'float': [112, 672, 1, 1]}, {'float': [112]}, {'float': [1, 112, 16, 16]}]	2852	506
1096_kernel_time	FusedConv	[{'float': [1, 336, 16, 16]}]	[{'float': [1, 336, 16, 16]}]	[{'float': [1, 112, 16, 16]}, {'float': [336, 112, 1, 1]}, {'float': [336]}]	[{'float': [1, 112, 16, 16]}, {'float': [336, 112, 1, 1]}, {'float': [336]}]	1413	307
1118_kernel_time	FusedConv	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 112, 16, 16]}]	[{'float': [1, 336, 16, 16]}, {'float': [112, 336, 1, 1]}, {'float': [112]}, {'float': [1, 112, 16, 16]}]	[{'float': [1, 336, 16, 16]}, {'float': [112, 336, 1, 1]}, {'float': [112]}, {'float': [1, 112, 16, 16]}]	1437	296
1129_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 112, 16, 16]}, {'float': [256, 112, 1, 1]}, {'float': [256]}]	[{'float': [1, 112, 16, 16]}, {'float': [256, 112, 1, 1]}, {'float': [256]}]	1042	234
1136/conv_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}]	708	139
1145_kernel_time	FusedConv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	2427	429
1153/conv_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}]	755	99
1162_kernel_time	FusedConv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	2377	360
1165_kernel_time	Slice	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'int64': [4]}, {'int64': [4]}, {'int64': [4]}]	[{'float': [1, 256, 16, 16]}, {'int64': [4]}, {'int64': [4]}, {'int64': [4]}]	39	39
1167_kernel_time	Concat	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [1, 64, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [1, 64, 16, 16]}]	39	39
1184_kernel_time	FusedConv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 320, 16, 16]}, {'float': [256, 320, 1, 1]}, {'float': [256]}]	[{'float': [1, 320, 16, 16]}, {'float': [256, 320, 1, 1]}, {'float': [256]}]	2987	463
1187_kernel_time	Slice	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'int64': [4]}, {'int64': [4]}, {'int64': [4]}]	[{'float': [1, 256, 16, 16]}, {'int64': [4]}, {'int64': [4]}, {'int64': [4]}]	28	19
1189_kernel_time	Concat	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [1, 64, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [1, 64, 16, 16]}]	37	31
1206_kernel_time	FusedConv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 320, 16, 16]}, {'float': [256, 320, 1, 1]}, {'float': [256]}]	[{'float': [1, 320, 16, 16]}, {'float': [256, 320, 1, 1]}, {'float': [256]}]	3026	451
1224_kernel_time	FusedConv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	2447	357
1242_kernel_time	FusedConv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 256, 1, 1]}, {'float': [256]}]	2383	343
1250/conv__1612_kernel_time	Transpose	[{'float': [1, 16, 16, 256]}]	[{'float': [1, 16, 16, 256]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	159	154
1293/conv__1648_kernel_time	Transpose	[{'float': [1, 16, 16, 256]}]	[{'float': [1, 16, 16, 256]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	159	162
616__1432_kernel_time	Transpose	[{'float': [1, 3, 256, 256]}]	[{'float': [1, 3, 256, 256]}]	[{'float': [1, 256, 256, 3]}]	[{'float': [1, 256, 256, 3]}]	274	291
616_kernel_time	FusedConv	[{'float': [1, 16, 128, 128]}]	[{'float': [1, 16, 128, 128]}]	[{'float': [1, 3, 256, 256]}, {'float': [16, 3, 3, 3]}, {'float': [16]}]	[{'float': [1, 3, 256, 256]}, {'float': [16, 3, 3, 3]}, {'float': [16]}]	2224	1594
638_kernel_time	FusedConv	[{'float': [1, 16, 128, 128]}]	[{'float': [1, 16, 128, 128]}]	[{'float': [1, 16, 128, 128]}, {'float': [16, 16, 1, 1]}, {'float': [16]}, {'float': [1, 16, 128, 128]}]	[{'float': [1, 16, 128, 128]}, {'float': [16, 16, 1, 1]}, {'float': [16]}, {'float': [1, 16, 128, 128]}]	1135	991
650_kernel_time	FusedConv	[{'float': [1, 96, 128, 128]}]	[{'float': [1, 96, 128, 128]}]	[{'float': [1, 16, 128, 128]}, {'float': [96, 16, 1, 1]}, {'float': [96]}]	[{'float': [1, 16, 128, 128]}, {'float': [96, 16, 1, 1]}, {'float': [96]}]	4785	3595
672_kernel_time	Conv	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 96, 64, 64]}, {'float': [24, 96, 1, 1]}, {'float': [24]}]	[{'float': [1, 96, 64, 64]}, {'float': [24, 96, 1, 1]}, {'float': [24]}]	1690	1698
693_kernel_time	FusedConv	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 24, 1, 1]}, {'float': [24]}, {'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 24, 1, 1]}, {'float': [24]}, {'float': [1, 24, 64, 64]}]	496	474
716_kernel_time	FusedConv	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 24, 1, 1]}, {'float': [24]}, {'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 24, 1, 1]}, {'float': [24]}, {'float': [1, 24, 64, 64]}]	498	504
728_kernel_time	FusedConv	[{'float': [1, 144, 64, 64]}]	[{'float': [1, 144, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [144, 24, 1, 1]}, {'float': [144]}]	[{'float': [1, 24, 64, 64]}, {'float': [144, 24, 1, 1]}, {'float': [144]}]	2402	1101
750_kernel_time	Conv	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 144, 32, 32]}, {'float': [32, 144, 1, 1]}, {'float': [32]}]	[{'float': [1, 144, 32, 32]}, {'float': [32, 144, 1, 1]}, {'float': [32]}]	784	285
760_kernel_time	FusedConv	[{'float': [1, 96, 32, 32]}]	[{'float': [1, 96, 32, 32]}]	[{'float': [1, 32, 32, 32]}, {'float': [96, 32, 1, 1]}, {'float': [96]}]	[{'float': [1, 32, 32, 32]}, {'float': [96, 32, 1, 1]}, {'float': [96]}]	512	279
782_kernel_time	FusedConv	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 96, 32, 32]}, {'float': [32, 96, 1, 1]}, {'float': [32]}, {'float': [1, 32, 32, 32]}]	[{'float': [1, 96, 32, 32]}, {'float': [32, 96, 1, 1]}, {'float': [32]}, {'float': [1, 32, 32, 32]}]	548	230
794_kernel_time	FusedConv	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 32, 32, 32]}, {'float': [192, 32, 1, 1]}, {'float': [192]}]	[{'float': [1, 32, 32, 32]}, {'float': [192, 32, 1, 1]}, {'float': [192]}]	997	518
816_kernel_time	FusedConv	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 192, 32, 32]}, {'float': [32, 192, 1, 1]}, {'float': [32]}, {'float': [1, 32, 32, 32]}]	[{'float': [1, 192, 32, 32]}, {'float': [32, 192, 1, 1]}, {'float': [32]}, {'float': [1, 32, 32, 32]}]	1104	384
828_kernel_time	FusedConv	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 32, 32, 32]}, {'float': [192, 32, 1, 1]}, {'float': [192]}]	[{'float': [1, 32, 32, 32]}, {'float': [192, 32, 1, 1]}, {'float': [192]}]	989	474
850_kernel_time	FusedConv	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 32, 32, 32]}]	[{'float': [1, 192, 32, 32]}, {'float': [32, 192, 1, 1]}, {'float': [32]}, {'float': [1, 32, 32, 32]}]	[{'float': [1, 192, 32, 32]}, {'float': [32, 192, 1, 1]}, {'float': [32]}, {'float': [1, 32, 32, 32]}]	1056	348
862_kernel_time	FusedConv	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 32, 32, 32]}, {'float': [192, 32, 1, 1]}, {'float': [192]}]	[{'float': [1, 32, 32, 32]}, {'float': [192, 32, 1, 1]}, {'float': [192]}]	1002	467
884_kernel_time	Conv	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 192, 16, 16]}, {'float': [64, 192, 1, 1]}, {'float': [64]}]	[{'float': [1, 192, 16, 16]}, {'float': [64, 192, 1, 1]}, {'float': [64]}]	494	170
894_kernel_time	FusedConv	[{'float': [1, 192, 16, 16]}]	[{'float': [1, 192, 16, 16]}]	[{'float': [1, 64, 16, 16]}, {'float': [192, 64, 1, 1]}, {'float': [192]}]	[{'float': [1, 64, 16, 16]}, {'float': [192, 64, 1, 1]}, {'float': [192]}]	472	199
916_kernel_time	FusedConv	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 192, 16, 16]}, {'float': [64, 192, 1, 1]}, {'float': [64]}, {'float': [1, 64, 16, 16]}]	[{'float': [1, 192, 16, 16]}, {'float': [64, 192, 1, 1]}, {'float': [64]}, {'float': [1, 64, 16, 16]}]	485	164
928_kernel_time	FusedConv	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 64, 16, 16]}, {'float': [384, 64, 1, 1]}, {'float': [384]}]	[{'float': [1, 64, 16, 16]}, {'float': [384, 64, 1, 1]}, {'float': [384]}]	943	309
950_kernel_time	FusedConv	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [64, 384, 1, 1]}, {'float': [64]}, {'float': [1, 64, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [64, 384, 1, 1]}, {'float': [64]}, {'float': [1, 64, 16, 16]}]	1036	279
962_kernel_time	FusedConv	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 64, 16, 16]}, {'float': [384, 64, 1, 1]}, {'float': [384]}]	[{'float': [1, 64, 16, 16]}, {'float': [384, 64, 1, 1]}, {'float': [384]}]	915	310
984_kernel_time	FusedConv	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 64, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [64, 384, 1, 1]}, {'float': [64]}, {'float': [1, 64, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [64, 384, 1, 1]}, {'float': [64]}, {'float': [1, 64, 16, 16]}]	970	260
996_kernel_time	FusedConv	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 64, 16, 16]}, {'float': [384, 64, 1, 1]}, {'float': [384]}]	[{'float': [1, 64, 16, 16]}, {'float': [384, 64, 1, 1]}, {'float': [384]}]	925	293
Conv__1651_kernel_time	FusedConv	[{'float': [1, 16, 128, 128]}]	[{'float': [1, 16, 128, 128]}]	[{'float': [1, 16, 128, 128]}, {'float': [16, 1, 3, 3]}, {'float': [16]}]	[{'float': [1, 16, 128, 128]}, {'float': [16, 1, 3, 3]}, {'float': [16]}]	1757	510
Conv__1652_kernel_time	FusedConv	[{'float': [1, 96, 64, 64]}]	[{'float': [1, 96, 64, 64]}]	[{'float': [1, 96, 128, 128]}, {'float': [96, 1, 3, 3]}, {'float': [96]}]	[{'float': [1, 96, 128, 128]}, {'float': [96, 1, 3, 3]}, {'float': [96]}]	7095	6880
Conv__1655_kernel_time	FusedConv	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 1, 3, 3]}, {'float': [24]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 1, 3, 3]}, {'float': [24]}]	767	247
Conv__1658_kernel_time	FusedConv	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 1, 3, 3]}, {'float': [24]}]	[{'float': [1, 24, 64, 64]}, {'float': [24, 1, 3, 3]}, {'float': [24]}]	770	211
Conv__1659_kernel_time	FusedConv	[{'float': [1, 144, 32, 32]}]	[{'float': [1, 144, 32, 32]}]	[{'float': [1, 144, 64, 64]}, {'float': [144, 1, 5, 5]}, {'float': [144]}]	[{'float': [1, 144, 64, 64]}, {'float': [144, 1, 5, 5]}, {'float': [144]}]	7319	7366
Conv__1662_kernel_time	FusedConv	[{'float': [1, 96, 32, 32]}]	[{'float': [1, 96, 32, 32]}]	[{'float': [1, 96, 32, 32]}, {'float': [96, 1, 5, 5]}, {'float': [96]}]	[{'float': [1, 96, 32, 32]}, {'float': [96, 1, 5, 5]}, {'float': [96]}]	2147	2279
Conv__1665_kernel_time	FusedConv	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}, {'float': [192, 1, 5, 5]}, {'float': [192]}]	[{'float': [1, 192, 32, 32]}, {'float': [192, 1, 5, 5]}, {'float': [192]}]	4282	4237
Conv__1668_kernel_time	FusedConv	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}]	[{'float': [1, 192, 32, 32]}, {'float': [192, 1, 3, 3]}, {'float': [192]}]	[{'float': [1, 192, 32, 32]}, {'float': [192, 1, 3, 3]}, {'float': [192]}]	1621	480
Conv__1669_kernel_time	FusedConv	[{'float': [1, 192, 16, 16]}]	[{'float': [1, 192, 16, 16]}]	[{'float': [1, 192, 32, 32]}, {'float': [192, 1, 5, 5]}, {'float': [192]}]	[{'float': [1, 192, 32, 32]}, {'float': [192, 1, 5, 5]}, {'float': [192]}]	2648	2594
Conv__1672_kernel_time	FusedConv	[{'float': [1, 192, 16, 16]}]	[{'float': [1, 192, 16, 16]}]	[{'float': [1, 192, 16, 16]}, {'float': [192, 1, 5, 5]}, {'float': [192]}]	[{'float': [1, 192, 16, 16]}, {'float': [192, 1, 5, 5]}, {'float': [192]}]	1529	1580
Conv__1675_kernel_time	FusedConv	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [384, 1, 5, 5]}, {'float': [384]}]	[{'float': [1, 384, 16, 16]}, {'float': [384, 1, 5, 5]}, {'float': [384]}]	3090	2953
Conv__1678_kernel_time	FusedConv	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [384, 1, 5, 5]}, {'float': [384]}]	[{'float': [1, 384, 16, 16]}, {'float': [384, 1, 5, 5]}, {'float': [384]}]	3065	3045
Conv__1679_kernel_time	FusedConv	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}]	[{'float': [1, 384, 16, 16]}, {'float': [384, 1, 5, 5]}, {'float': [384]}]	[{'float': [1, 384, 16, 16]}, {'float': [384, 1, 5, 5]}, {'float': [384]}]	3022	2963
Conv__1682_kernel_time	FusedConv	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 672, 16, 16]}, {'float': [672, 1, 5, 5]}, {'float': [672]}]	[{'float': [1, 672, 16, 16]}, {'float': [672, 1, 5, 5]}, {'float': [672]}]	5238	5168
Conv__1685_kernel_time	FusedConv	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 672, 16, 16]}]	[{'float': [1, 672, 16, 16]}, {'float': [672, 1, 5, 5]}, {'float': [672]}]	[{'float': [1, 672, 16, 16]}, {'float': [672, 1, 5, 5]}, {'float': [672]}]	5343	5203
Conv__1688_kernel_time	FusedConv	[{'float': [1, 336, 16, 16]}]	[{'float': [1, 336, 16, 16]}]	[{'float': [1, 336, 16, 16]}, {'float': [336, 1, 5, 5]}, {'float': [336]}]	[{'float': [1, 336, 16, 16]}, {'float': [336, 1, 5, 5]}, {'float': [336]}]	2695	2629
Conv__1693_kernel_time	Conv	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 320, 16, 16]}, {'float': [320, 1, 3, 3]}, {'float': [320]}]	[{'float': [1, 320, 16, 16]}, {'float': [320, 1, 3, 3]}, {'float': [320]}]	933	122
Conv__1694_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	749	115
Conv__1695_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	736	108
Conv__1696_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	750	105
Conv__1699_kernel_time	Conv	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 320, 16, 16]}]	[{'float': [1, 320, 16, 16]}, {'float': [320, 1, 3, 3]}, {'float': [320]}]	[{'float': [1, 320, 16, 16]}, {'float': [320, 1, 3, 3]}, {'float': [320]}]	964	156
Conv__1700_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	737	109
Conv__1701_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	746	103
Conv__1702_kernel_time	Conv	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	[{'float': [1, 256, 16, 16]}, {'float': [256, 1, 3, 3]}, {'float': [256]}]	756	103
SequentialExecutor::Execute	N/A	N/A	N/A	N/A	N/A	126116	73023
model_loading_uri	N/A	N/A	N/A	N/A	N/A	16375	28287
model_run	N/A	N/A	N/A	N/A	N/A	126261	73176
session_initialization	N/A	N/A	N/A	N/A	N/A	24448	35312
"""

# ==== 2. Utilities to parse the list-of-dicts fields ====

def parse_shape_list(field_str):
    field_str = field_str.strip()
    if not field_str or field_str == "N/A":
        return []
    try:
        return ast.literal_eval(field_str)
    except Exception as e:
        print("Failed to parse shape list:", field_str, "error:", e)
        return []

def get_first_float_4d(shapes):
    for entry in shapes:
        if "float" in entry:
            shape = entry["float"]
            if isinstance(shape, list) and len(shape) == 4:
                return shape
    return None

def get_weight_and_bias_shapes(input_shapes):
    float_entries = [e for e in input_shapes if "float" in e]
    if len(float_entries) < 2:
        return None, None

    weight_shape = float_entries[1]["float"]
    bias_shape = None
    if len(float_entries) >= 3:
        candidate = float_entries[2]["float"]
        if isinstance(candidate, list) and len(candidate) == 1:
            bias_shape = candidate

    return weight_shape, bias_shape

def infer_stride_and_padding(in_hw, out_hw, kernel_hw):
    H_in, W_in = in_hw
    H_out, W_out = out_hw
    kH, kW = kernel_hw

    pad_h = (kH - 1) // 2
    pad_w = (kW - 1) // 2

    stride_h = 1
    stride_w = 1

    if H_out > 0 and W_out > 0:
        if H_in % H_out == 0:
            stride_h = H_in // H_out
        if W_in % W_out == 0:
            stride_w = W_in // W_out

    if stride_h < 1:
        stride_h = 1
    if stride_w < 1:
        stride_w = 1

    pads = [pad_h, pad_w, pad_h, pad_w]
    strides = [stride_h, stride_w]
    return strides, pads

# ==== 3. Parse the CSV and collect Conv/FusedConv specs ====

layers = []

csv_file = io.StringIO(PROFILER_CSV.strip())
reader = csv.DictReader(csv_file, delimiter='\t')

for row in reader:
    op_name = row["op_name"]
    if op_name not in ("Conv", "FusedConv"):
        continue

    name = row["name"]

    out_shapes = parse_shape_list(row["output_type_shape_NEON"])
    in_shapes = parse_shape_list(row["input_type_shape_NEON"])

    act_in_shape = get_first_float_4d(in_shapes)
    act_out_shape = get_first_float_4d(out_shapes)
    if act_in_shape is None or act_out_shape is None:
        print(f"Skipping {name} due to missing 4D activation shapes")
        continue

    weight_shape, bias_shape = get_weight_and_bias_shapes(in_shapes)
    if weight_shape is None:
        print(f"Skipping {name} due to missing weight shape")
        continue

    if len(weight_shape) != 4:
        print(f"Skipping {name}: weight not 4D {weight_shape}")
        continue

    layers.append({
        "name": name,
        "op_type": op_name,
        "input_shape": act_in_shape,
        "output_shape": act_out_shape,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
    })

print(f"Parsed {len(layers)} Conv/FusedConv kernels")

# ==== 4. Build ONNX graph: each kernel is an independent Conv node ====

graph_inputs = []
graph_outputs = []
initializers = []
nodes = []

for layer in layers:
    lname = layer["name"]

    N_in, C_in, H_in, W_in = layer["input_shape"]
    N_out, C_out, H_out, W_out = layer["output_shape"]
    Cout_w, Cin_w, kH, kW = layer["weight_shape"]

    # --- Infer groups (handle depthwise convs) ---
    groups = 1
    # Depthwise pattern: act [N,C,H,W], W [C,1,kH,kW]
    if Cin_w == 1 and C_in == C_out == Cout_w:
        groups = C_in

    cin_effective = Cin_w * groups
    if cin_effective != C_in:
        print(
            f"Warning: channel mismatch in {lname}: "
            f"Cin={C_in}, Cin_w={Cin_w}, groups={groups}, "
            f"Cin_w*groups={cin_effective}, Cout={C_out}, Cout_w={Cout_w}"
        )

    strides, pads = infer_stride_and_padding(
        (H_in, W_in),
        (H_out, W_out),
        (kH, kW),
    )

    input_name = f"{lname}_input"
    weight_name = f"{lname}_W"
    bias_name = f"{lname}_B" if layer["bias_shape"] is not None else None
    output_name = f"{lname}_output"

    inp_vi = helper.make_tensor_value_info(
        input_name,
        TensorProto.FLOAT,
        [N_in, C_in, H_in, W_in],
    )
    graph_inputs.append(inp_vi)

    out_vi = helper.make_tensor_value_info(
        output_name,
        TensorProto.FLOAT,
        [N_out, C_out, H_out, W_out],
    )
    graph_outputs.append(out_vi)

    weight_data = np.random.randn(Cout_w, Cin_w, kH, kW).astype(np.float32) * 0.01
    weight_init = numpy_helper.from_array(weight_data, name=weight_name)
    initializers.append(weight_init)

    if bias_name is not None:
        bias_len = layer["bias_shape"][0]
        bias_data = np.random.randn(bias_len).astype(np.float32) * 0.01
        bias_init = numpy_helper.from_array(bias_data, name=bias_name)
        initializers.append(bias_init)

    conv_inputs = [input_name, weight_name]
    if bias_name is not None:
        conv_inputs.append(bias_name)

    conv_node = helper.make_node(
        "Conv",
        inputs=conv_inputs,
        outputs=[output_name],
        name=lname,
        strides=strides,
        pads=pads,
        group=groups,
    )

    nodes.append(conv_node)

# ==== 5. Assemble and save model ====

graph = helper.make_graph(
    nodes=nodes,
    name="ProfilerKernelsGraph",
    inputs=graph_inputs,
    outputs=graph_outputs,
    initializer=initializers,
)

model = helper.make_model(graph, producer_name="profiler_to_onnx")
onnx.checker.check_model(model)

onnx.save(model, "profiler_kernels.onnx")
print("Saved profiler_kernels.onnx")
