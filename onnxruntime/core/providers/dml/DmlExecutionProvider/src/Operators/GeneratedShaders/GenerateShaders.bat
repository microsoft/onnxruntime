@echo off

if "%1" == "DEBUG" (
    echo "WARNING: Compiling shaders for DEBUG configuration; do not check generated header files into the repo!"
    fxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_5_0 /DTBUFFER=float /Zi /Od /Fh stockham.h
    dxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh stockham_fp16.h

    fxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_5_0 /DTBUFFER=float /Zi /Od /Fh bluestein_chirp.h
    dxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh bluestein_chirp_fp16.h

    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint16_t -DTBUFFER2=float -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint16_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=uint /DTBUFFER2=float /Zi /Od /Fh grid_sample_uint_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint64_t -DTBUFFER2=float -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint64_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int16_t -DTBUFFER2=float -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int16_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=int /DTBUFFER2=float /Zi /Od /Fh grid_sample_int_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int64_t -DTBUFFER2=float -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int64_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float16_t -DTBUFFER2=float -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_fp16_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=float /DTBUFFER2=float /Zi /Od /Fh grid_sample_float_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=double /DTBUFFER2=float /Zi /Od /Fh grid_sample_double_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=bool /DTBUFFER2=float /Zi /Od /Fh grid_sample_bool_float.h

    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint16_t -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint16_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint64_t -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint64_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int16_t -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int16_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int64_t -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int64_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float16_t -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_fp16_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_float_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=double -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_double_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=bool -DTBUFFER2=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_bool_fp16.h

    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint16_t -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint16_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint64_t -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_uint64_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int16_t -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int16_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int64_t -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_int64_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float16_t -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_fp16_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_float_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=double -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_double_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=bool -DTBUFFER2=double -enable-16bit-types -Zi -Od -Qembed_debug -Fh grid_sample_bool_double.h

) else (
    fxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_5_0 /DTBUFFER=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh stockham.h
    dxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh stockham_fp16.h

    fxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_5_0 /DTBUFFER=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh bluestein_chirp.h
    dxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh bluestein_chirp_fp16.h

    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint16_t -DTBUFFER2=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint16_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=uint /DTBUFFER2=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh grid_sample_uint_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint64_t -DTBUFFER2=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint64_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int16_t -DTBUFFER2=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int16_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=int /DTBUFFER2=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh grid_sample_int_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int64_t -DTBUFFER2=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int64_float.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float16_t -DTBUFFER2=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_fp16_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=float /DTBUFFER2=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh grid_sample_float_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=double /DTBUFFER2=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh grid_sample_double_float.h
    fxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_5_0 /DTBUFFER1=bool /DTBUFFER2=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh grid_sample_bool_float.h

    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint16_t -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint16_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint64_t -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint64_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int16_t -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int16_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int64_t -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int64_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float16_t -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_fp16_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_float_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=double -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_double_fp16.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=bool -DTBUFFER2=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_bool_fp16.h

    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint16_t -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint16_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=uint64_t -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_uint64_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int16_t -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int16_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=int64_t -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_int64_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float16_t -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_fp16_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=float -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_float_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=double -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_double_double.h
    dxc.exe ..\Shaders\grid_sample.hlsl -E GridSample -T cs_6_2 -DTBUFFER1=bool -DTBUFFER2=double -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh grid_sample_bool_double.h

)
