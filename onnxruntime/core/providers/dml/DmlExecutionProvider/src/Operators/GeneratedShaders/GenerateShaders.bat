@echo off

if "%1" == "DEBUG" (
    echo "WARNING: Compiling shaders for DEBUG configuration; do not check generated header files into the repo!"
    fxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_5_0 /DTBUFFER=float /Zi /Od /Fh stockham.h
    dxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh stockham_fp16.h

    fxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_5_0 /DTBUFFER=float /Zi /Od /Fh bluestein_chirp.h
    dxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh bluestein_chirp_fp16.h
) else (
    fxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_5_0 /DTBUFFER=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh stockham.h
    dxc.exe ..\Shaders\stockham.hlsl -E DFT -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh stockham_fp16.h

    fxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_5_0 /DTBUFFER=float /O3 /Qstrip_reflect /Qstrip_debug /Qstrip_rootsignature /Qstrip_priv /Fh bluestein_chirp.h
    dxc.exe ..\Shaders\bluestein_chirp.hlsl -E BluesteinZChirp -T cs_6_2 -DTBUFFER=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh bluestein_chirp_fp16.h
)
