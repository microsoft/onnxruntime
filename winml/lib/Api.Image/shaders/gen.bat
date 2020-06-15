  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor-SurfaceToTensorBGR8.h         /E "SurfaceToTensorBGR8"       /Vn g_csSurfaceToTensorBGR8       SurfaceToTensorFloat.hlsl
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor-SurfaceToTensorGRAY8.h        /E "SurfaceToTensorGRAY8"      /Vn g_csSurfaceToTensorGRAY8      SurfaceToTensorFloat.hlsl
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor-SurfaceToTensorRGB8.h         /E "SurfaceToTensorRGB8"       /Vn g_csSurfaceToTensorRGB8       SurfaceToTensorFloat.hlsl
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor-SurfaceGRAY8ToTensorBGR8.h    /E "SurfaceGRAY8ToTensorBGR8"  /Vn g_csSurfaceGRAY8ToTensorBGR8  SurfaceToTensorFloat.hlsl
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor-SurfaceGRAY8ToTensorGRAY8.h   /E "SurfaceGRAY8ToTensorGRAY8" /Vn g_csSurfaceGRAY8ToTensorGRAY8 SurfaceToTensorFloat.hlsl

  fxc.exe /Tcs_5_0 /Fh  TensorToSurface-TensorBGR8ToSurface.h         /E "TensorBGR8ToSurface"       /Vn g_csTensorBGR8ToSurface       TensorFloatToSurface.hlsl
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface-TensorRGB8ToSurface.h         /E "TensorRGB8ToSurface"       /Vn g_csTensorRGB8ToSurface       TensorFloatToSurface.hlsl
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface-TensorGRAY8ToSurface.h        /E "TensorGRAY8ToSurface"      /Vn g_csTensorGRAY8ToSurface      TensorFloatToSurface.hlsl
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface-TensorBGR8ToSurfaceGRAY8.h    /E "TensorBGR8ToSurfaceGRAY8"  /Vn g_csTensorBGR8ToSurfaceGRAY8  TensorFloatToSurface.hlsl
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface-TensorRGB8ToSurfaceGRAY8.h    /E "TensorRGB8ToSurfaceGRAY8"  /Vn g_csTensorRGB8ToSurfaceGRAY8  TensorFloatToSurface.hlsl
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface-TensorGRAY8ToSurfaceGRAY8.h   /E "TensorGRAY8ToSurfaceGRAY8" /Vn g_csTensorGRAY8ToSurfaceGRAY8 TensorFloatToSurface.hlsl

  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor16-SurfaceToTensorBGR8.h       /E "SurfaceToTensorBGR8"       /Vn g_csSurfaceToTensorBGR8       SurfaceToTensorFloat.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor16-SurfaceToTensorGRAY8.h      /E "SurfaceToTensorGRAY8"      /Vn g_csSurfaceToTensorGRAY8      SurfaceToTensorFloat.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor16-SurfaceToTensorRGB8.h       /E "SurfaceToTensorRGB8"       /Vn g_csSurfaceToTensorRGB8       SurfaceToTensorFloat.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor16-SurfaceGRAY8ToTensorBGR8.h  /E "SurfaceGRAY8ToTensorBGR8"  /Vn g_csSurfaceGRAY8ToTensorBGR8  SurfaceToTensorFloat.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  SurfaceToTensor16-SurfaceGRAY8ToTensorGRAY8.h /E "SurfaceGRAY8ToTensorGRAY8" /Vn g_csSurfaceGRAY8ToTensorGRAY8 SurfaceToTensorFloat.hlsl /DFP16

  fxc.exe /Tcs_5_0 /Fh  TensorToSurface16-TensorBGR8ToSurface.h       /E "TensorBGR8ToSurface"       /Vn g_csTensorBGR8ToSurface       TensorFloatToSurface.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface16-TensorRGB8ToSurface.h       /E "TensorRGB8ToSurface"       /Vn g_csTensorRGB8ToSurface       TensorFloatToSurface.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface16-TensorGRAY8ToSurface.h      /E "TensorGRAY8ToSurface"      /Vn g_csTensorGRAY8ToSurface      TensorFloatToSurface.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface16-TensorBGR8ToSurfaceGRAY8.h  /E "TensorBGR8ToSurfaceGRAY8"  /Vn g_csTensorBGR8ToSurfaceGRAY8  TensorFloatToSurface.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface16-TensorRGB8ToSurfaceGRAY8.h  /E "TensorRGB8ToSurfaceGRAY8"  /Vn g_csTensorRGB8ToSurfaceGRAY8  TensorFloatToSurface.hlsl /DFP16
  fxc.exe /Tcs_5_0 /Fh  TensorToSurface16-TensorGRAY8ToSurfaceGRAY8.h /E "TensorGRAY8ToSurfaceGRAY8" /Vn g_csTensorGRAY8ToSurfaceGRAY8 TensorFloatToSurface.hlsl /DFP16