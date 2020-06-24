<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <Import Project="packages\microsoft.windows.cppwinrt\2.0.200615.7\build\native\Microsoft.Windows.CppWinRT.targets" Condition="Exists('packages\microsoft.windows.cppwinrt\2.0.200615.7\build\native\Microsoft.Windows.CppWinRT.targets')"/>
  <Import Project="packages\microsoft.ai.machinelearning\[PackageVersion]\build\native\Microsoft.AI.MachineLearning.targets" Condition="Exists('packages\microsoft.ai.machinelearning\[PackageVersion]\build\native\Microsoft.AI.MachineLearning.targets')"/>
</Project>
