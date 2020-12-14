<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net5.0-windows10.0.19041.0</TargetFramework>
    <Platforms>AnyCPU;x64</Platforms>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.AI.MachineLearning" Version="[PackageVersion]" />
  </ItemGroup>

  <ItemGroup>
    <None Include="..\..\testdata\squeezenet.onnx">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <Visible>true</Visible>
    </None>
    <None Include="..\..\..\winml\test\collateral\images\kitten_224.png">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <Visible>true</Visible>
    </None>
  </ItemGroup>

</Project>
