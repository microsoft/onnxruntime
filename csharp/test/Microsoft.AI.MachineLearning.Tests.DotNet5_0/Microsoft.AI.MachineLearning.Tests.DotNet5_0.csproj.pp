<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net5.0-windows10.0.19041.0</TargetFramework>
    <Platforms>x86;x64</Platforms>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.AI.MachineLearning" Version="[PackageVersion]" />
    <PackageReference Include="Microsoft.Windows.CsWinRT" Version="1.5.0" />
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

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="16.7.1" />
    <PackageReference Include="MSTest.TestAdapter" Version="2.1.1" />
    <PackageReference Include="MSTest.TestFramework" Version="2.1.1" />
    <PackageReference Include="coverlet.collector" Version="1.3.0" />
  </ItemGroup>
</Project>
