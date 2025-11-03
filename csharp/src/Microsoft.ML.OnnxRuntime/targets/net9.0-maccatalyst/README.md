### Notes for maccatalyst .NET targets:

We only add a blank file for the target framework folder here and thus will be including blank TFM under build/ and buildTransitive/ in the Nuget package. The reason is for Mac Catalyst platform, it directly will resolve the xcframework from the runtimes/native/ios folder based on this [RuntimeidentifierGraph](https://github.com/dotnet/sdk/blob/main/src/Layout/redist/PortableRuntimeIdentifierGraph.json#L300-L304)
