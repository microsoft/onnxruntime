#!/bin/bash

export DevKit=$(realpath ../nuget_root/BrainSlice.v3.DevKit.Release.3.0.0/)
export PkgBond_Cpp=$(realpath ../nuget_root/Bond.Cpp.7.0.5/)

echo "exporting DevKit='$DevKit'"
echo "exporting PkgBond_Cpp='$PkgBond_Cpp'"
