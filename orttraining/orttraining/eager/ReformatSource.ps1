<#
.SYNOPSIS
Formats .h and .cpp files in current directory and below.

.DESCRIPTION
Formats .h and .cpp files in current directory and below. It will ensure trailing newline at EOF, and run
clang-format in-place.

This is very similar to onnxruntime/ReformatSource.ps1, but we want to tweak the rules on files in the eager
directory to include the column limit in the style. In the version of clang-format included with ubuntu 20.04,
clang-format does not support inheriting rules from another file, so we add the extra options to the clang-format
command.

Later versions of clang-format do support inheriting style from another .clang-format file. When the baseline
environment includes a version of clang-format that supports this functionality, we can refactor this.
#>


function EnsureTrailingEmptyLine {

    <#
    .SYNOPSIS
    Ensures trailing newline for file.

    .DESCRIPTION
    If file is missing trailing newline, will append newline to end of file
    #>
    param (
        $Path
    )

    $content = [System.IO.File]::ReadAllText($Path)

    if ($content[$content.Length - 1] -ne [Environment]::NewLine) {
        [System.IO.File]::AppendAllText($Path, [Environment]::NewLine)
    }
}

gci -Recurse -Include  *.h, *.cpp | foreach {
    Write-Host "Updating " $_.FullName
    EnsureTrailingEmptyLine $_
    clang-format -i --style="{ColumnLimit: 120, SpacesBeforeTrailingComments : 2, AccessModifierOffset: -1}" $_
}
