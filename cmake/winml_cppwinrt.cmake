# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script adds cppwinrt support for VS-generated projects.
#
# target_cppwinrt(foo bar.idl)
#
# Calling target_midl function runs midlrt.exe and produces bar.h
# Calling target_cppwinrt function does two things:
#
# 1) Adds a target "bar.cppwinrt", which performs the midl and cppwinrt
#    builds and produces:
#       bar.h
#       bar.winmd
#       bar.tlb
#       module.g.cpp
#
# 2) Adds a dependency to the new custom target "bar.cppwinrt"

function(target_midl
    target_name
    idl_file
    output_name       # output name of the generated headers, winmd and tlb
    sdk_folder        # sdk kit directory
    sdk_version       # sdk version
    folder_name
    midl_options      # defines for the midl compiler
    )
    if (MSVC)
        # get sdk include paths for midl
        get_sdk_include_folder(${sdk_folder} ${sdk_version} sdk_include_folder)
        set(um_sdk_directory "${sdk_include_folder}/um")
        set(shared_sdk_directory "${sdk_include_folder}/shared")
        set(winrt_sdk_directory "${sdk_include_folder}/winrt")

        # get sdk metadata path
        get_sdk_metadata_folder(${sdk_folder} ${sdk_version} sdk_metadata_directory_forward_slashes)
        convert_forward_slashes_to_back(${sdk_metadata_directory_forward_slashes} sdk_metadata_directory)

        # get midl
        get_sdk_midl_exe(${sdk_folder} ${sdk_version} midl_exe)

        # Filename variables
        set(header_filename ${output_name}.h)
        convert_forward_slashes_to_back(${idl_file} idl_file_forward_slash)

        # using add_custom_command trick to prevent rerunning script unless ${file} is changed
        add_custom_command(
            OUTPUT ${header_filename}
            COMMAND ${midl_exe}
                /metadata_dir ${sdk_metadata_directory}
                /W1 /char signed /nologo /winrt
                /no_settings_comment /no_def_idir /target "NT60"
                /I ${um_sdk_directory}
                /I ${shared_sdk_directory}
                /I ${winrt_sdk_directory}
                /I ${CMAKE_CURRENT_SOURCE_DIR}
                /h ${header_filename}
                ${midl_options}
                ${idl_file_forward_slash}
            DEPENDS ${idl_file}
        )

        add_custom_target(
            ${target_name}
            ALL
            DEPENDS ${header_filename}
        )

        set_target_properties(${target_name} PROPERTIES FOLDER ${folder_name})
    endif()
endfunction()

function(target_cppwinrt
    target_name          # the name of the target to add
    file                 # name of the idl file to compile
    output_name          # output name of the generated headers, winmd and tlb
    out_sources_folder   # path where generated sources will be placed
    sdk_folder           # sdk kit directory
    sdk_version          # sdk version
    folder_name          # folder this target will be placed
    midl_options         # defines for the midl compiler
    set_ns_prefix        # set ns_prefix option
    add_ref              # set additional cppwinrt ref path
)
    if (MSVC)
        # get sdk include paths for midl
        get_sdk_include_folder(${sdk_folder} ${sdk_version} sdk_include_folder)
        set(um_sdk_directory "${sdk_include_folder}/um")
        set(shared_sdk_directory "${sdk_include_folder}/shared")
        set(winrt_sdk_directory "${sdk_include_folder}/winrt")

        # get sdk metadata path
        get_sdk_metadata_folder(${sdk_folder} ${sdk_version} sdk_metadata_directory_forward_slashes)
        convert_forward_slashes_to_back(${sdk_metadata_directory_forward_slashes} sdk_metadata_directory)

        # get midl
        get_sdk_midl_exe(${sdk_folder} ${sdk_version} midl_exe)

        # get cppwinrt
        get_sdk_cppwinrt_exe(${sdk_folder} ${sdk_version} cppwinrt_exe)

        # Filename variables
        set(header_filename ${output_name}.h)
        set(winmd_filename ${output_name}.winmd)
        set(tlb_filename ${output_name}.tlb)

        # Get directory
        get_filename_component(idl_source_directory ${file} DIRECTORY)

        if (NOT "${add_ref}" STREQUAL "")
            convert_forward_slashes_to_back(${add_ref} add_ref)
        endif()

        set(target_outputs ${CMAKE_CURRENT_BINARY_DIR}/${target_name})
        convert_forward_slashes_to_back(${target_outputs}/comp output_dir_back_slash)
        convert_forward_slashes_to_back(${target_outputs}/temp temp_dir_back_slash)
        convert_forward_slashes_to_back(${target_outputs}/comp_generated generated_dir_back_slash)
        convert_forward_slashes_to_back(${generated_dir_back_slash}/module.g.cpp module_g_cpp_back_slash)
        convert_forward_slashes_to_back(${generated_dir_back_slash}/module.g.excl.cpp module_g_ecxl_cpp_back_slash)
        if (set_ns_prefix)
          set(ns_prefix "/ns_prefix")
        else()
          set(ns_prefix "")
        endif()

        # Get name
        set(renamed_idl_filename ${output_name}.idl)
        set(renamed_idl_fullpath ${target_outputs}/${renamed_idl_filename})

        get_filename_component(idl_source_filename ${file} NAME)
        set(copied_idl_fullpath ${target_outputs}/${idl_source_filename})

        file(COPY ${file} DESTINATION ${target_outputs})
        file(RENAME ${copied_idl_fullpath} ${renamed_idl_fullpath})

        convert_forward_slashes_to_back(${renamed_idl_fullpath} renamed_idl_fullpath_back_slash)

        # using add_custom_command trick to prevent rerunning script unless ${file} is changed
        add_custom_command(
            OUTPUT ${header_filename} ${winmd_filename}
            DEPENDS ${file}
            COMMAND ${midl_exe}
                /metadata_dir ${sdk_metadata_directory}
                /W1 /char signed /nomidl /nologo /winrt
                /no_settings_comment /no_def_idir /target "NT60"
                /I ${idl_source_directory}
                /I ${um_sdk_directory}
                /I ${shared_sdk_directory}
                /I ${winrt_sdk_directory}
                /winmd ${winmd_filename}
                ${ns_prefix}
                /h ${header_filename}
                /tlb ${tlb_filename}
                ${midl_options}
                ${renamed_idl_fullpath_back_slash}
            COMMAND
                    ${cppwinrt_exe} -in ${winmd_filename} -comp ${output_dir_back_slash} -ref ${sdk_metadata_directory} ${add_ref} -out ${generated_dir_back_slash} -verbose
            COMMAND
                    # copy the generated component files into a temporary directory where headers exclusions will be applied
                    xcopy ${output_dir_back_slash} ${temp_dir_back_slash}\\ /Y /D
            COMMAND
                    # for each file in the temp directory, ensure it is not in the exclusions list.
                    # if it is, then we need to delete it.
                    cmd /C "@echo off \
                    for /f %I in ('dir /b ${temp_dir_back_slash}') \
                    do \
                    ( \
                        for /f %E in (${CPPWINRT_COMPONENT_EXCLUSION_LIST}) \
                        do \
                        ( \
                            if %E == %I \
                            ( \
                                del ${temp_dir_back_slash}\\%I \
                            ) \
                        ) \
                    )"
            COMMAND
                    # for each file in the temp directory, copy the file back into the source tree
                    # unless the file already exists
                    cmd /C "@echo off \
                    for /f %I in ('dir /b ${temp_dir_back_slash}') \
                    do \
                    ( \
                        if not exist ${out_sources_folder}\\%I \
                        ( \
                            copy ${temp_dir_back_slash}\\%I ${out_sources_folder}\\%I \
                        ) \
                    )"
            COMMAND
                    # open the generated module.g.cpp and strip all the includes (lines) containing excluded headers
                    # write the new file out to module.g.excl.cpp.
                    powershell -Command "& { \
                        $exclusions = get-content '${CPPWINRT_COMPONENT_EXCLUSION_LIST}'; \
                        (get-content '${module_g_cpp_back_slash}') \
                        | where { \
                            $str = $_; \
                            $matches = ($exclusions | where { $str -match $_ }); \
                            $matches.Length -eq 0 } \
                        | Out-File '${module_g_ecxl_cpp_back_slash}' \
                    }"
            BYPRODUCTS
                    ${generated_dir_back_slash}/module.g.excl.cpp
            VERBATIM
        )

        add_custom_target(
            ${target_name}
            ALL
            DEPENDS ${header_filename} ${winmd_filename}
        )

        set_target_properties(${target_name} PROPERTIES FOLDER ${folder_name})
    endif()
endfunction()

function(add_generate_cppwinrt_sdk_headers_target
    target_name     # the name of the target to add
    sdk_folder      # sdk kit directory
    sdk_version     # sdk version
    sdk_directory   # the name of the folder to output the sdk headers to
    folder_name     # folder this target will be placed
)
    if (MSVC)
        # get the current nuget sdk's metadata directory
        get_sdk_metadata_folder(${sdk_folder} ${sdk_version} metadata_folder)

        # get cppwinrt
        get_sdk_cppwinrt_exe(${sdk_folder} ${sdk_version} cppwinrt_exe)

        # windows.winmd is consumed by cppwinrt to produce the sdk headers
        set(windows_winmd "${metadata_folder}/windows.winmd")

        # base.h along with the other winrt sdk headers are produced by this command
        set(base_h "${sdk_directory}/winrt/base.h")

        # using add_custom_command trick to prevent rerunning script unless ${windows_winmd} is changed
        add_custom_command(
            OUTPUT ${base_h}
            DEPENDS ${windows_winmd}
            COMMAND ${cppwinrt_exe} -in \"${metadata_folder}\" -out \"${sdk_directory}\" -verbose
        )

        # add the target
        add_custom_target(${target_name} ALL DEPENDS ${base_h})

        set_target_properties(${target_name} PROPERTIES FOLDER ${folder_name})
    endif()
endfunction()