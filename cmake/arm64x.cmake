set(arm64ReproDir "${CMAKE_SOURCE_DIR}/repros")

if("${BUILD_AS_ARM64X}" STREQUAL "ARM64")
	foreach (n ${ARM64X_TARGETS})
		add_custom_target(mkdirs_${n} ALL COMMAND cmd /c (if exist \"${arm64ReproDir}/${n}_temp/\" rmdir /s /q \"${arm64ReproDir}/${n}_temp\") && mkdir \"${arm64ReproDir}/${n}_temp\" )
		add_dependencies(${n} mkdirs_${n})
		target_link_options(${n} PRIVATE "/LINKREPRO:${arm64ReproDir}/${n}_temp")
		add_custom_target(${n}_checkRepro ALL COMMAND cmd /c if exist \"${n}_temp/*.obj\" if exist \"${n}\" rmdir /s /q \"${n}\" 2>nul && if not exist \"${n}\" ren \"${n}_temp\" \"${n}\" DEPENDS ${n}
		WORKING_DIRECTORY ${arm64ReproDir})
	endforeach()


elseif("${BUILD_AS_ARM64X}" STREQUAL "ARM64EC")
	foreach (n ${ARM64X_TARGETS})
		set(ARM64_LIBS)
		set(ARM64_OBJS)
		set(ARM64_DEF)

		file(GLOB ARM64_OBJS "${arm64ReproDir}/${n}/*.obj")
		file(GLOB ARM64_DEF "${arm64ReproDir}/${n}/*.def")
		file(GLOB ARM64_LIBS "${arm64ReproDir}/${n}/*.LIB")

		if(NOT "${ARM64_DEF}" STREQUAL "")
			set(ARM64_DEF "/defArm64Native:${ARM64_DEF}")
		endif()
		target_sources(${n} PRIVATE ${ARM64_OBJS})
		target_link_options(${n} PRIVATE /machine:arm64x "${ARM64_DEF}")

		if(NOT "${ARM64_LIBS}" STREQUAL "")
			target_link_libraries(${n} PUBLIC ${ARM64_LIBS})
		endif()
	endforeach()
endif()
