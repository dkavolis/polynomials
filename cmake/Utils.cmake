#
# Print a message only if the `VERBOSE_OUTPUT` option is on
#

function(verbose_message content)
    if(${PROJECT_NAME}_VERBOSE_OUTPUT)
        message(${MESSAGE_TYPE} ${content})
    endif()
endfunction()

#
# Add a target for formating the project using `clang-format` (i.e: cmake --build build --target clang-format)
#

function(add_clang_format_target)
    if(NOT ${PROJECT_NAME}_CLANG_FORMAT_BINARY)
        find_program(${PROJECT_NAME}_CLANG_FORMAT_BINARY clang-format)
    endif()
    if(${PROJECT_NAME}_CLANG_FORMAT_BINARY)
        file(GLOB_RECURSE sources ${CMAKE_SOURCE_DIR}/src/**/*.cpp)
        file(GLOB_RECURSE headers ${CMAKE_SOURCE_DIR}/src/**/*.hpp)
        add_custom_target(clang-format
                COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY} -i ${sources} ${headers})
        message("Format the project using the `clang-format` target (i.e: cmake --build build --target clang-format).\n")
    endif()
endfunction()
