#
# Compiler options
#

option(${PROJECT_NAME}_WARNINGS_AS_ERRORS "Treat compiler warnings as errors."
       OFF)

#
# Static analyzers
#
# Currently supporting: Clang-Tidy, Cppcheck.

option(${PROJECT_NAME}_ENABLE_CLANG_TIDY
       "Enable static analysis with Clang-Tidy." OFF)
option(${PROJECT_NAME}_ENABLE_CPPCHECK "Enable static analysis with Cppcheck."
       OFF)

#
# Code coverage
#

option(${PROJECT_NAME}_ENABLE_CODE_COVERAGE "Enable code coverage through GCC."
       OFF)

#
# Miscelanious options
#

# Generate compile_commands.json for clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
  ${PROJECT_NAME}_VERBOSE_OUTPUT
  "Enable verbose output, allowing for a better understanding of each step taken."
  ON)

option(${PROJECT_NAME}_ENABLE_CCACHE
       "Enable the usage of CCache, in order to speed up build times." ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()
