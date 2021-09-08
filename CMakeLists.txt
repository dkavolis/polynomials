cmake_minimum_required(VERSION 3.16)
project(
  polynomials
  VERSION 0.1.0
  LANGUAGES CXX)

#
# Set project options
#

include(cmake/StandardSettings.cmake)
include(cmake/Utils.cmake)
message("\nStarted CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")

#
# Prevent building in the source directory
#

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

find_package(Python 3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)

add_library(polynomials OBJECT src/polynomials/polynomials.cpp)
target_compile_definitions(polynomials PUBLIC CMAKE_CXX_STANDARD=20)
target_include_directories(polynomials PUBLIC src/polynomials/include
                                              ${Boost_INCLUDE_DIR})
target_compile_definitions(polynomials PUBLIC POLY_HEADER_ONLY=0)

#
# Set the project standard and warnings
#

target_compile_features(polynomials PUBLIC cxx_std_20)

include(cmake/CompilerWarnings.cmake)
target_set_warnings(polynomials)

pybind11_add_module(polynomials_cpp src/polynomials/polynomials_cpp.cpp)
target_link_libraries(polynomials_cpp PRIVATE polynomials)

#
# Format the project using the `clang-format` target (i.e: cmake --build build
# --target clang-format)
#

add_clang_format_target()

include(cmake/StaticAnalyzers.cmake)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
