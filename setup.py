#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
    Setup file for polynomials.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

import os
import subprocess
import sys
from pkg_resources import VersionConflict, require
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    user_options = build_ext.user_options
    user_options.extend(
        [
            ("cmake-options=", None, "Additional options to pass to cmake"),
            ("vcpkg_dir", None, "Path to vcpkg"),
            ("vcpkg_triplet", None, "Triplet to use with vcpkg"),
            ("cxx_compiler", None, "Compiler to use for building the extension"),
            ("cmake_generator", None, "CMake generator to use"),
        ]
    )

    def initialize_options(self):
        super().initialize_options()
        self.cmake_options = ""
        self.vcpkg_dir = ""
        self.vcpkg_triplet = ""
        self.cxx_compiler = ""
        self.cmake_generator = ""

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(self.extensions[0].name))
        )
        if self.debug:
            cfg = "Debug"
            print("Building C++ extensions in debug mode")
        else:
            cfg = "Release"

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=" + cfg,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
        ]
        if self.cmake_options:
            cmake_args.extend(self.cmake_options.split(" "))

        if self.vcpkg_dir:
            cmake_args.append(
                "-DCMAKE_TOOLCHAIN_FILE={}/scripts/buildsystems/vcpkg.cmake".format(
                    self.vcpkg_dir
                )
            )
            if self.vcpkg_triplet:
                cmake_args.append(f"-DVCPKG_TARGET_TRIPLET={self.vcpkg_triplet}")

        if self.cxx_compiler:
            cmake_args.append(f"-DCMAKE_CXX_COMPILER={self.cxx_compiler}")
        if self.cmake_generator:
            cmake_args.extend(("-G", self.cmake_generator))

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print("-" * 10, "Running CMake prepare", "-" * 40)
        subprocess.check_call(
            ["cmake", cmake_list_dir] + cmake_args, cwd=self.build_temp
        )

        print("-" * 10, "Building extensions", "-" * 40)
        cmake_cmd = ["cmake", "--build", ".", "--config", cfg]
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)


try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        ext_modules=[CMakeExtension("polynomials/boost")],
        cmdclass=dict(build_ext=CMakeBuild),
    )
