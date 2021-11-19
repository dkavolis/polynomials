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
from typing import Any, Callable, List, Optional, Tuple
from typing_extensions import ParamSpec
from pkg_resources import VersionConflict, require
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from mypy import stubgen
import pathlib
import shutil
import functools


class CMakeExtension(Extension):
    def __init__(self, name: str):
        Extension.__init__(self, name, sources=[])


P = ParamSpec("P")


def path_str(path: pathlib.Path) -> str:
    return str(path).replace("\\", "/")


def stringify(f: Callable[P, Optional[pathlib.Path]]) -> Callable[P, str]:
    @functools.wraps(f)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> str:
        path = f(*args, **kwargs)

        if path is None:
            return ""

        return path_str(path)

    return wrapped


class CMakeBuild(build_ext):
    user_options: List[Tuple[str, Any, str]] = build_ext.user_options
    user_options.extend(
        [
            ("cmake-options=", None, "Additional options to pass to cmake"),
            ("vcpkg_dir", None, "Path to vcpkg"),
            ("vcpkg_triplet", None, "Triplet to use with vcpkg"),
            ("cxx_compiler", None, "Compiler to use for building the extension"),
            ("cmake_generator", None, "CMake generator to use"),
        ]
    )

    def initialize_options(self) -> None:
        super().initialize_options()
        self.cmake_options = ""
        self.vcpkg_dir = ""
        self.vcpkg_triplet = ""
        self.cxx_compiler = ""
        self.cmake_generator = ""

    @staticmethod
    def architecture() -> str:
        import ctypes

        ptr_size = ctypes.sizeof(ctypes.c_void_p)

        if ptr_size == 8:
            return "x64"
        elif ptr_size == 4:
            return "x86"
        raise RuntimeError(f"Unsupported architecture with pointer size of {ptr_size}")

    @staticmethod
    def _search_msvc(pattern: str) -> Optional[pathlib.Path]:
        search_paths = [
            "C:/Program Files/Microsoft Visual Studio",
            "C:/Program Files (x86)/Microsoft Visual Studio",
        ]

        for path in search_paths:
            for compiler in pathlib.Path(path).glob(pattern):
                return compiler

        return None

    @staticmethod
    def msvc_compiler() -> Optional[pathlib.Path]:
        cl_pattern = "**/VC/Tools/MSVC/*/bin/Host{0}/{0}/cl.exe".format(
            CMakeBuild.architecture()
        )

        return CMakeBuild._search_msvc(cl_pattern)

    @staticmethod
    def msvc_vars_script() -> Optional[pathlib.Path]:
        return CMakeBuild._search_msvc("**/VC/Auxiliary/Build/vcvarsall.bat")

    @staticmethod
    def _which(name: str) -> Optional[pathlib.Path]:
        path = shutil.which(name)
        if path is None:
            return None

        return pathlib.Path(path)

    @staticmethod
    def gcc_compiler() -> Optional[pathlib.Path]:
        return CMakeBuild._which("g++")

    @staticmethod
    def clang_compiler() -> Optional[pathlib.Path]:
        return CMakeBuild._which("clang++")

    @staticmethod
    def clang_cl_compiler() -> Optional[pathlib.Path]:
        path = pathlib.Path("C:/Program Files/LLVM/bin/clang-cl.exe")

        if not path.exists():
            return None

        return path

    def _building_msvc(self) -> bool:
        import platform

        py_compiler = platform.python_compiler()
        return py_compiler.startswith("MSC")

    @stringify
    def compiler_path(self) -> Optional[pathlib.Path]:
        if self.cxx_compiler:
            return pathlib.Path(self.cxx_compiler)

        if self._building_msvc():
            # python compiled with MSVC compiler, use windows compilers
            compiler = self.msvc_compiler()
            if compiler is not None:
                return compiler

            return self.clang_cl_compiler()

        compiler = self.gcc_compiler()
        if compiler is not None:
            return compiler

        return self.clang_compiler()

    @stringify
    def cmake_executable(self) -> Optional[pathlib.Path]:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        return self._which("cmake")

    def run(self) -> None:
        cmake_executable = self.cmake_executable()

        extension_name: str = self.extensions[0].name
        extdir = (
            pathlib.Path(self.get_ext_fullpath(extension_name))  # type: ignore
            .absolute()
            .parent
        )
        if self.debug:
            cfg = "Debug"
            print("Building C++ extensions in debug mode")
        else:
            cfg = "Release"

        cmake_args: List[str] = []

        if self.cmake_generator:
            cmake_args.extend(("-G", self.cmake_generator))

        cmake_args.extend(
            [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH={extdir}",
                f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
                "-DCMAKE_BUILD_TYPE:STRING=" + cfg,
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}:PATH={extdir}",
            ]
        )
        if self.cmake_options:
            cmake_args.extend(self.cmake_options.split(" "))

        if self.vcpkg_dir:
            # cmake doesn't like toolchain quoted file paths for some reason...
            cmake_args.append(
                f"-DCMAKE_TOOLCHAIN_FILE:FILEPATH={self.vcpkg_dir}"
                "/scripts/buildsystems/vcpkg.cmake"
            )
            if self.vcpkg_triplet:
                cmake_args.append(f"-DVCPKG_TARGET_TRIPLET:STRING={self.vcpkg_triplet}")

        compiler = self.compiler_path()
        if compiler is not None:
            cmake_args.append(f"-DCMAKE_CXX_COMPILER:FILEPATH={compiler}")
        else:
            raise ValueError("Could not find an existing compiler")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = path_str(pathlib.Path(__file__).parent)
        msvc_vars = self.msvc_vars_script()
        print("-" * 10, "Running CMake prepare", "-" * 40)
        cmake_cmd = [cmake_executable, cmake_list_dir] + cmake_args
        if msvc_vars is not None:
            cmake_cmd[:0] = [path_str(msvc_vars), self.architecture(), "&&"]
        cmake_cmd += ["&&", cmake_executable, "--build", ".", "--config", cfg]
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)

        sys.path.append(str(extdir))
        options = stubgen.parse_options(
            [
                "-o",
                str(extdir),
                "-p",
                extension_name.split("/")[-1],
                "--include-private",
                "-v",
            ]
        )
        stubgen.generate_stubs(options)


try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        ext_modules=[CMakeExtension("polynomials/polynomials_cpp")],
        cmdclass={"build_ext": CMakeBuild},
    )
