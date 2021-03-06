# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import tempfile

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

__all__ = ["build_ext"]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        f.flush()
        try:
            obj = compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
        if not os.path.exists(obj[0]):
            return False
    return True

def has_library(compiler, libname):
    """Return a boolean indicating whether a library is found."""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as srcfile:
        srcfile.write("int main (int argc, char **argv) { return 0; }")
        srcfile.flush()
        outfn = srcfile.name + ".so"
        try:
            compiler.link_executable(
                [srcfile.name],
                outfn,
                libraries=[libname],
            )
        except setuptools.distutils.errors.LinkError:
            return False
        if not os.path.exists(outfn):
            return False
        os.remove(outfn)
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError("Unsupported compiler -- at least C++11 support "
                           "is needed!")

class build_ext(_build_ext):
    """
    A custom extension builder that finds the include directories for Eigen
    before compiling.

    """

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    def build_extensions(self):
        # Add the numpy and pybind11 include directories
        import numpy
        import pybind11
        include_dirs = [
            numpy.get_include(),
            pybind11.get_include(False),
            pybind11.get_include(True),
        ]

        # Find FFTW headers
        dirs = include_dirs + self.compiler.include_dirs
        for ext in self.extensions:
            dirs += ext.include_dirs
        dirs += [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
                sys.executable))), "include")]
        print(dirs)
        found_fftw = False
        for d in dirs:
            if os.path.exists(os.path.join(d, "fftw3.h")):
                print("found 'fftw3' in '{0}'".format(d))
                include_dirs += [d]
                found_fftw = True
                break
        if not found_fftw:
            raise RuntimeError("could not find the required library 'fftw3'")

        for ext in self.extensions:
            ext.include_dirs += include_dirs

        # Set up pybind11
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append("-DVERSION_INFO=\"{0:s}\""
                        .format(self.distribution.get_version()))

            print("testing C++14/C++11 support")
            opts.append(cpp_flag(self.compiler))

            libraries = ["fftw3", "m", "stdc++", "c++"]

            # Check for OpenMP support first.
            if has_flag(self.compiler, "-fopenmp"):
                print("found omp...")
                libraries += ["gomp", "fftw3_threads", "fftw3_omp"]

            # Add the libraries
            print("checking libraries...")
            libraries = [lib for lib in libraries
                         if has_library(self.compiler, lib)]
            print("libraries: {0}".format(libraries))
            for ext in self.extensions:
                ext.libraries += libraries

            flags = ["-O3", "-Ofast", "-stdlib=libc++", "-fvisibility=hidden",
                     "-Wno-unused-function", "-Wno-uninitialized",
                     "-Wno-unused-local-typedefs", "-funroll-loops",
                     "-fopenmp"]

            # Mac specific flags and libraries
            if sys.platform == "darwin":
                flags += ["-march=native", "-mmacosx-version-min=10.9"]
                for ext in self.extensions:
                    ext.extra_link_args += ["-mmacosx-version-min=10.9",
                                            "-march=native"]

            # Check the flags
            print("testing compiler flags")
            for flag in flags:
                if has_flag(self.compiler, flag):
                    opts.append(flag)

        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"{0:s}\\"'
                        .format(self.distribution.get_version()))
        for ext in self.extensions:
            ext.extra_compile_args += opts

        # Run the standard build procedure.
        _build_ext.build_extensions(self)
