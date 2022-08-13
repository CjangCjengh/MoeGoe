import os
import platform
from distutils.core import Extension, setup

import numpy as np
from Cython.Build import cythonize

# make openmp available
system = platform.system()
if system == "Windows":
    extra_compile_args = []
    extra_link_args = ["/openmp"]
elif system == "Linux":
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]
elif system == "Darwin":
    os.system("brew install libomp")
    extra_compile_args = ["-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-L/usr/local/lib", "-lomp"]
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        name="monotonic_align.core",
        sources=["core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "cdivision": True,
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
