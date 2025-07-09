from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Minimal setup.py for Cython extension with NumPy includes
# All other project configuration is in pyproject.toml
extensions = [
    Extension(
        "modules.color_ops",
        ["modules/color_ops.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)