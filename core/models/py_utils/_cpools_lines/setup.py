from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="cpools_lines",
    ext_modules=[
        CppExtension("horizontal_line_pool", ["src/horizontal_line_pool.cpp"]),
        CppExtension("vertical_line_pool", ["src/vertical_line_pool.cpp"])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
