from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="libtorch_agn_scalar",
    version="0.0",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            "libtorch_agn_scalar._C",
            [str(Path(__file__).parent / "csrc" / "scalar.cpp")],
            py_limited_api=True,
            extra_compile_args=["-DTORCH_TARGET_VERSION=0x0210000000000000"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
