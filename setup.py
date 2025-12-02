import glob, os
from setuptools import setup, Extension
from Cython.Build import cythonize

pyx_files = glob.glob("sea/**/*.pyx", recursive=True)  # include subdirs

extensions = [
    Extension(
        ".".join(f.split(os.sep)[:-1] + [os.path.splitext(os.path.basename(f))[0]]),
        [f],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
    for f in pyx_files
]
print("Building extensions:", [ext.name for ext in extensions])

setup(
    name="sea",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        build_dir="build",
        compiler_directives={"binding": True},
    ),
)
