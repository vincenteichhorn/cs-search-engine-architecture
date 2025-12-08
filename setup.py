import glob, os
from setuptools import setup, Extension
from Cython.Build import cythonize

os.makedirs("vendor", exist_ok=True)
if not os.path.exists("vendor/libstemmer_c"):
    os.system("git clone https://github.com/indexdata/libstemmer_c vendor/libstemmer_c")

pyx_files = glob.glob("sea/**/*.pyx", recursive=True)  # include subdirs
fast_stemmer_path = ""
for f in pyx_files:
    if "fast_stemmer" in f:
        fast_stemmer_path = f
        break
lib_stemmer_sources = [
    "vendor/libstemmer_c/libstemmer/libstemmer_utf8.c",
    "vendor/libstemmer_c/runtime/api.c",
    "vendor/libstemmer_c/runtime/utilities.c",
]
lib_stemmer_sources += glob.glob("vendor/libstemmer_c/src_c/stem_UTF_8_*.c")
print("Libstemmer sources:", lib_stemmer_sources)
extensions = []
extensions.append(
    Extension(
        ".".join(
            fast_stemmer_path.split(os.sep)[:-1]
            + [os.path.splitext(os.path.basename(fast_stemmer_path))[0]]
        ),
        [fast_stemmer_path] + lib_stemmer_sources,
        include_dirs=[
            "vendor/libstemmer_c/include",
            "vendor/libstemmer_c/libstemmer",
            "vendor/libstemmer_c/libstemmer/src_c",
        ],
        language="c",
        # extra_compile_args=["-std=c++11"],
    )
)

extensions.extend(
    [
        Extension(
            ".".join(f.split(os.sep)[:-1] + [os.path.splitext(os.path.basename(f))[0]]),
            [f],
            include_dirs=["vendor/libstemmer_c/include"],
            language="c++",
            extra_compile_args=["-std=c++11"],
        )
        for f in pyx_files
        if not "stemmer" in f
    ]
)
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
