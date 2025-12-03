import sys
import os
import importlib

SETUP_COMMAND = "poetry run python3 setup.py build_ext"
MODULE_NAME = "sea"
LIB = "../build/lib.linux-x86_64-cpython-312"
FILE_EXT = ".cpython-312-x86_64-linux-gnu.so"
MODULES = [
    "util.memory",
    "util.fast_stemmer",
    "util.disk_array",
    "tokenizer",
    "corpus",
    "indexer",
]

ret = os.system(SETUP_COMMAND)
if ret != 0:
    raise RuntimeError("Failed to build Cython extensions.")

current_dir = os.path.dirname(__file__)
build_sea_dir = os.path.abspath(os.path.join(current_dir, f"{LIB}"))

for module_name in MODULES:
    module_path = module_name.replace(".", "/")
    so_file = os.path.join(build_sea_dir, MODULE_NAME, f"{module_path}{FILE_EXT}")
    if not os.path.isfile(so_file):
        raise FileNotFoundError(f"Compiled module not found: {so_file}")
    spec = importlib.util.spec_from_file_location(f"{MODULE_NAME}.{module_name}", so_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{MODULE_NAME}.{module_name}"] = module
    spec.loader.exec_module(module)
