"""SM80 entry point: applies tilelang fallbacks, then runs reference generate.py."""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # generate.py/model.py/kernel.py live here
PATCHES = "/work/patches" if os.path.isdir("/work/patches") else os.path.abspath(
    os.path.join(HERE, "..", "..", "patches"))
sys.path.insert(0, PATCHES)
ENCODING = "/ref/encoding" if os.path.isdir("/ref/encoding") else os.path.abspath(
    os.path.join(HERE, "..", "encoding"))
sys.path.insert(0, ENCODING)

import sm80_fallbacks  # noqa: E402

# Patch any already-imported kernel module; model.py will import patched names.
sm80_fallbacks.apply()

# generate.py lives in the same dir; argv passes through.
sys.argv = [os.path.join(HERE, "generate.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
