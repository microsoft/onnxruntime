"""Minimal setup.py to mark the wheel as platform-specific (non-pure).

pyproject.toml alone cannot express the non-pure wheel requirement, so this
companion setup.py defines a BinaryDistribution that ensures the wheel gets
the correct platform tag (e.g., win_amd64, manylinux_x86_64) instead of
py3-none-any.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
