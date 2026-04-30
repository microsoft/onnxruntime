"""Minimal setup.py to produce a platform-specific wheel.

The package contains pre-built native libraries (not CPython extension modules),
so the wheel tag should be py3-none-{platform} rather than cp3XX-cp3XX-{platform}.
This means a single wheel works across all supported Python versions.
"""

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution


class PlatformBdistWheel(bdist_wheel):
    """Override wheel tags to py3-none-{platform}."""

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformBdistWheel})
