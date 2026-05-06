"""Minimal setup.py to produce a platform-specific wheel."""

from setuptools import setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel


class PlatformBdistWheel(bdist_wheel):
    """Override wheel tags to py3-none-{platform}."""

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformBdistWheel})
