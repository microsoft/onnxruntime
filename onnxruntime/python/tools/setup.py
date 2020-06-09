from distutils.core import setup
from setuptools import find_packages

packages = find_packages()

setup(
    name='quantization',
    version='0.1',
    description="Converts Machine Learning models to ONNX for use in Windows ML",
    long_description='',
    long_description_content_type='text/markdown',
    license='MIT License',
    author='Microsoft Corporation',
    author_email='winmlcvt@microsoft.com',
    url='https://github.com/onnx/keras-onnx',
    packages=packages,
    include_package_data=True,
    install_requires=[],
    tests_require=['pytest', 'pytest-cov'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License']
)