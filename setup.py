"""Setup for the chocobo package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Luca Moschella",
    author_email="nopaste94@gmail.com",
    name='wraplot',
    license="GPUv3",
    description='wraplot is a matplotlib wrap to easily create plots, subplots and animations',
    version='v0.0.1',
    long_description=README,
    url='https://github.com/LucaMoschella/wraplot',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)