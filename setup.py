#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='DICpy',
    version="0.0.1",
    url='https://github.com/eesd-epfl/DICpy',
    description="PyCrack is a Python toolbox for analysis and quantification of cracks structural systems using optical methods.",
    author="Ketson R. M. dos Santos",
    author_email="ketson.santos@epfl.ch",
    license='MIT',
    platforms=["OSX", "Windows", "Linux"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.pdf"]},
    install_requires=[
        "numpy", "scipy", "matplotlib", "scikit-learn", "scikit-image"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
)
