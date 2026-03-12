#!/usr/bin/env python3
"""
Setup configuration for AeroGraphRX package.
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="aerographrx",
    version="1.0.0",
    description="AeroGraphRX: Simulation of a GSP Framework for Cooperative Multi-Station Signal Detection and Tracking",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Usha A, Noel George",
    author_email="usha@allianceuniversity.edu.in, noel@allianceuniversity.edu.in",
    url="https://github.com/aerographrx/aerographrx",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Signal Processing",
    ],
)
