#!/usr/bin/env python3
"""Setup script for Real-Time Voice Cloning."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="real-time-voice-cloning",
    version="1.0.0",
    author="Corentin Jemine",
    author_email="",
    description="Real-time voice cloning using SV2TTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CorentinJ/Real-Time-Voice-Cloning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "pytest>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rtvc-cli=demo_cli:main",
            "rtvc-toolbox=demo_toolbox:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
)