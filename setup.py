from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nasa-exoplanet-detection-ai",
    version="1.0.0",
    author="Mohsen Keshavarzian",
    author_email="Kermkoosh@gmail.com",
    description="Advanced Machine Learning system for automated exoplanet detection using NASA's space mission data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aiolearn04/Galactic-Vanguard.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "exoplanet-ai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.pkl", "*.json", "*.png", "*.jpg", "*.jpeg"],
    },
)
