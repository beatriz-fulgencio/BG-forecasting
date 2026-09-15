from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bg-forecasting-benchmark",
    version="0.1.0",
    author="Blood Glucose Forecasting Research Group",
    author_email="beatrizfulgencio03@gmail.com",
    description="A reproducible benchmark framework for blood glucose forecasting models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beatriz-fulgencio/BG-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    package_data={
        "benchmark": ["configs/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "bg-forecast=benchmark.cli:main",
        ],
    },
)
