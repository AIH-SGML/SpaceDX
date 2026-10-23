from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="spacedx",
    version="0.1.0",
    description="Spatial differential expression in spatial transcriptomics data",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Niklas Stotzem & Francesco Paolo Casale",
    license="MIT",
    packages=find_packages(include=["spacedx", "spacedx.*"]),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "anndata==0.10.9",
        "matplotlib==3.8.0",
        "numpy>=1.24,<2.0",
        "pandas>=2.0,<3.0",
        "pytorch_minimize==0.0.2",
        "scikit_learn==1.3.0",
        "scipy==1.11.4",
        "seaborn==0.13.2",
        "torch==2.1.0",
        "tqdm==4.65.0",
    ],
    include_package_data=True,
)