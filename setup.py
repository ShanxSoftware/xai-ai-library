# setup.py in your AI library root
from setuptools import setup, find_packages

setup(
    name="xai-ai-library",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "xai-sdk>=1.1",
        "numpy>=2.3",
        "pydantic>=2.11",
        "sentence-transformers>=5.1",
        "pymupdf>=1.26",
        "tenacity>=9.1",
        "pyyaml>=6.0",
        "grpcio>=1.75",
        "grpcio-tools>=1.75"
    ],
)
