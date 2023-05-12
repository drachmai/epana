from setuptools import setup, find_packages

setup(
    name="epana",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1",
        "huggingface-hub==0.14.1",
        "transformers==4.29.0",
    ],
)