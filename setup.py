from setuptools import setup, find_packages

setup(
    name="hutils",
    packages=find_packages(),
    author="胡玮文",
    author_email="huww98@outlook.com",
    url="https://github.com/huww98/hutils",
    description="Utilities for machine learning",
    long_description="",
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "jsonnet",
    ]
)
