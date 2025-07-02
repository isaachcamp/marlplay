from setuptools import setup, find_packages

setup(
    name="twoStwoR",
    version="0.1.0",
    description="MARL environment for Two Species, Two Resources (2S2R) problem",
    author="Isaac Campbell",
    author_email="isaac.campbell@wolfson.ox.ac.uk",
    packages=find_packages(),
    install_requires=[
        "ipykernel>=6.29.5",
        "jax>=0.6.2",
        "jax-dataclasses>=1.6.2",
        "tensorflow-probability>=0.25.0",
    ],
    python_requires=">=3.12",
    url="https://github.com/isaachcamp/marlplay.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
    ],
)