import setuptools

setuptools.setup(
    name="list_wise_reformer",
    version="0.0.1",
    author="Gustavo Penha",
    author_email="guzpenha10@gmail.com",
    description="A python package for learning to rank with pairwise and listwise reformer.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)