import setuptools

setuptools.setup(
    name="rec_probing",
    version="0.0.1",
    author="Gustavo Penha",
    author_email="guzpenha10@gmail.com",
    description="A python package for probing BERT models for recommendation knowledge.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)