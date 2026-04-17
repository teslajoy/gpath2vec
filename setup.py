from setuptools import setup, find_packages

__version__ = "2.0.0"

setup(
    name="gpath2vec",
    version=__version__,
    packages=find_packages(),
    description="gene-set to biological pathway embeddings with enrichment analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/teslajoy/gpath2vec",
    author="Nasim Sanati",
    author_email="nasim@plenary.org",
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "networkx",
        "requests",
        "click",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "gpath2vec=gpath2vec.cli:main",
        ],
    },
    keywords="bioinformatics pathways embeddings enrichment-analysis metapath2vec reactome",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/teslajoy/gpath2vec/issues",
        "Source": "https://github.com/teslajoy/gpath2vec",
    },
)
