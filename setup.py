from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name="gpath2vec",
    version=__version__,
    packages=find_packages(),
    description="Pathway to embedding vectors",
    long_description=open('README.md').read(),
    url='https://github.com/teslajoy/gpath2vec',
    author='Nasim Sanati',
    install_requires=[
        "torch",
        "networkx",
        "requests",
        "click",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'gpath2vec=gpath2vec.cli:main',
        ],
    },
    author_email="nasim@plenary.org",
    keywords="bioinformatics, pathways, embeddings, enrichment-analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
