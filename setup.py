from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name="pathway2vec",
    version=__version__,
    packages=find_packages(),
    description="Pathway to embedding vectors",
    long_description=open('README.md').read(),
    url='https://github.com/teslajoy/pathway2vec',
    author='Nasim Sanati',
    install_requires=[
        "torch",
        "networkx",
        "requests",
        "click"
    ],
    entry_points={
        'console_scripts': [
            'pathway2vec=pathway2vec.cli:main',
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