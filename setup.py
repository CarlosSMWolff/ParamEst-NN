# Setup file for the paramest_nn package.

import setuptools

# setup details
setuptools.setup(
    name="paramest_nn",
    version="0.1",
    author="Carlos Sánchez Muñoz",
    description="A package to study quantum parameter estimation using neural networks",
    packages=setuptools.find_packages(where='src'),
    install_requires=[
        "tensorflow",
        "matplotlib",
        "numpy",
        "jupyter",
        "qutip",
        "pandas",
        "tqdm",
        "scipy",
        "scikit-learn",
        "black",
        "seaborn",
        "cython==0.29.36",
        "sbi",
        "fire",
        "ultranest",
    ],
    python_requires=">=3.9",
)