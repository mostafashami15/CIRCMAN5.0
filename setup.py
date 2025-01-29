from setuptools import setup, find_packages

setup(
    name="circman5",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn>=1.6.1",
        "pandas>=2.0.0",
        "numpy>=1.21.0",
    ],
)
