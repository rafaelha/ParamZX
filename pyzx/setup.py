from setuptools import setup, find_packages

setup(
    name="pyzx",
    version="0.7.3",
    description="Python library for quantum circuit rewriting and optimisation using the ZX-calculus",
    author="Aleks Kissinger and John van de Wetering",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "sympy",
        "lark",
    ],
    python_requires=">=3.10",
)




