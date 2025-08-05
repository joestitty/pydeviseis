
#!/usr/bin/env python3
from setuptools import setup

# List every standalone .py (without “.py” extension) you want at top level:
PY_MODULES = [
    "__pyDaskObject",
    "__pyDaskOperator",
    "__pyDaskVector",
    "_pylops_interface",
    "dask_util",
    "pyADMMsolver",
    "pyCuOperator",
    "pyCuVector",
    "pyDaskOperator",
    "pyDaskProblem",
    "pyDaskVector",
    "pyLinearSolver",
    "pyNonLinearSolver",
    "pyNpOperator",
    "pyOperator",
    "pyParOperator",
    "pyParquetOperator",
    "pyParquetVector",
    "pyProblem",
    "pyProblemConstrained",
    "pyProxOperator",
    "pySolver",
    "pySolverConstrained",
    "pySparseSolver",
    "pyStepper",
    "pyStopper",
    "pyVector",
    "sep_util",
    "sys_util",
]

setup(
    name="GenericSolver",
    version="0.1.0",
    description="Pysolver core modules (operators, vectors, solvers, utils)",
    packages=["GenericSolver"],            # installs the GenericSolver package
    package_dir={"GenericSolver": ""},   # that package is this folder
    py_modules=PY_MODULES,               # installs all these as top-level modules
    install_requires=[
        "numpy>=1.18.0",
        # add any other runtime dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)