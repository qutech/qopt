# qopt: A Simulation and Quantum Optimal Control Package
[![Build Status](https://github.com/qutech/qopt/actions/workflows/python-test.yml/badge.svg)](https://github.com/qutech/qopt/actions/workflows/python-test.yml)
[![Documentation Status](https://img.shields.io/readthedocs/qopt)](https://qopt.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/qopt)](https://pypi.org/project/qopt/)
[![License](https://img.shields.io/github/license/qutech/qopt)](https://github.com/qutech/qopt/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/qutech/qopt/branch/master/graph/badge.svg)](https://app.codecov.io/gh/qutech/qopt)

## Documentation
The documentation can be found on 
[readthedocs](https://qopt.readthedocs.io/en/latest/index.html). 
It features an API documentation and an introduction in the 
form of jupyter notebooks demonstrating how to utilize the package. A 
complementary theoretical introduction is given in the
qopt paper on 
[Phys. Rev. Applied](https://doi.org/10.1103/PhysRevApplied.17.034036) and  an 
older version can be found on the 
[Arxiv](https://arxiv.org/abs/2110.05873).

## Applications

We set up another open-source repository named 
[qopt-applications](https://github.com/qutech/qopt-applications) to save and 
exchange quantum simulation and optimal control applications implemented using
qopt.

## Introduction
Realistic modelling of qubit systems including noise and constraints imposed
by control hardware is required for performance prediction and control
optimization of quantum processors.
qopt is a software framework for simulating qubit dynamics and
robust quantum optimal control considering  common experimental situations.
It supports modelling of open and closed qubit systems with a focus on the
simulation of realistic noise characteristics and experimental constraints.
Specifically, the influence of noise can be calculated using Monte Carlo
methods, effective master equations or with the filter function formalism,
enabling the investigation and mitigation of auto-correlated noise. In
addition, limitations of control electronics including finite bandwidth
effects can be considered. The calculation of gradients based on analytic
results is implemented to facilitate the efficient optimization of control
pulses. The software is published under an open source license, well-tested
and features a detailed documentation.

## Installation

Qopt is available on github and the python index Pypi.
To install qopt directly from the python index, you can use pip: 

    pip install qopt

or alternatively download the source code, navigate to the folder containing
qopt and install by

    pip install . 

or append the command -e to install qopt with symlinks

    pip -e install . 

The -e stands for edible as the symlinks allow you to make local changes to
the sourcecode.

### Optional packages

If you wish to use the plotting features of the quantum toolbox in pythen 
(QuTiP), then you need to install additional dependencies:

    conda install cython pytest pytest-cov jupyter

Then open a conda forge channel:

    conda config --append channels conda-forge
    
and install QuTiP:

    conda install qutip

Another optional package is simanneal for the use of simulated annealing for
discrete optimization:

    conda install simanneal

## Feature Requests

If you require an additional feature for your work, then please open an issue
on github or reach out to me via e-mail j.teske@fz-juelich.de.
There is a list in markdown format with possible extensions to the package.

## Patch Notes

You can find the patch Notes in a markdown file in the root folder of the 
package. You can also find it on 
[github](https://github.com/qutech/qopt/blob/master/patch_notes.md).

## Citing

If you are using qopt for your work then please cite the 
[qopt paper](https://doi.org/10.1103/PhysRevApplied.17.034036), as the funding 
of the development depends on the public impact.
