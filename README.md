# qopt: A Simulation and Quantum Optimal Control Package

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
In current quantum computer prototypes information is stored an processed in 
quantum bits or qubits which are controlled by electric pulses. In order to 
find the optimal control pulse for a given operation, this package simulates 
the system under control and applies optimization algorithms to the pulses. 
These include gradient based algorithms generalizing the GRAPE algorithm [1].

The package sets a focus on realistic noise models to enable noise mitigation
through pulse tailoring. Imperfections of the control electronics can also be
included in the simulations.

## Installation
The recommended way is to use conda for the installation.
To avoid difficulties, QuTiP needs to be installed first. To do so, follow 
[their instructions](http://qutip.org/docs/latest/installation.html) or these
instructions. Usually it is most convenient to create a new environment. The 
package was written and tested using python 3.7.

    conda create --name qopt_env python=3.7
    conda activate qopt_env

Start with all required dependencies including 
[filter_functions package](https://github.com/qutech/filter_functions): 

    conda install numpy scipy matplotlib
    pip install filter_functions

### Optional packages

If you wish to use the plotting features of QuTiP, then install additionally:

    conda install cython pytest pytest-cov jupyter

Then open a conda forge channel:

    conda config --append channels conda-forge
    
and install QuTiP:

    conda install qutip

Another optional package is simanneal for the use of simulated annealing for
discrete optimization:

    conda install simanneal

### qopt installation
    
Either install qopt via pip 

    pip install qopt

or alternatively download the source code and use
`python setup.py develop` to install using symlinks or 
`python setup.py install` without.

## Feature Requests

If you require an additional feature for your work, then please open an issue
on github or reach out to me via e-mail j.teske@fz-juelich.de.
There is a list in markdown format with possible extensions in the package.

## Patch Notes

You can find the patch Notes in a markdown list in the package. Please be aware
that the github repo is updated more frequently than the version on pypi.

## Citing

If you are using qopt for your work then please cite the 
[qopt paper](https://arxiv.org/abs/2110.05873), as the funding of the 
development depends on the public impact.

## References
[1]: Khaneja, N., Reiss, T., Kehlet, C., Schulte-Herbrüggen, T., Glaser, S.
(2004). Optimal control of coupled spin dynamics: design of NMR pulse sequences
gy gradient ascent algorithms 
[https://doi.org/10.1016/j.jmr.2004.11.004](https://doi.org/10.1016/j.jmr.2004.11.004)