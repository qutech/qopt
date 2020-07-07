# qopt: A Simulation and Quantum Optimal Control Package

## Version
This is the open alpha version. The name might be changed in the future. The
contend will be extended based on feedback concerning the needs and demands of
(possible) users. If you miss an important feature, you can request it by 
opening a issue in the git repository or write me directly at 
j.teske@fz-juelich.de.

## Introduction
In current quantum computer prototypes information is stored an processed in 
quantum bits or qubits which are controlled by electric pulses. In order to 
find the optimal control pulse for a given operation, this package simulates 
the system under control and applies optimization algorithms to the pulses. 
These include gradient based algorithms generalizing the GRAPE algorithm [1].

The package sets a focus on realistic noise models to enable noise mitigation
through pulse tailoring. Imperfections of the control electronics can also be
included in the simulations.

The implementation was inspired by the optimal control package of 
[QuTiP](http://qutip.org/).

## Installation
To avoid difficulties, QuTiP needs to be installed first. To do so, follow 
[their instructions](http://qutip.org/docs/latest/installation.html) or these
instructions. Start with all recommended dependencies: 

    conda install numpy scipy cython matplotlib pytest pytest-cov jupyter

Then open a conda forge channel:

    conda config --append channels conda-forge
    
and install QuTiP:

    conda install qutip

Then install the 
[filter_functions package](https://github.com/qutech/filter_functions) via pip:

    pip install filter_functions 
 
And two remaining dependencies by:

    conda install pandas simanneal
    pip install filter_functions

Then download the source code and use
`python setup.py develop` to install using symlinks or 
`python setup.py install` without.

## Documentation
The documentation can be found on 
[readthedocs](https://qopt.readthedocs.io/en/latest/index.html). 
It features an API documentation and an introduction in the 
form of jupyter notebooks demonstrating how to utilize the package.

## References
[1]: Khaneja, N., Reiss, T., Kehlet, C., Schulte-Herbrüggen, T., Glaser, S.
(2004). Optimal control of coupled spin dynamics: design of NMR pulse sequences
gy gradient ascent algorithms 
[https://doi.org/10.1016/j.jmr.2004.11.004](https://doi.org/10.1016/j.jmr.2004.11.004)