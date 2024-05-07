.. qopt documentation master file, created by
   sphinx-quickstart on Tue Apr 28 12:03:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

qopt
====

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Contents:


Welcome to ``qopt``. In this documentation you will find everything you need to
know about qopt to simulate qubits and apply optimal control techniques. You
can find the source code and install instructions on
https://github.com/qutech/qopt.

A complementary publication about the software can be found at Phys. Rev.
Applied https://doi.org/10.1103/PhysRevApplied.17.034036 and an older version
on the Arxiv
https://arxiv.org/abs/2110.05873. This paper gives a sound introduction to the
topic of qubit simulation and quantum optimal control. It also provides a
systematic theoretical introduction of simulation methods to give a profound
understanding of each method's capabilities.

Abstract
========

Realistic modelling of qubit systems including noise and constraints imposed
by control hardware is required for performance prediction and control
optimization of quantum processors.
We introduce qopt, a software framework for simulating qubit dynamics and
robust quantum optimal control considering  common experimental situations.
To this end, we model open and closed qubit systems with a focus on the
simulation of realistic noise characteristics and experimental constraints.
Specifically, the influence of noise can be calculated using Monte Carlo
methods, effective master equations or with the filter function formalism,
enabling the investigation and mitigation of auto-correlated noise. In
addition, limitations of control electronics including finite bandwidth
effects can be considered. The calculation of gradients based on analytic
results is implemented to facilitate the efficient optimization of control
pulses. The software is published under an open source license, well-tested
and features a detailed documentation.

Summary
=======

This python package is designed to facilitate the simulation of
state-of-the-art quantum bits (qubits) including realistic experimental
operation conditions, and for the optimization of noise-robust control pulses.
For this purpose, an extensive set of noise simulation tools is
included and complemented by methods to describe the limitations posed by the
electronics steering the quantum operations. Compared to other available
simulation packages qopt stands out by the ability to efficiently simulate the
effects of fast Non-Markovian noise, while providing a general interface to
define control pulses for any qubit type making qopt platform-independent.
The simulation interfaces to optimization algorithms to apply
optimal quantum control theory, the field of study which optimizes the
accuracy of quantum operations by intelligent steering methods.

The functionalities can be coarsely divided into simulation and optimization of
quantum operations. Various cost functions can be evaluated on the simulated
evolution of the qubits such as an error rate, a gate or state fidelity or a
leakage rate. Since gradient-based optimization algorithms perform
extremely well in minimization problems, we implemented the derivatives of the
cost functions by the optimization parameters based on analytical calculations.

Simulation
----------

The qopt package simulates closed or open quantum systems on a pulse level by
solving the corresponding partial differential equations being the Schroedinger
equation or a Lindblad master equation.
Thereby, pulses are discretized in time and the differential equations are
solved using matrix exponentials. The total propagator is then available for
every time step to analyse the dynamics of the qubit system.


Noise
-----

The realistic simulation of noise is one of qopt's key features. The various
methods are therefore mentioned in more detail, and in a brief overview is
given stating the advantages and requirements of each method.

**Monte Carlo Simulations**

The most forward way to simulate noise is to draw samples from the noise
distribution and repeat the simulation for each of those noise realizations.
Any cost function is then averaged over the repetitions.
The sampling can be based on pseudo random number generators.
Monte Carlo simulations are universally applicable but computationally
expensive for high-frequency noise.


**Lindblad Master Equation**

In order to include dissipation effects in the simulation, the qubit and its
environment can be modeled as open quantum system, described by a master
equation in Lindblad form. The solution of the master equation is in
the general case not unitary unlike the propagators calculated from
Schroedinger's equation, such that it can also describe the loss of energy or
information into the environment. This approach is numerically efficient but
only applicable to systems subjected to Markovian noise.

**Filter Functions**

The filter function formalism is a mathematical approach which allows the
estimation of fidelities in the presence of universal classical noise. It is
numerically very efficient for low numbers of qubits and widely applicable.
This package interfaces to the open source
filter function package (https://github.com/qutech/filter_functions)
written by Tobias Hangleiter.

**Leakage**

Leakage occurs when the qubit leaves the computational space spanned by the
computational states. To take this kind of error into
account, the Hilbert space must be expanded as vector space sum by the leakage
levels. The simulation is then performed on the larger Hilbert space and
needs to be truncated to the computational states for evaluation. The Leakage
rate or transition rate into the leakage states can be used to quantify the
error rate caused to leakage.

Pulse Parametrization
---------------------

The pulse parameterization translates a mathematically described pulse function
into discrete-time control amplitudes that appear in the Hamiltonian describing
the qubit model. This comprises sampling a potentially continuously defined
control pulse, evaluating the physical function that describes the relation
between pulse values and the control amplitudes, and including the hardware
limitations of the control electronics that generate the control pulse.


**Amplitude Functions**

The amplitude functions encode a differential relationship between the
optimization parameters, which describe the pulse, and the control amplitudes,
which appear in the Hamiltonian and describe the dynamics of the quantum
system. An example would be a sinusoidal pulse that drives resonant excitations
of a qubit. The optimization parameters would be the pulse length, the pulse
frequency and the amplitude. The amplitude function samples the continuous
pulse and maps the voltage values to energy values from the Hamiltonian,
which are the control amplitudes in this example.


**Transfer Functions**

To model realistic control electronics the package includes transfer functions
mapping the ideal pulse to the actual provided voltages. This can include
for example exponential saturation to consider finite voltage rise times in
pulse generators, Gaussian smoothing of pulses to mimic bandwidth limitations
on arbitrary waveform generators, linear transformations or even
the measured response of an arbitrary waveform generator to a set of input
voltages. The transfer functions then map the ideal pulse to the actually
generated pulse.

Optimization
------------

To leverage a given noisy quantum computer to its full potential, optimal
control techniques can be applied to mitigate the detrimental effects of noise.
The package allows the use of different optimization algorithms by a strong
modularity in the implementation.

**Analytical Derivatives**

Gradient based optimization algorithms such as GRAPE have proven to be
versatile and reliable for the application in pulse optimization. For the
efficient calculation of gradients, the package implements analytical
derivatives for the solution of the Schroedinger equation, the master
equation in Lindblad form and all calculations used to estimate fidelities.


Documentation
=============

The documentation is structured in the three parts 'Features',
'Example Applications' and the 'API Documentation'.

**qopt Features**

The first part introduces the qopt functionalities step by step. Refer to this
chapter for an introduction to the simulation package.

**Example Applications**

The 'Example
Applications' combines an educational introduction to physical phenomena with
simulation techniques and theoretical background information. They demonstrate
how the package is used and treat FAQs on the way. They can also serve as
blueprints for applications.

**API Documentation**

You can find the full API Documentation in the last section. Each class is
implemented in a single module and each module is a subsection in the
auto-generated API documentation. These subsections start with a general
description of the purpose of the respective class. If you want to gain a quick
overview of the class structure, I advise you to read through these
descriptions. During the implementation of a simulation using qopt, you can
frequently jump to the classes and functions your are using to look up the
signatures.


.. toctree::
   :maxdepth: 2
   :numbered:

   qopt_features/qopt_features
   examples/examples
   qopt API Documentation <qopt>

Citing
======

If you are using qopt for your work then please cite the
[qopt paper](https://doi.org/10.1103/PhysRevApplied.17.034036), as the funding
of the development depends on the public impact.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
