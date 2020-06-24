# Feature Requests

Since this version covers by no means all possible features a quantum optimal
control package can have, this document gathers features requested by 
(potential) users. If require a new feature urgently, you can reach out to me
via email: julian.teske@rwth-aachen.de

## Transfer Function

### Convolution as transfer function
Increases the number of computable time steps.

## Solver Algorithms

### Control noise in the Lindblad master equation
Automatically estimate the lindblad operators when handling noise on the control
hamiltonian. Also provide a way to automate the calculation of derivatives from
the resulting amplitudes.


### Parallelization by noise traces 
Multi processing to decrease run time.

### Propagate only initial State
To use the package only for simulation or for the optimization of a state
fidelity.

### Handle the time steps
Avoid setting time steps twice. i. e. for the solver and the transfer function.

## Cost function

### State fidelity
