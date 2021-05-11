# Feature Requests

Since this version covers by no means all possible features a quantum optimal
control package can have, this document gathers features requested by 
(potential) users. If require a new feature urgently, you can reach out to me
via email: julian.teske@rwth-aachen.de

## Transfer Function

### Convolution as transfer function

## Solver Algorithms

### Control noise in the Lindblad master equation
Automatically estimate the lindblad operators when handling noise on the control
hamiltonian. Also provide a way to automate the calculation of derivatives from
the resulting amplitudes.

### Propagate only initial State
To use the package only for simulation or for the optimization of a state
fidelity.


### Dynamically change between lab and rotating frame
(priority low) In the special case of Rabi driving change between two frames 
connected by a unitary evolution. (i.e. combine single gates in rot frame with
two qubit gates in the lab frame.)

## Cost function

### State fidelity
