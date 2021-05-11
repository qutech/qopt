# qopt Tests

The purpose of this file is to contain a structure for the test coverage of 
qopt classes.

## Unittests

- Matrix operations
- Propagator Calculation
- Transfer Functions
- Fidelity Functions
- Noise Distribution Sampling

## Integration Tests

### Tests with Analytical Results

- T2-star pure dephasing with quasi static noise
- T1 depolarization decay with white noise
- T2 non-Markovian literature analytic calculation
- T2 Markovian noise

### Tests with Numerical Results

- Test gradients vs. finite differences

### Consistency

- FF vs. Lindblad vs. MC for white noise
- FF vs MC for colored noise
- Use multiple optimization algorithms of simple problem to ensure convergence

### Testing per Module

To ensure coverage of all critical features, we also go through each module
and list its coverage in a test.

#### Amplitude Functions
Correct gradient calculation tested by finite differences.

#### Cost Functions
Correct gradient calculation tested by finite differences.


#### Matrix
Unittests for the calculation of Matrix exponential. And other methods.

#### Noise
Fast noise tested with periodogram. Slow noise with analytic pure dephasing.

#### Optimize
Use multiple optimization algorithms of simple problem to ensure convergence.

#### Simulator
Covered in all integration tests using also optimizers.

#### Solver Algorithms
Unittests for the correct calculation of Propagators. Gradient calculation 
checked for all classes with finite differences. Used in almost every 
integration test.

#### Transfer Functions
Correct gradient calculation tested by finite differences.


## Exempted from Testing

The less critical modules are not explicitly considered in the testing.

- data_container
- energy_spectrum
- optimization_data
- parallel
- performance_statistics
- plotting
- util