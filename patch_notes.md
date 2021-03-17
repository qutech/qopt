### Version 0.1 to 1.0

Parallelization
- Monte Carlo Simulations can be run in parallel
- Optimization with multiple seeds can be run in parallel

Filter Function Derivatives
- New Filter function cost function can be used with analytical gradients
for the optimization

Transfer Function
- New Base class MatrixTF to distinguish between transfer functions implemented
as matrix multiplication and other transfer functions
- implementation of gaussian convolution as transfer function

Solver Algorithms
- the times are now set automatically to the transfer functions. The Solver
must now be instantiated with the untransferred times
- drift Hamiltonians can be set to constant my setting only a single element
otherwise you need one element for each transferred time step.

Optimizer
- scalar optimization algorithms available
- gradient free nelder mead algorithm available
- cost function weights must now be given in the optimizer class

Matrix
- implementation of the division by scalar values

Documentation:
- Rabi Driving in detail as example
- Noise notebooks describing T1, T2-star and T2 in the case of Markoian
and non-Markovian noise
