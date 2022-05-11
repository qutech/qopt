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

### Version 1.0 to 1.1

Cost Functions
- refactoring of the angle axis representation

Matrix
- implements the division by scalar by div operator

Documentation
- Extends the documentation by the notebook series examples

### Version 1.1 to 1.2

General:
- Updates in the README, including paper reference and bug fixes
- changed the URL from git-ce to github

Imports
- reduce the list of required imports. QuTiP and simanneal can only be used
if they were installed.
  
Documentation
- Improvement of the docstrings at module level and for the feature notebooks.

Transfer Function:
- adds the custom ConvolutionTF

OperatorMatrix:
- adds a function to vectorize density matrices
- Adds the calculation of the partial trace

CostFunction:
- Implements leakage and entanglement fidelity with truncation to computational
states in Liouville space

Optimizer:
- improve storage. The optimizer is only stored in the result on request.

### Version 1.2 to 1.3

GaussianMTF:
- made the deprecation explicit

Transfer Function:
- new internal check function has more explicit error messages than previous
  assertions.

Energy Spectrum:
- plotting reworked to be applicable to a larger number of dimensions in the 
  Hamiltonian
- Plotting function returns the figure and axis.

Solver Algorithm:
- the filter_function_h_n and noise_coeffs_derivatives change their signature, 
now then are called with the optimization parameters, the transferred 
parameters and the control amplitudes. Previously only with the control 
amplitudes.

Read the docs:
- Add a new notebook about the basic use of filter functions in optimal control
- Rework the notebook about the numerics. Now focused on the matrix class and 
put at the start.
- Add a new notebook about the use of filter functions in the optimization of 
amplitude, frequency and phase noise.

Optimizer:
- Fix a false dimension check for the cost function weights.
