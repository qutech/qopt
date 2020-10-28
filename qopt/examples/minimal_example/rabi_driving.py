from qopt import *
import numpy


noise_gen = NTGQuasiStatic(
    n_samples_per_trace=20,
    n_traces=10,
    standard_deviation=[.5, ]
)


solver = SchroedingerSMonteCarlo(
    h_ctrl=[.5 * DenseOperator.pauli_x(),
            .5 * DenseOperator.pauli_y()],
    h_drift=[0 * DenseOperator.pauli_x()],
    tau=.25 * numpy.ones(20),
    h_noise=[.5 * DenseOperator.pauli_z()],
    noise_trace_generator=noise_gen
)

cost_fkt = OperationNoiseInfidelity(
    solver=solver,
    target=(DenseOperator.pauli_x()).exp(.25j * numpy.pi),
    neglect_systematic_errors=False
)

optimizer = ScalarMinimizingOptimizer(
    system_simulator=Simulator(
        solvers=[solver, ],
        cost_fktns=[cost_fkt, ]
    ),
    bounds=[[-.5 * numpy.pi, .5 * numpy.pi], ] * 40
)

numpy.random.seed(0)
data = run_optimization_parallel(optimizer, numpy.random.rand(10, 20, 2))
analyser = Analyser(data)
analyser.plot_absolute_costs()
