import qutip.control_2.rabi_driving.setup as rabi
from qutip.control_2.tslotcomp import TSCompSaveAll
from qutip.control_2.matrix import ControlDense
from qutip.control_2.cost_functions import OperationInfidelity, \
    OperationNoiseInfidelity
from qutip.control_2.dynamics import Dynamics
import numpy as np

ntg = rabi.ntg_quasi_static

noise_samples = ntg.noise_samples[0, :, 0]

t_slot_comps_drift = []
for drift_level in noise_samples:
    t_slot_comps_drift.append(TSCompSaveAll(
        h_drift=[drift_level * rabi.h_drift, ] * rabi.n_time_samples
        * rabi.oversampling,
        h_ctrl=rabi.h_ctrl,
        initial_state=ControlDense(np.eye(2)),
        tau=[rabi.time_step / rabi.oversampling, ] * rabi.n_time_samples
        * rabi.oversampling,
        is_skew_hermitian=True,
        exponential_method='spectral',
        transfer_function=rabi.exponential_transfer_function,
        amplitude_function=rabi.lin_amp_func
    ))

amp_bound = rabi.rabi_frequency_max / rabi.lin_freq_rel
bounds_xy = [-amp_bound, amp_bound]
np.random.seed(0)
initial_pulse = amp_bound * (2 * np.random.rand(rabi.n_time_samples, 2) - 1)

tslot_noise = rabi.time_slot_comp_qs_noise_xy
"""
tslot_noise.set_optimization_parameters(initial_pulse)
for tslot_comp in t_slot_comps_drift:
    tslot_comp.set_optimization_parameters(initial_pulse)

grad_noise = tslot_noise.frechet_deriv_propagators[0][10].data
grad_drifts = []
for tslot_comp in t_slot_comps_drift:
    grad_drifts.append(tslot_comp.frechet_deriv_propagators[0][10].data)
grad_drift = np.mean(grad_drifts, axis=0)
"""

ent_fids = []
for i in range(len(noise_samples)):
    ent_fids.append(OperationInfidelity(
        t_slot_comp=t_slot_comps_drift[i],
        target=rabi.x_half,
        fidelity_measure='entanglement',
        index=['Entanglement Fidelity XY-Control']
    ))

entanglement_infid_qs_noise_xy = OperationNoiseInfidelity(
    t_slot_comp=tslot_noise,
    target=rabi.x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity QS-Noise XY-Control'],
    neglect_systematic_errors=True
)

dynamics = Dynamics(
    tslot_comps=t_slot_comps_drift + [tslot_noise],
    cost_fktns=ent_fids + [entanglement_infid_qs_noise_xy, ]
)

dynamics_only_noise = Dynamics(
    tslot_comps=[tslot_noise, ],
    cost_fktns=[entanglement_infid_qs_noise_xy]
)

num_grad = dynamics.numeric_gradient(initial_pulse)
ana_grad = dynamics.wrapped_jac_function(initial_pulse)

# The gradients sum up correctly if neglect systematic errors = False
np.mean(ana_grad[:, :3, :], axis=1) - ana_grad[:, 3, :]
np.mean(num_grad[:, :3, :], axis=1) - num_grad[:, 3, :]


(num_grad - ana_grad)[:, 3, :]

