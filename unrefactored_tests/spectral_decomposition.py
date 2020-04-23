from qutip.control_2.matrix import ControlDense

import numpy as np


sigma_x = ControlDense(np.asarray([[0, 1], [1, 0]]))
sigma_z = ControlDense(np.asarray([[1, 0], [0, -1]]))

a_x = 1
a_z = 1e-5
m = a_x * sigma_x + a_z * sigma_z

exp_fr = (1j * m).exp(
    tau=1,
    method='Frechet'
)

exp_sp = (1j * m).exp(
    tau=1,
    method='spectral',
    is_skew_hermitian=True
)
