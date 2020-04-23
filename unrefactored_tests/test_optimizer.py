import qutip.control_2.optimize
import qutip.control.optimresult
import scipy.optimize
import numpy as np
import unittest


class OptimizationTest(unittest.TestCase):
    def test_trf_rosenbock(self):
        optim = qutip.control_2.optimize.OptimizerLeastSquaresOld(error=scipy.optimize.rosen, x0=np.arange(4))
        optim.approx_grad = True
        optim.termination_conditions['min_gradient_norm'] = 1e-8
        result = qutip.control.optimresult.OptimResult()
        optim.run_optimization(result)

        assert result.goal_achieved
        self.assertAlmostEqual(float(result.final_cost), 0, places=5)
        self.assertAlmostEqual(float(result.final_cost), scipy.optimize.rosen(result.final_x))
        if result.grad_norm_min_reached:
            self.assertLess(result.grad_norm_final, optim.termination_conditions['min_gradient_norm'] * 10)

    def test_termination_by_wall_time(self):
        optim = qutip.control_2.optimize.OptimizerLeastSquaresOld(error=scipy.optimize.rosen, x0=np.arange(4))
        optim.approx_grad = True
        optim.termination_conditions["max_wall_time"] = .1
        result = qutip.control.optimresult.OptimResult()
        optim.run_optimization(result)

        self.assertEqual('max_wall_time', result.termination_reason)

