import numpy as np
from scipy.stats import norm
from typing import Tuple

from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from conditioning import get_cond_state
from solution import task2 as task2_solu


def get_conds(state: MultiVarGauss2d,
              sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
              sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
              ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    cond_state_c = get_cond_state(state, sens_model_c, meas_c)
    cond_state_r = get_cond_state(state, sens_model_r, meas_r)

    cond_c = MultiVarGauss2d(cond_state_c.mean, cond_state_c.cov)
    cond_r = MultiVarGauss2d(cond_state_r.mean, cond_state_r.cov)
 
    return cond_c, cond_r


def get_double_conds(state: MultiVarGauss2d,
                     sens_model_c: LinearSensorModel2d, meas_c: Measurement2d,
                     sens_model_r: LinearSensorModel2d, meas_r: Measurement2d
                     ) -> Tuple[MultiVarGauss2d, MultiVarGauss2d]:

    # p(x | z^c, z^r)       -       Camera, then lidar
    cond_state_c = get_cond_state(state, sens_model_c, meas_c)
    cond_state_cr = get_cond_state(cond_state_c, sens_model_r, meas_r)
    cond_cr = MultiVarGauss2d(cond_state_cr.mean, cond_state_cr.cov)

    # p(x | z^r, z^c)       -       Lidar, then camera
    cond_state_r = get_cond_state(state, sens_model_r, meas_r)
    cond_state_rc = get_cond_state(cond_state_r, sens_model_c, meas_c)
    cond_rc = MultiVarGauss2d(cond_state_rc.mean, cond_state_rc.cov)
    
    return cond_cr, cond_rc


def get_prob_over_line(gauss: MultiVarGauss2d) -> float:

    F = np.array([-1, 1])

    y = MultiVarGauss2d.get_transformed(gauss, F)
    y_mean = y.mean
    y_std = np.sqrt(y.cov)

    cdf_y = norm.cdf(5, y_mean, y_std)

    # since cdf gives Pr(y < 5) and we want Pr(y > 5), we get
    prob = 1 - cdf_y
    
    return prob
