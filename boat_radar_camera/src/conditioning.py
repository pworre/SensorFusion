import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:
    pred_meas = sens_modl.get_pred_meas(state)  # TODO
    kalman_gain = state.cov @ sens_modl.H.T @ np.linalg.inv(pred_meas.cov)  # TODO
    innovation = meas.value - pred_meas.mean  # TODO
    cond_mean = state.mean + kalman_gain @ innovation  # TODO    - "Posterier = conditional"
    cond_cov = (np.eye(2) - kalman_gain @ sens_modl.H) @ state.cov  # TODO

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)
    
    return cond_state
