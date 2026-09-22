import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu


def get_init_CV_state(meas0: Measurement2d, meas1: Measurement2d,
                      ekf_params: EKFParams) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt
    z0, z1 = meas0.value, meas1.value
    sigma_a = ekf_params.sigma_a    # process
    sigma_z = ekf_params.sigma_z    # Measure

    R = sigma_z**2 * np.eye(2)
    Q = sigma_a**2 * np.block([
        [dt**3/3 * np.eye(2), dt**2/2 * np.eye(2)],
        [dt**2/2 * np.eye(2), dt * np.eye(2)]
    ])      # eq. 4.64 in book

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
                 
    F = np.block([[np.eye(2), dt * np.eye(2)],
                  [np.zeros((2,2)), np.eye(2)]])
    F_inv = np.linalg.inv(F)

    mean = np.array([*z1, *(z1 - z0)/dt])  # TODO
    cov11 = R
    cov12 = R/dt
    cov21 = R/dt
    cov22 = (2*R + H @ F_inv @ Q @ F_inv.T @ H.T)/(dt**2)
    cov = np.block([[cov11, cov12],
                   [cov21, cov22]])  # TODO

    init_state = MultiVarGauss(mean, cov)

    return init_state
