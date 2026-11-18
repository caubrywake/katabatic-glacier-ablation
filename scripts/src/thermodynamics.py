"""Thermodynamic helper functions ported from the legacy MATLAB workflow.

The goal of this module is first to reproduce the manuscript-era calculations
faithfully, while making units explicit and allowing revised choices later.
"""
from __future__ import annotations
import numpy as np

T0 = 273.15
E0_HPA = 6.113
LV = 2.5e6
LICE = 2.83e6
RV = 461.5


def saturation_vapor_pressure_hpa(T_k):
    """Legacy saturation vapour pressure used by satvaporpressure.m.

    Parameters
    ----------
    T_k : array-like
        Air temperature [K].

    Returns
    -------
    ndarray
        Saturation vapour pressure [hPa].

    Notes
    -----
    The MATLAB function switches to latent heat of sublimation below 273.15 K.
    """
    T = np.asarray(T_k, dtype=float)
    L = np.where(T < T0, LICE, LV)
    return E0_HPA * np.exp((L / RV) * (1.0 / T0 - 1.0 / T))


def pressure_from_altitude_hpa(altitude_m):
    """Standard-atmosphere pressure expression used in the MATLAB code [hPa]."""
    return 1013.25 * (1.0 - 2.25577e-5 * float(altitude_m)) ** 5.25588


def moist_air_density_kgm3(T_k, ea_hpa, p_hpa):
    """Moist-air density following the expression in the MATLAB functions."""
    T = np.asarray(T_k, dtype=float)
    ea = np.asarray(ea_hpa, dtype=float)
    p = np.asarray(p_hpa, dtype=float)
    Rd = 287.05
    Rv = 461.495
    return ((p - ea) * 100.0 / (Rd * T)) + (ea * 100.0 / (Rv * T))
