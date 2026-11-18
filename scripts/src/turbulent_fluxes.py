"""Bulk-transfer and katabatic turbulent-flux parameterizations.

These functions are deliberately close to the legacy MATLAB implementation.
The baseline configuration is intended to reproduce the current manuscript
before any reviewer-motivated changes are introduced.
"""
from __future__ import annotations

import numpy as np
from .thermodynamics import (
    saturation_vapor_pressure_hpa,
    pressure_from_altitude_hpa,
    moist_air_density_kgm3,
)

TS = 273.15
KAPPA = 0.4
CP = 1006.0
G = 9.8
LV = 2.5e6
EPS = 0.622
Z0M_DEFAULT = 0.00241


def stability_correction(zeta):
    """Port of stabilitycorrection.m (Dyer 1974 / Beljaars-Holtslag branch)."""
    z = float(zeta)
    if z < 0:
        x = (1.0 - 16.0 * z) ** 0.25
        psi_m = np.log(((1.0 + x*x) / 2.0) * ((1.0 + x) / 2.0) ** 2) - 2.0*np.arctan(x) + np.pi/2.0
        psi_h = 2.0 * np.log((1.0 + x*x) / 2.0)
    else:
        a, b, c, d = 1.0, 0.667, 5.0, 0.35
        psi_m = -(a*z) + b*(z - c/d)*np.exp(-d*z) + (b*c/d)
        psi_h = -(1.0 + (2.0/3.0)*z) ** 1.5 + b*(z - c/d)*np.exp(-d*z) + (b*c/d) - 1.0
    return psi_m, psi_h


def bulk_transfer(
    T_k,
    U_ms,
    RH_pct,
    altitude_m,
    z_m,
    z0m=Z0M_DEFAULT,
    max_iter=50,
    convergence_m=0.03,
):
    """Legacy bulk-transfer calculation.

    Returns a dict of numpy arrays: CH, CE, rho, ea, zeta, ustar, Qh, Qe.
    """
    T = np.asarray(T_k, dtype=float)
    U = np.asarray(U_ms, dtype=float)
    RH = np.asarray(RH_pct, dtype=float)
    if not (T.shape == U.shape == RH.shape):
        raise ValueError("T, U and RH must have matching shapes.")

    p = pressure_from_altitude_hpa(altitude_m)
    esat = saturation_vapor_pressure_hpa(T)
    Ea = esat * RH / 100.0

    out = {k: np.full(T.shape, np.nan) for k in ["CH","CE","rho","ea","zeta","ustar","Qh","Qe"]}

    for idx in np.ndindex(T.shape):
        Tr, Ur, ea = T[idx], U[idx], Ea[idx]
        if not np.isfinite(Tr) or not np.isfinite(Ur) or not np.isfinite(ea):
            continue
        if Ur < 0:
            continue

        rho = moist_air_density_kgm3(Tr, ea, p)
        if not np.isfinite(rho) or rho <= 0:
            continue
        nu = 1.983e-5 / rho

        L_mo = 9999.0
        CH = CE = zeta = ustar = Qh = Qe = np.nan

        for _ in range(max_iter):
            zeta = z_m / L_mo
            if zeta >= 5:
                zeta = 5.0

            psi_m, psi_h = stability_correction(zeta)
            denom_m = np.log(z_m / z0m) + psi_m
            if not np.isfinite(denom_m) or denom_m == 0:
                break

            ustar = (KAPPA * Ur) / denom_m
            Re = (ustar * z0m) / nu
            if not np.isfinite(Re) or Re <= 0:
                break

            lRe = np.log(Re)
            z0h = np.exp(0.317 - 0.565*lRe - 0.183*lRe*lRe + np.log(z0m))
            z0e = np.exp(0.396 - 0.512*lRe - 0.180*lRe*lRe + np.log(z0m))

            CH = KAPPA**2 / ((np.log(z_m/z0m)+psi_m) * (np.log(z_m/z0h)+psi_h))
            CE = KAPPA**2 / ((np.log(z_m/z0m)+psi_m) * (np.log(z_m/z0e)+psi_h))

            Qh = rho * CP * CH * Ur * (Tr - TS)
            es_surface = 6.1130
            Qe = ((rho * EPS * LV) / p) * CE * Ur * (ea - es_surface)

            # Reproduce MATLAB Monin-Obukhov update. Avoid division by zero.
            if Qh == 0 or not np.isfinite(Qh):
                break
            L_new = (ustar**3 * rho * CP * Tr) / (KAPPA * G * Qh)
            if not np.isfinite(L_new):
                break
            if abs(L_new - L_mo) < convergence_m:
                L_mo = L_new
                break
            L_mo = L_new

        out["CH"][idx] = CH
        out["CE"][idx] = CE
        out["rho"][idx] = rho
        out["ea"][idx] = ea
        out["zeta"][idx] = zeta
        out["ustar"][idx] = ustar
        out["Qh"][idx] = Qh
        out["Qe"][idx] = Qe

    return out


def katabatic_transfer(
    T_k,
    RH_pct,
    altitude_m,
    Kb=0.01,
    Pr=5.0,
    gamma=0.005,
    K=0.0004,
    pressure_mode="legacy_fixed",
    legacy_fixed_pressure_hpa=781.55,
):
    """Oerlemans-Grisogono / Munro-style katabatic parameterization.

    Parameters
    ----------
    pressure_mode : {"legacy_fixed", "altitude"}
        ``legacy_fixed`` reproduces the active manuscript-era
        ``ogparamterization.m``, which hard-coded p = 781.55 hPa.
        ``altitude`` uses the standard-atmosphere expression found in the
        archived 2021 version and in ogparamterization_withgam.m.

    Notes
    -----
    Wind speed does not enter this formulation in the legacy code.
    """
    T = np.asarray(T_k, dtype=float)
    RH = np.asarray(RH_pct, dtype=float)
    if T.shape != RH.shape:
        raise ValueError("T and RH must have matching shapes.")

    if pressure_mode == "legacy_fixed":
        p = float(legacy_fixed_pressure_hpa)
    elif pressure_mode == "altitude":
        p = pressure_from_altitude_hpa(altitude_m)
    else:
        raise ValueError("pressure_mode must be 'legacy_fixed' or 'altitude'.")

    esat = saturation_vapor_pressure_hpa(T)
    ea = esat * RH / 100.0
    es = float(saturation_vapor_pressure_hpa(np.array(TS)))

    rho = moist_air_density_kgm3(T, ea, p)
    Kkat = K * (T - TS) * np.sqrt(G / (273.0 * gamma * Pr))
    conductance = (Kb + Kkat) / 2.0

    Qh = rho * CP * conductance * (T - TS)
    Qe = (EPS * rho * LV / p) * conductance * (ea - es)

    bad = ~(np.isfinite(T) & np.isfinite(RH))
    for arr in (rho, ea, Kkat, Qh, Qe):
        arr[bad] = np.nan

    return {
        "rho": rho,
        "ea": ea,
        "Kkat": Kkat,
        "conductance": conductance,
        "Qh": Qh,
        "Qe": Qe,
    }


def fit_kb_zero_mean_bias(T_k, RH_pct, Qh_ec, Qe_ec, altitude_m, Pr=5.0, gamma=0.005,
                          pressure_mode="legacy_fixed"):
    """Fit Kb so mean modeled (Qh+Qe) equals mean EC (Qh+Qe).

    This directly implements the interpretation of the reviewer request that
    period-mean total turbulent flux should match EC.
    """
    # Evaluate at Kb=0 and Kb=1; the response is linear in Kb.
    y = np.asarray(Qh_ec, float) + np.asarray(Qe_ec, float)
    m0 = katabatic_transfer(T_k, RH_pct, altitude_m, Kb=0.0, Pr=Pr, gamma=gamma,
                            pressure_mode=pressure_mode)
    m1 = katabatic_transfer(T_k, RH_pct, altitude_m, Kb=1.0, Pr=Pr, gamma=gamma,
                            pressure_mode=pressure_mode)
    c = m0["Qh"] + m0["Qe"]
    x = (m1["Qh"] + m1["Qe"]) - c
    mask = np.isfinite(y) & np.isfinite(c) & np.isfinite(x)
    if mask.sum() == 0 or np.nanmean(x[mask]) == 0:
        return np.nan
    return (np.nanmean(y[mask]) - np.nanmean(c[mask])) / np.nanmean(x[mask])


def fit_kb_least_squares(T_k, RH_pct, Qh_ec, Qe_ec, altitude_m, Pr=5.0, gamma=0.005,
                         pressure_mode="legacy_fixed", constrain_nonnegative=True):
    """Least-squares Kb fit to EC total turbulent flux Qh+Qe."""
    y = np.asarray(Qh_ec, float) + np.asarray(Qe_ec, float)
    m0 = katabatic_transfer(T_k, RH_pct, altitude_m, Kb=0.0, Pr=Pr, gamma=gamma,
                            pressure_mode=pressure_mode)
    m1 = katabatic_transfer(T_k, RH_pct, altitude_m, Kb=1.0, Pr=Pr, gamma=gamma,
                            pressure_mode=pressure_mode)
    c = m0["Qh"] + m0["Qe"]
    x = (m1["Qh"] + m1["Qe"]) - c
    mask = np.isfinite(y) & np.isfinite(c) & np.isfinite(x)
    if mask.sum() == 0 or np.sum(x[mask]**2) == 0:
        return np.nan
    kb = np.sum(x[mask] * (y[mask] - c[mask])) / np.sum(x[mask]**2)
    if constrain_nonnegative:
        kb = max(0.0, float(kb))
    return float(kb)
