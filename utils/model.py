"""
Conditional Convergence Model (CCM) — projection engine.

Core projection formula:
  Δln(rel_prod_t) = β × (lss_log − ln(rel_prod_t))

GDPPC projection:
  gdppc_t = gdppc_base
            × (rel_prod_t / rel_prod_base)      ← productivity convergence
            × exp(g_usa × Δt)                   ← US frontier growth
            × (wap_ratio_t / wap_ratio_base)     ← demographic dividend/drag

GDP (billions) = GDPPC × Population(thousands) / 1,000,000
"""

import numpy as np
import pandas as pd


# ─── Kernel regression ───────────────────────────────────────────────────────

def gaussian_kernel_estimate(
    gci_target: float,
    gci_data: np.ndarray,
    log_prod_data: np.ndarray,
    bandwidth: float,
) -> float:
    """Gaussian kernel regression: estimate LSS for a given GCI score."""
    weights = np.exp(-((gci_target - gci_data) ** 2) / (2 * bandwidth ** 2))
    total = weights.sum()
    if total < 1e-12:
        return float(np.mean(log_prod_data))
    return float(np.dot(weights, log_prod_data) / total)


def kernel_curve(
    gci_grid: np.ndarray,
    gci_data: np.ndarray,
    log_prod_data: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Evaluate kernel regression over a grid of GCI values."""
    return np.array([
        gaussian_kernel_estimate(x, gci_data, log_prod_data, bandwidth)
        for x in gci_grid
    ])


def reestimate_lss(kernel_df: pd.DataFrame, comp_df: pd.DataFrame, bandwidth: float) -> dict:
    """
    Re-estimate LSS for all countries using the given bandwidth.
    Returns dict: {country_code: lss_log}.
    Falls back to stored Excel value if GCI is missing.
    """
    valid = comp_df.dropna(subset=["gci", "log_prod"])
    gci_data  = valid["gci"].values.astype(float)
    prod_data = valid["log_prod"].values.astype(float)

    result = {}
    for code, row in kernel_df.iterrows():
        if row["gci"] is not None and not np.isnan(float(row["gci"]) if row["gci"] else float("nan")):
            lss = gaussian_kernel_estimate(float(row["gci"]), gci_data, prod_data, bandwidth)
        else:
            # No GCI → use stored Excel value
            lss = row["lss_log"] if row["lss_log"] is not None else np.nan
        result[code] = lss
    return result


# ─── Convergence projection ───────────────────────────────────────────────────

def _convergence_trajectory(
    log_rel_start: float,
    lss_log: float,
    beta: float,
    n_steps: int,
) -> np.ndarray:
    """
    Project log(productivity relative to USA) via discrete convergence.
    Returns array of length n_steps + 1 (includes starting value).
    """
    traj = np.empty(n_steps + 1)
    traj[0] = log_rel_start
    for i in range(1, n_steps + 1):
        traj[i] = traj[i - 1] + beta * (lss_log - traj[i - 1])
    return traj


def project_country(
    country_code: str,
    gdppc_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    wap_df: pd.DataFrame,
    rel_prod_df: pd.DataFrame,
    lss_log: float,
    beta: float,
    g_usa: float,
    base_year: int = 2024,
    end_year: int = 2050,
) -> pd.DataFrame | None:
    """
    Project GDPPC and GDP for a single country from base_year to end_year.

    Returns DataFrame with columns: gdppc, gdp_bn
    indexed by year (base_year … end_year).
    Returns None if required base data is missing.
    """
    try:
        gdppc_base   = float(gdppc_df.loc[country_code, base_year])
        wap_base     = float(wap_df.loc[country_code, base_year])
        pop_base     = float(pop_df.loc[country_code, base_year])
        rel_prod_raw = float(rel_prod_df.loc[country_code, base_year])
    except (KeyError, TypeError, ValueError):
        return None

    if not all(np.isfinite([gdppc_base, wap_base, pop_base, rel_prod_raw])):
        return None
    if rel_prod_raw <= 0 or wap_base <= 0 or pop_base <= 0:
        return None

    log_rel_base  = np.log(rel_prod_raw)
    wap_ratio_base = wap_base / pop_base
    n_steps        = end_year - base_year
    years          = list(range(base_year, end_year + 1))

    log_rel_traj = _convergence_trajectory(log_rel_base, lss_log, beta, n_steps)

    gdppc_proj = []
    gdp_proj   = []

    for i, yr in enumerate(years):
        # WAP/Pop ratio (uses Excel projections where available, else flat)
        wap_yr = _get_val(wap_df, country_code, yr, wap_base)
        pop_yr = _get_val(pop_df, country_code, yr, pop_base)
        wap_ratio_yr = wap_yr / pop_yr if pop_yr > 0 else wap_ratio_base

        gdppc = (
            gdppc_base
            * (np.exp(log_rel_traj[i]) / rel_prod_raw)   # productivity gain vs USA
            * np.exp(g_usa * i)                           # US frontier growth
            * (wap_ratio_yr / wap_ratio_base)             # demographic effect
        )
        gdp = gdppc * pop_yr / 1_000_000   # billions (GDPPC in USD, pop in thousands)

        gdppc_proj.append(gdppc)
        gdp_proj.append(gdp)

    return pd.DataFrame({"gdppc": gdppc_proj, "gdp_bn": gdp_proj}, index=pd.Index(years, name="year"))


def _get_val(df: pd.DataFrame, code: str, year: int, fallback: float) -> float:
    """Safe cell lookup with fallback."""
    try:
        v = df.loc[code, year]
        return float(v) if np.isfinite(float(v)) else fallback
    except (KeyError, TypeError, ValueError):
        return fallback


# ─── Regional aggregation ─────────────────────────────────────────────────────

def regional_gdp(
    region_codes: list[str],
    gdp_df: pd.DataFrame,
    years: list[int],
) -> pd.Series:
    """Sum GDP across a list of country codes for given years."""
    available = [c for c in region_codes if c in gdp_df.index]
    if not available:
        return pd.Series(0.0, index=years)
    sub = gdp_df.loc[available, [y for y in years if y in gdp_df.columns]]
    return sub.sum(axis=0).reindex(years)
