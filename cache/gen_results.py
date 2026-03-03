"""
gen_results.py — Aggregate experiment results for GBC book chapters.

Reads numbers files from hopper/code results (already copied to cache/),
produces summary DataFrames, and saves CSVs for easy loading in chapters.

Usage:
    python cache/gen_results.py
"""

import os
import re
import pandas as pd
import numpy as np

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_numbers_file(path):
    """Parse a key = value numbers file, ignoring comments and non-numeric values."""
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if val in ("NA", "infeasible"):
                results[key] = np.nan
            else:
                try:
                    results[key] = float(val)
                except ValueError:
                    continue
    return results


def build_motorcycle_summary():
    """Build motorcycle benchmark summary from moto_ens256_numbers.txt."""
    nums = parse_numbers_file(os.path.join(CACHE_DIR, "moto_ens256_numbers.txt"))
    rows = [
        {
            "Method": "hetGP (Matern-5/2)",
            "RMSE": 23.81,
            "RMSE_SE": 0.49,
            "CRPS": nums.get("hetgp_crps", 12.56),
            "CRPS_SE": nums.get("hetgp_crps_se", 0.24),
            "Coverage_90": 0.875,
        },
        {
            "Method": "GBC (IQN, K=5 ensemble)",
            "RMSE": nums.get("gbc_ens_rmse", 23.88),
            "RMSE_SE": nums.get("gbc_ens_rmse_se", 0.57),
            "CRPS": nums.get("gbc_ens_crps", 12.49),
            "CRPS_SE": nums.get("gbc_ens_crps_se", 0.29),
            "Coverage_90": nums.get("gbc_ens_cov", 0.83),
        },
    ]
    return pd.DataFrame(rows)


def build_bgp_summary():
    """Build BGP benchmark summary from bgp_numbers.txt."""
    nums = parse_numbers_file(os.path.join(CACHE_DIR, "bgp_numbers.txt"))
    rows = []
    for d in [2, 3, 4]:
        rows.append(
            {
                "d": d,
                "GP_RMSE": nums.get(f"bgp_d{d}_gp_rmse"),
                "GP_RMSE_SE": nums.get(f"bgp_d{d}_gp_rmse_se"),
                "GP_CRPS": nums.get(f"bgp_d{d}_gp_crps"),
                "GP_CRPS_SE": nums.get(f"bgp_d{d}_gp_crps_se"),
                "GBC_RMSE": nums.get(f"bgp_d{d}_gbc_rmse"),
                "GBC_RMSE_SE": nums.get(f"bgp_d{d}_gbc_rmse_se"),
                "GBC_CRPS": nums.get(f"bgp_d{d}_gbc_crps"),
                "GBC_CRPS_SE": nums.get(f"bgp_d{d}_gbc_crps_se"),
                "GBC_Coverage_90": nums.get(f"bgp_d{d}_gbc_cov"),
            }
        )
    return pd.DataFrame(rows)


def build_flowers_summary():
    """Build Flowers benchmark summary (Phantom, Star, AMHV, Michalewicz)."""
    nums = parse_numbers_file(os.path.join(CACHE_DIR, "flowers_numbers.txt"))
    phant = parse_numbers_file(os.path.join(CACHE_DIR, "augiqn_phantom_numbers.txt"))
    star = parse_numbers_file(os.path.join(CACHE_DIR, "augiqn_star_numbers.txt"))

    rows = [
        # Phantom
        {
            "Dataset": "Phantom",
            "Method": "MJGP",
            "RMSE": nums.get("phantom_mjgp_rmse"),
            "CRPS": nums.get("phantom_mjgp_crps"),
        },
        {
            "Dataset": "Phantom",
            "Method": "Plain GBC",
            "RMSE": nums.get("phantom_gbc_v5_rmse"),
            "CRPS": nums.get("phantom_gbc_v5_crps"),
        },
        {
            "Dataset": "Phantom",
            "Method": "Aug-IQN",
            "RMSE": nums.get("phantom_aug_rmse"),
            "CRPS": nums.get("phantom_aug_crps"),
        },
        # Star
        {
            "Dataset": "Star",
            "Method": "MJGP",
            "RMSE": nums.get("star_mjgp_rmse"),
            "CRPS": nums.get("star_mjgp_crps"),
        },
        {
            "Dataset": "Star",
            "Method": "Plain GBC",
            "RMSE": nums.get("star_gbc_v5_rmse"),
            "CRPS": nums.get("star_gbc_v5_crps"),
        },
        {
            "Dataset": "Star",
            "Method": "Aug-IQN",
            "RMSE": nums.get("star_aug_rmse"),
            "CRPS": nums.get("star_aug_crps"),
        },
        # AMHV
        {
            "Dataset": "AMHV",
            "Method": "MJGP",
            "RMSE": nums.get("amhv_mjgp_rmse"),
            "CRPS": nums.get("amhv_mjgp_crps"),
        },
        {
            "Dataset": "AMHV",
            "Method": "Plain GBC",
            "RMSE": nums.get("amhv_gbc_default_rmse"),
            "CRPS": nums.get("amhv_gbc_default_crps"),
        },
    ]
    return pd.DataFrame(rows)


def build_al_summary():
    """Build active learning summary from al_numbers.txt."""
    nums = parse_numbers_file(os.path.join(CACHE_DIR, "al_numbers.txt"))
    rows = [
        {
            "Benchmark": "Rocket LGBB",
            "Method": "GBC-AL",
            "RMSE": nums.get("gbc_alc_rmse", 0.00723),
            "RMSE_SE": nums.get("gbc_alc_rmse_se"),
        },
    ]
    return pd.DataFrame(rows)


def main():
    print("Aggregating results...")

    # Motorcycle
    moto = build_motorcycle_summary()
    moto.to_csv(os.path.join(CACHE_DIR, "summary_motorcycle.csv"), index=False)
    print(f"  motorcycle: {len(moto)} rows -> summary_motorcycle.csv")

    # BGP
    bgp = build_bgp_summary()
    bgp.to_csv(os.path.join(CACHE_DIR, "summary_bgp.csv"), index=False)
    print(f"  BGP: {len(bgp)} rows -> summary_bgp.csv")

    # Flowers (Phantom, Star, AMHV)
    flowers = build_flowers_summary()
    flowers.to_csv(os.path.join(CACHE_DIR, "summary_flowers.csv"), index=False)
    print(f"  Flowers: {len(flowers)} rows -> summary_flowers.csv")

    # Active learning
    al = build_al_summary()
    al.to_csv(os.path.join(CACHE_DIR, "summary_al.csv"), index=False)
    print(f"  AL: {len(al)} rows -> summary_al.csv")

    print("Done.")


if __name__ == "__main__":
    main()
