#!/usr/bin/env python
"""A repair, not just a diagnosis: calibrate A_f per unit instead of against one shared threshold.

The artifact arises because one global threshold is applied to a statistic whose sampling variance
differs per feature (a feature active on 5 loci has a far noisier weighted median than one active on
21). The fix is to studentise: bootstrap each unit's own per-locus scores to get SE(A_f), and
threshold z_f = A_f / SE_f instead of A_f, with the null directions treated identically. Units are
then compared on a scale where coverage has already been accounted for.

Reports the tail fractions before and after, and whether calibration removes the coverage
dependence. Reads only cached artifacts. Emits `calibrate_layer{L}.json` in the artifacts directory.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/24_calibrate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dna_interp import causal, controls  # noqa: E402
from dna_interp.data import SatmutElement, locus_groups  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402

B = 200
rng = np.random.default_rng(0)


def studentise(df):
    """A_f and its bootstrap SE, per unit, over that unit's own loci."""
    out = {}
    for f, g in df.groupby("feature_id"):
        v, w = g.a_f.to_numpy(), g.mass.to_numpy()
        n = len(v)
        a = causal.weighted_median(v, w)
        if n < 3:
            out[int(f)] = (a, np.nan, np.nan, n); continue
        bs = np.empty(B)
        for b in range(B):
            i = rng.integers(0, n, n)
            bs[b] = causal.weighted_median(v[i], w[i])
        se = bs.std(ddof=1)
        out[int(f)] = (a, se, a / se if se > 1e-9 else np.nan, n)
    return pd.DataFrame(out, index=["A_f", "se", "z", "n_loci"]).T


def frac(x, hi, lo):
    x = x[~np.isnan(x)]
    return float((x > hi).mean()), float((x < lo).mean())


def main():
    cfg = load_config(sys.argv[1:])
    art = Path(cfg.paths["artifacts"])
    L = int((art / "chosen_layer.txt").read_text().strip())

    d = np.load(art / "satmut.npz", allow_pickle=True)
    el = [SatmutElement(str(i), str(s), np.asarray(m), np.asarray(sg), np.asarray(mx))
          for i, s, m, sg, mx in zip(d["ids"], d["seqs"], d["E_mean"], d["E_signed"], d["E_max"])]
    g = locus_groups(el)
    real = controls.collapse_to_loci(
        pd.read_parquet(art / f"align_topk_seed{cfg.seed}_layer{L}.parquet"), g)
    null = controls.collapse_to_loci(pd.read_parquet(art / f"align_null_layer{L}.parquet"), g)

    R, N = studentise(real), studentise(null)

    print("=== BEFORE: threshold on A_f against the shared null ===")
    hi, lo = np.nanpercentile(N.A_f, 95), np.nanpercentile(N.A_f, 5)
    up, dn = frac(R.A_f.to_numpy(), hi, lo)
    print(f"  thresholds {lo:+.4f} / {hi:+.4f} | above {up:.3f}  below {dn:.3f}  (excess {up-dn:+.3f})")
    # Units on <3 loci get no bootstrap SE, so z is NaN for them and AFTER is computed without
    # them. They are the sparsest, most extreme units, so compare BEFORE on the same set too.
    has_se = ~R.z.isna()
    mup, mdn = frac(R.A_f[has_se].to_numpy(), hi, lo)
    print(f"  same {int(has_se.sum())} units that have an SE (>=3 loci): "
          f"above {mup:.3f}  below {mdn:.3f}  ({int((~has_se).sum())} units on <3 loci excluded)")

    print("\n=== AFTER: threshold on z = A_f / bootstrap SE, against the same null studentised ===")
    zhi, zlo = np.nanpercentile(N.z, 95), np.nanpercentile(N.z, 5)
    zup, zdn = frac(R.z.to_numpy(), zhi, zlo)
    print(f"  thresholds {zlo:+.3f} / {zhi:+.3f} | above {zup:.3f}  below {zdn:.3f}  (excess {zup-zdn:+.3f})")

    print("\n=== does calibration remove the coverage dependence? ===")
    for lab, col in (("A_f  ", "A_f"), ("z    ", "z")):
        v = R[col].to_numpy(); nl = R.n_loci.to_numpy()
        ok = ~np.isnan(v)
        print(f"  Spearman(n_loci, |{lab.strip()}|) = "
              f"{causal.spearman_rho(nl[ok], np.abs(v[ok])):+.3f}")
    rho_A_matched = causal.spearman_rho(R.n_loci[has_se], np.abs(R.A_f[has_se]))
    print(f"  Spearman(n_loci, |A_f|) on the same units as z = {rho_A_matched:+.3f}")

    # 1-2 loci is split off from 3-8 because those units have no SE: every bin from 3 up then
    # compares A_f and z on exactly the same units.
    print("\n=== tail fraction by coverage, before (A_f) and after (z) calibration ===")
    print(f"  {'n_loci':>7} {'units':>6} {'sd(A_f)':>8}   A_f above / below   z above / below")
    by_cov = []
    for a, b in [(1, 2), (3, 8), (9, 12), (13, 16), (17, 19), (20, 21)]:
        m = R[(R.n_loci >= a) & (R.n_loci <= b)]
        if len(m) < 20: continue
        ua, wa = frac(m.A_f.to_numpy(), hi, lo)
        n_se = int(m.z.notna().sum())                 # units with a bootstrap SE (>=3 loci)
        uz, wz = frac(m.z.to_numpy(), zhi, zlo) if n_se else (float("nan"), float("nan"))
        by_cov.append({"bin": f"{a}-{b}", "n": len(m), "sd_A": float(m.A_f.std()),
                       "A_above": ua, "A_below": wa, "n_with_se": n_se,
                       "z_above": uz, "z_below": wz})
        print(f"  {a:>3}-{b:<3} {len(m):>6} {m.A_f.std():>8.4f}   {ua:>7.3f} / {wa:.3f}   "
              f"{uz:>7.3f} / {wz:.3f}")

    out = {"layer": L, "n_boot": B,
           "before": {"above": up, "below": dn},
           "before_same_units": {"above": mup, "below": mdn, "n_units": int(has_se.sum()),
                                 "n_excluded_lt3_loci": int((~has_se).sum()),
                                 "sd_A": float(R.A_f[has_se].std())},
           "after": {"above": zup, "below": zdn},
           "spearman_nloci_absA": causal.spearman_rho(R.n_loci, np.abs(R.A_f)),
           "spearman_nloci_absA_same_units": float(rho_A_matched),
           "spearman_nloci_absZ": float(causal.spearman_rho(
               R.n_loci[~R.z.isna()], np.abs(R.z[~R.z.isna()]))),
           "by_coverage": by_cov}
    (art / f"calibrate_layer{L}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {art / f'calibrate_layer{L}.json'}")


if __name__ == "__main__":
    main()
