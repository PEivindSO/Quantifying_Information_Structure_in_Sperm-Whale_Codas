import numpy as np
from collections import Counter, defaultdict
import pandas as pd


def _is_valid_unit(u):
    try:
        rt = u[0] if isinstance(u, (list, tuple)) else u
        return (rt is not None) and (str(rt) != "rh_unk")
    except Exception:
        return False



def _to_hashable(x):
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    if isinstance(x, (list, tuple)):
        return tuple(_to_hashable(xx) for xx in x)
    return x

def entropy_from_counts(counts):
    N = int(sum(counts))
    if N <= 0:
        return 0.0, 0.0
    ps = np.array([c / N for c in counts if c > 0], float)
    H_raw = -np.sum(ps * np.log2(ps))
    K = len(ps)
    H_mm = H_raw + (K - 1) / (2 * N * np.log(2))
    return float(H_mm), float(H_raw)


def entropy_of(labels, ignore_rh_unk=True):
    # Filter out rh_unk-labeled units if requested
    clean = []
    for l in labels:
        if ignore_rh_unk and not _is_valid_unit(l):
            continue
        clean.append(_to_hashable(l))
    return entropy_from_counts(Counter(clean).values())



def conditional_entropy(xs, ys, ignore_rh_unk=True, dropna_y=True):
    xs_list = list(xs)
    ys_list = list(ys)

    # Drop rows with missing Y (NaN / <NA> / None)
    if dropna_y:
        mask = [not pd.isna(y) for y in ys_list]
        xs_list = [x for x, m in zip(xs_list, mask) if m]
        ys_list = [y for y, m in zip(ys_list, mask) if m]

    # Optionally drop rh_unk rows
    if ignore_rh_unk:
        mask = [_is_valid_unit(x) for x in xs_list]
        xs_list = [x for x, m in zip(xs_list, mask) if m]
        ys_list = [y for y, m in zip(ys_list, mask) if m]

    N = len(xs_list)
    if N == 0:
        return 0.0

    # Group by Y and weight per-group entropy by p(Y)
    
    buckets = defaultdict(list)
    for x, y in zip(xs_list, ys_list):
        buckets[y].append(_to_hashable(x))

    Hc = 0.0
    for y, bucket in buckets.items():
        ny = len(bucket)
        if ny == 0:
            continue
        H_y, _ = entropy_from_counts(Counter(bucket).values())
        Hc += (ny / N) * H_y
    return float(Hc)

def entropy_rate_order1(units, groups=None):
    #Order-1 entropy rate H(U_t | U_{t-1}) without bridging across group boundaries.
    trans = defaultdict(Counter)
    total = 0

    if groups is None:
        seq = list(units)
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a][b] += 1
            total += 1
        
    else:
        # accumulate transitions separately within each group
        prev_g = object()
        buf = []
        for u, g in zip(units, groups):
            if g != prev_g:
                # flush previous group
                if len(buf) >= 2:
                    for a, b in zip(buf[:-1], buf[1:]):
                        trans[a][b] += 1
                        total += 1
                buf = [u]
                prev_g = g
            else:
                buf.append(u)
        if len(buf) >= 2:
            for a, b in zip(buf[:-1], buf[1:]):
                trans[a][b] += 1
                total += 1

    if total == 0:
        return 0.0, 0.0

    H = 0.0
    H_raw = 0.0
    for a, row in trans.items():
        counts = list(row.values())
        p_a = sum(counts) / total
        H_row, H_row_raw = entropy_from_counts(counts)
        H += p_a * H_row
        H_raw += p_a * H_row_raw
    return float(H), float(H_raw)


def entropy_rate_null(df, R=200, rng=None, group_col="exchange_id"):
    #Order-1 entropy rate after shuffling codas within each exchange

    # Build null values by shuffling *within exchanges* (if available)
    null_vals = []

    if group_col in df.columns and df[group_col].notna().any():
        for _ in range(R):
            units_shuf = []
            groups_shuf = []
            for gid, sub in df.groupby(group_col, sort=False):
                arr = sub["unit"].to_numpy(copy=True)
                rng.shuffle(arr)
                units_shuf.extend(arr.tolist())
                groups_shuf.extend([gid] * len(arr))
            null_vals.append(
                entropy_rate_order1(units_shuf, groups=groups_shuf)[1]
            )
    else:
        arr = df["unit"].to_numpy(copy=True)
        for _ in range(R):
            rng.shuffle(arr)
            null_vals.append(
                entropy_rate_order1(arr.tolist(), groups=None)[1]
            )

    null_vals = np.asarray(null_vals, dtype=float)
    lo, hi = np.quantile(null_vals, [0.025, 0.975])
    null_mean = float(null_vals.mean())
    lo = float(lo); hi = float(hi)
    return null_mean, (lo, hi)
