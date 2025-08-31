import pandas as pd, numpy as np, pickle


def _icis_from_row(row, nclicks_col, max_icis=64):
    # Build ICI list ICI1..ICI{n-1}
    #nClicks_col: use with or without ornamentation
    n = int(row[nclicks_col])
    icis = []
    for k in range(1, min(max_icis, n)):
        c = f'{"ICI"}{k}'
        if c not in row.index:
            break
        v = row[c]
        if pd.isna(v):
            break
        icis.append(float(v))
    return icis

def _recompute_without_terminal_click(icis):
    #For ornamented codas: compute icis, click times and duration without final click. 

    if not isinstance(icis, (list, tuple)) or len(icis) == 0:
        return icis, None, None, None
    ct = [0.0]
    s = 0.0
    for x in icis:
        s += float(x)
        ct.append(s)
    if len(ct) <= 2:
        return icis, ct, len(ct), s
    ct_wo = ct[:-1]
    icis_wo = [ct_wo[i+1] - ct_wo[i] for i in range(len(ct_wo)-1)]
    dur_wo = ct_wo[-1] if len(ct_wo) else float('nan')
    return icis_wo, ct_wo, len(ct_wo), dur_wo



def load_dialogues(csv_path, ornaments_p):
    #Create a df with a single tuple with icis, add adjusted values when ornamented
    df = pd.read_csv(csv_path)
    icis_list = [_icis_from_row(r, nclicks_col="nClicks") for _, r in df.iterrows()]
    duration = df["Duration"]
    click_times = [([0.0] + np.cumsum(ici).tolist()) if len(ici)>0 else None for ici in icis_list]

    out = pd.DataFrame({
        'coda_id': df.index.astype(int).astype(str).map(lambda i: f'D2_{i}'),
        'whale_id': df.get('Whale', pd.Series([None]*len(df))),
        'exchange_id': df.get('REC', pd.Series([None]*len(df))),
        'click_times': click_times,
        'icis': icis_list,
        'n_clicks': df['nClicks'].astype(int),
        'duration': duration,
        'TsTo': df.get('TsTo', pd.Series([None]*len(df)))
    })

    #ornamentation
    with open(ornaments_p, 'rb') as f:
        orn = pickle.load(f)
    if len(orn) != len(out):
        raise ValueError('ornaments length mismatch')
    out['ornament_flag'] = pd.Series(orn, dtype=int).values
    # Provide *_wo_ornament convenience columns that drop a terminal ornament click.
    icis_wo, ct_wo, n_wo, dur_wo = [], [], [], []
    for icis, flag in zip(out['icis'], out['ornament_flag']):
        if int(flag) == 1 and isinstance(icis, list) and len(icis) > 0:
            ic_w, ct_w, n_w, du_w = _recompute_without_terminal_click(icis)
            icis_wo.append(ic_w)
            ct_wo.append(ct_w)
            n_wo.append(n_w if n_w is not None else 0)
            dur_wo.append(du_w if du_w is not None else float('nan'))
        else:
            # No ornament indicated; keep originals
            icis_wo.append(icis)
            ct_wo.append([0.0] + np.cumsum(icis).tolist() if isinstance(icis, list) and len(icis)>0 else None)
            n_wo.append((len(icis) + 1) if isinstance(icis, list) else 0)
            dur_wo.append(float(np.sum(icis)) if isinstance(icis, list) and len(icis)>0 else float('nan'))
    out['icis_wo_ornament'] = icis_wo
    out['click_times_wo_ornament'] = ct_wo
    out['n_clicks_wo_ornament'] = n_wo
    out['duration_wo_ornament'] = dur_wo

    return out
