import numpy as np, pandas as pd
from numpy.random import default_rng
from src.sw_loader import load_dialogues
from src.assign_rhythm_by_means import assign_rhythm_by_means
from src.entropies import entropy_of, conditional_entropy, entropy_rate_order1, entropy_rate_null

#use if calculating confidence intervals:
#from src.bootstrap import bootstrap_ci

if __name__ == "__main__":

    rng = default_rng(0)

    DIALOGUES_CSV = 'data/sperm-whale-dialogues.csv'
    ORNAMENTS_P   = 'data/ornaments.p'   
    MEAN_CODAS_P  = 'data/mean_codas.p'

    df = load_dialogues(DIALOGUES_CSV, ORNAMENTS_P)
    if 'exchange_id' in df.columns and 'TsTo' in df.columns:
        df = df.sort_values(['exchange_id','TsTo']).reset_index(drop=True)

    #Rhythm: assign via mean templates from Nature paper
    df = assign_rhythm_by_means(df, MEAN_CODAS_P)

    #Tempo typing: fixed 5 modes from Nature paper (seconds)
    TEMPO_CENTERS = np.array([0.33, 0.51, 0.80, 1.02, 1.26], dtype=float)

    #assign the nearest fixed center
    dur_wo = df["duration_wo_ornament"].to_numpy()
    df["tempo_type"] = np.abs(dur_wo[:, None] - TEMPO_CENTERS[None, :]).argmin(axis=1).astype(int)
    

    #Rubato

    dur_col = "duration_wo_ornament"
    # Sort by time within exchange
    df["_orig_order"] = np.arange(len(df))
    keys = ["exchange_id"]
    if "TsTo" in df.columns:
        keys.append("TsTo")
    keys.append("_orig_order")
    df = df.sort_values(keys, kind="mergesort").reset_index(drop=True)

    # Previous row (within the same the exchange)
    grp = df.groupby("exchange_id", sort=False)
    prev_whale  = grp["whale_id"].shift(1)
    prev_rhythm = grp["rhythm_type"].shift(1)
    prev_tempo  = grp["tempo_type"].shift(1)
    prev_dur    = grp[dur_col].shift(1)

    # Valid only if the immediately previous coda is same whale & same (rhythm, tempo)
    adj_mask = (
        prev_whale.notna()
        & (prev_whale == df["whale_id"])
        & (prev_rhythm == df["rhythm_type"])
        & (prev_tempo  == df["tempo_type"])
    )

    rubato_diff = np.where(adj_mask, df[dur_col].to_numpy() - prev_dur.to_numpy(), np.nan)

    # Discretize: -1 (decreasing), 0 (stable within threshold), +1 (increasing)
    RUBATO_ABS_THRESH = 0.05  # seconds. Chosen based on figure 8 in supp. discussion (Nature)
    rub_bin = np.full(len(df), np.nan)
    rub_bin[adj_mask & (rubato_diff < -RUBATO_ABS_THRESH)] = -1
    rub_bin[adj_mask & (np.abs(rubato_diff) <= RUBATO_ABS_THRESH)] =  0
    rub_bin[adj_mask & (rubato_diff >  RUBATO_ABS_THRESH)] =  1

    df["rubato_scalar"] = rubato_diff
    df["rubato_bin"]    = pd.array(rub_bin, dtype="Int8") 
    df.drop(columns="_orig_order", inplace=True)

    # Define Unit = (rhythm_type, tempo_type)
    df["rhythm_type"] = df["rhythm_type"].astype("string")
    df["tempo_type"]  = pd.to_numeric(df["tempo_type"], errors="coerce").fillna(-1).astype(int)
    df["unit"] = list(zip(df["rhythm_type"], df["tempo_type"]))

    #for calculations which only use codas with valid rubato
    mask_rub = df["rubato_bin"].notna()

    #Use line below to calculate with all 4 features as unit
    #df["unit"] = list(zip(df["rhythm_type"], df["tempo_type"], df["rubato_bin"], df["ornament_flag"]))

    #save df
    df.to_csv("data/dialogues_with_units.csv", index=False)

    #Compute metrics
    H_unit, _ = entropy_of(df["unit"])
    H_unit_rub, _  = entropy_of(df.loc[mask_rub, "unit"])
    H_u_given_rub = conditional_entropy(df.loc[mask_rub, "unit"], df.loc[mask_rub, "rubato_bin"].astype(int))
    H_u_given_orn = conditional_entropy(df["unit"], df["ornament_flag"])
    H_u_given_both = conditional_entropy(df.loc[mask_rub, "unit"], list(zip(df.loc[mask_rub, "rubato_bin"].astype(int), df.loc[mask_rub, "ornament_flag"])))
    H_rate, H_rate_raw = entropy_rate_order1(df["unit"].tolist(), df["exchange_id"].tolist())
    

    I_u_rub  = H_unit_rub - H_u_given_rub
    I_u_orn  = H_unit - H_u_given_orn
    I_u_both = H_unit_rub - H_u_given_both

    #Prints
    print(f'Codas: {len(df)} | Rhythms: {df["rhythm_type"].nunique()} | Tempo types: {df["tempo_type"].nunique()}')
    print(f'H(Unit): {H_unit:.3f} bits')
    print(f'Only for codas with valid rubato H(Unit): {H_unit_rub:.3f} bits')
    print(f'H(Unit|rubato): {H_u_given_rub:.3f}  |  I(Unit; rubato): {I_u_rub:.3f} bits')
    print(f'H(Unit|ornamentation): {H_u_given_orn:.3f}  |  I(Unit; ornamentation): {I_u_orn:.3f} bits')
    print(f'H(Unit|rubato, ornamentation): {H_u_given_both:.3f}  |  I(Unit; rubato, ornamentation): {I_u_both:.3f} bits')
    print(f"Entropy rate observed(order-1): {H_rate_raw:.3f} bits")

    obs_entropy_null, (lo, hi) = entropy_rate_null(df, rng=rng)
    print(f"Entropy-rate under null hypothesis (mean): {obs_entropy_null:.3f} | null 95% [{lo:.3f}, {hi:.3f}]")
    mean_dur = float(df["duration_wo_ornament"].mean())
    print(f"Mean duration: {mean_dur:.3f}s")
    print(f"Average information/Surprisal per second: {entropy_of(df['unit'])[1]/mean_dur:.2f} bits/s")

    # CIs for entropies/conditionals. In the end I did not use these calculations, based on an 
    # assumption that coda choices are not independent

    # # H(Unit) on FULL data
    # m_H_full, lo_H_full, hi_H_full = bootstrap_ci(df, lambda d: entropy_of(d['unit'])[1], rng=rng)

    # # H(Unit | ornament) on FULL data
    # m_H_given_orn, lo_H_given_orn, hi_H_given_orn = bootstrap_ci(
    #     df, lambda d: conditional_entropy(d['unit'], d['ornament_flag']), rng=rng
    # )

    

    # # H(Unit) on RUBATO-VALID subset
    # m_H_rub_base, lo_H_rub_base, hi_H_rub_base = bootstrap_ci(
    #     df, lambda d: (lambda m: entropy_of(d.loc[m,'unit'])[1])(d['rubato_bin'].notna()), rng=rng
    # )

    # # H(Unit | rubato) on RUBATO-VALID subset
    # m_H_given_rub, lo_H_given_rub, hi_H_given_rub = bootstrap_ci(
    #     df, lambda d: (lambda m: conditional_entropy(
    #         d.loc[m,'unit'],
    #         d.loc[m,'rubato_bin'].astype(int)
    #     ))(d['rubato_bin'].notna()), rng=rng
    # )

    # # H(Unit | rubato, ornament) on RUBATO-VALID subset
    # m_H_given_rub_orn, lo_H_given_rub_orn, hi_H_given_rub_orn = bootstrap_ci(
    #     df, lambda d: (lambda m: conditional_entropy(
    #         d.loc[m,'unit'],
    #         list(zip(d.loc[m,'rubato_bin'].astype(int), d.loc[m,'ornament_flag']))
    #     ))(d['rubato_bin'].notna()), rng=rng
    # )

    # #Mutual information

    # CI for MI directly (accounts for covariance inside each resample)
    # mi_ornament_mean, mi_ornament_lo, mi_ornament_hi = bootstrap_ci(
    #     df, lambda d: entropy_of(d['unit'])[1] - conditional_entropy(d['unit'], d['ornament_flag']), rng=rng
    # )
    
    # # I(Unit; rubato) on RUBATO-VALID subset
    # m_I_rubato_mean, mi_I_rubato_lo, mi_I_rubato_hi = bootstrap_ci(
    #     df, lambda d: (lambda m:
    #         entropy_of(d.loc[m,'unit'])[1]
    #         - conditional_entropy(d.loc[m,'unit'], d.loc[m,'rubato_bin'].astype(int))
    #     )(d['rubato_bin'].notna()), rng=rng
    # )

    # # I(Unit; rubato, ornament) on RUBATO-VALID subset
    # m_I_rub_orn, lo_I_rub_orn, hi_I_rub_orn = bootstrap_ci(
    #     df, lambda d: (lambda m:
    #         entropy_of(d.loc[m,'unit'])[1]
    #         - conditional_entropy(
    #             d.loc[m,'unit'],
    #             list(zip(d.loc[m,'rubato_bin'].astype(int), d.loc[m,'ornament_flag']))
    #         )
    #     )(d['rubato_bin'].notna()), rng=rng
    # )


    # print("CIs:")
    # print(f"  H(Unit):              {m_H_full:.3f}  [{lo_H_full:.3f}, {hi_H_full:.3f}]")
    # print(f"  H(Unit|ornamentation):        {m_H_given_orn:.3f}  [{lo_H_given_orn:.3f}, {hi_H_given_orn:.3f}]")
    # print(f"  H(Unit) [rubato subset]:      {m_H_rub_base:.3f}  [{lo_H_rub_base:.3f}, {hi_H_rub_base:.3f}]")
    # print(f"  H(Unit|rubato):               {m_H_given_rub:.3f}  [{lo_H_given_rub:.3f}, {hi_H_given_rub:.3f}]")
    # print(f"  H(Unit|rubato, ornamentation): {m_H_given_rub_orn:.3f}  [{lo_H_given_rub_orn:.3f}, {hi_H_given_rub_orn:.3f}]")
    # print(f"  I(Unit; ornamentation, rubato):   {m_I_rub_orn:.3f}  [{lo_I_rub_orn:.3f}, {hi_I_rub_orn:.3f}]")
    # print(f"  I(Unit; ornamentation):       {mi_ornament_mean:.3f}  [{mi_ornament_lo:.3f}, {mi_ornament_hi:.3f}]")
    # print(f"  I(Unit; rubato):       {m_I_rubato_mean:.3f}  [{mi_I_rubato_lo:.3f}, {mi_I_rubato_hi:.3f}]")
