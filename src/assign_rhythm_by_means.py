import pickle
import numpy as np
import pandas as pd

def assign_rhythm_by_means(df, mean_codas_path="data/mean_codas.p"):
    """
    Assign a rhythm type to each row by comparing normalized cumulative click times
    against precomputed mean rhythm templates from Nature paper.

    Output: dataframe with an extra column 'rhythm_type'
    """
    
    # Load mean templates
    with open(mean_codas_path, "rb") as f:
        MEAN_CODAS = pickle.load(f)

    rhythm_types = []

    for _, row in df.iterrows():
        
        icis = row["icis_wo_ornament"]
        n_clicks = row["n_clicks_wo_ornament"]
        total_time = row["duration_wo_ornament"]
        click_times = row["click_times_wo_ornament"]

        

        #normalize click times to match mean_codas
        click_times_array = np.array(click_times, dtype=float)
        norm_cum = click_times_array / total_time

        # Find best matching template (must have same number of clicks) by mean square error
        best_index = -2
        best_score = float("inf")

        for i, template in enumerate(MEAN_CODAS):
            if len(template) != n_clicks:
                continue
            tmpl_arr = np.asarray(template, dtype=float)
            mse = float(np.mean((norm_cum - tmpl_arr) ** 2))
            if mse < best_score:
                best_score = mse
                best_index = i

        if best_index >= 0: #drops unknown
            rhythm_types.append(f"rh_{best_index+1:02d}")
        else:
            rhythm_types.append(None)  # mark as missing
    
    df["rhythm_type"] = pd.Series(rhythm_types, dtype="string")
    df = df[df["rhythm_type"].notna()].reset_index(drop=True)  # drop unknowns

    return df
