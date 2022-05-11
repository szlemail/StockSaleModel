import pandas as pd
import numpy as np


def run(x):
    data = x.split("\n")
    data = [d.split(" ") for d in data if d.strip() != ""]
    df = pd.DataFrame(data)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
            print(f"{df[c].mean():0.4f}", end=' ')
        except:
            print(df[c][0], end=' ')


if __name__ == "__main__":
    x = """
   ROUND 0, 5YEAR: val_sell_auc: 0.6791 - val_bsc_auc: 0.5714 - val_bso_auc: 0.5451 - val_bs_auc: 0.5631 - val_bsl_auc: 0.6194 - val_bgc2_auc: 0.6080 - val_bgc5_auc: 0.6578
   ROUND 1, 1YEAR: val_sell_auc: 0.6702 - val_bsc_auc: 0.5436 - val_bso_auc: 0.5383 - val_bs_auc: 0.5352 - val_bsl_auc: 0.6183 - val_bgc2_auc: 0.6165 - val_bgc5_auc: 0.6751
   ROUND 2, 1YEAR: val_sell_auc: 0.6831 - val_bsc_auc: 0.5744 - val_bso_auc: 0.5681 - val_bs_auc: 0.5766 - val_bsl_auc: 0.6510 - val_bgc2_auc: 0.6691 - val_bgc5_auc: 0.7388
   ROUND 3, 1YEAR: val_sell_auc: 0.6752 - val_bsc_auc: 0.5723 - val_bso_auc: 0.5632 - val_bs_auc: 0.5731 - val_bsl_auc: 0.6492 - val_bgc2_auc: 0.6870 - val_bgc5_auc: 0.7505
   ROUND 4, 1YEAR: val_sell_auc: 0.6794 - val_bsc_auc: 0.5699 - val_bso_auc: 0.5581 - val_bs_auc: 0.5750 - val_bsl_auc: 0.6466 - val_bgc2_auc: 0.6850 - val_bgc5_auc: 0.7606
   """
    run(x)
