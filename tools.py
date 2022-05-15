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


def killall():
    import os
    cmdout = os.popen("ps aux | grep 'python main.py'").read()
    print(cmdout)
    pid = [s.split("  ")[4] for s in cmdout.split("\n")[0:-1]]
    for p in pid:
        try:
            print(os.popen(f"kill -9 {p}").read())
        except:
            print(f"kill pid:{p} error")


if __name__ == "__main__":
    x = """
   ROUND 0, 5YEAR: loss: 0.6081 - auc: 0.7170 - val_loss: 0.6089 - val_auc: 0.6907
   ROUND 1, 1YEAR: loss: 0.5864 - auc_1: 0.7148 - val_loss: 0.5993 - val_auc_1: 0.6965
   ROUND 2, 1YEAR: loss: 0.5952 - auc_1: 0.7224 - val_loss: 0.6304 - val_auc_1: 0.6804
   ROUND 3, 1YEAR: loss: 0.6073 - auc_1: 0.7254 - val_loss: 0.6027 - val_auc_1: 0.6903
   ROUND 4, 1YEAR: loss: 0.5978 - auc_1: 0.7177 - val_loss: 0.6299 - val_auc_1: 0.6887
   """
    run(x)
    # killall()
